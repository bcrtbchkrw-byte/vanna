"""
Trading Environment for Reinforcement Learning.

Uses REAL historical data from *_rl.parquet files (enriched).
40 market features + 7 position features = 47 total features.
"""
from typing import Any, SupportsFloat, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.logger import get_logger

logger = get_logger()


class TradingEnvironment(gym.Env):
    """
    Gymnasium environment using REAL historical data.
    
    Loads data from *_1min_rl.parquet files (enriched with ML outputs).
    
    State: 47 features (40 market + 7 position)
    Actions: 0=HOLD, 1=OPEN, 2=CLOSE, 3=INCREASE, 4=DECREASE
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    # Market features from normalized *_rl.parquet (63 total)
    # NO OHLC, NO timestamp - all normalized/scaled
    MARKET_FEATURES = [
        # Time (6) - already normalized [-1, 1]
        'sin_time', 'cos_time', 'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy',
        # VIX (8) - ratios and z-scores, comparable across symbols
        'vix_ratio', 'vix_in_contango', 'vix_change_1d', 'vix_change_5d',
        'vix_percentile', 'vix_zscore', 'vix_norm', 'vix3m_norm',
        # Regime (1)
        'regime',
        # Options (3) - normalized
        'options_iv_atm', 'options_put_call_ratio', 'options_volume_norm',
        # Price (5) - returns/ratios, already normalized
        'return_1m', 'return_5m', 'volatility_20', 'momentum_20', 'range_pct',
        # Greeks (7) - already scale-invariant
        'delta', 'gamma', 'theta', 'vega', 'vanna', 'charm', 'volga',
        # ML Regime outputs (4)
        'regime_ml', 'regime_adj_position', 'regime_adj_delta', 'regime_adj_dte',
        # ML DTE outputs (2) - normalized
        'dte_confidence', 'optimal_dte_norm',
        # ML Trade outputs (1)
        'trade_prob',
        # Binary signals (5) - 0/1
        'signal_high_prob', 'signal_low_vol', 'signal_crisis',
        'signal_contango', 'signal_backwardation',
        # Earnings features (4)
        'days_to_earnings', 'is_earnings_week', 'is_earnings_month', 'earnings_iv_multiplier',
        # Daily features injected from 1day (17) - uses YESTERDAY's data (no lookahead!)
        'day_sma_200', 'day_sma_50', 'day_sma_20',
        'day_price_vs_sma200', 'day_price_vs_sma50',
        'day_rsi_14',
        'day_atr_14', 'day_atr_pct',
        'day_bb_position',
        'day_macd', 'day_macd_hist',
        'day_above_sma200', 'day_above_sma50',
        'day_sma_50_200_ratio',
        'day_days_to_earnings', 'day_is_earnings_week', 'day_earnings_iv_boost',
    ]
    
    # Position features added at runtime (7)
    POSITION_FEATURES = [
        'pnl_pct', 'days_held', 'position_flag', 'capital_ratio',
        'trade_count', 'bid_ask_spread', 'market_open'
    ]
    
    N_MARKET_FEATURES = 63  # Updated: 46 + 17 daily
    N_POSITION_FEATURES = 7
    N_FEATURES = 70  # 63 market + 7 position
    
    def __init__(
        self,
        data_dir: str = "data/vanna_ml",
        symbols: List[str] = None,
        initial_capital: float = 10000.0,
        episode_length: int = 390,  # 1 trading day (390 minutes)
        render_mode: str | None = None
    ) -> None:
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.symbols = symbols or ['SPY', 'QQQ']
        self.initial_capital = initial_capital
        self.episode_length = episode_length
        self.render_mode = render_mode
        
        # Load all data
        self._load_data()
        
        # Observation space: 32 features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.N_FEATURES,),
            dtype=np.float32
        )
        
        # Action space: 5 discrete
        self.action_space = spaces.Discrete(5)
        
        # Initialize state
        self._reset_state()
    
    def _load_data(self) -> None:
        """Load parquet files for all symbols. Prefers *_rl.parquet, fallback to *_vanna.parquet."""
        self.data = {}
        
        for symbol in self.symbols:
            # Try enriched RL file first
            rl_path = self.data_dir / f"{symbol}_1min_rl.parquet"
            vanna_path = self.data_dir / f"{symbol}_1min_vanna.parquet"
            
            if rl_path.exists():
                df = pd.read_parquet(rl_path)
                logger.info(f"Loaded {symbol} (RL): {len(df):,} rows, {len(df.columns)} cols")
            elif vanna_path.exists():
                df = pd.read_parquet(vanna_path)
                logger.info(f"Loaded {symbol} (Vanna): {len(df):,} rows")
            else:
                logger.warning(f"No data found for {symbol}")
                continue
            
            # Fill NaN
            df = df.fillna(0)
            self.data[symbol] = df
        
        if not self.data:
            raise ValueError(f"No data found in {self.data_dir}")
        
        # Set current symbol
        self.current_symbol = list(self.data.keys())[0]
        self.df = self.data[self.current_symbol]
    
    def _reset_state(self) -> None:
        """Reset environment state for new episode."""
        self.capital = self.initial_capital
        self.position_size = 0
        self.cumulative_pnl = 0.0  # Track return-based P/L
        self.entry_step = 0
        self.current_step = 0
        self.episode_start_idx = 0
        self.total_pnl = 0.0
        self.trades = 0
        self.winning_trades = 0
    
    def _sample_episode_start(self) -> int:
        """Sample random starting point for episode."""
        max_start = len(self.df) - self.episode_length - 1
        if max_start <= 0:
            return 0
        return np.random.randint(0, max_start)
    
    def _get_market_features(self, idx: int) -> np.ndarray:
        """Get market features from parquet row."""
        row = self.df.iloc[idx]
        
        features = []
        for col in self.MARKET_FEATURES:
            if col in row:
                val = row[col]
                if pd.isna(val):
                    val = 0.0
            else:
                val = 0.0
            features.append(float(val))
        
        return np.array(features, dtype=np.float32)
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation (49 features)."""
        idx = self.episode_start_idx + self.current_step
        idx = min(idx, len(self.df) - 1)
        
        # Get 42 market features from data
        market_features = self._get_market_features(idx)
        
        # Calculate 7 position features (no close price needed!)
        position_features = np.array([
            self.cumulative_pnl * 10,                # pnl_pct (scaled)
            min((self.current_step - self.entry_step) / 390, 1.0) if self.position_size > 0 else 0.0,  # days_held
            1.0 if self.position_size > 0 else 0.0,  # position_flag
            self.capital / self.initial_capital,     # capital_ratio
            min(self.trades / 10, 1.0),              # trade_count (normalized)
            0.02,                                    # bid_ask_spread (placeholder)
            1.0,                                     # market_open
        ], dtype=np.float32)
        
        # Combine: 42 market + 7 position = 49
        obs = np.concatenate([market_features, position_features])
        
        # Ensure exactly N_FEATURES
        if len(obs) < self.N_FEATURES:
            obs = np.pad(obs, (0, self.N_FEATURES - len(obs)))
        elif len(obs) > self.N_FEATURES:
            obs = obs[:self.N_FEATURES]
        
        return obs.astype(np.float32)
    
    def _get_info(self) -> dict:
        """Get info dict."""
        idx = self.episode_start_idx + self.current_step
        idx = min(idx, len(self.df) - 1)
        row = self.df.iloc[idx]
        
        return {
            "capital": self.capital,
            "position_size": self.position_size,
            "total_pnl": self.total_pnl,
            "trades": self.trades,
            "win_rate": self.winning_trades / self.trades if self.trades > 0 else 0,
            "symbol": self.current_symbol,
            "step": self.current_step,
            "data_idx": idx,
            "vix_norm": row.get('vix_norm', 0),
            "vanna": row.get('vanna', 0),
            "return_1m": row.get('return_1m', 0),
        }
    
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset for new episode."""
        super().reset(seed=seed)
        
        # Optionally switch symbol
        if options and 'symbol' in options:
            if options['symbol'] in self.data:
                self.current_symbol = options['symbol']
                self.df = self.data[self.current_symbol]
        else:
            # Random symbol selection
            self.current_symbol = np.random.choice(list(self.data.keys()))
            self.df = self.data[self.current_symbol]
        
        self._reset_state()
        self.episode_start_idx = self._sample_episode_start()
        
        return self._get_obs(), self._get_info()
    
    def step(
        self,
        action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """
        Execute one step.
        
        Uses return_1m from data for P/L calculation (no absolute prices needed).
        """
        reward = 0.0
        terminated = False
        truncated = False
        
        idx = self.episode_start_idx + self.current_step
        idx = min(idx, len(self.df) - 1)
        
        # Get return from data (already normalized)
        current_return = self.df.iloc[idx].get('return_1m', 0)
        
        # Execute action
        if action == 0:  # HOLD
            # If in position, accumulate return
            if self.position_size > 0:
                self.cumulative_pnl += current_return * self.position_size
                reward = current_return * 10  # Reward based on return
            else:
                reward = -0.001  # Small penalty for doing nothing
            
        elif action == 1:  # OPEN
            if self.position_size == 0:
                self.position_size = 1
                self.entry_step = self.current_step
                self.cumulative_pnl = 0  # Reset for new position
                self.trades += 1
                reward = 0.01  # Small reward for action
            else:
                reward = -0.05  # Penalty for invalid action
        
        elif action == 2:  # CLOSE
            if self.position_size > 0:
                # Final P/L is accumulated return
                profit = self.cumulative_pnl * self.capital * 0.1
                
                self.capital += profit
                self.total_pnl += profit
                
                # Improved reward function with risk-adjustment
                hold_time = self.current_step - self.entry_step
                hold_penalty = -0.001 * hold_time  # Penalize long holds
                
                # Base reward proportional to P/L
                base_reward = self.cumulative_pnl * 15
                
                # Sharpe-like adjustment: reward consistency
                # (higher reward if profit, lower penalty if small loss)
                if profit > 0:
                    self.winning_trades += 1
                    # Quick profit bonus
                    speed_bonus = 0.1 if hold_time < 30 else 0
                    reward = base_reward + speed_bonus + hold_penalty
                else:
                    # Cut losses quickly is better
                    quick_cut_bonus = 0.05 if hold_time < 60 else 0
                    reward = base_reward + quick_cut_bonus + hold_penalty
                
                self.position_size = 0
                self.cumulative_pnl = 0
            else:
                reward = -0.05
        
        elif action == 3:  # INCREASE
            if self.position_size > 0:
                self.position_size += 1
                reward = 0.005
            else:
                reward = -0.02
        
        elif action == 4:  # DECREASE
            if self.position_size > 1:
                self.position_size -= 1
                reward = 0.005
            else:
                reward = -0.02
        
        # Advance step
        self.current_step += 1
        
        # Check termination
        if self.current_step >= self.episode_length:
            truncated = True
        
        if self.episode_start_idx + self.current_step >= len(self.df) - 1:
            truncated = True
        
        if self.capital <= 0:
            terminated = True
            reward = -100
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self) -> str | None:
        """Render environment."""
        if self.render_mode == "ansi":
            info = self._get_info()
            return (
                f"Step {info['step']}/{self.episode_length} | "
                f"{info['symbol']} ${info['price']:.2f} | "
                f"VIX {info['vix']:.1f} | Vanna {info['vanna']:.4f} | "
                f"Capital ${self.capital:.2f} | P&L ${self.total_pnl:.2f}"
            )
        return None


def make_trading_env(
    data_dir: str = "data/vanna_ml",
    symbols: List[str] = None,
    initial_capital: float = 10000.0,
    episode_length: int = 390
) -> TradingEnvironment:
    """Factory function."""
    return TradingEnvironment(
        data_dir=data_dir,
        symbols=symbols,
        initial_capital=initial_capital,
        episode_length=episode_length
    )
