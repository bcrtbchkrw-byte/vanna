"""
Trading Environment for Reinforcement Learning.

Uses REAL historical data from *_rl.parquet files (enriched).
77 market features + 7 position features = 84 total features.
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
    
    State: 97 features (83 market + 14 position)
    Actions: MultiDiscrete([3, 2, 2, 3])
        - Direction: 0=HOLD, 1=OPEN, 2=CLOSE
        - Option Type: 0=CALL, 1=PUT
        - Side: 0=BUY (long), 1=SELL (short)
        - DTE Bucket: 0=0DTE, 1=WEEKLY(1-7), 2=MONTHLY(14-60)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    # Market features from normalized *_rl.parquet (67 → 77 total)
    # NO OHLC, NO timestamp - all normalized/scaled
    MARKET_FEATURES = [
        # Time (6 + 3 NEW = 9) - already normalized [-1, 1]
        'sin_time', 'cos_time', 'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy',
        'hour_of_day',  # NEW: 0-23 for time-of-day patterns
        'is_market_open_hour',  # NEW: 1 if first/last hour (high volatility)
        'is_lunch_hour',  # NEW: 1 if lunch hour (low volume)
        
        # VIX (8) - ratios and z-scores, comparable across symbols
        'vix_ratio', 'vix_in_contango', 'vix_change_1d', 'vix_change_5d',
        'vix_percentile', 'vix_zscore', 'vix_norm', 'vix3m_norm',
        
        # Regime (1)
        'regime',
        
        # Options (3) - normalized
        'options_iv_atm', 'options_put_call_ratio', 'options_volume_norm',
        
        # Price (5 + 5 NEW = 10) - returns/ratios, already normalized
        'return_1m', 'return_5m', 'volatility_20', 'momentum_20', 'range_pct',
        'volume_ratio',  # NEW: Volume vs 20-bar avg (spot unusual volume)
        'high_low_range',  # NEW: Alias for range_pct (consistency)
        'price_acceleration',  # NEW: Change in momentum (trend shifts)
        'return_1m_lag1',  # NEW: Previous bar return (momentum continuation)
        'return_1m_lag5',  # NEW: 5 bars ago return (pattern recognition)
        'volatility_20_lag1',  # NEW: Previous volatility (vol clustering)
        
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
        
        # Major event features (4) - FOMC/CPI for bonds, mega-cap earnings for equities
        'days_to_major_event', 'is_event_week', 'is_event_day', 'event_iv_boost',
        
        # Daily features injected from 1day (21) - uses YESTERDAY's data (no lookahead!)
        'day_sma_200', 'day_sma_50', 'day_sma_20',
        'day_price_vs_sma200', 'day_price_vs_sma50',
        'day_rsi_14',
        'day_atr_14', 'day_atr_pct',
        'day_bb_position', 'day_bb_upper', 'day_bb_lower',  # Bollinger Bands
        'day_macd', 'day_macd_signal', 'day_macd_hist',  # MACD complete
        'day_above_sma200', 'day_above_sma50',
        'day_sma_50_200_ratio',
        'day_days_to_major_event', 'day_is_event_week', 'day_is_event_day', 'day_event_iv_boost',
        
        # DTE features for multi-strategy (6) - NEW for Phase 18
        'target_dte_norm',      # Normalized DTE (0-60 → 0-1)
        'is_0dte',              # Binary: 1 if DTE = 0
        'is_weekly',            # Binary: 1 if DTE < 7
        'is_monthly',           # Binary: 1 if DTE >= 14
        'theta_to_premium',     # Theta / Premium ratio (for sell strategies)
        'gamma_exposure',       # Gamma * position size (risk metric)
    ]
    
    # Position features added at runtime (14) - Extended for multi-strategy
    POSITION_FEATURES = [
        'pnl_pct', 'days_held', 'position_flag', 'capital_ratio',
        'trade_count', 'bid_ask_spread', 'market_open',
        # Multi-strategy state (7) - NEW
        'current_option_type',  # 0=none, 1=CALL, 2=PUT
        'current_side',         # 0=none, 1=BUY, 2=SELL
        'current_dte_bucket',   # 0=none, 1=0DTE, 2=WEEKLY, 3=MONTHLY
        'theta_pnl',            # Accumulated theta P/L (for sell positions)
        'gamma_pnl',            # Accumulated gamma P/L
        'position_delta',       # Current position delta exposure
        'days_to_expiry',       # Actual DTE remaining
    ]
    
    N_MARKET_FEATURES = 83  # 77 + 6 DTE features
    N_POSITION_FEATURES = 14  # 7 + 7 multi-strategy
    N_FEATURES = 97  # 83 market + 14 position
    
    def __init__(
        self,
        data_dir: str = "data/enriched",
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
        
        # Action space: MultiDiscrete for multi-strategy trading
        # [Direction, OptionType, Side, DTEBucket]
        # Direction: 0=HOLD, 1=OPEN, 2=CLOSE
        # OptionType: 0=CALL, 1=PUT
        # Side: 0=BUY, 1=SELL
        # DTEBucket: 0=0DTE, 1=WEEKLY, 2=MONTHLY
        self.action_space = spaces.MultiDiscrete([3, 2, 2, 3])
        
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
        
        # For Sharpe-based reward
        self.recent_returns = []  # Track last N returns for volatility
        self.peak_capital = self.initial_capital  # Track max capital for drawdown
        self.episode_returns = []  # All returns in episode
        
        # Multi-strategy position state (NEW Phase 18)
        self.current_option_type = 0  # 0=none, 1=CALL, 2=PUT
        self.current_side = 0         # 0=none, 1=BUY, 2=SELL
        self.current_dte_bucket = 0   # 0=none, 1=0DTE, 2=WEEKLY, 3=MONTHLY
        self.theta_pnl = 0.0          # Accumulated theta P/L
        self.gamma_pnl = 0.0          # Accumulated gamma P/L
        self.position_delta = 0.0     # Current delta exposure
        self.days_to_expiry = 0       # Remaining DTE
    
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
        """Get current observation (97 features: 83 market + 14 position)."""
        idx = self.episode_start_idx + self.current_step
        idx = min(idx, len(self.df) - 1)
        
        # Get 83 market features from data
        market_features = self._get_market_features(idx)
        
        # Get current row for delta/theta/gamma
        row = self.df.iloc[idx]
        current_delta = row.get('delta', 0.0)
        current_theta = row.get('theta', 0.0)
        current_gamma = row.get('gamma', 0.0)
        
        # Calculate 14 position features
        position_features = np.array([
            # Original 7 features
            self.cumulative_pnl * 10,                # pnl_pct (scaled)
            min((self.current_step - self.entry_step) / 390, 1.0) if self.position_size > 0 else 0.0,  # days_held
            1.0 if self.position_size > 0 else 0.0,  # position_flag
            self.capital / self.initial_capital,     # capital_ratio
            min(self.trades / 10, 1.0),              # trade_count (normalized)
            0.02,                                    # bid_ask_spread (placeholder)
            1.0,                                     # market_open
            # Multi-strategy state (7 NEW)
            float(self.current_option_type) / 2.0,   # Normalized: 0=none, 0.5=CALL, 1=PUT
            float(self.current_side) / 2.0,          # Normalized: 0=none, 0.5=BUY, 1=SELL
            float(self.current_dte_bucket) / 3.0,    # Normalized: 0=none, 0.33=0DTE, 0.66=WEEKLY, 1=MONTHLY
            self.theta_pnl * 10,                     # Theta P/L (scaled)
            self.gamma_pnl * 10,                     # Gamma P/L (scaled)
            self.position_delta,                     # Current delta exposure
            min(self.days_to_expiry / 60.0, 1.0),    # DTE remaining (normalized to 60)
        ], dtype=np.float32)
        
        # Combine: 83 market + 14 position = 97
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
        
        # Map internal state to human-readable strings
        opt_type_map = {0: "NONE", 1: "CALL", 2: "PUT"}
        side_map = {0: "NONE", 1: "BUY", 2: "SELL"}
        dte_map = {0: "NONE", 1: "0DTE", 2: "WEEKLY", 3: "MONTHLY"}
        
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
            # Multi-strategy info (NEW)
            "option_type": opt_type_map.get(self.current_option_type, "NONE"),
            "side": side_map.get(self.current_side, "NONE"),
            "dte_bucket": dte_map.get(self.current_dte_bucket, "NONE"),
            "theta_pnl": self.theta_pnl,
            "gamma_pnl": self.gamma_pnl,
            "position_delta": self.position_delta,
            "days_to_expiry": self.days_to_expiry,
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
        action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """
        Execute one step with Multi-Discrete action and strategy-specific rewards.
        
        Action format: [direction, option_type, side, dte_bucket]
        - direction: 0=HOLD, 1=OPEN, 2=CLOSE
        - option_type: 0=CALL, 1=PUT
        - side: 0=BUY (long), 1=SELL (short)
        - dte_bucket: 0=0DTE, 1=WEEKLY, 2=MONTHLY
        
        Reward logic:
        - BUY positions: profit from price movement (delta * return) - theta decay
        - SELL positions: profit from theta income - adverse price movement
        """
        reward = 0.0
        terminated = False
        truncated = False
        
        # Decode Multi-Discrete action
        direction = action[0]      # 0=HOLD, 1=OPEN, 2=CLOSE
        opt_type = action[1] + 1   # Convert to internal: 1=CALL, 2=PUT
        side = action[2] + 1       # Convert to internal: 1=BUY, 2=SELL
        dte_bucket = action[3] + 1 # Convert to internal: 1=0DTE, 2=WEEKLY, 3=MONTHLY
        
        idx = self.episode_start_idx + self.current_step
        idx = min(idx, len(self.df) - 1)
        row = self.df.iloc[idx]
        
        # Get Greeks from data
        delta = row.get('delta', 0.3)
        theta = row.get('theta', -0.01)
        gamma = row.get('gamma', 0.05)
        
        # CRITICAL: Use FUTURE return to avoid data leakage
        future_idx = min(idx + 1, len(self.df) - 1)
        future_return = self.df.iloc[future_idx].get('return_1m', 0)
        current_return = row.get('return_1m', 0)
        
        # Track returns for volatility calculation
        self.episode_returns.append(current_return)
        if len(self.recent_returns) >= 20:
            self.recent_returns.pop(0)
        self.recent_returns.append(current_return)
        
        # DTE risk multipliers
        dte_risk_mult = {1: 2.0, 2: 1.0, 3: 0.5}  # 0DTE high risk, monthly low
        dte_theta_mult = {1: 3.0, 2: 1.0, 3: 0.3}  # 0DTE fast theta decay
        
        # =====================================================================
        # ACTION EXECUTION
        # =====================================================================
        
        if direction == 0:  # HOLD
            if self.position_size > 0:
                # Calculate ongoing P/L based on position type
                if self.current_side == 1:  # BUY (long option)
                    # Long profits from delta * price move, loses theta
                    delta_pnl = future_return * self.position_delta * 50
                    theta_cost = abs(theta) * dte_theta_mult.get(self.current_dte_bucket, 1.0)
                    step_pnl = delta_pnl - theta_cost
                    self.theta_pnl -= theta_cost
                    self.gamma_pnl += gamma * (future_return ** 2) * 100
                else:  # SELL (short option)
                    # Short profits from theta, loses on adverse moves
                    theta_income = abs(theta) * dte_theta_mult.get(self.current_dte_bucket, 1.0)
                    adverse_move = abs(future_return) * abs(self.position_delta) * 50
                    step_pnl = theta_income - adverse_move
                    self.theta_pnl += theta_income
                    self.gamma_pnl -= gamma * (future_return ** 2) * 100
                
                self.cumulative_pnl += step_pnl / 100
                
                # Decay DTE
                self.days_to_expiry = max(0, self.days_to_expiry - 1/390)
                
                # Reward
                if len(self.recent_returns) > 5:
                    volatility = np.std(self.recent_returns) + 1e-6
                    reward = step_pnl - (volatility * 0.2)
                else:
                    reward = step_pnl
            else:
                reward = -0.005  # Small penalty for idle capital
        
        elif direction == 1:  # OPEN
            if self.position_size == 0:
                self.position_size = 1
                self.entry_step = self.current_step
                self.cumulative_pnl = 0
                self.trades += 1
                
                # Set position attributes
                self.current_option_type = opt_type  # 1=CALL, 2=PUT
                self.current_side = side             # 1=BUY, 2=SELL
                self.current_dte_bucket = dte_bucket # 1=0DTE, 2=WEEKLY, 3=MONTHLY
                
                # Set DTE based on bucket
                dte_values = {1: 0, 2: 5, 3: 30}  # 0DTE, ~5 days, ~30 days
                self.days_to_expiry = dte_values.get(dte_bucket, 5)
                
                # Set delta based on option type
                if opt_type == 1:  # CALL
                    self.position_delta = delta if side == 1 else -delta  # BUY=positive, SELL=negative
                else:  # PUT
                    self.position_delta = -delta if side == 1 else delta  # PUT is naturally negative delta
                
                # Reset Greeks P/L
                self.theta_pnl = 0.0
                self.gamma_pnl = 0.0
                
                # Small reward for taking action
                reward = 0.01
            else:
                reward = -0.05  # Invalid: already in position
        
        elif direction == 2:  # CLOSE
            if self.position_size > 0:
                # Calculate final P/L
                profit = self.cumulative_pnl * self.capital * 0.1
                self.capital += profit
                self.total_pnl += profit
                
                # Update peak capital
                if self.capital > self.peak_capital:
                    self.peak_capital = self.capital
                
                hold_time = self.current_step - self.entry_step
                
                # Base reward from P/L
                base_reward = self.cumulative_pnl * 100
                
                # Strategy-specific bonus
                strategy_bonus = 0
                if self.current_side == 2:  # SELL strategy
                    if self.theta_pnl > 0:
                        strategy_bonus = self.theta_pnl * 5  # Reward theta capture
                else:  # BUY strategy
                    if self.gamma_pnl > 0:
                        strategy_bonus = self.gamma_pnl * 2  # Reward gamma capture
                
                # DTE-specific adjustments
                if self.current_dte_bucket == 1:  # 0DTE
                    if profit > 0:
                        strategy_bonus += 0.3  # Bonus for profitable 0DTE
                    else:
                        base_reward *= 1.5  # Larger loss penalty for risky 0DTE
                
                # Sharpe-like bonus
                sharpe_bonus = 0
                if len(self.recent_returns) >= 10:
                    mean_ret = np.mean(self.recent_returns[-20:])
                    std_ret = np.std(self.recent_returns[-20:]) + 1e-6
                    sharpe_bonus = (mean_ret / std_ret) * 3.0
                
                # Drawdown penalty
                drawdown = (self.peak_capital - self.capital) / self.peak_capital if self.peak_capital > 0 else 0
                drawdown_penalty = drawdown * 2.0
                
                # Time penalty (encourage action)
                time_penalty = 0.003 * hold_time
                
                # Final reward
                reward = (
                    base_reward 
                    + strategy_bonus
                    + sharpe_bonus
                    - drawdown_penalty
                    - time_penalty
                )
                
                if profit > 0:
                    self.winning_trades += 1
                    if hold_time < 30:
                        reward += 0.2  # Quick profit bonus
                
                # Reset position state
                self.position_size = 0
                self.cumulative_pnl = 0
                self.current_option_type = 0
                self.current_side = 0
                self.current_dte_bucket = 0
                self.position_delta = 0.0
                self.theta_pnl = 0.0
                self.gamma_pnl = 0.0
                self.days_to_expiry = 0
            else:
                reward = -0.05  # Invalid: no position to close
        
        # Advance step
        self.current_step += 1
        
        # Check termination
        if self.current_step >= self.episode_length:
            truncated = True
        
        if self.episode_start_idx + self.current_step >= len(self.df) - 1:
            truncated = True
        
        if self.capital <= 0:
            terminated = True
            reward = -100  # Large penalty for bankruptcy
        
        # 0DTE expiry check
        if self.position_size > 0 and self.current_dte_bucket == 1 and self.days_to_expiry <= 0:
            # Force close at expiry
            if self.cumulative_pnl < 0:
                reward -= 1.0  # Extra penalty for expiring worthless
            self.position_size = 0
            self.cumulative_pnl = 0
            self.current_option_type = 0
            self.current_side = 0
            self.current_dte_bucket = 0
        
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
    data_dir: str = "data/enriched",
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
