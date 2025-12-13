"""
Trading Environment for Reinforcement Learning.

Gymnasium-compatible environment for training RL agents on:
- Position sizing decisions
- Entry/exit timing
- Strategy selection

Uses 32 features matching TradeSuccessPredictor.
"""
from typing import Any, SupportsFloat, Optional
from datetime import datetime
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.logger import get_logger


class TradingEnvironment(gym.Env):
    """
    Gymnasium environment for options trading.
    
    State: 32 features including Greeks (Vanna, Charm, Volga)
    
    Features (32 total):
    - VIX metrics (7): vix, vix_ratio, vix_in_contango, vix_change_1d, vix_percentile, vix_zscore, regime
    - Time features (4): sin_time, cos_time, sin_dow, cos_dow
    - Price features (4): return_1m, return_5m, volatility_20, momentum_20
    - Greeks (7): delta, gamma, theta, vega, vanna, charm, volga
    - Position features (5): pnl_pct, days_held, position_flag, capital_ratio, trade_count
    - ATM Options (3): atm_iv, atm_vanna, atm_volume
    - Additional (2): bid_ask_spread, market_open
    
    Actions:
    0 = HOLD (do nothing)
    1 = OPEN (open new position if none)
    2 = CLOSE (close existing position)
    3 = INCREASE (add to position)
    4 = DECREASE (reduce position)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    # Feature names for documentation/debugging
    FEATURE_NAMES = [
        # VIX metrics (7)
        'vix', 'vix_ratio', 'vix_in_contango', 'vix_change_1d', 
        'vix_percentile', 'vix_zscore', 'regime',
        # Time features (4)
        'sin_time', 'cos_time', 'sin_dow', 'cos_dow',
        # Price features (4)
        'return_1m', 'return_5m', 'volatility_20', 'momentum_20',
        # Greeks (7)
        'delta', 'gamma', 'theta', 'vega', 'vanna', 'charm', 'volga',
        # Position features (5)
        'pnl_pct', 'days_held', 'position_flag', 'capital_ratio', 'trade_count',
        # ATM Options (3)
        'atm_iv', 'atm_vanna', 'atm_volume',
        # Additional (2)
        'bid_ask_spread', 'market_open'
    ]
    
    N_FEATURES = 32
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_steps: int = 252,  # ~1 year of trading days
        price_volatility: float = 0.02,
        render_mode: str | None = None
    ) -> None:
        super().__init__()
        
        self.logger = get_logger()
        self.initial_capital = initial_capital
        self.max_steps = max_steps
        self.price_volatility = price_volatility
        self.render_mode = render_mode
        
        # State space: 32 continuous features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.N_FEATURES,),
            dtype=np.float32
        )
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        # Environment state
        self._reset_state()
    
    def _reset_state(self) -> None:
        """Reset all environment state."""
        self.capital = self.initial_capital
        self.position_size = 0
        self.position_price = 0.0
        self.current_price = 100.0  # Starting "stock" price
        self.prev_price = 100.0
        self.vix = 18.0  # Starting VIX
        self.vix3m = 20.0  # VIX3M for term structure
        self.current_step = 0
        self.total_pnl = 0.0
        self.trades = 0
        self.winning_trades = 0
        
        # Historical data for features
        self.vix_history = [18.0] * 20
        self.price_history = [100.0] * 20
        self.return_history = [0.0] * 5
        
        # Greeks simulation (updated at each step)
        self.atm_delta = -0.50
        self.atm_gamma = 0.03
        self.atm_theta = -0.05
        self.atm_vega = 0.15
        self.atm_vanna = 0.02
        self.atm_charm = -0.01
        self.atm_volga = 0.08
        self.atm_iv = 0.25
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation - 32 features."""
        
        # Calculate derived features
        if self.position_size > 0:
            pnl_pct = (self.current_price - self.position_price) / self.position_price
            days_held = self.current_step
        else:
            pnl_pct = 0.0
            days_held = 0
        
        # Time features (simulate market time)
        minute_of_day = (self.current_step * 5) % 390  # 5-min bars
        day_of_week = (self.current_step // 78) % 5  # 78 bars per day
        
        sin_time = math.sin(2 * math.pi * minute_of_day / 390)
        cos_time = math.cos(2 * math.pi * minute_of_day / 390)
        sin_dow = math.sin(2 * math.pi * day_of_week / 5)
        cos_dow = math.cos(2 * math.pi * day_of_week / 5)
        
        # VIX features
        vix_ratio = self.vix / self.vix3m if self.vix3m > 0 else 1.0
        vix_in_contango = 1.0 if vix_ratio < 1.0 else 0.0
        vix_change_1d = (self.vix - self.vix_history[-1]) / max(self.vix_history[-1], 1)
        
        # VIX percentile (simplified)
        vix_percentile = min(1.0, max(0.0, (self.vix - 10) / 40))
        vix_zscore = (self.vix - np.mean(self.vix_history)) / max(np.std(self.vix_history), 0.1)
        
        # Regime (0=low vol, 1=normal, 2=high vol)
        if self.vix < 15:
            regime = 0
        elif self.vix > 25:
            regime = 2
        else:
            regime = 1
        
        # Price features
        return_1m = (self.current_price - self.prev_price) / self.prev_price if self.prev_price > 0 else 0
        return_5m = np.mean(self.return_history[-5:]) if len(self.return_history) >= 5 else 0
        volatility_20 = np.std(self.return_history[-20:]) if len(self.return_history) >= 20 else 0.02
        momentum_20 = (self.current_price - self.price_history[0]) / self.price_history[0] if self.price_history[0] > 0 else 0
        
        # ATM Vanna (key feature!) - varies with VIX
        atm_vanna = self.atm_vanna * (1 + 0.1 * (self.vix - 18) / 10)
        
        # Build 32-feature observation
        obs = np.array([
            # VIX metrics (7)
            (self.vix - 18) / 10,           # Normalized VIX
            vix_ratio - 1.0,                # Centered ratio
            vix_in_contango,
            vix_change_1d * 10,             # Scaled
            vix_percentile,
            np.clip(vix_zscore, -3, 3),     # Clipped z-score
            regime / 2.0,                   # Normalized regime
            
            # Time features (4)
            sin_time,
            cos_time,
            sin_dow,
            cos_dow,
            
            # Price features (4)
            return_1m * 100,                # Scaled returns
            return_5m * 100,
            volatility_20 * 100,
            momentum_20 * 10,
            
            # Greeks (7) - THE KEY FEATURES
            self.atm_delta,                 # Already -1 to 0 for puts
            self.atm_gamma * 10,            # Scaled
            self.atm_theta * 10,            # Scaled (negative)
            self.atm_vega,                  # 0 to 0.5 typically
            atm_vanna,                      # ATM Vanna!
            self.atm_charm * 10,            # Scaled
            self.atm_volga,                 # Vega of vega
            
            # Position features (5)
            pnl_pct * 10,                   # Scaled P&L
            days_held / 30,                 # Normalized days
            1.0 if self.position_size > 0 else 0.0,
            self.capital / self.initial_capital,  # Capital ratio
            min(self.trades / 10, 1.0),     # Trade count normalized
            
            # ATM Options (3)
            self.atm_iv,                    # 0.1 to 0.5 typically
            atm_vanna,                      # Duplicate for emphasis
            min(1.0, abs(return_1m) * 100), # Volume proxy
            
            # Additional (2)
            0.02,                           # Bid-ask spread placeholder
            1.0,                            # Market open flag
            
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> dict[str, Any]:
        """Get current info dict."""
        return {
            "capital": self.capital,
            "position_size": self.position_size,
            "total_pnl": self.total_pnl,
            "trades": self.trades,
            "win_rate": self.winning_trades / self.trades if self.trades > 0 else 0,
            "vix": self.vix,
            "atm_vanna": self.atm_vanna
        }
    
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self._reset_state()
        
        return self._get_obs(), self._get_info()
    
    def step(
        self,
        action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Returns: (observation, reward, terminated, truncated, info)
        """
        reward = 0.0
        terminated = False
        truncated = False
        
        # Store previous price
        self.prev_price = self.current_price
        
        # Simulate price movement
        price_change = np.random.normal(0, self.price_volatility)
        self.current_price *= (1 + price_change)
        
        # Simulate VIX movement (mean-reverting)
        vix_change = np.random.normal(0, 0.5) - 0.05 * (self.vix - 18)
        self.vix = max(10, min(50, self.vix + vix_change))
        self.vix3m = max(12, min(45, self.vix3m + np.random.normal(0, 0.3)))
        
        # Update histories
        self.vix_history.append(self.vix)
        self.vix_history = self.vix_history[-20:]
        self.price_history.append(self.current_price)
        self.price_history = self.price_history[-20:]
        self.return_history.append(price_change)
        self.return_history = self.return_history[-20:]
        
        # Update Greeks based on VIX and price
        self._update_greeks()
        
        # Execute action
        if action == 0:  # HOLD
            reward = -0.001
            
        elif action == 1:  # OPEN
            if self.position_size == 0:
                position_cost = self.capital * 0.1
                if position_cost < self.capital:
                    self.position_size = 1
                    self.position_price = self.current_price
                    self.capital -= position_cost * 0.01
                    self.trades += 1
                    reward = 0.01
                else:
                    reward = -0.1
            else:
                reward = -0.05
        
        elif action == 2:  # CLOSE
            if self.position_size > 0:
                pnl = (self.current_price - self.position_price) / self.position_price
                profit = pnl * self.capital * 0.1
                
                self.capital += profit
                self.total_pnl += profit
                
                if profit > 0:
                    self.winning_trades += 1
                    reward = pnl * 10
                else:
                    reward = pnl * 5
                
                self.position_size = 0
                self.position_price = 0.0
            else:
                reward = -0.05
        
        elif action == 3:  # INCREASE
            reward = -0.001
        
        elif action == 4:  # DECREASE
            reward = -0.001
        
        # Update step counter
        self.current_step += 1
        
        # Check termination conditions
        if self.current_step >= self.max_steps:
            truncated = True
        
        if self.capital <= 0:
            terminated = True
            reward = -100
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _update_greeks(self) -> None:
        """Update simulated ATM Greeks based on market conditions."""
        # Greeks vary with VIX and price
        vix_factor = self.vix / 18.0
        
        self.atm_delta = -0.50 + np.random.normal(0, 0.02)
        self.atm_gamma = 0.03 * vix_factor + np.random.normal(0, 0.005)
        self.atm_theta = -0.05 * vix_factor + np.random.normal(0, 0.01)
        self.atm_vega = 0.15 * vix_factor + np.random.normal(0, 0.02)
        
        # Vanna increases with VIX (key insight!)
        self.atm_vanna = 0.02 * vix_factor + np.random.normal(0, 0.005)
        self.atm_charm = -0.01 + np.random.normal(0, 0.002)
        self.atm_volga = 0.08 * vix_factor + np.random.normal(0, 0.01)
        
        self.atm_iv = 0.15 + 0.01 * (self.vix - 18) + np.random.normal(0, 0.01)
    
    def render(self) -> str | None:  # type: ignore[override]
        """Render the environment."""
        if self.render_mode == "ansi":
            return (
                f"Step: {self.current_step}, Capital: ${self.capital:.2f}, "
                f"Price: ${self.current_price:.2f}, VIX: {self.vix:.1f}, "
                f"Position: {self.position_size}, P&L: ${self.total_pnl:.2f}, "
                f"Vanna: {self.atm_vanna:.4f}"
            )
        return None


def make_trading_env(
    initial_capital: float = 10000.0,
    max_steps: int = 252
) -> TradingEnvironment:
    """Factory function to create trading environment."""
    return TradingEnvironment(
        initial_capital=initial_capital,
        max_steps=max_steps
    )

