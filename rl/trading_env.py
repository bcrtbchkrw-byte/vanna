"""
Trading Environment for Reinforcement Learning.

Gymnasium-compatible environment for training RL agents on:
- Position sizing decisions
- Entry/exit timing
- Strategy selection

Uses simulated market data for training.
"""
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.logger import get_logger


class TradingEnvironment(gym.Env):
    """
    Gymnasium environment for options trading.
    
    State: [price_pct_change, vix, delta, pnl_pct, days_held, position_flag]
    
    Actions:
    0 = HOLD (do nothing)
    1 = OPEN (open new position if none)
    2 = CLOSE (close existing position)
    3 = INCREASE (add to position)
    4 = DECREASE (reduce position)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
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
        
        # State space: 6 continuous features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
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
        self.vix = 18.0  # Starting VIX
        self.current_step = 0
        self.total_pnl = 0.0
        self.trades = 0
        self.winning_trades = 0
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        # Calculate derived features
        if self.position_size > 0:
            pnl_pct = (self.current_price - self.position_price) / self.position_price
            days_held = self.current_step  # Simplified
        else:
            pnl_pct = 0.0
            days_held = 0
        
        price_change = np.random.normal(0, self.price_volatility)
        
        return np.array([
            price_change,  # Recent price change
            (self.vix - 18) / 10,  # Normalized VIX
            -0.15,  # Placeholder delta
            pnl_pct,  # Current P&L %
            days_held / 30,  # Days held normalized
            1.0 if self.position_size > 0 else 0.0  # Position flag
        ], dtype=np.float32)
    
    def _get_info(self) -> dict[str, Any]:
        """Get current info dict."""
        return {
            "capital": self.capital,
            "position_size": self.position_size,
            "total_pnl": self.total_pnl,
            "trades": self.trades,
            "win_rate": self.winning_trades / self.trades if self.trades > 0 else 0
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
        
        # Simulate price movement
        price_change = np.random.normal(0, self.price_volatility)
        self.current_price *= (1 + price_change)
        
        # Simulate VIX movement
        vix_change = np.random.normal(0, 0.5)
        self.vix = max(10, min(50, self.vix + vix_change))
        
        # Execute action
        if action == 0:  # HOLD
            # Holding cost for capital
            reward = -0.001
            
        elif action == 1:  # OPEN
            if self.position_size == 0:
                # Open new position
                position_cost = self.capital * 0.1  # 10% of capital
                if position_cost < self.capital:
                    self.position_size = 1
                    self.position_price = self.current_price
                    self.capital -= position_cost * 0.01  # Small transaction cost
                    self.trades += 1
                    reward = 0.01  # Small reward for taking action
                else:
                    reward = -0.1  # Penalty for invalid action
            else:
                reward = -0.05  # Penalty for trying to open with position
        
        elif action == 2:  # CLOSE
            if self.position_size > 0:
                # Calculate P&L
                pnl = (self.current_price - self.position_price) / self.position_price
                profit = pnl * self.capital * 0.1
                
                self.capital += profit
                self.total_pnl += profit
                
                if profit > 0:
                    self.winning_trades += 1
                    reward = pnl * 10  # Reward proportional to profit
                else:
                    reward = pnl * 5  # Smaller penalty for losses
                
                self.position_size = 0
                self.position_price = 0.0
            else:
                reward = -0.05  # Penalty for closing when no position
        
        elif action == 3:  # INCREASE (currently simplified to HOLD)
            reward = -0.001
        
        elif action == 4:  # DECREASE (currently simplified to HOLD)
            reward = -0.001
        
        # Update step counter
        self.current_step += 1
        
        # Check termination conditions
        if self.current_step >= self.max_steps:
            truncated = True
        
        if self.capital <= 0:
            terminated = True
            reward = -100  # Large penalty for losing all capital
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self) -> str | None:  # type: ignore[override]
        """Render the environment."""

        if self.render_mode == "ansi":
            return (
                f"Step: {self.current_step}, Capital: ${self.capital:.2f}, "
                f"Price: ${self.current_price:.2f}, VIX: {self.vix:.1f}, "
                f"Position: {self.position_size}, P&L: ${self.total_pnl:.2f}"
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
