"""
PPO Agent for Vanna Trading Bot.

Wraps stable-baselines3 PPO for trading decisions.
Supports:
- Training on historical/simulated data
- Inference for live trading suggestions
- Model persistence
"""
import os
from typing import Any

from core.logger import get_logger


class PPOTradingAgent:
    """
    PPO-based trading agent using stable-baselines3.
    
    Used for learning optimal position sizing and timing
    from market simulations.
    """
    
    def __init__(
        self,
        model_path: str | None = None,
        learning_rate: float = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10
    ) -> None:
        self.logger = get_logger()
        self.model_path = model_path or "models/ppo_trading.zip"
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self._model: Any = None
        self._model_loaded = False
        
        self._try_load_model()
    
    def _try_load_model(self) -> bool:
        """Try to load a pre-trained model."""
        if not os.path.exists(self.model_path):
            self.logger.info(f"No trained model at {self.model_path}")
            return False
        
        try:
            from stable_baselines3 import PPO
            self._model = PPO.load(self.model_path)
            self._model_loaded = True
            self.logger.info(f"Loaded PPO model from {self.model_path}")
            return True
        except ImportError:
            self.logger.warning(
                "stable-baselines3 not installed. "
                "Run: pip install stable-baselines3"
            )
            return False
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def train(
        self,
        total_timesteps: int = 100000,
        save_path: str | None = None
    ) -> dict[str, Any]:
        """
        Train the agent on the trading environment.
        
        Args:
            total_timesteps: Number of training steps
            save_path: Where to save the trained model
            
        Returns:
            Training statistics
        """
        try:
            from stable_baselines3 import PPO

            from rl.trading_env import make_trading_env

            
            # Create environment
            env = make_trading_env()
            
            # Create or update model
            if self._model is None:
                self._model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=self.learning_rate,
                    n_steps=self.n_steps,
                    batch_size=self.batch_size,
                    n_epochs=self.n_epochs,
                    verbose=1
                )
            else:
                self._model.set_env(env)
            
            # Train
            self.logger.info(f"Starting training for {total_timesteps} timesteps")
            self._model.learn(total_timesteps=total_timesteps)
            
            # Save
            save_to = save_path or self.model_path
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            self._model.save(save_to)
            self._model_loaded = True
            
            self.logger.info(f"Model saved to {save_to}")
            
            return {
                "status": "success",
                "timesteps": total_timesteps,
                "model_path": save_to
            }
            
        except ImportError:
            self.logger.error("stable-baselines3 not available")
            return {"status": "error", "reason": "missing dependency"}
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {"status": "error", "reason": str(e)}
    
    def predict(self, observation: list[float]) -> tuple[int, float]:
        """
        Get action prediction for given observation.
        
        Args:
            observation: State observation [price_change, vix, delta, pnl, days, position]
            
        Returns:
            Tuple of (action, confidence)
        """
        if not self._model_loaded or self._model is None:
            # Fallback to heuristic
            return self._heuristic_predict(observation)
        
        import numpy as np
        obs_array = np.array(observation).reshape(1, -1)
        action, _ = self._model.predict(obs_array, deterministic=True)
        
        return int(action[0]), 0.7  # Fixed confidence for now
    
    def _heuristic_predict(self, observation: list[float]) -> tuple[int, float]:
        """
        Heuristic fallback when no model is available.
        
        Simple rules:
        - Open if VIX favorable and no position
        - Close if profitable or stop-loss hit
        """
        _, vix_norm, _, pnl_pct, _, position_flag = observation
        
        vix = vix_norm * 10 + 18  # Denormalize
        has_position = position_flag > 0.5
        
        if has_position:
            # Position management
            if pnl_pct > 0.5:  # 50% profit
                return 2, 0.8  # CLOSE with high confidence
            elif pnl_pct < -0.3:  # 30% loss
                return 2, 0.7  # CLOSE (stop loss)
            else:
                return 0, 0.5  # HOLD
        else:
            # Entry decision
            if 15 <= vix <= 25:
                return 1, 0.6  # OPEN in favorable conditions
            else:
                return 0, 0.6  # HOLD
    
    def get_action_name(self, action: int) -> str:
        """Convert action number to readable name."""
        actions = ["HOLD", "OPEN", "CLOSE", "INCREASE", "DECREASE"]
        return actions[action] if 0 <= action < len(actions) else "UNKNOWN"


# Singleton
_agent: PPOTradingAgent | None = None


def get_ppo_agent() -> PPOTradingAgent:
    """Get global PPO agent instance."""
    global _agent
    if _agent is None:
        _agent = PPOTradingAgent()
    return _agent
