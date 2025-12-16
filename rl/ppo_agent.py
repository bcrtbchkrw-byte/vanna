"""
PPO Agent for Options Trading

Uses stable-baselines3 PPO with vectorized multi-symbol environment.
Trained on SPY, QQQ historical data with Vanna features.
"""
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import VecNormalize

from rl.vec_env import make_vec_env, get_available_symbols
from core.logger import get_logger

logger = get_logger()


class TradingAgent:
    """
    PPO-based trading agent using stable-baselines3.
    
    Actions:
    0 = HOLD
    1 = OPEN position
    2 = CLOSE position
    3 = INCREASE position
    4 = DECREASE position
    """
    
    def __init__(
        self,
        model_path: str = "data/models/ppo_trading_agent",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        ent_coef: float = 0.01,
        verbose: int = 1
    ):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.verbose = verbose
        
        self.model: Optional[PPO] = None
        self.vec_env: Optional[VecNormalize] = None
        
        logger.info("TradingAgent initialized")
    
    def create_env(
        self,
        symbols: list = None,
        n_envs_per_symbol: int = 1
    ) -> VecNormalize:
        """Create vectorized training environment."""
        if symbols is None:
            symbols = get_available_symbols()
        
        self.vec_env = make_vec_env(
            symbols=symbols,
            n_envs_per_symbol=n_envs_per_symbol,
            use_subproc=True,
            normalize=True
        )
        
        logger.info(f"Created env with {self.vec_env.num_envs} parallel environments")
        return self.vec_env
    
    def create_model(self, env: VecNormalize = None) -> PPO:
        """Create PPO model."""
        if env is None:
            env = self.vec_env or self.create_env()
        
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            ent_coef=self.ent_coef,
            verbose=self.verbose,
            tensorboard_log=str(self.model_path.parent / "tensorboard")
        )
        
        logger.info(f"Created PPO model with MlpPolicy")
        return self.model
    
    def train(
        self,
        total_timesteps: int = 100_000,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        checkpoint_freq: int = 25_000
    ) -> Dict[str, Any]:
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total training steps
            eval_freq: How often to evaluate
            n_eval_episodes: Episodes per evaluation
            checkpoint_freq: How often to save checkpoints
        """
        if self.model is None:
            self.create_model()
        
        logger.info(f"Starting training for {total_timesteps:,} timesteps...")
        
        # Create evaluation env
        eval_env = make_vec_env(
            symbols=get_available_symbols()[:1],  # SPY only for eval
            n_envs_per_symbol=1,
            use_subproc=False,
            normalize=True
        )
        
        # Callbacks
        callbacks = CallbackList([
            EvalCallback(
                eval_env,
                best_model_save_path=str(self.model_path / "best"),
                log_path=str(self.model_path / "logs"),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True
            ),
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(self.model_path / "checkpoints"),
                name_prefix="ppo_trading"
            )
        ])
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        self.save()
        
        logger.info("Training complete!")
        
        return {
            "total_timesteps": total_timesteps,
            "model_path": str(self.model_path)
        }
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Predict action for given observation."""
        if self.model is None:
            self.load()
        
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def predict_with_confidence(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, float]:
        """Predict action with confidence score."""
        if self.model is None:
            self.load()
            
        # Get action
        action, _ = self.model.predict(obs, deterministic=deterministic)
        action_int = int(action)
        
        try:
            # Get probabilities from policy
            import torch
            with torch.no_grad():
                obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
                distribution = self.model.policy.get_distribution(obs_tensor)
                probs = distribution.distribution.probs
                
                # Get probability of selected action
                confidence = float(probs[0][action_int].item())
                return action_int, confidence
        except Exception as e:
            logger.warning(f"Could not get confidence: {e}")
            return action_int, 0.0
    
    def save(self):
        """Save model and normalizer."""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        self.model.save(str(self.model_path / "ppo_trading"))
        
        if self.vec_env is not None:
            self.vec_env.save(str(self.model_path / "vec_normalize.pkl"))
        
        logger.info(f"Saved model to {self.model_path}")
    
    def load(self) -> bool:
        """Load saved model."""
        model_file = self.model_path / "ppo_trading.zip"
        
        if not model_file.exists():
            logger.warning(f"Model not found: {model_file}")
            return False
        
        # Load normalizer first if exists
        norm_file = self.model_path / "vec_normalize.pkl"
        if norm_file.exists():
            # Load normalizer with proper symbols
            logger.info("Initializing lightweight dummy environment for loading normalization stats...")
            dummy_env = make_vec_env(
                symbols=get_available_symbols()[:1],
                n_envs_per_symbol=1,
                use_subproc=False,
                normalize=False  # Don't normalize twice
            )
            self.vec_env = VecNormalize.load(str(norm_file), dummy_env)
        
        self.model = PPO.load(str(model_file), env=self.vec_env)
        logger.info(f"Loaded model from {model_file}")
        
        return True


# Quick access function
def get_trading_agent() -> TradingAgent:
    """Get or create trading agent."""
    return TradingAgent()


# Training script
if __name__ == "__main__":
    from core.logger import setup_logger
    
    try:
        setup_logger(level="INFO")
    except Exception as e:
        print(f"Logger setup failed: {e}")
    
    print("=" * 60)
    print("PPO Trading Agent Training")
    print("=" * 60)
    
    agent = TradingAgent()
    
    # Create environment
    symbols = get_available_symbols()
    print(f"Symbols: {symbols}")
    
    if not symbols:
        print("No data files found!")
        exit(1)
    
    agent.create_env(symbols=symbols)
    agent.create_model()
    
    # Train (short run for testing)
    print("\nStarting training...")
    agent.train(total_timesteps=10_000)
    
    print("\n✅ Training complete!")
