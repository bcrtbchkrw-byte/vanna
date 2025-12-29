"""
PPO Agent for Options Trading

Uses stable-baselines3 PPO with vectorized multi-symbol environment.
Trained on SPY, QQQ historical data with Vanna features.
"""
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from rl.vec_env import make_vec_env, get_available_symbols, make_lightweight_env
from core.logger import get_logger

logger = get_logger()


class TradingAgent:
    """
    PPO-based trading agent using stable-baselines3.
    
    Multi-Discrete Action Space:
    [direction, option_type, side, dte_bucket]
    
    direction: 0=HOLD, 1=OPEN, 2=CLOSE
    option_type: 0=CALL, 1=PUT
    side: 0=BUY (long), 1=SELL (short)
    dte_bucket: 0=0DTE, 1=WEEKLY, 2=MONTHLY
    """
    
    def __init__(
        self,
        model_path: str = "data/models/ppo_trading_agent",
        learning_rate: float = 1e-4,      # Reduced from 3e-4 for stability in noisy data
        n_steps: int = 1024,               # REDUCED from 2048 for RPi 5 stability
        batch_size: int = 512,             # INCREASED from 256 for smoother gradients
        n_epochs: int = 10,                # Unchanged
        gamma: float = 0.99,               # Unchanged - discount factor
        gae_lambda: float = 0.95,          # NEW - smooth advantage estimates
        ent_coef: float = 0.01,           # Unchanged - exploration
        clip_range: float = 0.2,           # NEW - conservative policy updates
        verbose: int = 1
    ):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        self.verbose = verbose
        
        self.model: Optional[PPO] = None
        self.vec_env: Optional[VecNormalize] = None
        
        logger.info("TradingAgent initialized with noise-resistant hyperparameters")
    
    def create_env(
        self,
        symbols: list = None,
        n_envs_per_symbol: int = 1  # RPi 5: 2 symbols × 1 = 2 cores (safe, prevents crashes)
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
        """Create PPO model with noise-resistant hyperparameters."""
        if env is None:
            env = self.vec_env or self.create_env()
        
        # Network architecture: 128-128 (upgraded from default 64-64)
        # Better capacity for 84 input features
        policy_kwargs = dict(
            net_arch=[128, 128],  # 2 hidden layers, 128 neurons each
            activation_fn=torch.nn.ReLU,
        )
        
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.learning_rate,    # 1e-4 for stability
            n_steps=self.n_steps,                 # 2048 steps per update
            batch_size=self.batch_size,           # 512 for gradient smoothing
            n_epochs=self.n_epochs,               # 10 epochs
            gamma=self.gamma,                     # 0.99 discount
            gae_lambda=self.gae_lambda,           # 0.95 advantage smoothing
            ent_coef=self.ent_coef,              # 0.01 exploration
            clip_range=self.clip_range,           # 0.2 conservative updates
            policy_kwargs=policy_kwargs,
            verbose=self.verbose,
            tensorboard_log=str(self.model_path.parent / "tensorboard"),
            device='cpu'  # Force CPU to prevent Segfaults
        )
        
        logger.info(f"Created PPO model: 128-128 arch, LR={self.learning_rate}, batch={self.batch_size}")
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
    
    # Action decoding maps
    DIRECTION_MAP = ['HOLD', 'OPEN', 'CLOSE']
    OPTION_TYPE_MAP = ['CALL', 'PUT']
    SIDE_MAP = ['BUY', 'SELL']
    DTE_BUCKET_MAP = ['0DTE', 'WEEKLY', 'MONTHLY']
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> dict:
        """Predict action for given observation.
        
        Returns:
            dict with keys: direction, option_type, side, dte_bucket, raw_action
        """
        if self.model is None:
            self.load()
        
        action, _ = self.model.predict(obs, deterministic=deterministic)
        
        # Multi-Discrete: action is [direction, opt_type, side, dte_bucket]
        return {
            'direction': self.DIRECTION_MAP[action[0]],
            'option_type': self.OPTION_TYPE_MAP[action[1]],
            'side': self.SIDE_MAP[action[2]],
            'dte_bucket': self.DTE_BUCKET_MAP[action[3]],
            'raw_action': action
        }

    def predict_with_confidence(self, obs: np.ndarray, deterministic: bool = True) -> tuple[dict, float]:
        """Predict action with confidence score.
        
        Returns:
            (action_dict, confidence) where action_dict has direction, option_type, side, dte_bucket
        """
        if self.model is None:
            self.load()
            
        # Get action - cast to float32 to prevent binary incompatibility
        obs_f32 = obs.astype(np.float32)
        action, _ = self.model.predict(obs_f32, deterministic=deterministic)
        
        # Decode Multi-Discrete action
        action_dict = {
            'direction': self.DIRECTION_MAP[action[0]],
            'option_type': self.OPTION_TYPE_MAP[action[1]],
            'side': self.SIDE_MAP[action[2]],
            'dte_bucket': self.DTE_BUCKET_MAP[action[3]],
            'raw_action': action
        }
        
        try:
            # Get probabilities from policy (average across all sub-distributions)
            import torch
            with torch.no_grad():
                obs_tensor = self.model.policy.obs_to_tensor(obs_f32)[0]
                distribution = self.model.policy.get_distribution(obs_tensor)
                
                # Multi-Discrete: distribution has multiple sub-distributions
                # Get confidence as product of individual action probabilities
                if hasattr(distribution.distribution, 'distributions'):
                    # MultiCategorical distribution
                    confidences = []
                    for i, dist in enumerate(distribution.distribution.distributions):
                        prob = dist.probs[0][action[i]].item()
                        confidences.append(prob)
                    # Joint probability
                    confidence = float(np.prod(confidences))
                else:
                    # Fallback for single distribution
                    probs = distribution.distribution.probs
                    confidence = float(probs[0].max().item())
                    
                return action_dict, confidence
        except Exception as e:
            logger.warning(f"Could not get confidence: {e}")
            return action_dict, 0.5  # Default to 50% confidence

    
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
        
        # Stability: Force PyTorch to use 1 thread to avoid ARM64 Parallelism Crashes
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        if norm_file.exists():
            # Load normalizer with proper symbols
            logger.info("Initializing lightweight dummy environment for loading normalization stats...")
            # OPTIMIZATION: Use LightweightTradingEnv (no data loading)
            # This prevents OOM on Raspberry Pi by avoiding loading GBs of parquet data
            env_fns = [make_lightweight_env(n_envs=1) for _ in range(1)]
            dummy_env = DummyVecEnv(env_fns)
            
            self.vec_env = VecNormalize.load(str(norm_file), dummy_env)
        
        # Load model with forced CPU device to prevent AVX/CUDA crashes in Docker
        self.model = PPO.load(str(model_file), env=self.vec_env, device='cpu')
        logger.info(f"Loaded model from {model_file} (Device: CPU)")
        
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
    
    # Create environment (ALL available symbols)
    symbols = get_available_symbols()
    print(f"Training on {len(symbols)} symbols: {symbols}")
    
    if not symbols:
        print("ERROR: No data files found!")
        exit(1)
            
    agent.create_env(symbols=symbols)
    agent.create_model()
    
    # Train (Production Run: 2M steps)
    print("\nStarting PRODUCTION training (2,000,000 steps, ALL symbols)...")
    agent.train(total_timesteps=2_000_000)
    
    print("\n✅ Training complete!")
