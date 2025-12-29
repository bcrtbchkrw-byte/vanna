"""
Vectorized Environment for Multi-Symbol RL Training

Uses stable-baselines3 SubprocVecEnv to train on all symbols simultaneously:
- SPY, QQQ, IWM, GLD, TLT (5 parallel environments)

Each environment uses its own *_1min_vanna.parquet data.
"""
from typing import List, Optional, Callable
from pathlib import Path
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from core.logger import get_logger
from rl.trading_env import TradingEnvironment
from gymnasium import spaces
import numpy as np

logger = get_logger()


class LightweightTradingEnv(gym.Env):
    """
    Lightweight environment for inference/loading only.
    Loads NO data, just defines spaces.
    """
    def __init__(self, n_features: int = 84):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)  # Matches TradingEnvironment

    def reset(self, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, False, False, {}


def make_lightweight_env(n_envs: int = 1) -> Callable[[], gym.Env]:
    """Create lightweight env factory."""
    def _init():
        return LightweightTradingEnv()
    return _init


# Single Source of Truth: import from ml.symbols
from ml.symbols import TRAINING_SYMBOLS as DEFAULT_SYMBOLS


def make_env(
    symbol: str,
    data_dir: str = "data/vanna_ml",
    initial_capital: float = 10000.0,
    episode_length: int = 390,
    rank: int = 0,
    seed: int = 0
) -> Callable[[], gym.Env]:
    """
    Create a callable that returns a single-symbol environment.
    
    Args:
        symbol: Symbol to trade (SPY, QQQ, etc.)
        data_dir: Path to parquet files
        initial_capital: Starting capital
        episode_length: Steps per episode
        rank: Environment index for seeding
        seed: Base random seed
        
    Returns:
        Callable that creates the environment
    """
    def _init() -> gym.Env:
        env = TradingEnvironment(
            data_dir=data_dir,
            symbols=[symbol],  # Single symbol per env
            initial_capital=initial_capital,
            episode_length=episode_length
        )
        env.reset(seed=seed + rank)
        # Wrap with Monitor for logging
        env = Monitor(env)
        return env
    
    return _init


def make_vec_env(
    symbols: List[str] = None,
    data_dir: str = "data/vanna_ml",
    initial_capital: float = 10000.0,
    episode_length: int = 390,
    n_envs_per_symbol: int = 1,
    use_subproc: bool = True,
    seed: int = None,  # Dynamic by default (None = use timestamp)
    normalize: bool = True
) -> VecNormalize:
    """
    Create vectorized environment for multi-symbol training.
    
    Args:
        symbols: List of symbols to trade
        data_dir: Path to parquet files
        initial_capital: Starting capital per env
        episode_length: Steps per episode
        n_envs_per_symbol: Number of parallel envs per symbol
        use_subproc: Use SubprocVecEnv (parallel) vs DummyVecEnv (sequential)
        seed: Random seed (None = dynamic based on timestamp)
        normalize: Wrap with VecNormalize
        
    Returns:
        VecNormalize wrapped vectorized environment
    """
    if symbols is None:
        symbols = get_available_symbols(data_dir)
    
    if not symbols:
        raise ValueError(f"No symbols found with *_vanna.parquet in {data_dir}")
    
    # Dynamic seed if not specified (different each training run)
    if seed is None:
        import time
        seed = int(time.time()) % 2**31
        logger.info(f"Using dynamic seed for PPO environments: {seed}")
    
    logger.info(f"Creating {len(symbols)} × {n_envs_per_symbol} = {len(symbols) * n_envs_per_symbol} parallel environments")
    logger.info(f"Symbols: {symbols}")
    
    # Create env factories
    env_fns = []
    rank = 0
    
    for symbol in symbols:
        for _ in range(n_envs_per_symbol):
            env_fns.append(
                make_env(
                    symbol=symbol,
                    data_dir=data_dir,
                    initial_capital=initial_capital,
                    episode_length=episode_length,
                    rank=rank,
                    seed=seed
                )
            )
            rank += 1
    
    # Create vectorized environment
    if use_subproc and len(env_fns) > 1:
        vec_env = SubprocVecEnv(env_fns)
        logger.info("Using SubprocVecEnv (parallel processing)")
    else:
        vec_env = DummyVecEnv(env_fns)
        logger.info("Using DummyVecEnv (sequential)")
    
    # Wrap with VecNormalize for observation/reward normalization
    if normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        logger.info("Wrapped with VecNormalize")
    
    return vec_env


def get_available_symbols(data_dir: str = "data/vanna_ml") -> List[str]:
    """
    Get list of symbols that have RL training data (*_1min_rl.parquet).
    
    Only uses enriched RL data files - these are created by the 
    data pipeline after feature engineering and normalization.
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return []
    
    # Only look for *_1min_rl.parquet files (enriched RL training data)
    parquet_files = list(data_path.glob("*_1min_rl.parquet"))
    
    # Extract symbol names
    symbols = [f.stem.replace("_1min_rl", "") for f in parquet_files]
    
    if symbols:
        logger.info(f"Found {len(symbols)} symbols with RL data: {symbols}")
    else:
        logger.warning("No *_rl.parquet files found. Run data pipeline first.")
    
    return sorted(symbols)


def create_training_env(
    use_all_symbols: bool = True,
    n_envs: int = None
) -> VecNormalize:
    """
    Convenience function to create training environment.
    
    Args:
        use_all_symbols: If True, use all available symbols
        n_envs: Optional override for number of environments
        
    Returns:
        Ready-to-train vectorized environment
    """
    symbols = get_available_symbols() if use_all_symbols else ['SPY']
    
    if n_envs is None:
        # 1 env per symbol by default
        n_envs_per_symbol = 1
    else:
        # Distribute n_envs across symbols
        n_envs_per_symbol = max(1, n_envs // len(symbols))
    
    return make_vec_env(
        symbols=symbols,
        n_envs_per_symbol=n_envs_per_symbol,
        use_subproc=True,
        normalize=True
    )


# CLI for testing
if __name__ == "__main__":
    from core.logger import setup_logger
    
    try:
        setup_logger(level="INFO")
    except:
        pass
    
    print("=" * 60)
    print("Vectorized Environment Test")
    print("=" * 60)
    
    # Check available symbols
    symbols = get_available_symbols()
    print(f"Available symbols: {symbols}")
    
    if not symbols:
        print("No *_vanna.parquet files found!")
        exit(1)
    
    # Create vec env
    print("\nCreating vectorized environment...")
    vec_env = make_vec_env(
        symbols=symbols,
        n_envs_per_symbol=1,
        use_subproc=False,  # Use Dummy for testing
        normalize=True
    )
    
    print(f"Number of envs: {vec_env.num_envs}")
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")
    
    # Test reset
    obs = vec_env.reset()
    print(f"\nReset observations shape: {obs.shape}")
    
    # Test step
    actions = [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
    obs, rewards, dones, infos = vec_env.step(actions)
    
    print(f"Step observations shape: {obs.shape}")
    print(f"Rewards: {rewards}")
    
    # Cleanup
    vec_env.close()
    
    print("\n✅ Vectorized environment works!")
