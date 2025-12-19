"""
Validation utilities for PPO model testing.

Shared functions for walk-forward and cross-symbol validation.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from stable_baselines3.common.evaluation import evaluate_policy

from rl.ppo_agent import get_trading_agent
from rl.trading_env import TradingEnvironment
from stable_baselines3.common.monitor import Monitor
from core.logger import get_logger

logger = get_logger()


class ValidationThresholds:
    """
    Thresholds for model validation.
    
    These define minimum acceptable performance levels
    for different validation tests.
    """
    # Walk-forward validation
    WALK_FORWARD_MIN_REWARD = 0.0  # Must be positive
    WALK_FORWARD_MIN_RETENTION = 0.25  # 25% of train performance
    
    # Cross-symbol validation  
    CROSS_SYMBOL_MIN_REWARD = 0.0  # Must be positive
    CROSS_SYMBOL_MIN_RETENTION = 0.15  # 15% of train performance
    
    # Training baseline (from logs)
    BASELINE_TRAIN_REWARD = 88.3


def validate_model(
    test_type: str = "cross-symbol",
    n_eval_episodes: int = 20,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Run validation test on trained model.
    
    Args:
        test_type: "walk-forward" or "cross-symbol"
        n_eval_episodes: Number of episodes to evaluate
        verbose: Print results
        
    Returns:
        Dict with validation metrics
    """
    # Load model
    agent = get_trading_agent()
    model_path = Path("data/models/ppo_trading_agent/ppo_trading.zip")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    agent.load()
    
    if test_type == "cross-symbol":
        return _validate_cross_symbol(agent, n_eval_episodes, verbose)
    elif test_type == "walk-forward":
        return _validate_walk_forward(agent, n_eval_episodes, verbose)
    else:
        raise ValueError(f"Unknown test type: {test_type}")


def _validate_cross_symbol(
    agent,
    n_eval_episodes: int,
    verbose: bool
) -> Dict[str, float]:
    """Run cross-symbol validation."""
    from ml.symbols import TRAINING_SYMBOLS
    
    # Load available symbols
    data_dir = Path("data/vanna_ml")
    available_symbols = [
        f.stem.replace("_1min_rl", "")
        for f in data_dir.glob("*_1min_rl.parquet")
    ]
    
    # Test on first symbol (leave-one-out style)
    test_symbol = available_symbols[0] if available_symbols else "SPY"
    
    if verbose:
        logger.info(f"🧪 Cross-Symbol Validation: {test_symbol}")
    
    # Create test env
    test_env = TradingEnvironment(
        data_dir=str(data_dir),
        symbols=[test_symbol],
        episode_length=390
    )
    test_env = Monitor(test_env)
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        agent.model,
        test_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    
    retention = (mean_reward / ValidationThresholds.BASELINE_TRAIN_REWARD) * 100
    
    if verbose:
        logger.info(f"   Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"   Retention: {retention:.1f}%")
    
    return {
        "test_type": "cross-symbol",
        "reward": mean_reward,
        "std": std_reward,
        "retention_pct": retention,
        "passed": mean_reward > ValidationThresholds.CROSS_SYMBOL_MIN_REWARD
    }


def _validate_walk_forward(
    agent,
    n_eval_episodes: int,
    verbose: bool
) -> Dict[str, float]:
    """Run walk-forward validation."""
    import pandas as pd
    import shutil
    
    data_dir = Path("data/vanna_ml")
    symbol = "SPY"
    
    rl_path = data_dir / f"{symbol}_1min_rl.parquet"
    if not rl_path.exists():
        raise FileNotFoundError(f"Data not found: {rl_path}")
    
    df_full = pd.read_parquet(rl_path)
    
    # Split: first 50% train, last 50% test
    split_idx = len(df_full) // 2
    test_df = df_full.iloc[split_idx:].copy()
    
    if verbose:
        logger.info(f"🧪 Walk-Forward Validation: {symbol}")
        logger.info(f"   Test data: {len(test_df):,} rows (last 50%)")
    
    # Create temp test data
    temp_dir = Path("data/vanna_ml/temp_walk_forward")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"{symbol}_1min_rl.parquet"
    test_df.to_parquet(temp_path, index=False)
    
    try:
        # Create test env
        test_env = TradingEnvironment(
            data_dir=str(temp_dir),
            symbols=[symbol],
            episode_length=390
        )
        test_env = Monitor(test_env)
        
        # Evaluate
        mean_reward, std_reward = evaluate_policy(
            agent.model,
            test_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True
        )
        
        retention = (mean_reward / ValidationThresholds.BASELINE_TRAIN_REWARD) * 100
        
        if verbose:
            logger.info(f"   Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            logger.info(f"   Retention: {retention:.1f}%")
        
        return {
            "test_type": "walk-forward",
            "reward": mean_reward,
            "std": std_reward,
            "retention_pct": retention,
            "passed": mean_reward > ValidationThresholds.WALK_FORWARD_MIN_REWARD
        }
    
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def run_full_validation(verbose: bool = True) -> Dict[str, Dict]:
    """
    Run complete validation suite.
    
    Returns:
        Dict with results for each test
    """
    results = {}
    
    if verbose:
        logger.info("=" * 70)
        logger.info("🧪 PPO MODEL VALIDATION SUITE")
        logger.info("=" * 70)
    
    # Walk-forward
    try:
        results["walk_forward"] = validate_model("walk-forward", verbose=verbose)
    except Exception as e:
        logger.error(f"Walk-forward validation failed: {e}")
        results["walk_forward"] = {"passed": False, "error": str(e)}
    
    # Cross-symbol
    try:
        results["cross_symbol"] = validate_model("cross-symbol", verbose=verbose)
    except Exception as e:
        logger.error(f"Cross-symbol validation failed: {e}")
        results["cross_symbol"] = {"passed": False, "error": str(e)}
    
    # Summary
    all_passed = all(r.get("passed", False) for r in results.values())
    
    if verbose:
        logger.info("=" * 70)
        if all_passed:
            logger.info("✅ ALL VALIDATIONS PASSED")
        else:
            logger.error("❌ SOME VALIDATIONS FAILED")
            logger.warning("⚠️  Model may be overfitted - review results")
        logger.info("=" * 70)
    
    results["summary"] = {
        "all_passed": all_passed,
        "passed_count": sum(1 for r in results.values() if isinstance(r, dict) and r.get("passed", False)),
        "total_count": len([r for r in results.values() if isinstance(r, dict) and "passed" in r])
    }
    
    return results
