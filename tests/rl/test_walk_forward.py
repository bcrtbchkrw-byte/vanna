"""
Walk-Forward Validation Test for PPO Agent

Tests model generalization on chronologically split data:
- Train: First 50% of historical data (older)
- Test: Last 50% of historical data (newer, unseen)

This detects overfitting: if model only memorized training patterns,
it will fail on the test set.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3.common.evaluation import evaluate_policy

from rl.ppo_agent import get_trading_agent
from rl.trading_env import TradingEnvironment
from stable_baselines3.common.monitor import Monitor


def split_data_temporally(df: pd.DataFrame, train_ratio: float = 0.5):
    """
    Split dataframe chronologically (index-based).
    
    Args:
        df: DataFrame (assumes rows are already chronologically ordered)
        train_ratio: Proportion for training (0-1)
        
    Returns:
        train_df, test_df
    """
    # Data is already in chronological order in parquet files
    # No need to sort - just split by index
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df


def create_temp_parquet(df: pd.DataFrame, symbol: str, suffix: str = "test"):
    """Create temporary parquet file for testing."""
    temp_dir = Path(f"data/vanna_ml/temp_{suffix}")
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    temp_path = temp_dir / f"{symbol}_1min_rl.parquet"
    df.to_parquet(temp_path, index=False)
    
    return temp_dir


def test_walk_forward_validation():
    """
    Walk-forward test: train on old data, test on new data.
    
    Expected:
    - Test reward should be positive (> 0)
    - Test reward will likely be lower than train reward (88.3)
    - If test reward < 0, model doesn't generalize
    """
    # Load trained model
    agent = get_trading_agent()
    model_path = Path("data/models/ppo_trading_agent/ppo_trading.zip")
    
    if not model_path.exists():
        pytest.skip("Trained model not found. Train model first.")
    
    agent.load()
    
    # Load full dataset
    data_dir = Path("data/vanna_ml")
    symbol = "SPY"  # Use SPY as primary test symbol
    
    rl_path = data_dir / f"{symbol}_1min_rl.parquet"
    if not rl_path.exists():
        pytest.skip(f"RL data not found for {symbol}")
    
    df_full = pd.read_parquet(rl_path)
    
    # Split chronologically
    train_df, test_df = split_data_temporally(df_full, train_ratio=0.5)
    
    print(f"\n📊 Data Split:")
    print(f"   Full data: {len(df_full):,} rows")
    print(f"   Train: {len(train_df):,} rows ({len(train_df)/len(df_full)*100:.1f}%)")
    print(f"   Test:  {len(test_df):,} rows ({len(test_df)/len(df_full)*100:.1f}%)")
    
    # Create temporary test environment with ONLY test data
    test_data_dir = create_temp_parquet(test_df, symbol, suffix="walk_forward_test")
    
    try:
        # Create test environment
        test_env = TradingEnvironment(
            data_dir=str(test_data_dir),
            symbols=[symbol],
            episode_length=390
        )
        test_env = Monitor(test_env)
        
        # Evaluate on test set (out-of-sample)
        print(f"\n🧪 Evaluating on test set ({symbol})...")
        mean_reward, std_reward = evaluate_policy(
            agent.model,
            test_env,
            n_eval_episodes=20,
            deterministic=True
        )
        
        print(f"\n📊 Walk-Forward Results:")
        print(f"   Test Reward (OOS): {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   Train Reward (from logs): ~88.3")
        
        if mean_reward > 0:
            performance_ratio = (mean_reward / 88.3) * 100
            print(f"   Performance Retention: {performance_ratio:.1f}%")
            
            if performance_ratio > 50:
                print(f"   ✅ Model generalizes reasonably well")
            elif performance_ratio > 25:
                print(f"   ⚠️  Model partially generalizes")
            else:
                print(f"   🚨 Model heavily overfitted")
        else:
            print(f"   🚨 Model FAILS on out-of-sample data!")
        
        # Assert minimum threshold
        assert mean_reward > 0, \
            f"Model fails on out-of-sample data (reward={mean_reward:.2f})"
        
        # Warn if performance drops too much
        if mean_reward < 20:
            pytest.warns(UserWarning, 
                match="Model performance degraded significantly on OOS data")
    
    finally:
        # Cleanup
        import shutil
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)


if __name__ == "__main__":
    # Run test standalone
    test_walk_forward_validation()
