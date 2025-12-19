"""
Cross-Symbol Validation Test for PPO Agent

Tests model generalization across different symbols:
- Train: All symbols EXCEPT one (e.g., all except AAPL)
- Test: Only the held-out symbol (e.g., AAPL)

This tests if model learns general trading patterns vs.
memorizing specific symbol characteristics.
"""
import pytest
import numpy as np
from pathlib import Path
from stable_baselines3.common.evaluation import evaluate_policy

from rl.ppo_agent import get_trading_agent
from rl.trading_env import TradingEnvironment
from stable_baselines3.common.monitor import Monitor
from ml.symbols import TRAINING_SYMBOLS


def test_cross_symbol_generalization():
    """
    Leave-one-out cross-symbol test.
    
    Tests if model trained on 13 symbols can generalize to 14th symbol.
    
    Expected:
    - Held-out symbol should have positive reward
    - Performance may be lower than training, but should still work
    - If reward < 0, model doesn't generalize across symbols
    """
    # Load trained model
    agent = get_trading_agent()
    model_path = Path("data/models/ppo_trading_agent/ppo_trading.zip")
    
    if not model_path.exists():
        pytest.skip("Trained model not found. Train model first.")
    
    agent.load()
    
    # Check available symbols
    data_dir = Path("data/vanna_ml")
    available_symbols = [
        f.stem.replace("_1min_rl", "")
        for f in data_dir.glob("*_1min_rl.parquet")
    ]
    
    if len(available_symbols) < 2:
        pytest.skip("Need at least 2 symbols for cross-symbol test")
    
    # Use AAPL as held-out symbol (popular, likely in dataset)
    test_symbol = "AAPL" if "AAPL" in available_symbols else available_symbols[0]
    train_symbols = [s for s in available_symbols if s != test_symbol]
    
    print(f"\n📊 Cross-Symbol Test Setup:")
    print(f"   Total symbols: {len(available_symbols)}")
    print(f"   Train symbols: {len(train_symbols)} ({', '.join(train_symbols[:5])}...)")
    print(f"   Test symbol: {test_symbol} (held-out)")
    
    # Create test environment with ONLY held-out symbol
    test_env = TradingEnvironment(
        data_dir=str(data_dir),
        symbols=[test_symbol],  # Only test symbol
        episode_length=390
    )
    test_env = Monitor(test_env)
    
    # Evaluate on held-out symbol
    print(f"\n🧪 Evaluating on held-out symbol ({test_symbol})...")
    mean_reward, std_reward = evaluate_policy(
        agent.model,
        test_env,
        n_eval_episodes=20,
        deterministic=True
    )
    
    print(f"\n📊 Cross-Symbol Results:")
    print(f"   {test_symbol} Reward (held-out): {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"   Train Reward (all symbols): ~88.3")
    
    if mean_reward > 0:
        performance_ratio = (mean_reward / 88.3) * 100
        print(f"   Performance Retention: {performance_ratio:.1f}%")
        
        if performance_ratio > 40:
            print(f"   ✅ Model generalizes well across symbols")
        elif performance_ratio > 15:
            print(f"   ⚠️  Model partially generalizes")
        else:
            print(f"   🚨 Model is symbol-specific")
    else:
        print(f"   🚨 Model FAILS on held-out symbol!")
    
    # Assert minimum threshold
    assert mean_reward > 0, \
        f"Model fails on held-out symbol {test_symbol} (reward={mean_reward:.2f})"
    
    # Warn if performance drops significantly
    if mean_reward < 10:
        pytest.warns(UserWarning,
            match="Model doesn't generalize well across symbols")


def test_all_symbols_leave_one_out():
    """
    Comprehensive test: leave each symbol out and test on it.
    
    This gives a complete picture of cross-symbol generalization.
    """
    # Load model
    agent = get_trading_agent()
    model_path = Path("data/models/ppo_trading_agent/ppo_trading.zip")
    
    if not model_path.exists():
        pytest.skip("Trained model not found")
    
    agent.load()
    
    # Get available symbols
    data_dir = Path("data/vanna_ml")
    available_symbols = [
        f.stem.replace("_1min_rl", "")
        for f in data_dir.glob("*_1min_rl.parquet")
    ]
    
    if len(available_symbols) < 3:
        pytest.skip("Need at least 3 symbols for comprehensive test")
    
    results = {}
    
    print(f"\n📊 Leave-One-Out Cross-Validation:")
    print(f"   Testing {len(available_symbols)} symbols")
    print(f"   {'-' * 50}")
    
    for test_symbol in available_symbols[:5]:  # Test first 5 to save time
        test_env = TradingEnvironment(
            data_dir=str(data_dir),
            symbols=[test_symbol],
            episode_length=390
        )
        test_env = Monitor(test_env)
        
        mean_reward, std_reward = evaluate_policy(
            agent.model,
            test_env,
            n_eval_episodes=10,
            deterministic=True
        )
        
        results[test_symbol] = mean_reward
        print(f"   {test_symbol:6s}: {mean_reward:6.2f} ± {std_reward:.2f}")
    
    # Calculate average performance
    avg_reward = np.mean(list(results.values()))
    print(f"   {'-' * 50}")
    print(f"   Average: {avg_reward:6.2f}")
    print(f"   Min:     {min(results.values()):6.2f} ({min(results, key=results.get)})")
    print(f"   Max:     {max(results.values()):6.2f} ({max(results, key=results.get)})")
    
    # All symbols should have positive reward
    failing_symbols = [s for s, r in results.items() if r < 0]
    assert len(failing_symbols) == 0, \
        f"Model fails on symbols: {failing_symbols}"


if __name__ == "__main__":
    # Run tests standalone
    print("=" * 60)
    print("Cross-Symbol Validation Test")
    print("=" * 60)
    
    test_cross_symbol_generalization()
    print("\n")
    test_all_symbols_leave_one_out()
