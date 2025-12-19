"""
Debug script to analyze PPO agent actions.
Runs a single evaluation episode and prints detailed action logs.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.ppo_agent import get_trading_agent
from rl.trading_env import TradingEnvironment

def debug_actions():
    print("🔍 DEBUGGING MODEL ACTIONS")
    print("-" * 50)
    
    # Load model
    agent = get_trading_agent()
    # Correct path with subfolder
    model_path = Path("data/models/ppo_trading_agent/ppo_trading.zip")
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return

    print("✅ Loading model...")
    agent.load()
    
    # Setup Env (SPY only)
    data_dir = "data/vanna_ml"
    symbol = "SPY"
    
    env = TradingEnvironment(
        data_dir=data_dir,
        symbols=[symbol],
        episode_length=390  # 1 day
    )
    
    obs, _ = env.reset()
    done = False
    truncated = False
    
    # Metrics
    action_counts = {0:0, 1:0, 2:0, 3:0, 4:0}
    action_names = {0: "HOLD", 1: "OPEN", 2: "CLOSE", 3: "INC", 4: "DEC"}
    rewards = []
    
    print("\n🎬 Running Episode...")
    step = 0
    while not done and not truncated:
        action, _ = agent.model.predict(obs, deterministic=True)
        action_int = int(action)
        
        # Log first 20 steps detailed
        if step < 20:
             print(f"Step {step:3d}: Action={action_names[action_int]} ({action_int})")
        
        action_counts[action_int] += 1
        
        obs, reward, done, truncated, info = env.step(action_int)
        rewards.append(reward)
        step += 1
        
    print("-" * 50)
    print("📊 ACTION STATISTICS")
    total_steps = sum(action_counts.values())
    for act, count in action_counts.items():
        pct = (count / total_steps) * 100 if total_steps > 0 else 0
        print(f"{action_names[act]:5s}: {count:4d} ({pct:5.1f}%)")
        
    print(f"\n💰 Total Reward: {sum(rewards):.4f}")
    print("-" * 50)

if __name__ == "__main__":
    debug_actions()
