
"""
RL Training Script

Usage:
    python3 train.py
"""
import asyncio
from core.logger import setup_logger
from rl.ppo_agent import TradingAgent, get_available_symbols

def train():
    try:
        setup_logger(level="INFO")
    except:
        pass

    print("=" * 60)
    print("🚀 VANNA RL TRAINING")
    print("=" * 60)
    
    # 1. Check Data
    symbols = get_available_symbols()
    print(f"Found data for: {symbols}")
    
    if not symbols:
        print("❌ No training data found in data/vanna_ml!")
        print("   Please run initial data collection first.")
        return

    # 2. Initialize Agent
    print("\nInitializing PPO Agent...")
    agent = TradingAgent(
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64, 
        n_epochs=10
    )
    
    # 3. Create Environment
    print("Creating vectorized environment...")
    agent.create_env(symbols=symbols)
    
    # 4. Train
    print("\nStarting training loop (Target: 100,000 steps)...")
    try:
        agent.train(
            total_timesteps=100_000,
            eval_freq=10_000,
            checkpoint_freq=50_000
        )
        print("\n✅ Training Complete!")
        print(f"   Model saved to: {agent.model_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted. Saving current progress...")
        agent.save()
        print("✅ Saved.")

if __name__ == "__main__":
    train()
