
import os
import sys
import numpy as np
import xgboost as xgb
print(f"XGBoost: {xgb.__version__}")
import torch
import faulthandler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from rl.vec_env import make_lightweight_env
from pathlib import Path

faulthandler.enable()

print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CPU Threads: {torch.get_num_threads()}")

# Force threads
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print(f"Forced CPU Threads: {torch.get_num_threads()}")

MODEL_PATH = Path("data/models/ppo_trading_agent")
print(f"Model Path: {MODEL_PATH}")

def test_load():
    try:
        norm_file = MODEL_PATH / "vec_normalize.pkl"
        print(f"Loading Normalizer from {norm_file}...")
        
        env_fns = [make_lightweight_env(n_envs=1) for _ in range(1)]
        dummy_env = DummyVecEnv(env_fns)
        
        vec_env = VecNormalize.load(str(norm_file), dummy_env)
        print("✅ Normalizer loaded successfully!")
        
        model_file = MODEL_PATH / "ppo_trading.zip"
        print(f"Loading PPO Model from {model_file}...")
        
        # Load on CPU
        model = PPO.load(str(model_file), env=vec_env, device='cpu')
        print("✅ PPO Model loaded successfully!")
        
        # INSPECT SHAPE
        print(f"Model Policy Obs Space: {model.policy.observation_space}")
        try:
             print(f"Model Policy Obs Shape: {model.policy.observation_space.shape}")
        except:
             pass
             
        # Test Inference
        print("Testing Inference...")
        # Try both 84 (current) and likely candidates
        obs_shape = model.policy.observation_space.shape
        obs = np.zeros((1, obs_shape[0]), dtype=np.float32)
        print(f"Feeding observation of shape {obs.shape}")
        action, _ = model.predict(obs, deterministic=True)
        print(f"✅ Predict Result: {action}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
