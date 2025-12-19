import numpy as np
import os

path = "data/models/ppo_trading_agent/logs/evaluations.npz"

if not os.path.exists(path):
    print(f"File not found: {path}")
    exit(1)

data = np.load(path)
print("Files in archive:", data.files)

# 'results' shape is usually (n_evaluations, n_episodes)
# 'timesteps' shape is (n_evaluations,)

timesteps = data['timesteps']
results = data['results']

print(f"\nLoaded {len(timesteps)} evaluation points.")

for i, t in enumerate(timesteps):
    mean_reward = np.mean(results[i])
    std_reward = np.std(results[i])
    print(f"Step {t:7d}: Mean Reward = {mean_reward:8.2f} +/- {std_reward:6.2f}")

print("\nBest Model Logic:")
best_idx = np.argmax(np.mean(results, axis=1))
print(f"Best evaluation was at Step {timesteps[best_idx]} with Reward {np.mean(results[best_idx]):.2f}")
