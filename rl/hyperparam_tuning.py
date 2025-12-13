"""
Hyperparameter Tuning with RL Baselines3 Zoo

Uses Optuna for automated hyperparameter optimization.
Compatible with rl-baselines3-zoo train script.
"""
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import yaml
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from rl.vec_env import make_vec_env, get_available_symbols
from core.logger import get_logger

logger = get_logger()


# Default hyperparameter search space
HYPERPARAMS_SPACE = {
    "learning_rate": {
        "type": "float",
        "low": 1e-5,
        "high": 1e-3,
        "log": True
    },
    "n_steps": {
        "type": "categorical",
        "choices": [256, 512, 1024, 2048]
    },
    "batch_size": {
        "type": "categorical",
        "choices": [32, 64, 128, 256]
    },
    "n_epochs": {
        "type": "int",
        "low": 3,
        "high": 20
    },
    "gamma": {
        "type": "float",
        "low": 0.9,
        "high": 0.9999,
        "log": True
    },
    "gae_lambda": {
        "type": "float",
        "low": 0.9,
        "high": 0.999
    },
    "clip_range": {
        "type": "float",
        "low": 0.1,
        "high": 0.4
    },
    "ent_coef": {
        "type": "float",
        "low": 1e-8,
        "high": 0.1,
        "log": True
    },
    "vf_coef": {
        "type": "float",
        "low": 0.1,
        "high": 1.0
    },
    "max_grad_norm": {
        "type": "float",
        "low": 0.1,
        "high": 10.0
    }
}


def sample_hyperparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters using Optuna trial."""
    hyperparams = {}
    
    for name, config in HYPERPARAMS_SPACE.items():
        if config["type"] == "float":
            hyperparams[name] = trial.suggest_float(
                name,
                config["low"],
                config["high"],
                log=config.get("log", False)
            )
        elif config["type"] == "int":
            hyperparams[name] = trial.suggest_int(
                name,
                config["low"],
                config["high"]
            )
        elif config["type"] == "categorical":
            hyperparams[name] = trial.suggest_categorical(
                name,
                config["choices"]
            )
    
    return hyperparams


class HyperparameterTuner:
    """
    Hyperparameter tuning for PPO using Optuna.
    
    Compatible with RL Baselines3 Zoo methodology.
    """
    
    def __init__(
        self,
        n_trials: int = 100,
        n_timesteps: int = 50_000,
        n_eval_episodes: int = 5,
        eval_freq: int = 10_000,
        study_name: str = "ppo_trading",
        storage: str = None,
        output_dir: str = "data/tuning"
    ):
        self.n_trials = n_trials
        self.n_timesteps = n_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.study_name = study_name
        self.storage = storage
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: float = float("-inf")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Sample hyperparameters
        hyperparams = sample_hyperparams(trial)
        
        logger.info(f"Trial {trial.number}: {hyperparams}")
        
        try:
            # Create environments
            train_env = make_vec_env(
                symbols=get_available_symbols(),
                n_envs_per_symbol=1,
                use_subproc=False,
                normalize=True
            )
            
            eval_env = make_vec_env(
                symbols=get_available_symbols()[:1],
                n_envs_per_symbol=1,
                use_subproc=False,
                normalize=True
            )
            
            # Create model with sampled hyperparams
            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                verbose=0,
                **hyperparams
            )
            
            # Eval callback for pruning
            eval_callback = TrialEvalCallback(
                eval_env=eval_env,
                trial=trial,
                n_eval_episodes=self.n_eval_episodes,
                eval_freq=self.eval_freq
            )
            
            # Train
            model.learn(
                total_timesteps=self.n_timesteps,
                callback=eval_callback
            )
            
            # Get final reward
            mean_reward = eval_callback.last_mean_reward
            
            # Cleanup
            train_env.close()
            eval_env.close()
            
            return mean_reward
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
    
    def tune(self) -> Dict[str, Any]:
        """Run hyperparameter tuning."""
        logger.info(f"Starting tuning: {self.n_trials} trials, {self.n_timesteps} timesteps each")
        
        # Create Optuna study
        sampler = TPESampler(n_startup_trials=10, seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Best params
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        logger.info(f"Best value: {self.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        # Save results
        self._save_results(study)
        
        return self.best_params
    
    def _save_results(self, study: optuna.Study):
        """Save tuning results."""
        # Best params as YAML (RL Zoo format)
        yaml_path = self.output_dir / f"{self.study_name}_best.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump({
                "TradingEnv": {
                    "policy": "MlpPolicy",
                    **study.best_params
                }
            }, f, default_flow_style=False)
        
        logger.info(f"Saved best params to {yaml_path}")
        
        # All trials as CSV
        df = study.trials_dataframe()
        csv_path = self.output_dir / f"{self.study_name}_trials.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved trials to {csv_path}")


class TrialEvalCallback(EvalCallback):
    """Eval callback that reports to Optuna for pruning."""
    
    def __init__(
        self,
        eval_env: VecNormalize,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        **kwargs
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            best_model_save_path=None,
            log_path=None,
            **kwargs
        )
        self.trial = trial
        self.eval_idx = 0
        self.last_mean_reward = float("-inf")
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            result = super()._on_step()
            
            if self.last_mean_reward is not None:
                self.last_mean_reward = self.last_mean_reward
            
            # Report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            self.eval_idx += 1
            
            # Prune if needed
            if self.trial.should_prune():
                raise optuna.TrialPruned()
            
            return result
        return True


def create_zoo_config() -> str:
    """
    Create RL Baselines3 Zoo compatible config file.
    
    Returns path to config file.
    """
    config = {
        "TradingEnv": {
            "normalize": True,
            "n_envs": 2,
            "policy": "MlpPolicy",
            "n_timesteps": 100000,
            
            # PPO specific
            "learning_rate": "lin_3e-4",
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            
            # Hyperparameter tuning
            "optimize_hyperparameters": True,
            "n_trials": 100,
            "n_startup_trials": 10,
            "sampler": "tpe",
            "pruner": "median"
        }
    }
    
    config_path = Path("data/tuning/hyperparams.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created RL Zoo config: {config_path}")
    return str(config_path)


# Quick run
if __name__ == "__main__":
    from core.logger import setup_logger
    
    try:
        setup_logger(level="INFO")
    except:
        pass
    
    print("=" * 60)
    print("Hyperparameter Tuning")
    print("=" * 60)
    
    # Quick test with few trials
    tuner = HyperparameterTuner(
        n_trials=5,
        n_timesteps=5000
    )
    
    best = tuner.tune()
    print(f"\nBest params: {best}")
