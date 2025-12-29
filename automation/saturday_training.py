#!/usr/bin/env python3
"""
saturday_training_full.py

MAXIMUM TRAINING PIPELINE - USE ALL 16 YEARS OF DATA!

Strategie:
- XGBoost: VŠECH 16 let dat, hyperparameter tuning
- LSTM: VŠECH 16 let dat, architecture search
- PPO: Hyperparameter optimization + full training

Sobota = neomezený čas → maximální kvalita modelů!

Usage:
    python saturday_training_full.py --full          # Vše včetně HPO
    python saturday_training_full.py --quick         # Bez HPO (rychlejší)
    python saturday_training_full.py --hpo-only      # Jen hyperparameter search
"""
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import time
import json
import numpy as np
import pandas as pd

# Logging
import logging
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - MAXIMUM DATA USAGE
# =============================================================================

class FullTrainingConfig:
    """Konfigurace pro MAXIMUM využití dat."""
    
    # Paths
    DATA_DIR = Path("data/raw")
    MODELS_DIR = Path("data/models")
    HPO_RESULTS = Path("data/hpo_results.json")
    
    # XGBoost Configuration
    XGBOOST = {
        # Data
        'use_all_data': True,           # VŠECH 16 let!
        'test_size': 0.15,              # 15% pro test (cca 2.4 roku)
        'validation_size': 0.15,        # 15% pro validation
        
        # Hyperparameter search space
        'hpo_space': {
            'n_estimators': [100, 200, 300, 500, 1000],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [1, 1.5, 2, 3],
        },
        'hpo_iterations': 100,          # Počet kombinací k vyzkoušení
        'cv_folds': 5,                  # Cross-validation folds
    }
    
    # LSTM Configuration
    LSTM = {
        'use_all_data': True,
        'test_size': 0.15,
        'validation_size': 0.15,
        'sequence_length': 60,          # 60 minut kontextu
        
        # Architecture search
        'architectures': [
            {'lstm_units': [32], 'dense_units': 16},
            {'lstm_units': [64], 'dense_units': 32},
            {'lstm_units': [64, 32], 'dense_units': 32},
            {'lstm_units': [128, 64], 'dense_units': 64},
            {'lstm_units': [128, 64, 32], 'dense_units': 32},
        ],
        'learning_rates': [0.0001, 0.0005, 0.001],
        'batch_sizes': [32, 64, 128],
        'epochs': 50,
        'early_stopping_patience': 10,
    }
    
    # PPO Configuration
    PPO = {
        'use_all_data': True,
        
        # Hyperparameter search space
        'hpo_space': {
            'learning_rate': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
            'n_steps': [256, 512, 1024, 2048, 4096],
            'batch_size': [32, 64, 128, 256],
            'n_epochs': [3, 5, 10, 20],
            'gamma': [0.95, 0.97, 0.99, 0.995],
            'gae_lambda': [0.9, 0.95, 0.98],
            'clip_range': [0.1, 0.2, 0.3],
            'ent_coef': [0.0, 0.001, 0.01, 0.1],
            'vf_coef': [0.25, 0.5, 1.0],
            'max_grad_norm': [0.3, 0.5, 1.0],
        },
        'hpo_iterations': 50,             # Počet kombinací pro HPO
        'hpo_timesteps': 100_000,         # Timesteps per HPO trial
        
        # Training timesteps - VÝRAZNĚ ZVÝŠENO!
        # 14 symbolů × 16 let × 390 min = ~21M datových bodů
        # PPO potřebuje vidět data 5-10x pro stabilní učení
        'full_timesteps': 100_000_000,    # 100M pro kvartální FULL train
        'monthly_timesteps': 10_000_000,  # 10M pro měsíční continue
        'weekly_timesteps': 1_000_000,    # 1M pro týdenní quick update
        
        'eval_episodes': 100,             # Episodes pro evaluaci
    }
    
    # HPO Schedule - kdy hledat hyperparametry
    HPO_SCHEDULE = {
        # HPO jen KVARTÁLNĚ (měsíce 1, 4, 7, 10)
        'hpo_months': [1, 4, 7, 10],
        
        # Full retrain KVARTÁLNĚ
        'full_retrain_months': [1, 4, 7, 10],
        
        # Continue training MĚSÍČNĚ
        'continue_months': [2, 3, 5, 6, 8, 9, 11, 12],
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(data_dir: Path) -> pd.DataFrame:
    """
    Načte VŠECHNA data (16 let).
    
    Preferuje *_features.parquet (s 126+ features),
    fallback na *_vanna.parquet nebo *_1min.parquet.
    
    Returns:
        DataFrame se všemi daty
    """
    dfs = []
    
    # Priority: _features > _vanna > _1min
    for symbol_pattern in ['*_1min_features.parquet', '*_1min_vanna.parquet', '*_1min.parquet']:
        for f in data_dir.glob(symbol_pattern):
            # Skip if we already have this symbol
            symbol = f.stem.split('_')[0]
            if any(symbol in str(existing) for existing in [d.get('_source', '') for d in dfs if isinstance(d, dict)]):
                continue
            
            try:
                df = pd.read_parquet(f)
                df['symbol'] = symbol
                df['_source'] = str(f)
                dfs.append(df)
                logger.info(f"  Loaded {symbol}: {len(df):,} rows, {len(df.columns)} cols from {f.name}")
            except Exception as e:
                logger.error(f"  Failed {f.name}: {e}")
        
        # Stop if we found files
        if dfs:
            break
    
    if not dfs:
        raise ValueError("No data files found!")
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Sort by timestamp if available
    if 'timestamp' in combined.columns:
        combined = combined.sort_values('timestamp').reset_index(drop=True)
    
    # Remove helper column
    if '_source' in combined.columns:
        combined = combined.drop(columns=['_source'])
    
    logger.info(f"Total: {len(combined):,} rows from {len(dfs)} symbols, {len(combined.columns)} features")
    
    return combined


def temporal_train_test_split(df: pd.DataFrame, 
                               test_size: float = 0.15,
                               val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Časový split - NIKDY nemíchej budoucnost do tréninku!
    
    [====== TRAIN ======][== VAL ==][== TEST ==]
         70%                15%         15%
    """
    n = len(df)
    
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    logger.info(f"Split: Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")
    
    return train, val, test


# =============================================================================
# XGBOOST HYPERPARAMETER OPTIMIZATION
# =============================================================================

class XGBoostHPO:
    """
    XGBoost Hyperparameter Optimization.
    
    Používá RandomizedSearchCV s časovým cross-validation.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or FullTrainingConfig.XGBOOST
        self.best_params = None
        self.best_score = None
        self.results = []
    
    def run(self, df: pd.DataFrame, 
            target_col: str = 'is_successful',
            feature_cols: List[str] = None) -> Dict:
        """
        Spustí hyperparameter optimization.
        """
        import xgboost as xgb
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
        from sklearn.metrics import roc_auc_score, make_scorer
        
        logger.info("=" * 60)
        logger.info("🔍 XGBoost Hyperparameter Optimization")
        logger.info(f"   Data: {len(df):,} samples")
        logger.info(f"   Iterations: {self.config['hpo_iterations']}")
        logger.info("=" * 60)
        
        # Prepare features
        if feature_cols is None:
            feature_cols = [c for c in df.columns 
                          if c not in [target_col, 'timestamp', 'symbol', 'date']]
        
        X = df[feature_cols].fillna(0)
        y = df[target_col].astype(int)
        
        # Time series cross-validation (respektuje čas!)
        tscv = TimeSeriesSplit(n_splits=self.config['cv_folds'])
        
        # Base model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42
        )
        
        # Randomized search
        search = RandomizedSearchCV(
            base_model,
            param_distributions=self.config['hpo_space'],
            n_iter=self.config['hpo_iterations'],
            cv=tscv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=2,
            random_state=42
        )
        
        start = time.time()
        search.fit(X, y)
        elapsed = time.time() - start
        
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        
        # Log results
        logger.info(f"\n✅ HPO Complete in {elapsed/60:.1f} minutes")
        logger.info(f"   Best AUC: {self.best_score:.4f}")
        logger.info(f"   Best params:")
        for k, v in self.best_params.items():
            logger.info(f"      {k}: {v}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'elapsed_minutes': elapsed / 60,
            'n_iterations': self.config['hpo_iterations']
        }
    
    def train_final_model(self, train_df: pd.DataFrame, 
                          val_df: pd.DataFrame,
                          test_df: pd.DataFrame,
                          target_col: str = 'is_successful',
                          feature_cols: List[str] = None) -> Dict:
        """
        Natrénuje finální model s nejlepšími parametry.
        """
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
        
        if self.best_params is None:
            raise ValueError("Run HPO first!")
        
        logger.info("\n🌲 Training final XGBoost model...")
        
        if feature_cols is None:
            feature_cols = [c for c in train_df.columns 
                          if c not in [target_col, 'timestamp', 'symbol', 'date']]
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col].astype(int)
        
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df[target_col].astype(int)
        
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_col].astype(int)
        
        # Final model with best params
        model = xgb.XGBClassifier(
            **self.best_params,
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42
        )
        
        # Train with early stopping on validation
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        
        # Evaluate on TEST set (unseen data!)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_auc': roc_auc_score(y_test, y_proba),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else None
        }
        
        logger.info(f"\n✅ Final Model Results:")
        logger.info(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"   Test AUC: {metrics['test_auc']:.4f}")
        
        # Save model
        import joblib
        model_path = FullTrainingConfig.MODELS_DIR / "trade_success_predictor.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': model,
            'feature_names': feature_cols,
            'best_params': self.best_params,
            'metrics': metrics
        }, model_path)
        
        logger.info(f"   Saved to: {model_path}")
        
        return metrics


# =============================================================================
# LSTM ARCHITECTURE SEARCH
# =============================================================================

class LSTMArchitectureSearch:
    """
    LSTM Architecture Search.
    
    Hledá nejlepší kombinaci:
    - Počet LSTM vrstev
    - Units per layer
    - Learning rate
    - Batch size
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or FullTrainingConfig.LSTM
        self.best_architecture = None
        self.best_score = None
        self.results = []
    
    def run(self, df: pd.DataFrame,
            target_col: str = 'regime_target',
            feature_cols: List[str] = None) -> Dict:
        """
        Spustí architecture search.
        """
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, callbacks
        except ImportError:
            logger.error("TensorFlow not installed!")
            return {'error': 'TensorFlow not installed'}
        
        logger.info("=" * 60)
        logger.info("🧠 LSTM Architecture Search")
        logger.info(f"   Data: {len(df):,} samples")
        logger.info(f"   Architectures to try: {len(self.config['architectures'])}")
        logger.info("=" * 60)
        
        # Prepare features
        if feature_cols is None:
            feature_cols = [c for c in df.columns 
                          if c not in [target_col, 'timestamp', 'symbol', 'date', 'is_successful']]
        
        # Split data
        train_df, val_df, test_df = temporal_train_test_split(
            df, 
            test_size=self.config['test_size'],
            val_size=self.config['validation_size']
        )
        
        best_val_acc = 0
        best_config = None
        
        for arch in self.config['architectures']:
            for lr in self.config['learning_rates']:
                for batch_size in self.config['batch_sizes']:
                    
                    config_str = f"LSTM{arch['lstm_units']}_lr{lr}_bs{batch_size}"
                    logger.info(f"\n   Testing: {config_str}")
                    
                    try:
                        # Build model
                        model = self._build_model(
                            input_shape=(self.config['sequence_length'], len(feature_cols)),
                            lstm_units=arch['lstm_units'],
                            dense_units=arch['dense_units'],
                            learning_rate=lr,
                            num_classes=5  # 5 regimes
                        )
                        
                        # Prepare sequences
                        X_train, y_train = self._create_sequences(
                            train_df[feature_cols].values,
                            train_df[target_col].values,
                            self.config['sequence_length']
                        )
                        
                        X_val, y_val = self._create_sequences(
                            val_df[feature_cols].values,
                            val_df[target_col].values,
                            self.config['sequence_length']
                        )
                        
                        # Train
                        early_stop = callbacks.EarlyStopping(
                            monitor='val_accuracy',
                            patience=self.config['early_stopping_patience'],
                            restore_best_weights=True
                        )
                        
                        history = model.fit(
                            X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=self.config['epochs'],
                            batch_size=batch_size,
                            callbacks=[early_stop],
                            verbose=0
                        )
                        
                        val_acc = max(history.history['val_accuracy'])
                        
                        self.results.append({
                            'architecture': arch,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'val_accuracy': val_acc
                        })
                        
                        logger.info(f"      Val Accuracy: {val_acc:.4f}")
                        
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_config = {
                                'architecture': arch,
                                'learning_rate': lr,
                                'batch_size': batch_size
                            }
                            self.best_architecture = best_config
                            self.best_score = val_acc
                        
                        # Clear memory
                        del model
                        tf.keras.backend.clear_session()
                        
                    except Exception as e:
                        logger.error(f"      Failed: {e}")
        
        logger.info(f"\n✅ Architecture Search Complete")
        logger.info(f"   Best Val Accuracy: {self.best_score:.4f}")
        logger.info(f"   Best Config: {self.best_architecture}")
        
        return {
            'best_architecture': self.best_architecture,
            'best_score': self.best_score,
            'all_results': self.results
        }
    
    def _build_model(self, input_shape: Tuple, lstm_units: List[int],
                     dense_units: int, learning_rate: float,
                     num_classes: int):
        """Build LSTM model with given architecture."""
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.BatchNormalization())
        
        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            model.add(layers.LSTM(units, return_sequences=return_sequences, dropout=0.2))
            model.add(layers.BatchNormalization())
        
        # Dense layers
        model.add(layers.Dense(dense_units, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, 
                          seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create sequences
        X_seq = []
        y_seq = []
        
        for i in range(seq_length, len(X_scaled)):
            X_seq.append(X_scaled[i-seq_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)


# =============================================================================
# PPO HYPERPARAMETER OPTIMIZATION (OPTUNA)
# =============================================================================

class PPOHPO:
    """
    PPO Hyperparameter Optimization using OPTUNA.
    
    Výhody Optuna vs Random Search:
    - TPE Sampler (Tree-structured Parzen Estimator) - chytřejší sampling
    - MedianPruner - zastaví špatné trials brzy (šetří čas!)
    - SQLite storage - lze pokračovat po pádu
    - Paralelizace ready
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or FullTrainingConfig.PPO
        self.best_params = None
        self.best_score = None
        self.study = None
    
    def run(self, symbols: List[str] = None,
            n_trials: int = None,
            n_timesteps_per_trial: int = None) -> Dict:
        """
        Spustí HPO pomocí Optuna.
        
        Args:
            symbols: Seznam symbolů pro trénink
            n_trials: Počet trials (default z configu)
            n_timesteps_per_trial: Timesteps per trial
        """
        try:
            import optuna
            from optuna.pruners import MedianPruner
            from optuna.samplers import TPESampler
        except ImportError:
            logger.error("Optuna not installed! pip install optuna")
            return {'error': 'Optuna not installed'}
        
        n_trials = n_trials or self.config['hpo_iterations']
        n_timesteps = n_timesteps_per_trial or self.config['hpo_timesteps']
        
        logger.info("=" * 60)
        logger.info("🔍 PPO Hyperparameter Optimization (OPTUNA)")
        logger.info(f"   Trials: {n_trials}")
        logger.info(f"   Timesteps per trial: {n_timesteps:,}")
        logger.info(f"   Symbols: {symbols}")
        logger.info("=" * 60)
        
        # Store for objective function
        self._symbols = symbols
        self._n_timesteps = n_timesteps
        
        # Create Optuna study
        sampler = TPESampler(
            n_startup_trials=10,  # Random trials before TPE kicks in
            seed=42
        )
        pruner = MedianPruner(
            n_startup_trials=5,   # Trials before pruning
            n_warmup_steps=3      # Evaluations before pruning
        )
        
        # SQLite storage for resume capability
        storage_path = FullTrainingConfig.MODELS_DIR / "optuna_ppo.db"
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{storage_path}"
        
        study_name = f"ppo_hpo_{datetime.now().strftime('%Y%m')}"
        
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True  # Resume pokud existuje!
        )
        
        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            show_progress_bar=True,
            catch=(Exception,)  # Catch errors, don't crash
        )
        
        # Best params
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"\n✅ Optuna HPO Complete")
        logger.info(f"   Best Reward: {self.best_score:.2f}")
        logger.info(f"   Best Params:")
        for k, v in self.best_params.items():
            logger.info(f"      {k}: {v}")
        
        # Save results
        self._save_results()
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials),
            'study_name': study_name
        }
    
    def _objective(self, trial: 'optuna.Trial') -> float:
        """Optuna objective function."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.evaluation import evaluate_policy
        except ImportError:
            raise optuna.TrialPruned()
        
        # Sample hyperparameters
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'n_steps': trial.suggest_categorical('n_steps', [256, 512, 1024, 2048, 4096]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'n_epochs': trial.suggest_int('n_epochs', 3, 20),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999, log=True),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
            'ent_coef': trial.suggest_float('ent_coef', 1e-8, 0.1, log=True),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.1, 10.0),
        }
        
        logger.info(f"\n   Trial {trial.number}: lr={params['learning_rate']:.2e}, "
                   f"n_steps={params['n_steps']}, gamma={params['gamma']:.4f}")
        
        try:
            # Create environment
            from rl.vec_env import create_trading_vec_env
            env = create_trading_vec_env(
                symbols=self._symbols or ['SPY'],
                n_envs_per_symbol=1
            )
            
            # Create model
            model = PPO(
                "MlpPolicy",
                env,
                **params,
                verbose=0
            )
            
            # Train with intermediate evaluations for pruning
            eval_freq = self._n_timesteps // 5  # 5 evaluations per trial
            
            for step in range(5):
                model.learn(
                    total_timesteps=eval_freq,
                    reset_num_timesteps=False
                )
                
                # Evaluate
                mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
                
                # Report to Optuna for pruning
                trial.report(mean_reward, step)
                
                # Prune if needed
                if trial.should_prune():
                    logger.info(f"   Trial {trial.number} PRUNED at step {step}")
                    env.close()
                    raise optuna.TrialPruned()
            
            # Final evaluation
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
            
            logger.info(f"   Trial {trial.number} DONE: reward={mean_reward:.2f}")
            
            env.close()
            del model
            
            return mean_reward
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"   Trial {trial.number} FAILED: {e}")
            raise optuna.TrialPruned()
    
    def _save_results(self):
        """Save HPO results."""
        if self.study is None:
            return
        
        # Save best params as JSON
        results_path = FullTrainingConfig.MODELS_DIR / "ppo_best_params.json"
        with open(results_path, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_score': self.best_score,
                'timestamp': datetime.now().isoformat(),
                'n_trials': len(self.study.trials)
            }, f, indent=2)
        
        # Save all trials as CSV
        try:
            df = self.study.trials_dataframe()
            csv_path = FullTrainingConfig.MODELS_DIR / "ppo_hpo_trials.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"   Saved trials to {csv_path}")
        except:
            pass
        
        logger.info(f"   Saved best params to {results_path}")
    
    def load_best_params(self) -> Optional[Dict]:
        """Load best params from previous HPO."""
        results_path = FullTrainingConfig.MODELS_DIR / "ppo_best_params.json"
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                data = json.load(f)
                self.best_params = data.get('best_params')
                self.best_score = data.get('best_score')
                logger.info(f"Loaded best params from {results_path}")
                logger.info(f"   Score: {self.best_score}")
                return self.best_params
        
        return None
    
    def train_final_model(self, symbols: List[str] = None,
                          timesteps: int = None) -> Dict:
        """Train final PPO model with best params."""
        
        # Try to load best params if not set
        if self.best_params is None:
            loaded = self.load_best_params()
            if loaded is None:
                # Use default good params
                self.best_params = {
                    'learning_rate': 3e-4,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2,
                    'ent_coef': 0.01,
                    'vf_coef': 0.5,
                    'max_grad_norm': 0.5,
                }
                logger.info("Using default PPO params (no HPO results found)")
        
        # Use provided timesteps or default
        if timesteps is None:
            timesteps = self.config['full_timesteps']
        
        logger.info(f"\n🤖 Training final PPO model...")
        logger.info(f"   Timesteps: {timesteps:,}")
        logger.info(f"   Symbols: {len(symbols) if symbols else 'all'}")
        logger.info(f"   Params: {self.best_params}")
        
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import CheckpointCallback
            from rl.vec_env import create_trading_vec_env
            
            # Model path
            model_path = FullTrainingConfig.MODELS_DIR / "ppo_trading_agent.zip"
            
            # Create environment
            env = create_trading_vec_env(
                symbols=symbols,
                n_envs_per_symbol=2
            )
            
            # Load existing or create new
            if model_path.exists() and timesteps < 50_000_000:
                logger.info(f"   📥 Loading existing model for CONTINUE training")
                model = PPO.load(str(model_path), env=env)
                model.learning_rate = self.best_params.get('learning_rate', 3e-4)
            else:
                logger.info(f"   🆕 Creating NEW model for FULL training")
                model = PPO(
                    "MlpPolicy",
                    env,
                    **self.best_params,
                    verbose=1,
                    tensorboard_log="./tensorboard_logs/"
                )
            
            # Checkpoint callback
            checkpoint_dir = FullTrainingConfig.MODELS_DIR / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_callback = CheckpointCallback(
                save_freq=max(1_000_000, timesteps // 10),
                save_path=str(checkpoint_dir),
                name_prefix="ppo_checkpoint"
            )
            
            # Train
            model.learn(
                total_timesteps=timesteps,
                callback=[checkpoint_callback],
                progress_bar=True,
                reset_num_timesteps=False
            )
            
            # Save
            model.save(str(model_path))
            
            # Evaluate
            from stable_baselines3.common.evaluation import evaluate_policy
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
            
            logger.info(f"\n✅ Final PPO Model:")
            logger.info(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            logger.info(f"   Timesteps: {timesteps:,}")
            logger.info(f"   Saved to: {model_path}")
            
            env.close()
            
            return {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'timesteps': timesteps,
                'model_path': str(model_path),
                'params_used': self.best_params
            }
            
        except Exception as e:
            logger.error(f"Final training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}


# =============================================================================
# FULL TRAINING PIPELINE
# =============================================================================

class FullTrainingPipeline:
    """
    Kompletní trénovací pipeline s HPO.
    
    Schedule:
    - KVARTÁLNĚ (1,4,7,10): HPO + Full retrain (100M timesteps)
    - MĚSÍČNĚ (ostatní): Continue training (10M timesteps)
    - TÝDNĚ: Quick XGBoost update (1M timesteps)
    """
    
    def __init__(self, 
                 run_hpo: bool = None,      # None = auto-detect based on month
                 hpo_only: bool = False,
                 force_full: bool = False,
                 force_quick: bool = False):
        
        # Auto-detect based on month
        current_month = datetime.now().month
        is_hpo_month = current_month in FullTrainingConfig.HPO_SCHEDULE['hpo_months']
        
        if run_hpo is None:
            self.run_hpo = is_hpo_month
        else:
            self.run_hpo = run_hpo
        
        self.hpo_only = hpo_only
        self.force_full = force_full
        self.force_quick = force_quick
        self.is_hpo_month = is_hpo_month
        self.config = FullTrainingConfig  # Reference to config
        self.results = {}
        
        # Determine training mode
        if force_quick:
            self.training_mode = 'WEEKLY'
            self.ppo_timesteps = FullTrainingConfig.PPO['weekly_timesteps']
        elif force_full or is_hpo_month:
            self.training_mode = 'QUARTERLY_FULL'
            self.ppo_timesteps = FullTrainingConfig.PPO['full_timesteps']
        else:
            self.training_mode = 'MONTHLY_CONTINUE'
            self.ppo_timesteps = FullTrainingConfig.PPO['monthly_timesteps']
        
        logger.info(f"Training mode: {self.training_mode}")
        logger.info(f"PPO timesteps: {self.ppo_timesteps:,}")
    
    async def _step_feature_engineering(self):
        """
        STEP 0: Add all features to parquet files.
        
        Features added:
        - Technical (SMA, RSI, MACD, ATR, BB...) - 100+ features
        - Earnings (days_to_earnings, historical_move...) - 8 features
        - Time (hour, day_of_week, is_opex...) - 19 features
        
        TOTAL: 126+ features
        """
        data_dir = FullTrainingConfig.DATA_DIR
        
        # Import feature engineers
        try:
            from technical_features import FeatureEngineer
            tech_engineer = FeatureEngineer()
            has_tech = True
        except ImportError:
            logger.warning("technical_features.py not found, skipping technical features")
            has_tech = False
        
        try:
            from earnings_features import EarningsFeatureEngineer
            earn_engineer = EarningsFeatureEngineer()
            has_earn = True
        except ImportError:
            logger.warning("earnings_features.py not found, skipping earnings features")
            has_earn = False
        
        if not has_tech and not has_earn:
            logger.warning("No feature engineers available!")
            return
        
        processed = 0
        
        # Process each parquet file
        for parquet_file in data_dir.glob("*_1min.parquet"):
            # Skip already processed files
            if "_features" in parquet_file.name:
                continue
            
            symbol = parquet_file.stem.split('_')[0]
            output_file = data_dir / f"{symbol}_1min_features.parquet"
            
            try:
                logger.info(f"   Processing {symbol}...")
                
                df = pd.read_parquet(parquet_file)
                original_cols = len(df.columns)
                
                # Add technical features
                if has_tech:
                    df = tech_engineer.add_all_features(df, symbol=symbol)
                
                # Add earnings features
                if has_earn:
                    df = earn_engineer.add_features(df, symbol=symbol)
                
                # Save enriched data
                df.to_parquet(output_file, index=False, compression='snappy')
                
                new_cols = len(df.columns) - original_cols
                logger.info(f"   ✅ {symbol}: +{new_cols} features ({len(df):,} rows)")
                processed += 1
                
            except Exception as e:
                logger.error(f"   ❌ {symbol}: {e}")
        
        logger.info(f"\n✅ Feature engineering complete: {processed} symbols processed")
        self.results['feature_engineering'] = {'processed': processed}
    
    async def run(self) -> Dict:
        """Run full pipeline."""
        start = time.time()
        
        logger.info("=" * 70)
        logger.info("🚀 FULL TRAINING PIPELINE - 16 YEARS OF DATA")
        logger.info(f"   Started: {datetime.now()}")
        logger.info(f"   HPO: {'ON' if self.run_hpo else 'OFF'}")
        logger.info("=" * 70)
        
        # STEP 0: Feature Engineering
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 0: Feature Engineering (126+ features)")
        logger.info("=" * 50)
        await self._step_feature_engineering()
        
        # Load ALL data
        logger.info("\n📥 Loading ALL data (16 years)...")
        df = load_all_data(FullTrainingConfig.DATA_DIR)
        
        # Prepare labels
        df = self._prepare_labels(df)
        
        # Split data
        train_df, val_df, test_df = temporal_train_test_split(df)
        
        # 1. XGBoost
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 1: XGBoost")
        logger.info("=" * 50)
        
        xgb_hpo = XGBoostHPO()
        
        if self.run_hpo:
            self.results['xgboost_hpo'] = xgb_hpo.run(train_df)
        else:
            # Use default good params
            xgb_hpo.best_params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
        
        if not self.hpo_only:
            self.results['xgboost_final'] = xgb_hpo.train_final_model(
                train_df, val_df, test_df
            )
        
        # 2. LSTM
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 2: LSTM")
        logger.info("=" * 50)
        
        lstm_search = LSTMArchitectureSearch()
        
        if self.run_hpo:
            self.results['lstm_hpo'] = lstm_search.run(df)
        
        # 3. PPO
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 3: PPO")
        logger.info("=" * 50)
        
        try:
            from rl.vec_env import get_available_symbols
            symbols = get_available_symbols()
        except:
            symbols = ['SPY', 'QQQ', 'AAPL']
        
        ppo_hpo = PPOHPO()
        
        # Run HPO if this is HPO month
        if self.run_hpo:
            logger.info(f"Running Optuna HPO ({self.config.PPO['hpo_iterations']} trials)...")
            self.results['ppo_hpo'] = ppo_hpo.run(
                symbols=symbols[:3],  # HPO on subset for speed
                n_trials=FullTrainingConfig.PPO['hpo_iterations'],
                n_timesteps_per_trial=FullTrainingConfig.PPO['hpo_timesteps']
            )
        else:
            # Load best params from previous HPO
            ppo_hpo.load_best_params()
        
        # Train final model (unless HPO only mode)
        if not self.hpo_only:
            self.results['ppo_final'] = ppo_hpo.train_final_model(
                symbols=symbols,
                timesteps=self.ppo_timesteps
            )
        
        # Summary
        elapsed = time.time() - start
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"   Total time: {elapsed/3600:.1f} hours")
        logger.info(f"   XGBoost: {self.results.get('xgboost_final', {}).get('test_auc', 'N/A')}")
        logger.info(f"   LSTM: {self.results.get('lstm_hpo', {}).get('best_score', 'N/A')}")
        logger.info(f"   PPO: {self.results.get('ppo_final', {}).get('mean_reward', 'N/A')}")
        
        # Save results
        results_path = FullTrainingConfig.HPO_RESULTS
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"   Results saved to: {results_path}")
        
        return self.results
    
    def _prepare_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare training labels."""
        
        # XGBoost target: is_successful (Triple Barrier)
        if 'is_successful' not in df.columns:
            logger.info("   Creating Triple Barrier labels...")
            df['future_return'] = df.groupby('symbol')['close'].shift(-60) / df['close'] - 1
            df['is_successful'] = (df['future_return'] > 0.003).astype(int)
        
        # LSTM target: regime from VIX
        if 'regime_target' not in df.columns:
            def vix_to_regime(vix):
                if pd.isna(vix): return 1
                if vix < 15: return 0
                if vix < 20: return 1
                if vix < 25: return 2
                if vix < 35: return 3
                return 4
            
            df['regime_target'] = df['vix'].apply(vix_to_regime)
        
        return df.dropna(subset=['is_successful'])


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Full Training Pipeline with HPO")
    parser.add_argument("--full", action="store_true", help="Force FULL training (100M timesteps)")
    parser.add_argument("--quick", action="store_true", help="Quick weekly update (1M timesteps)")
    parser.add_argument("--hpo", action="store_true", help="Force HPO even if not HPO month")
    parser.add_argument("--hpo-only", action="store_true", help="Only run HPO, don't train final models")
    parser.add_argument("--no-hpo", action="store_true", help="Skip HPO even if HPO month")
    parser.add_argument("--force", action="store_true", help="Run even if not Saturday")
    parser.add_argument("--show-schedule", action="store_true", help="Show training schedule and exit")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Show schedule
    if args.show_schedule:
        current_month = datetime.now().month
        is_hpo = current_month in FullTrainingConfig.HPO_SCHEDULE['hpo_months']
        
        print("\n" + "=" * 60)
        print("📅 TRAINING SCHEDULE")
        print("=" * 60)
        print(f"\nCurrent month: {current_month}")
        print(f"Is HPO month: {'YES ✅' if is_hpo else 'NO'}")
        print(f"\nHPO months (quarterly): {FullTrainingConfig.HPO_SCHEDULE['hpo_months']}")
        print(f"Continue months: {FullTrainingConfig.HPO_SCHEDULE['continue_months']}")
        print(f"\nPPO Timesteps:")
        print(f"  Quarterly FULL: {FullTrainingConfig.PPO['full_timesteps']:,}")
        print(f"  Monthly continue: {FullTrainingConfig.PPO['monthly_timesteps']:,}")
        print(f"  Weekly quick: {FullTrainingConfig.PPO['weekly_timesteps']:,}")
        print(f"\nThis Saturday would run:")
        if is_hpo:
            print("  🔍 HPO (XGBoost, LSTM, PPO)")
            print(f"  🚀 Full PPO training: {FullTrainingConfig.PPO['full_timesteps']:,} timesteps")
            print("  ⏱️ Estimated time: 24-48 hours")
        else:
            print("  📈 Monthly continue training")
            print(f"  🚀 PPO continue: {FullTrainingConfig.PPO['monthly_timesteps']:,} timesteps")
            print("  ⏱️ Estimated time: 4-6 hours")
        exit(0)
    
    # Check Saturday
    if not args.force and datetime.now().weekday() != 5:
        logger.warning("Today is not Saturday. Use --force to run anyway.")
        exit(0)
    
    # Determine HPO
    if args.no_hpo:
        run_hpo = False
    elif args.hpo:
        run_hpo = True
    else:
        run_hpo = None  # Auto-detect
    
    # Run
    pipeline = FullTrainingPipeline(
        run_hpo=run_hpo,
        hpo_only=args.hpo_only,
        force_full=args.full,
        force_quick=args.quick
    )
    
    asyncio.run(pipeline.run())
