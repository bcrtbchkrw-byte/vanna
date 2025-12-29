"""
Trade Success Predictor with Vanna Features

XGBoost classifier to predict trade success probability.
Uses 22 features including Greeks (Delta, Vanna, Charm, Volga).

Acts as a Gatekeeper before AI analysis to save costs.
"""
import os
import joblib
import pandas as pd
import numpy as np
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from core.logger import get_logger

logger = get_logger()


class TradeSuccessPredictor:
    """
    XGBoost Classifier to predict trade success probability.
    
    Features (33 total):
    - Market: VIX, VIX3M, vix_ratio, regime
    - Time: sin_time, cos_time, sin_dow, cos_dow
    - Price: return_1m, return_5m, volatility_20, momentum_20
    - Greeks: delta, gamma, theta, vega, vanna, charm, volga
    - Options: iv_atm, volume, put_call_ratio, OI
    
    Acts as Gatekeeper between Screening and AI Analysis.
    """
    
    # Feature list - matches SPY_1min_vanna.parquet columns
    FEATURES = [
        # Market/VIX features
        'vix',
        'vix_ratio',
        'vix_in_contango',
        'vix_change_1d',
        'vix_percentile',
        'vix_zscore',
        'regime',
        
        # Time features (ENHANCED)
        'sin_time',
        'cos_time',
        'sin_dow',
        'cos_dow',
        'hour_of_day',           # NEW: 0-23
        'is_market_open_hour',   # NEW: 1 if within first/last hour
        'is_lunch_hour',         # NEW: 1 if 12-13 ET
        
        # Price/momentum features
        'return_1m',
        'return_5m',
        'volatility_20',
        'momentum_20',
        
        # NEW: Lagged features (previous bars)
        'return_1m_lag1',        # Return 1 bar ago
        'return_1m_lag5',        # Return 5 bars ago
        'volatility_20_lag1',    # Volatility 1 bar ago
        
        # NEW: Market microstructure
        'volume_ratio',          # Volume vs 20-bar avg
        'high_low_range',        # (High - Low) / Close
        
        # Greeks (7 features)
        'delta',
        'gamma',
        'theta',
        'vega',
        'vanna',
        'charm',
        'volga',
        
        # NOTE: Removed total_call_oi, total_put_oi, put_call_oi_ratio
        # These were hardcoded placeholders (100000.0), not real data
        # TODO: Implement real OI data fetching from IBKR
    ]
    
    def __init__(self, model_path: str = "data/models/trade_success_predictor.pkl"):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.feature_importances = None
        self.model_feature_names = None  # NEW: To store features expected by the loaded model
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model if exists."""
        try:
            if self.model_path.exists():
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.scaler = data.get('scaler')
                self.feature_importances = data.get('feature_importances')
                # Try to load feature names from metadata
                self.model_feature_names = data.get('feature_names')
                
                # If not in metadata, try XGBoost booster
                if not self.model_feature_names and hasattr(self.model, 'get_booster'):
                    try:
                        self.model_feature_names = self.model.get_booster().feature_names
                    except Exception:
                        pass
                
                logger.info(f"✅ Loaded TradeSuccessPredictor from {self.model_path}")
                if self.model_feature_names:
                    logger.info(f"   Model expects {len(self.model_feature_names)} features")
            else:
                logger.warning("⚠️ No trained model found. Gatekeeper OPEN (pass all).")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def train(
        self,
        df: pd.DataFrame,
        target_col: str = 'is_successful',
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the model on historical data.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column (1=success, 0=failure)
            test_size: Fraction for test set
            
        Returns:
            Training metrics dict
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            logger.error(f"XGBoost/Scikit-Learn not installed: {e}")
            return {}
        
        if df.empty:
            logger.warning("No data for training")
            return {}
        
        logger.info(f"📊 Training TradeSuccessPredictor on {len(df):,} samples...")
        
        # Prepare features
        available_features = [f for f in self.FEATURES if f in df.columns]
        missing_features = [f for f in self.FEATURES if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        X = df[available_features].copy()
        X = X.fillna(0)  # Simple imputation
        
        # Prepare target
        y = df[target_col].astype(int)
        
        logger.info(f"   Features: {len(available_features)}")
        logger.info(f"   Target distribution: {y.value_counts().to_dict()}")
        
        # Split with dynamic seed (different each training run)
        import time
        dynamic_seed = int(time.time()) % 2**31  # Use current timestamp as seed
        logger.info(f"   Using dynamic random seed: {dynamic_seed}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=dynamic_seed, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost with class imbalance handling
        # Calculate scale_pos_weight to balance classes
        n_negative = len(y[y == 0])
        n_positive = len(y[y == 1])
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
        
        logger.info(f"   Class balance: {n_negative} negative, {n_positive} positive (scale_pos_weight={scale_pos_weight:.2f})")
        
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,  # Handle imbalance
            eval_metric='logloss',
            random_state=dynamic_seed,  # Dynamic seed for different trees each run
            n_jobs=-1
        )
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'positive_rate': float(y.mean())
        }
        
        # Feature importances
        self.feature_importances = dict(zip(available_features, self.model.feature_importances_))
        top_features = sorted(self.feature_importances.items(), key=lambda x: -x[1])[:10]
        
        logger.info(f"✅ Model trained!")
        logger.info(f"   Accuracy: {metrics['accuracy']:.2%}")
        logger.info(f"   AUC: {metrics['auc']:.3f}")
        logger.info(f"   Top features: {[f[0] for f in top_features[:5]]}")
        
        # Save model
        self._save_model(available_features)
        
        return metrics
    
    def predict(self, features: Dict[str, Any]) -> float:
        """
        Predict probability of trade success.
        
        Args:
            features: Dict of feature values
            
        Returns:
            Probability (0.0 to 1.0). High = likely profit.
        """
        if self.model is None:
            return 0.5  # Neutral if no model
        
        try:
            # Create DataFrame
            df = pd.DataFrame([features])
            
            # Determine which features to use
            features_to_use = self.model_feature_names if self.model_feature_names else self.FEATURES
            
            # Ensure all feature columns exist
            for col in features_to_use:
                if col not in df.columns:
                    df[col] = 0
            
            # Select and order features - CRITICAL: Must match model input exactly
            X = df[features_to_use].fillna(0)
            
            # Scale
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
            
            # Predict probability of class 1 (Success)
            prob = self.model.predict_proba(X_scaled)[0][1]
            
            return float(prob)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5
    
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for multiple samples.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Array of probabilities
        """
        if self.model is None:
            return np.full(len(df), 0.5)
        
        try:
            # Determine which features to use
            features_to_use = self.model_feature_names if self.model_feature_names else self.FEATURES

            # Prepare features
            X = df[features_to_use].copy().fillna(0)
            
            # Scale
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
            
            # Predict
            probs = self.model.predict_proba(X_scaled)[:, 1]
            
            return probs
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return np.full(len(df), 0.5)
    
    def should_analyze(
        self,
        features: Dict[str, Any],
        threshold: float = 0.4
    ) -> tuple:
        """
        Gatekeeper decision: Should this trade go to AI analysis?
        
        Args:
            features: Trade features
            threshold: Minimum probability to pass (default 40%)
            
        Returns:
            (should_pass, probability, reason)
        """
        prob = self.predict(features)
        
        if prob >= threshold:
            return True, prob, f"Pass: {prob:.1%} success probability"
        else:
            return False, prob, f"Reject: {prob:.1%} < {threshold:.0%} threshold"
    
    def _save_model(self, feature_names: List[str]):
        """Save model to disk."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_importances': self.feature_importances,
                'feature_names': feature_names,
                'timestamp': datetime.now(),
                'version': '2.0-vanna'
            }, self.model_path)
            
            logger.info(f"💾 Saved model to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Get top feature importances."""
        if not self.feature_importances:
            return {}
        
        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: -x[1]
        )
        return dict(sorted_features[:top_n])
    
    def predict_probability(self, trade_setup: Dict[str, Any]) -> float:
        """
        Predict probability of profit from trade setup dict.
        
        This is the simple interface for quick predictions.
        
        Args:
            trade_setup: Dict with 'symbol', 'strategy', 'market_data'
            
        Returns:
            Float 0.0 to 1.0 (Probability of Success)
        """
        market = trade_setup.get('market_data', {})
        vix = market.get('vix', 18)
        iv_rank = market.get('iv_rank', 0.5)
        strategy = trade_setup.get('strategy', 'UNKNOWN')
        symbol = trade_setup.get('symbol', 'UNKNOWN')
        
        # Try using trained model
        if self.model is not None:
            try:
                # Build feature dict
                features = {
                    'vix': vix,
                    'vix_ratio': market.get('vix_ratio', 1.0),
                    'vix_in_contango': 1 if market.get('vix_ratio', 1.0) < 1 else 0,
                    'vix_change_1d': market.get('vix_change_1d', 0),
                    'vix_percentile': market.get('vix_percentile', 50),
                    'vix_zscore': market.get('vix_zscore', 0),
                    'regime': market.get('regime', 1),
                    'sin_time': 0, 'cos_time': 0, 'sin_dow': 0, 'cos_dow': 0,
                    'return_1m': market.get('return_1m', 0),
                    'return_5m': market.get('return_5m', 0),
                    'volatility_20': market.get('volatility_20', 0.02),
                    'momentum_20': market.get('momentum_20', 0),
                    'delta': market.get('delta', -0.16),
                    'gamma': market.get('gamma', 0),
                    'theta': market.get('theta', 0),
                    'vega': market.get('vega', 0),
                    'vanna': market.get('vanna', 0),
                    'charm': market.get('charm', 0),
                    'volga': market.get('volga', 0),
                }
                
                prob = self.predict_proba(features)
                logger.debug(f"🤖 ML Prediction ({symbol}/{strategy}): {prob:.1%}")
                return float(prob)
                
            except Exception as e:
                logger.debug(f"Model prediction failed, using fallback: {e}")
        
        # Rule-based fallback (no random!)
        prob = self._calculate_rule_based_prob(vix, iv_rank, strategy)
        logger.debug(f"📐 Rule-based Prediction ({symbol}/{strategy}): {prob:.1%}")
        return prob
    
    def _calculate_rule_based_prob(
        self, 
        vix: float, 
        iv_rank: float,
        strategy: str
    ) -> float:
        """
        Rule-based probability calculation.
        
        NO RANDOM! Uses deterministic rules based on market conditions.
        """
        base_prob = 0.50  # Neutral baseline
        
        # VIX adjustments for credit strategies
        if strategy in ['BULL_PUT_SPREAD', 'BEAR_CALL_SPREAD', 'IRON_CONDOR', 
                        'IRON_BUTTERFLY', 'JADE_LIZARD']:
            # High VIX = better for premium selling
            if vix > 25:
                base_prob += 0.15
            elif vix > 20:
                base_prob += 0.10
            elif vix < 15:
                base_prob -= 0.10
            
            # High IV rank = sell premium
            if iv_rank > 0.7:
                base_prob += 0.10
            elif iv_rank > 0.5:
                base_prob += 0.05
        
        # Debit strategies prefer low IV
        elif strategy in ['PUT_DEBIT_SPREAD', 'CALL_DEBIT_SPREAD', 
                          'POOR_MANS_COVERED_CALL']:
            if iv_rank < 0.3:
                base_prob += 0.10
            elif iv_rank > 0.7:
                base_prob -= 0.10
        
        # Clamp to valid range
        return max(0.20, min(0.85, base_prob))


# Singleton (thread-safe)
_predictor: Optional[TradeSuccessPredictor] = None
_predictor_lock = threading.Lock()


def get_trade_success_predictor(
    model_path: str = "data/models/trade_success_predictor.pkl"
) -> TradeSuccessPredictor:
    """Get or create singleton predictor (thread-safe)."""
    global _predictor
    # Double-checked locking pattern for thread safety
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = TradeSuccessPredictor(model_path)
    return _predictor


# Backward compatibility aliases
TradePredictor = TradeSuccessPredictor  # Alias for old code
get_trade_predictor = get_trade_success_predictor  # Alias for old imports


# ============================================================
# Training Script
# ============================================================

def create_synthetic_training_data(parquet_path: str) -> pd.DataFrame:
    """
    Create synthetic training labels using Triple Barrier Method.
    
    Standard quantitative finance labeling method:
    1. Upper Barrier (Take Profit): Entry + vol * 2 (or fixed %)
    2. Lower Barrier (Stop Loss): Entry - vol * 1 (or fixed %)
    3. Vertical Barrier (Time): Max hold time
    
    Target 1 (Success) if Upper Barrier hit first.
    Target 0 (Failure) if Lower Barrier hit first or Time Limit reached.
    
    Args:
        parquet_path: Path to input parquet file
        
    Returns:
        DataFrame with features and 'is_successful' target
    """
    df = pd.read_parquet(parquet_path)
    
    if 'close' not in df.columns:
        logger.error("No 'close' column found!")
        raise ValueError("Missing 'close' column")

    logger.info("Applying Triple Barrier Method labeling...")
    
    # Parameters
    time_limit = 60      # Max hold 60 mins
    tp_pct = 0.003       # Take Profit +0.3% (approx $1.50 on SPY)
    sl_pct = 0.0015      # Stop Loss -0.15% (Risk Reward 1:2)
    
    closes = df['close'].values
    n = len(closes)
    
    labels = np.zeros(n, dtype=int)
    
    # Vectorized loop is hard for path dependency, using optimized iteration
    # Check max 60 bars into future
    
    # We can pre-calculate rolling max/min windows, but "first touch" is tricky vectorized.
    # Let's use a reasonably fast loop with numba if available, or just careful python logic.
    # Given dataset size (~150k params), simple loop is slow. 
    # Optimized Strategy: Look at window of 60.
    
    # For speed, we will use a simplified vectorized approach:
    # 1. Calc explicit Price targets for every row
    # 2. Check if High[t:t+60] > TP
    # 3. Check if Low[t:t+60] < SL
    # 4. If both, check WHICH came first (argmax)
    
    # Since we don't have High/Low in simplified RL parquet (sometimes), use Close.
    # If High/Low available, use them.
    use_hl = 'high' in df.columns and 'low' in df.columns
    series = df['close'].values
    highs = df['high'].values if use_hl else series
    lows = df['low'].values if use_hl else series
    
    # Iterate (optimized)
    # This might take 10-20s for 150k rows, acceptable for offline training script
    
    for i in range(n - time_limit):
        entry_price = series[i]
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
        
        # Slice future window
        window_highs = highs[i+1 : i+1+time_limit]
        window_lows = lows[i+1 : i+1+time_limit]
        
        # Find exceedances
        # np.argmax returns index of first True. If no True, returns 0 (risk!)
        # So we must check max() first
        
        tp_hit = window_highs >= tp_price
        sl_hit = window_lows <= sl_price
        
        has_tp = tp_hit.any()
        has_sl = sl_hit.any()
        
        if has_tp and not has_sl:
            labels[i] = 1 # Profit, no stop
        elif not has_tp and has_sl:
            labels[i] = 0 # Stop, no profit
        elif has_tp and has_sl:
            # Both hit, check which first
            first_tp_idx = np.argmax(tp_hit)
            first_sl_idx = np.argmax(sl_hit)
            
            if first_tp_idx < first_sl_idx:
                labels[i] = 1 # TP before SL
            else:
                labels[i] = 0 # SL before TP
        else:
            labels[i] = 0 # Time expiry (Vertical Barrier) - Treat as fail (opportunity cost)
            
    df['is_successful'] = labels
    
    # Drop the last 'time_limit' rows which weren't labeled
    df = df.iloc[:-time_limit]
    
    # Filter for only "Active" times (9:30 - 16:00) to reduce overnight noise
    # Parquet usually has is_market_open or we parse timestamp
    # Assuming standard trading hours data mostly
    
    success_rate = df['is_successful'].mean()
    logger.info(f"Triple Barrier Labeling complete. Success Rate: {success_rate:.1%}")
    
    return df


if __name__ == "__main__":
    from core.logger import setup_logger
    import glob
    
    # Try to setup logger, fallback to print
    try:
        setup_logger(level="INFO")
    except Exception as e:
        print(f"Logger setup failed: {e}")
    
    print("=" * 60)
    print("TradeSuccessPredictor Training (Multi-Symbol)")
    print("=" * 60)
    
    data_dir = Path("data/enriched")
    files = list(data_dir.glob("*_1min_vanna.parquet"))
    
    if not files:
        print(f"❌ No data files found in {data_dir}")
        exit(1)
        
    print(f"Found {len(files)} files: {[f.name for f in files]}")
    
    dfs = []
    total_samples = 0
    
    for file_path in files:
        symbol = file_path.name.split('_')[0]
        print(f"\nProcessing {symbol}...")
        try:
            df_part = create_synthetic_training_data(str(file_path))
            if not df_part.empty:
                dfs.append(df_part)
                total_samples += len(df_part)
                print(f"  -> Added {len(df_part):,} samples (Success rate: {df_part['is_successful'].mean():.1%})")
        except Exception as e:
            print(f"  -> Failed: {e}")
            
    if not dfs:
        print("❌ No valid data loaded.")
        exit(1)
        
    print(f"\nMerging {len(dfs)} datasets...")
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total Training Samples: {len(full_df):,}")
    print(f"Overall Target Dist: {full_df['is_successful'].value_counts().to_dict()}")
    
    # Train model
    predictor = TradeSuccessPredictor()
    metrics = predictor.train(full_df, target_col='is_successful')
    
    print("\nTraining Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\nTop 15 Feature Importances:")
    for feat, imp in predictor.get_feature_importance(15).items():
        print(f"  {feat}: {imp:.4f}")
