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
    
    Features (22 total):
    - Market: VIX, VIX3M, vix_ratio, regime
    - Time: sin_time, cos_time, sin_dow, cos_dow
    - Price: return_1m, return_5m, volatility_20, momentum_20
    - Greeks: delta, gamma, theta, vega, vanna, charm, volga
    - Options: iv_atm, volume, put_call_ratio
    
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
        
        # Time features
        'sin_time',
        'cos_time',
        'sin_dow',
        'cos_dow',
        
        # Price/momentum features
        'return_1m',
        'return_5m',
        'volatility_20',
        'momentum_20',
        
        # Greeks (7 new!)
        'delta',
        'gamma',
        'theta',
        'vega',
        'vanna',
        'charm',
        'volga',
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
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model if exists."""
        try:
            if self.model_path.exists():
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.scaler = data.get('scaler')
                self.feature_importances = data.get('feature_importances')
                logger.info(f"✅ Loaded TradeSuccessPredictor from {self.model_path}")
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
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
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
            
            # Ensure all feature columns exist
            for col in self.FEATURES:
                if col not in df.columns:
                    df[col] = 0
            
            # Select and order features
            X = df[self.FEATURES].fillna(0)
            
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
            # Prepare features
            X = df[self.FEATURES].copy().fillna(0)
            
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
    Create synthetic training labels from historical data.
    
    Labels based on forward returns:
    - is_successful = 1 if return_5m > 0 (for long positions)
    - Adjusts for regime (bull/bear)
    
    This is a simplified approach - real labels would come from
    actual trade outcomes.
    """
    df = pd.read_parquet(parquet_path)
    
    # Simple labeling: positive forward return = success
    # (In reality, this would come from actual trade P/L)
    if 'return_5m' in df.columns:
        # For puts (negative delta strategies), negative return = success
        df['is_successful'] = (df['return_5m'] < 0).astype(int)
    else:
        # Random 50/50 if no return column
        df['is_successful'] = np.random.randint(0, 2, len(df))
    
    return df


if __name__ == "__main__":
    from core.logger import setup_logger
    
    # Try to setup logger, fallback to print
    try:
        setup_logger(level="INFO")
    except Exception as e:
        print(f"Logger setup failed: {e}")
    
    print("=" * 60)
    print("TradeSuccessPredictor Training")
    print("=" * 60)
    
    # Load data with Greeks
    data_path = "data/vanna_ml/SPY_1min_vanna.parquet"
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        exit(1)
    
    print(f"Loading {data_path}...")
    df = create_synthetic_training_data(data_path)
    print(f"Loaded {len(df):,} samples")
    print(f"Target distribution: {df['is_successful'].value_counts().to_dict()}")
    
    # Train model
    predictor = TradeSuccessPredictor()
    metrics = predictor.train(df, target_col='is_successful')
    
    print("\nTraining Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\nTop 10 Feature Importances:")
    for feat, imp in predictor.get_feature_importance(10).items():
        print(f"  {feat}: {imp:.4f}")
