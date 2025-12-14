"""
Regime Classifier - Market Regime Detection

XGBoost classifier for detecting market regimes:
- Bull (trending up)
- Bear (trending down)  
- Sideways/Choppy
- High Volatility
- Low Volatility
"""
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from core.logger import get_logger

logger = get_logger()


@dataclass
class RegimeResult:
    """Regime classification result."""
    regime: int
    regime_name: str
    confidence: float
    probabilities: Dict[str, float]


class RegimeClassifier:
    """
    XGBoost-based market regime classifier.
    
    Regimes:
    0 = Low Volatility (VIX < 15)
    1 = Normal (VIX 15-20)
    2 = Elevated (VIX 20-25)
    3 = High Volatility (VIX 25-35)
    4 = Crisis (VIX > 35)
    
    Features:
    - VIX level and changes
    - VIX term structure
    - Price momentum
    - Realized volatility
    """
    
    REGIME_NAMES = {
        0: 'low_vol',
        1: 'normal',
        2: 'elevated',
        3: 'high_vol',
        4: 'crisis'
    }
    
    FEATURES = [
        'vix', 'vix_ratio', 'vix_change_1d', 'vix_zscore',
        'return_1m', 'return_5m', 'volatility_20', 'momentum_20'
    ]
    
    def __init__(
        self,
        model_path: str = "data/models/regime_classifier.pkl"
    ):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        
        self._load_model()
        logger.info("RegimeClassifier initialized")
    
    def _load_model(self):
        """Load trained model if exists."""
        if self.model_path.exists():
            try:
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.scaler = data.get('scaler')
                logger.info(f"Loaded model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
    
    def classify_by_vix(self, vix: float) -> RegimeResult:
        """
        Simple rule-based classification by VIX level.
        
        Fallback when no ML model is available.
        
        Args:
            vix: Current VIX level
            
        Returns:
            RegimeResult
        """
        if vix < 15:
            regime = 0
            confidence = 0.9
        elif vix < 20:
            regime = 1
            confidence = 0.85
        elif vix < 25:
            regime = 2
            confidence = 0.8
        elif vix < 35:
            regime = 3
            confidence = 0.75
        else:
            regime = 4
            confidence = 0.9
        
        return RegimeResult(
            regime=regime,
            regime_name=self.REGIME_NAMES[regime],
            confidence=confidence,
            probabilities={self.REGIME_NAMES[regime]: confidence}
        )
    
    def classify(self, features: Dict[str, float]) -> RegimeResult:
        """
        Classify regime using ML model or fallback to VIX rules.
        
        Args:
            features: Dict with feature values
            
        Returns:
            RegimeResult
        """
        vix = features.get('vix', 18)
        
        # Use rule-based if no model
        if self.model is None:
            return self.classify_by_vix(vix)
        
        try:
            # Prepare features as DataFrame with column names (avoids sklearn warning)
            feature_values = [[features.get(f, 0) for f in self.FEATURES]]
            X = pd.DataFrame(feature_values, columns=self.FEATURES)
            
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Predict
            regime = int(self.model.predict(X)[0])
            proba = self.model.predict_proba(X)[0]
            
            confidence = float(max(proba))
            probabilities = {
                self.REGIME_NAMES[i]: float(p)
                for i, p in enumerate(proba)
            }
            
            return RegimeResult(
                regime=regime,
                regime_name=self.REGIME_NAMES.get(regime, 'unknown'),
                confidence=confidence,
                probabilities=probabilities
            )
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return self.classify_by_vix(vix)
    
    def train(
        self,
        df: pd.DataFrame,
        target_col: str = 'regime'
    ) -> Dict[str, float]:
        """
        Train the regime classifier.
        
        Args:
            df: DataFrame with features and regime labels
            target_col: Name of target column
            
        Returns:
            Training metrics
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            return {}
        
        # Prepare data
        available = [f for f in self.FEATURES if f in df.columns]
        X = df[available].fillna(0)
        y = df[target_col].astype(int)
        
        logger.info(f"Training on {len(X)} samples, {len(available)} features")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Accuracy: {accuracy:.2%}")
        
        # Save
        self._save_model(available)
        
        return {'accuracy': accuracy}
    
    def _save_model(self, feature_names: List[str]):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': feature_names,
            'regime_names': self.REGIME_NAMES
        }, self.model_path)
        
        logger.info(f"Saved model to {self.model_path}")
    
    def get_strategy_adjustment(self, regime: int) -> Dict[str, float]:
        """
        Get strategy parameter adjustments for regime.
        
        Returns multipliers for:
        - position_size: Scale positions
        - delta_target: Adjust delta
        - dte_adjustment: Add/subtract DTE
        """
        adjustments = {
            0: {'position_size': 1.2, 'delta_target': -0.20, 'dte_adjustment': 10},  # Low vol: bigger, shorter
            1: {'position_size': 1.0, 'delta_target': -0.16, 'dte_adjustment': 0},   # Normal
            2: {'position_size': 0.8, 'delta_target': -0.12, 'dte_adjustment': 5},   # Elevated: smaller
            3: {'position_size': 0.5, 'delta_target': -0.10, 'dte_adjustment': -5},  # High vol: defensive
            4: {'position_size': 0.25, 'delta_target': -0.08, 'dte_adjustment': -10}, # Crisis: minimal
        }
        
        return adjustments.get(regime, adjustments[1])


# Singleton
_classifier: Optional[RegimeClassifier] = None


def get_regime_classifier() -> RegimeClassifier:
    """Get or create regime classifier."""
    global _classifier
    if _classifier is None:
        _classifier = RegimeClassifier()
    return _classifier
