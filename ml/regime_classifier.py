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
import threading
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
        # VIX features (NO raw VIX to prevent data leakage!)
        'vix_ratio',           # VIX3M / VIX (term structure)
        'vix_change_1d',       # VIX daily change
        'vix_zscore',          # VIX z-score over lookback
        
        # Price/momentum features
        'return_1m',           # 1-minute return
        'return_5m',           # 5-minute return
        'volatility_20',       # 20-period realized vol
        'momentum_20',         # 20-period momentum
        
        # Market microstructure features (NEW)
        'volume_ratio',        # Current volume vs 20-period avg
        'high_low_range',      # (High - Low) / Close
        'price_acceleration',  # Change in momentum (d(momentum)/dt)
    ]
    
    SEQUENCE_LENGTH = 60
    
    def __init__(
        self,
        model_path: str = "data/models/regime_classifier"
    ):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        
        self._load_model()
        logger.info("RegimeClassifier (LSTM) initialized")
    
    def _load_model(self):
        """Load trained Keras model if exists."""
        keras_path = self.model_path / "regime_model.keras"
        scaler_path = self.model_path / "scaler.pkl"
        
        if keras_path.exists():
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(keras_path))
                if scaler_path.exists():
                    self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded LSTM model from {keras_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
    
    def build_model(self, input_shape):
        """Build LSTM architecture for classification."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
        except ImportError:
            logger.error("TensorFlow not installed")
            return None
            
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=True, dropout=0.2),
            layers.LSTM(32, dropout=0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(5, activation='softmax') # 5 classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

    def classify_by_vix(self, vix: float) -> RegimeResult:
        """Fallback rule-based classification."""
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
    
    def classify(self, sequence: np.ndarray) -> RegimeResult:
        """
        Classify regime using LSTM or fallback.
        
        Args:
            sequence: Array of shape (seq_len, features) or (1, seq_len, features)
                      If 1D/2D invalid shape is passed, tries to handle or fallback.
            
        Returns:
            RegimeResult
        """
        # Fallback if no model
        if self.model is None:
            # Try to estimate VIX from sequence if possible, otherwise 18
            return self.classify_by_vix(18.0)
            
        try:
            # Ensure shape (1, 60, features)
            if len(sequence.shape) == 2:
                sequence = sequence.reshape(1, *sequence.shape)
            
            proba = self.model.predict(sequence, verbose=0)[0]
            regime = int(np.argmax(proba))
            confidence = float(proba[regime])
            
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
            logger.error(f"LSTM Classification error: {e}")
            return self.classify_by_vix(18.0)
    
    def train(
        self,
        data: pd.DataFrame,
        target_col: str = 'regime_target',
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the LSTM model.
        
        Args:
            data: DataFrame with features
            target_col: Name of target column
            test_size: Fraction of data for validation
            
        Returns:
            Metrics dict
        """
        if self.model is None:
            # Build model based on feature count
            input_shape = (self.SEQUENCE_LENGTH, len(self.FEATURES))
            self.build_model(input_shape=input_shape)
            
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # Prepare features
        feature_data = data[self.FEATURES].values
        targets = data[target_col].values
        
        # Scale features
        self.scaler = StandardScaler()
        feature_data_scaled = self.scaler.fit_transform(feature_data)
        
        # Generate sequences
        X, y = [], []
        # Need enough data for at least one sequence
        if len(feature_data_scaled) > self.SEQUENCE_LENGTH:
            for i in range(self.SEQUENCE_LENGTH, len(feature_data_scaled)):
                X.append(feature_data_scaled[i-self.SEQUENCE_LENGTH:i])
                y.append(targets[i])
            
            X = np.array(X)
            y = np.array(y)
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False # Time series!
            )
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=20,
                batch_size=64,
                callbacks=[early_stopping],
                verbose=1
            )
            
            acc = float(history.history['val_accuracy'][-1])
            logger.info(f"Model trained. Val Accuracy: {acc:.2%}")
            
            self._save_model()
            return {'accuracy': acc}
        else:
            logger.warning("Not enough data for sequence generation")
            return {'accuracy': 0.0}
    
    def _save_model(self, feature_names: List[str] = None):
        """Save Keras model."""
        self.model.save(self.model_path / "regime_model.keras")
        if self.scaler:
            joblib.dump(self.scaler, self.model_path / "scaler.pkl")
        logger.info(f"Saved LSTM model to {self.model_path}")
    
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


# Singleton (thread-safe)
_classifier: Optional[RegimeClassifier] = None
_classifier_lock = threading.Lock()


def get_regime_classifier() -> RegimeClassifier:
    """Get or create regime classifier (thread-safe)."""
    global _classifier
    # Double-checked locking pattern for thread safety
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = RegimeClassifier()
    return _classifier
