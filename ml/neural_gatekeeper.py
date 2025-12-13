"""
Neural Gatekeeper - LSTM-based Market Predictor

Deep learning gatekeeper for trade filtering.
Can host RL agent policy for action decisions.
"""
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np

from core.logger import get_logger

logger = get_logger()


class NeuralGatekeeper:
    """
    LSTM-based neural network for trade gatekeeping.
    
    Uses TensorFlow/Keras for:
    - Price direction prediction
    - Trade probability gating
    - RL policy integration
    
    Features:
    - 32-feature input (matching TradingEnv)
    - LSTM layers for temporal patterns
    - TFLite export for Raspberry Pi
    """
    
    SEQUENCE_LENGTH = 60  # 60 minutes lookback
    N_FEATURES = 32
    
    def __init__(
        self,
        model_path: str = "data/models/neural_gatekeeper",
        threshold: float = 0.6
    ):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        
        self.model = None
        self.scaler = None
        
        self._load_model()
        logger.info(f"NeuralGatekeeper initialized (threshold={threshold})")
    
    def _load_model(self):
        """Load trained model if exists."""
        keras_path = self.model_path / "gatekeeper.keras"
        
        if keras_path.exists():
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(keras_path))
                logger.info(f"Loaded model from {keras_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
    
    def build_model(self) -> Any:
        """Build LSTM model architecture."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
        except ImportError:
            logger.error("TensorFlow not installed")
            return None
        
        model = models.Sequential([
            # Input: (batch, sequence_length, features)
            layers.Input(shape=(self.SEQUENCE_LENGTH, self.N_FEATURES)),
            
            # LSTM layers
            layers.LSTM(64, return_sequences=True, dropout=0.2),
            layers.LSTM(32, dropout=0.2),
            
            # Dense layers
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            
            # Output: probability
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        self.model = model
        logger.info("Built LSTM model")
        logger.info(model.summary())
        
        return model
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Train the gatekeeper model.
        
        Args:
            X: Input sequences (samples, seq_len, features)
            y: Binary labels (0=bad trade, 1=good trade)
            validation_split: Fraction for validation
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Training metrics
        """
        if self.model is None:
            self.build_model()
        
        try:
            import tensorflow as tf
        except ImportError:
            return {"error": "TensorFlow not installed"}
        
        logger.info(f"Training on {len(X)} samples...")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5
            )
        ]
        
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        self.save()
        
        return {
            "accuracy": float(history.history['accuracy'][-1]),
            "val_accuracy": float(history.history.get('val_accuracy', [0])[-1]),
            "auc": float(history.history.get('auc', [0])[-1])
        }
    
    def predict(self, sequence: np.ndarray) -> float:
        """
        Predict trade success probability.
        
        Args:
            sequence: (seq_len, features) or (1, seq_len, features)
            
        Returns:
            Probability 0-1
        """
        if self.model is None:
            return 0.5  # Neutral if no model
        
        # Ensure 3D shape
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, *sequence.shape)
        
        prob = float(self.model.predict(sequence, verbose=0)[0, 0])
        return prob
    
    def should_trade(self, sequence: np.ndarray) -> Tuple[bool, float, str]:
        """
        Gatekeeper decision: Should we trade?
        
        Args:
            sequence: Market data sequence
            
        Returns:
            (should_trade, probability, reason)
        """
        prob = self.predict(sequence)
        
        if prob >= self.threshold:
            return True, prob, f"✅ Pass: {prob:.1%} >= {self.threshold:.0%}"
        else:
            return False, prob, f"❌ Reject: {prob:.1%} < {self.threshold:.0%}"
    
    def save(self):
        """Save model."""
        if self.model is None:
            return
        
        keras_path = self.model_path / "gatekeeper.keras"
        self.model.save(str(keras_path))
        logger.info(f"Saved model to {keras_path}")
    
    def export_tflite(self) -> Optional[Path]:
        """Export to TFLite for Raspberry Pi."""
        if self.model is None:
            return None
        
        try:
            import tensorflow as tf
            
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            tflite_path = self.model_path / "gatekeeper.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Exported TFLite to {tflite_path}")
            return tflite_path
            
        except Exception as e:
            logger.error(f"TFLite export failed: {e}")
            return None


# Singleton
_gatekeeper: Optional[NeuralGatekeeper] = None


def get_neural_gatekeeper() -> NeuralGatekeeper:
    """Get or create gatekeeper."""
    global _gatekeeper
    if _gatekeeper is None:
        _gatekeeper = NeuralGatekeeper()
    return _gatekeeper
