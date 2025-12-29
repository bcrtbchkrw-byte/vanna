"""
Price Forecaster - LSTM-based Price Prediction

Part of Multi-Model Trading Architecture.
Predicts future returns and volatility.

Outputs:
- return_5min: Expected return in 5 minutes
- return_15min: Expected return in 15 minutes
- direction: UP/DOWN/FLAT
- vol_forecast: Expected volatility
- confidence: Prediction confidence
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import joblib

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
except ImportError:
    HAS_TF = False

from core.logger import get_logger

logger = get_logger()


@dataclass
class PriceForecast:
    """Price forecaster output."""
    return_5min: float      # Expected 5-min return (e.g., 0.001 = +0.1%)
    return_15min: float     # Expected 15-min return
    direction: str          # UP/DOWN/FLAT
    vol_forecast: float     # Expected volatility
    confidence: float       # Prediction confidence
    
    def to_dict(self) -> dict:
        return {
            'lstm_return_5min': self.return_5min,
            'lstm_return_15min': self.return_15min,
            'lstm_direction': self.direction,
            'lstm_vol_forecast': self.vol_forecast,
            'lstm_confidence': self.confidence,
        }


class PriceForecaster:
    """
    LSTM-based price and volatility forecaster.
    
    Uses sequence of past prices/features to predict future returns.
    """
    
    # Feature columns for LSTM input
    SEQUENCE_FEATURES = [
        'close_return',      # Price return
        'volume_ma_ratio',   # Volume vs MA
        'iv',                # Implied volatility
        'delta',             # Delta
        'gamma',             # Gamma
        'vix_normalized',    # VIX normalized
    ]
    
    SEQUENCE_LENGTH = 60  # 60 time bars (e.g., minutes)
    
    def __init__(self, model_path: str = "data/models/price_forecaster.keras"):
        """
        Initialize forecaster.
        
        Args:
            model_path: Path to saved LSTM model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.is_ready = False
        
        if HAS_TF:
            self._load_model()
        else:
            logger.warning("TensorFlow not available - LSTM disabled")
    
    def _load_model(self):
        """Load trained model if exists."""
        if self.model_path.exists():
            try:
                self.model = load_model(self.model_path)
                scaler_path = self.model_path.with_suffix('.scaler.pkl')
                if scaler_path.exists():
                    self.scaler = joblib.load(scaler_path)
                self.is_ready = True
                logger.info(f"LSTM model loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load LSTM model: {e}")
        else:
            logger.info("No LSTM model found - will use fallback predictions")
    
    def build_model(self, n_features: int, n_outputs: int = 2) -> 'tf.keras.Model':
        """Build LSTM architecture."""
        if not HAS_TF:
            raise RuntimeError("TensorFlow required for model building")
        
        model = Sequential([
            LSTM(64, return_sequences=True, 
                 input_shape=(self.SEQUENCE_LENGTH, n_features)),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dense(n_outputs)  # [return_5min, return_15min]
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(self, df: pd.DataFrame, 
                         target_cols: List[str] = ['return_5min', 'return_15min']
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            df: DataFrame with features
            target_cols: Columns to predict
            
        Returns:
            (X, y) where X is (samples, seq_length, features)
        """
        # Calculate derived features
        df = df.copy()
        
        # Close returns
        if 'close_return' not in df.columns:
            df['close_return'] = df['close'].pct_change().fillna(0)
        
        # Volume MA ratio
        if 'volume_ma_ratio' not in df.columns:
            vol_ma = df['volume'].rolling(20).mean()
            df['volume_ma_ratio'] = (df['volume'] / vol_ma).fillna(1)
        
        # VIX normalized
        if 'vix_normalized' not in df.columns and 'vix' in df.columns:
            df['vix_normalized'] = df['vix'] / 20  # Normalize around typical VIX
        elif 'vix_normalized' not in df.columns:
            df['vix_normalized'] = 1.0
        
        # Fill missing feature columns
        for col in self.SEQUENCE_FEATURES:
            if col not in df.columns:
                df[col] = 0
        
        # Create future returns as targets
        if 'return_5min' not in df.columns:
            df['return_5min'] = df['close'].pct_change(5).shift(-5)
        if 'return_15min' not in df.columns:
            df['return_15min'] = df['close'].pct_change(15).shift(-15)
        
        # Drop NaN
        df = df.dropna(subset=target_cols + self.SEQUENCE_FEATURES)
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        if self.scaler is None:
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(df[self.SEQUENCE_FEATURES])
        else:
            features_scaled = self.scaler.transform(df[self.SEQUENCE_FEATURES])
        
        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - self.SEQUENCE_LENGTH):
            X.append(features_scaled[i:i + self.SEQUENCE_LENGTH])
            y.append(df[target_cols].iloc[i + self.SEQUENCE_LENGTH].values)
        
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame, epochs: int = 50, 
              validation_split: float = 0.2) -> dict:
        """
        Train LSTM on historical data.
        
        Args:
            df: DataFrame with features and price data
            epochs: Training epochs
            validation_split: Validation fraction
            
        Returns:
            Training history dict
        """
        if not HAS_TF:
            raise RuntimeError("TensorFlow required for training")
        
        logger.info("Preparing LSTM training sequences...")
        X, y = self.prepare_sequences(df)
        
        logger.info(f"Training data: X={X.shape}, y={y.shape}")
        
        # Build model
        self.model = self.build_model(n_features=X.shape[2], n_outputs=y.shape[1])
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5),
        ]
        
        # Train
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            batch_size=64,
            verbose=1
        )
        
        # Save model
        self._save_model()
        
        return {
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss']),
        }
    
    def _save_model(self):
        """Save model and scaler."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_path)
        
        if self.scaler:
            scaler_path = self.model_path.with_suffix('.scaler.pkl')
            joblib.save(self.scaler, scaler_path)
        
        logger.info(f"LSTM model saved to {self.model_path}")
    
    def predict(self, sequence: np.ndarray) -> PriceForecast:
        """
        Predict from a single sequence.
        
        Args:
            sequence: Array of shape (seq_length, features)
            
        Returns:
            PriceForecast with predictions
        """
        if not self.is_ready or self.model is None:
            return self._fallback_prediction()
        
        try:
            # Reshape for batch prediction
            X = sequence.reshape(1, *sequence.shape)
            
            # Predict
            pred = self.model.predict(X, verbose=0)[0]
            return_5min = float(pred[0])
            return_15min = float(pred[1]) if len(pred) > 1 else return_5min * 3
            
            # Determine direction
            if return_5min > 0.001:
                direction = "UP"
            elif return_5min < -0.001:
                direction = "DOWN"
            else:
                direction = "FLAT"
            
            # Estimate volatility from recent sequence
            vol_forecast = float(np.std(sequence[:, 0]) * np.sqrt(252 * 390))  # Annualized
            
            # Confidence based on prediction magnitude (stronger = more confident)
            confidence = min(abs(return_5min) * 100, 0.9)
            
            return PriceForecast(
                return_5min=return_5min,
                return_15min=return_15min,
                direction=direction,
                vol_forecast=vol_forecast,
                confidence=confidence
            )
            
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}")
            return self._fallback_prediction()
    
    def predict_from_features(self, features: Dict[str, Any], 
                              history: List[Dict] = None) -> PriceForecast:
        """
        Predict from feature dict and optional history.
        
        Args:
            features: Current market features
            history: List of past feature dicts (at least 60)
            
        Returns:
            PriceForecast
        """
        if not self.is_ready or history is None or len(history) < self.SEQUENCE_LENGTH:
            return self._fallback_prediction()
        
        try:
            # Build sequence from history
            sequence_data = []
            for h in history[-self.SEQUENCE_LENGTH:]:
                row = [h.get(col, 0) for col in self.SEQUENCE_FEATURES]
                sequence_data.append(row)
            
            sequence = np.array(sequence_data)
            
            # Scale
            if self.scaler:
                sequence = self.scaler.transform(sequence)
            
            return self.predict(sequence)
            
        except Exception as e:
            logger.warning(f"LSTM predict_from_features failed: {e}")
            return self._fallback_prediction()
    
    def _fallback_prediction(self) -> PriceForecast:
        """Fallback when model not available."""
        return PriceForecast(
            return_5min=0.0,
            return_15min=0.0,
            direction="FLAT",
            vol_forecast=0.2,  # 20% typical
            confidence=0.1     # Low confidence
        )
    
    def add_predictions_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add LSTM predictions to DataFrame.
        
        For each row, uses previous SEQUENCE_LENGTH rows as input.
        """
        if not self.is_ready:
            logger.warning("LSTM not ready - adding fallback predictions")
            df['lstm_return_5min'] = 0.0
            df['lstm_return_15min'] = 0.0
            df['lstm_direction'] = "FLAT"
            df['lstm_vol_forecast'] = 0.2
            df['lstm_confidence'] = 0.1
            return df
        
        # Prepare sequences
        X, _ = self.prepare_sequences(df, target_cols=['close'])  # Dummy target
        
        # Predict all
        predictions = self.model.predict(X, verbose=0)
        
        # Add to dataframe (offset by SEQUENCE_LENGTH)
        pred_df = pd.DataFrame({
            'lstm_return_5min': predictions[:, 0],
            'lstm_return_15min': predictions[:, 1] if predictions.shape[1] > 1 else predictions[:, 0] * 3,
        })
        
        # Pad beginning with NaN
        padding = pd.DataFrame({
            'lstm_return_5min': [np.nan] * self.SEQUENCE_LENGTH,
            'lstm_return_15min': [np.nan] * self.SEQUENCE_LENGTH,
        })
        
        pred_df = pd.concat([padding, pred_df], ignore_index=True)
        
        # Add direction
        pred_df['lstm_direction'] = pred_df['lstm_return_5min'].apply(
            lambda x: "UP" if x > 0.001 else ("DOWN" if x < -0.001 else "FLAT")
        )
        
        # Truncate to match original length
        pred_df = pred_df.head(len(df))
        
        # Merge
        for col in pred_df.columns:
            df[col] = pred_df[col].values
        
        return df


# Singleton
_forecaster: Optional[PriceForecaster] = None

def get_price_forecaster() -> PriceForecaster:
    """Get singleton PriceForecaster."""
    global _forecaster
    if _forecaster is None:
        _forecaster = PriceForecaster()
    return _forecaster


# Quick test
if __name__ == "__main__":
    from core.logger import setup_logger
    setup_logger(level="INFO")
    
    forecaster = PriceForecaster()
    
    # Test fallback
    pred = forecaster._fallback_prediction()
    print(f"Fallback prediction: {pred}")
    
    # Test with synthetic data
    print("\nTesting sequence preparation...")
    np.random.seed(42)
    df = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(200) * 0.1),
        'volume': np.random.randint(1000, 10000, 200),
        'iv': np.random.uniform(0.15, 0.35, 200),
        'delta': np.random.uniform(0.3, 0.7, 200),
        'gamma': np.random.uniform(0.01, 0.05, 200),
        'vix': np.random.uniform(15, 25, 200),
    })
    
    try:
        X, y = forecaster.prepare_sequences(df)
        print(f"Prepared sequences: X={X.shape}, y={y.shape}")
    except Exception as e:
        print(f"Sequence prep failed (expected without full data): {e}")
