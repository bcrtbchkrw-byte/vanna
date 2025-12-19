import pandas as pd
import numpy as np
import glob
from pathlib import Path
from core.logger import setup_logger, get_logger
from ml.regime_classifier import RegimeClassifier

logger = get_logger()

def create_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'True' Regime labels based on FUTURE looking metrics.
    We want the model to predict the regime of the NEXT hour/day.
    
    Logic:
    0 (Low Vol): Future Realized Vol < 10% AND No Crash
    1 (Normal): Future Realized Vol 10-20%
    2 (Elevated): Future Realized Vol > 20%
    3 (High Vol): Future Realized Vol > 30%
    4 (Crisis): Future Drawdown > 2% in next hour OR Vol > 50%
    """
    df = df.copy()
    
    # 1. Calculate Future Realized Volatility (Next 60 min)
    # Volatility of returns over next 60 points * sqrt(252*390) for annualized
    # Simplified: Rolling std dev of returns shifted backwards
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=60)
    
    # Need return series first
    if 'return_1m' not in df.columns:
        if 'close' in df.columns:
            df['return_1m'] = df['close'].pct_change()
        else:
            return pd.DataFrame() # fail
            
    # Annualized Realized Volatility (next 60 mins)
    # 1-min returns std dev * sqrt(252 trading days * 390 minutes) -> approx * 313
    df['future_vol'] = df['return_1m'].rolling(window=indexer).std() * 313
    
    # 2. Calculate Future Drawdown (Max loss in next 60 mins)
    # Min(Low) in next 60 vs Current Close
    if 'low' in df.columns:
        future_low = df['low'].rolling(window=indexer).min()
        df['future_drawdown'] = (future_low - df['close']) / df['close']
    else:
        df['future_drawdown'] = 0.0
        
    # 3. Labeling Rules (Vectorized)
    conditions = [
        (df['future_drawdown'] < -0.02) | (df['future_vol'] > 0.50), # Crisis
        (df['future_vol'] > 0.30), # High Vol
        (df['future_vol'] > 0.20), # Elevated
        (df['future_vol'] > 0.10), # Normal
        (df['future_vol'] <= 0.10) # Low Vol
    ]
    
    choices = [4, 3, 2, 1, 0]
    
    df['regime'] = np.select(conditions, choices, default=1)
    
    # Drop NaNs at the end
    df = df.dropna(subset=['future_vol', 'regime'])
    
    return df

# ---------------------------------------------------------
# DataGenerator for efficient memory usage
# ---------------------------------------------------------
import tensorflow as tf

class RegimeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_data, y_data, seq_len=60, batch_size=64, indices=None, shuffle=True):
        self.x = x_data
        self.y = y_data
        self.seq_len = seq_len
        self.batch_size = batch_size
        # If indices not provided, use all valid indices
        if indices is None:
            self.indices = np.arange(seq_len, len(x_data))
        else:
            self.indices = indices
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        X_batch = np.empty((self.batch_size, self.seq_len, self.x.shape[1]))
        y_batch = np.empty((self.batch_size), dtype=int)
        
        for i, idx in enumerate(batch_indices):
            # idx is the end position
            X_batch[i] = self.x[idx-self.seq_len:idx]
            y_batch[i] = self.y[idx]
            
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def run_training() -> float:
    """
    Run full training pipeline for Regime Classifier.
    Returns validation accuracy.
    """
    setup_logger(level="INFO")
    logger.info("=" * 60)
    logger.info("RegimeClassifier Training (Smart Labeling)")
    logger.info("=" * 60)
    
    data_dir = Path("data/vanna_ml")
    files = list(data_dir.glob("*_1min_vanna.parquet"))
    
    if not files:
        logger.error("No data files found")
        return 0.0
        
    dfs = []
    
    for f in files:
        logger.info(f"Processing {f.name}...")
        try:
            df = pd.read_parquet(f)
            df = create_regime_labels(df)
            if not df.empty:
                dfs.append(df)
                logger.info(f"  -> Added {len(df)} samples")
        except Exception as e:
            logger.error(f"Failed {f.name}: {e}")
            
    if not dfs:
        return 0.0
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Feature Selection & Scaling
    features = RegimeClassifier.FEATURES
    # Ensure all features exist
    for f in features:
        if f not in full_df.columns:
            full_df[f] = 0.0
            
    X_raw = full_df[features].values
    y_raw = full_df['regime'].values.astype(int)
    
    # Robust cleanup
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info("Scaling features...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Clip extreme values to prevent gradient explosion
    X_scaled = np.clip(X_scaled, -10.0, 10.0)
    
    # Create indices for Train/Test split
    SEQ_LEN = 60
    total_samples = len(X_scaled)
    # Valid indices start from SEQ_LEN
    valid_indices = np.arange(SEQ_LEN, total_samples)
    
    # Split indices
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(valid_indices, test_size=0.2, shuffle=True, random_state=42)
    
    logger.info(f"Train samples: {len(train_idx):,}, Test samples: {len(test_idx):,}")
    
    # Generators
    train_gen = RegimeDataGenerator(X_scaled, y_raw, seq_len=SEQ_LEN, batch_size=128, indices=train_idx)
    test_gen = RegimeDataGenerator(X_scaled, y_raw, seq_len=SEQ_LEN, batch_size=128, indices=test_idx, shuffle=False)
    
    classifier = RegimeClassifier()
    classifier.scaler = scaler
    
    # Build model first
    classifier.build_model(input_shape=(SEQ_LEN, X_scaled.shape[1]))
    
    # Train using generator
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    
    history = classifier.model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=15,
        callbacks=[early_stopping],
        verbose=1
    )
    
    acc = float(history.history['val_accuracy'][-1])
    logger.info(f"Model trained. Val Accuracy: {acc:.2%}")
    
    classifier._save_model()
    return acc

if __name__ == "__main__":
    run_training()
