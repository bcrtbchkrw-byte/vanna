#!/usr/bin/env python3
"""
Add Technical Indicators to Daily Parquet Files

Calculates and adds:
- SMA 20, 50, 200
- RSI 14
- ATR 14
- Bollinger Bands
- MACD
- Major event features (placeholder)

These are required by DailyFeatureInjector for RL training.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=1).mean()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs.fillna(0)))


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = high - low
    high_close_prev = (high - close.shift()).abs()
    low_close_prev = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bollinger Bands - returns (upper, middle, lower, position)."""
    middle = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    # Position: 0 = at lower band, 1 = at upper band
    position = (series - lower) / (upper - lower + 1e-8)
    return upper, middle, lower, position.clip(0, 1)


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD - returns (macd_line, signal_line, histogram)."""
    ema_fast = series.ewm(span=fast, min_periods=1).mean()
    ema_slow = series.ewm(span=slow, min_periods=1).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=1).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to daily DataFrame."""
    df = df.copy()
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # SMAs
    df['sma_20'] = calculate_sma(close, 20)
    df['sma_50'] = calculate_sma(close, 50)
    df['sma_200'] = calculate_sma(close, 200)
    
    # Price vs SMA (normalized to 0-centered ratio)
    df['price_vs_sma200'] = (close / df['sma_200']) - 1
    df['price_vs_sma50'] = (close / df['sma_50']) - 1
    
    # Above SMA (binary)
    df['above_sma200'] = (close > df['sma_200']).astype(int)
    df['above_sma50'] = (close > df['sma_50']).astype(int)
    
    # SMA ratio
    df['sma_50_200_ratio'] = df['sma_50'] / df['sma_200'].replace(0, np.nan)
    df['sma_50_200_ratio'] = df['sma_50_200_ratio'].fillna(1.0)
    
    # RSI
    df['rsi_14'] = calculate_rsi(close, 14)
    
    # ATR
    df['atr_14'] = calculate_atr(high, low, close, 14)
    df['atr_pct'] = df['atr_14'] / close  # ATR as percentage of price
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower, bb_position = calculate_bollinger_bands(close)
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_position'] = bb_position
    
    # MACD
    macd_line, macd_signal, macd_hist = calculate_macd(close)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    
    # Major event features (placeholder defaults)
    # These would ideally be computed from earnings/FOMC calendars
    df['days_to_major_event'] = 30  # Default: ~1 month
    df['is_event_week'] = 0
    df['is_event_day'] = 0
    df['event_iv_boost'] = 1.0
    
    # Fill any NaN
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].ffill().bfill().fillna(0)
    
    return df


def process_all_daily_files():
    """Add technical indicators to all *_1day.parquet files."""
    data_dir = Path('data/vanna_ml')
    
    print('=' * 60)
    print('ADDING TECHNICAL INDICATORS TO DAILY PARQUET FILES')
    print('=' * 60)
    
    for filepath in sorted(data_dir.glob('*_1day.parquet')):
        symbol = filepath.stem.replace('_1day', '')
        
        df = pd.read_parquet(filepath)
        n_before = len(df.columns)
        
        df = add_daily_indicators(df)
        n_after = len(df.columns)
        
        df.to_parquet(filepath, index=False, compression='snappy')
        
        print(f'✅ {filepath.name}: {n_before} → {n_after} columns (+{n_after - n_before})')
    
    print()
    print('✅ All daily files updated!')


if __name__ == "__main__":
    process_all_daily_files()
