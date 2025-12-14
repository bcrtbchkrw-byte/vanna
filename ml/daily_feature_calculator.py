"""
Daily Feature Calculator

Calculates daily technical indicators and adds them to *_1day.parquet:
- SMA 200, 50, 20
- RSI 14
- ATR 14
- Bollinger Bands

These features are later injected into 1-min data for ML/RL training.
"""
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np

from loguru import logger


class DailyFeatureCalculator:
    """
    Calculate daily technical indicators for *_1day.parquet files.
    
    Features added:
    - sma_200, sma_50, sma_20
    - rsi_14
    - atr_14
    - bb_upper, bb_lower, bb_middle
    - price_vs_sma200 (ratio)
    """
    
    def __init__(self, data_dir: str = "data/vanna_ml"):
        self.data_dir = Path(data_dir)
        logger.info(f"DailyFeatureCalculator initialized (dir: {self.data_dir})")
    
    def calculate_for_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Calculate daily features for a single symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'SPY')
            
        Returns:
            Updated DataFrame or None if file not found
        """
        filepath = self.data_dir / f"{symbol}_1day.parquet"
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None
        
        logger.info(f"Calculating daily features for {symbol}...")
        
        # Load data
        df = pd.read_parquet(filepath)
        
        # Ensure we have required columns
        if 'close' not in df.columns:
            logger.error(f"{symbol}: Missing 'close' column")
            return None
        
        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        n_before = len(df.columns)
        
        # ================================================================
        # 1. Moving Averages
        # ================================================================
        df['sma_200'] = df['close'].rolling(window=200, min_periods=50).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=20).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=10).mean()
        
        # Price vs SMA ratios (useful for trend detection)
        df['price_vs_sma200'] = df['close'] / df['sma_200']
        df['price_vs_sma50'] = df['close'] / df['sma_50']
        
        # ================================================================
        # 2. RSI (Relative Strength Index)
        # ================================================================
        df['rsi_14'] = self._calculate_rsi(df['close'], period=14)
        
        # ================================================================
        # 3. ATR (Average True Range)
        # ================================================================
        if 'high' in df.columns and 'low' in df.columns:
            df['atr_14'] = self._calculate_atr(df, period=14)
            df['atr_pct'] = df['atr_14'] / df['close']  # ATR as % of price
        
        # ================================================================
        # 4. Bollinger Bands
        # ================================================================
        df['bb_middle'] = df['sma_20']
        df['bb_std'] = df['close'].rolling(window=20, min_periods=10).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # BB position: 0 = at lower, 1 = at upper
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_position'] = df['bb_position'].clip(0, 1)
        
        # ================================================================
        # 5. MACD
        # ================================================================
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ================================================================
        # 6. Trend indicators
        # ================================================================
        # Above/below key SMAs (binary)
        df['above_sma200'] = (df['close'] > df['sma_200']).astype(int)
        df['above_sma50'] = (df['close'] > df['sma_50']).astype(int)
        
        # Golden/Death cross signals
        df['sma_50_200_ratio'] = df['sma_50'] / df['sma_200']
        
        n_after = len(df.columns)
        logger.info(f"  {symbol}: Added {n_after - n_before} daily features ({n_after} total)")
        
        # Save updated parquet
        df.to_parquet(filepath, index=False)
        logger.info(f"  ✅ Saved to {filepath}")
        
        return df
    
    def calculate_all(self, symbols: Optional[List[str]] = None) -> dict:
        """
        Calculate daily features for all symbols.
        
        Args:
            symbols: List of symbols or None for auto-detect
            
        Returns:
            Dict of results per symbol
        """
        if symbols is None:
            # Auto-detect from files
            symbols = [
                f.stem.replace('_1day', '')
                for f in self.data_dir.glob('*_1day.parquet')
            ]
        
        logger.info(f"Calculating daily features for {len(symbols)} symbols...")
        
        results = {}
        for symbol in symbols:
            try:
                df = self.calculate_for_symbol(symbol)
                results[symbol] = df is not None
            except Exception as e:
                logger.error(f"Error calculating {symbol}: {e}")
                results[symbol] = False
        
        success = sum(results.values())
        logger.info(f"✅ Daily features calculated: {success}/{len(symbols)} symbols")
        
        return results
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # Use EMA for smoother RSI
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral RSI for missing
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr


# Singleton
_calculator: Optional[DailyFeatureCalculator] = None


def get_daily_feature_calculator() -> DailyFeatureCalculator:
    """Get or create daily feature calculator."""
    global _calculator
    if _calculator is None:
        _calculator = DailyFeatureCalculator()
    return _calculator


# CLI
if __name__ == "__main__":
    from core.logger import setup_logger
    setup_logger(level="INFO")
    
    calc = get_daily_feature_calculator()
    calc.calculate_all()
