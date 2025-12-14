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
        
        # Initialize earnings fetcher
        try:
            from ml.yahoo_earnings import get_yahoo_earnings_fetcher
            self.earnings_fetcher = get_yahoo_earnings_fetcher()
        except ImportError:
            self.earnings_fetcher = None
            logger.warning("YahooEarningsFetcher not available, earnings features will be placeholders")

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
        
    @staticmethod
    def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical features (SMA, RSI, MACD, etc.).
        Single Source of Truth for both Training and Live Trading.
        
        Args:
            df: DataFrame with 'close', 'high', 'low'
            
        Returns:
            DataFrame with added technical columns
        """
        df = df.copy()
        
        # Ensure sufficient history for calculation
        if len(df) < 50:
            return df
            
        # 1. Moving Averages
        df['sma_200'] = df['close'].rolling(window=200, min_periods=50).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=20).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=10).mean()
        
        # Price vs SMA ratios
        df['price_vs_sma200'] = df['close'] / df['sma_200']
        df['price_vs_sma50'] = df['close'] / df['sma_50']
        df['sma_50_200_ratio'] = df['sma_50'] / df['sma_200']
        
        # 2. RSI (EMA-based, standard)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
        rs = gain / loss.replace(0, 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50)
        
        # 3. ATR
        if 'high' in df.columns and 'low' in df.columns:
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - df['close'].shift(1)).abs()
            tr3 = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(window=14).mean()
            df['atr_pct'] = df['atr_14'] / df['close']
        
        # 4. Bollinger Bands
        df['bb_middle'] = df['sma_20']
        df['bb_std'] = df['close'].rolling(window=20, min_periods=10).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # BB position
        diff = df['bb_upper'] - df['bb_lower']
        # Avoid division by zero
        diff = diff.replace(0, 1e-9) 
        df['bb_position'] = (df['close'] - df['bb_lower']) / diff
        df['bb_position'] = df['bb_position'].clip(0, 1)
        
        # 5. MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 6. Trend binaries
        df['above_sma200'] = (df['close'] > df['sma_200']).astype(int)
        df['above_sma50'] = (df['close'] > df['sma_50']).astype(int)
        
        return df

    def calculate_for_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        # ... (implementation using add_technical_features)
        filepath = self.data_dir / f"{symbol}_1day.parquet"
        if not filepath.exists():
            return None
            
        logger.info(f"Calculating daily features for {symbol}...")
        df = pd.read_parquet(filepath)
        
        if 'close' not in df.columns:
            return None
            
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
            
        n_before = len(df.columns)
        
        # Use centralized calculation
        df = self.add_technical_features(df)
        
        # 7. Earnings / Major Events calculation (keeps existing logic)
        # ... (rest of earnings logic)
        # This part requires 'self' so it stays here or we pass fetcher to static method
        # Keeping it here for now as it relies on self.earnings_fetcher
        
        # Default values
        if 'days_to_major_event' not in df.columns:
             df['days_to_major_event'] = 45
             df['is_event_week'] = 0
             df['is_event_day'] = 0
             df['event_iv_boost'] = 1.0
             df['major_event_type'] = 'none' # Default
        
        if self.earnings_fetcher:
            # ... (Earnings logic from previous step, preserved)
            next_earnings = self.earnings_fetcher.get_next_earnings(symbol)
            if next_earnings:
                last_date = None
                if 'trade_date' in df.columns:
                    last_date = df['trade_date'].iloc[-1]
                elif 'timestamp' in df.columns:
                    last_date = df['timestamp'].iloc[-1].date()
                
                if last_date:
                    days = (next_earnings - last_date).days
                    df.loc[df.index[-1], 'days_to_major_event'] = max(0, days)
                    df.loc[df.index[-1], 'is_event_week'] = 1 if 0 <= days <= 7 else 0
                    df.loc[df.index[-1], 'is_event_day'] = 1 if days == 0 else 0
                    df.loc[df.index[-1], 'major_event_type'] = 'earnings' # Set type
                    
                    # Backfill simulation logic
                    n = len(df)
                    offset = (days % 90)
                    idx = np.arange(n)
                    days_array = (days + (n - 1 - idx)) % 91
                    df['days_to_major_event'] = days_array.astype(int)
                    df['is_event_week'] = (df['days_to_major_event'] <= 5).astype(int)
                    df['is_event_day'] = (df['days_to_major_event'] == 0).astype(int)
                    df['event_iv_boost'] = 1.0 + (0.5 * np.exp(-df['days_to_major_event'] / 5))
                    df['major_event_type'] = 'earnings' # Backfill type
            else:
                 if symbol in ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']:
                    n = len(df)
                    idx = np.arange(n)
                    days_array = idx % 22
                    df['days_to_major_event'] = days_array.astype(int)
                    df['is_event_week'] = 0
                    df['event_iv_boost'] = 1.0
                    df['major_event_type'] = 'macro'
        
        n_after = len(df.columns)
        logger.info(f"  {symbol}: Added {n_after - n_before} daily features ({n_after} total)")
        
        df.to_parquet(filepath, index=False)
        return df
        
    def _calculate_rsi(self, prices, period=14):
        # Deprecated, used add_technical_features
        pass
    
    def _calculate_atr(self, df, period=14):
        # Deprecated
        pass


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
