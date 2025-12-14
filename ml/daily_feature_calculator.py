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
        # Golden/Death cross signals
        df['sma_50_200_ratio'] = df['sma_50'] / df['sma_200']
        
        # ================================================================
        # 7. Earnings / Major Events
        # ================================================================
        # Default values
        df['days_to_major_event'] = 45
        df['is_event_week'] = 0
        df['is_event_day'] = 0
        df['event_iv_boost'] = 1.0
        
        if self.earnings_fetcher:
            # Get next earnings date (REAL)
            next_earnings = self.earnings_fetcher.get_next_earnings(symbol)
            
            if next_earnings:
                # Calculate for latest data point (most important for live trading)
                last_date = df['trade_date'].iloc[-1] if 'trade_date' in df.columns else None
                # If no trade_date col, try to derive from index or timestamp
                if last_date is None and 'timestamp' in df.columns:
                     last_date = df['timestamp'].iloc[-1].date()

                if last_date:
                    days = (next_earnings - last_date).days
                    # Update last row features
                    df.loc[df.index[-1], 'days_to_major_event'] = max(0, days)
                    df.loc[df.index[-1], 'is_event_week'] = 1 if 0 <= days <= 7 else 0
                    df.loc[df.index[-1], 'is_event_day'] = 1 if days == 0 else 0
                    
                    # Simple decay for historical backfill (approximate previous quarters)
                    # We assume earnings every ~90 days
                    # This creates a sawtooth pattern 90 -> 0 -> 90 -> 0 which is good enough for RL training on history
                    # to learn "cycles", even if dates aren't perfect historically.
                    
                    # Vectorized sawtooth wave for history
                    n = len(df)
                    # Create a decaying sawtooth from 90 down to 0
                    # Offset so the last value matches the 'real' days to earnings
                    offset = (days % 90)
                    # Create sequence: ... 90, 89, ..., 0, 90, 89 ...
                    # We use negative index to look back from the known future date
                    idx = np.arange(n)
                    # Cycle is approx 63 trading days (quarter)
                    cycle_len = 63 
                    
                    # Pattern: Days decreases by 1 each day until event, then resets
                    # We reconstruct past "days to earnings" by projecting backwards
                    
                    # Linear projection: days_historical = (days_real + (n - 1 - i)) % 90
                    # But accurate "days" should be trading days approx.
                    
                    days_array = (days + (n - 1 - idx)) % 91 # Modulo 91 to wrap 90->0
                    
                    df['days_to_major_event'] = days_array.astype(int)
                    df['is_event_week'] = (df['days_to_major_event'] <= 5).astype(int)
                    df['is_event_day'] = (df['days_to_major_event'] == 0).astype(int)
                    
                    # IV Boost (simulated): Higher close to earnings
                    # 1.0 baseline, up to 1.5 near event
                    df['event_iv_boost'] = 1.0 + (0.5 * np.exp(-df['days_to_major_event'] / 5))
            
            else:
                # If no earnings date found (e.g. ETF), check for FOMC logic override
                if symbol in ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']:
                    # ETFs don't have earnings, they have FOMC/CPI
                    # We simulate a monthly cycle (CPI) + 6-week cycle (FOMC) mix
                    # Just a 30-day cycle for "Macro Event"
                    n = len(df)
                    idx = np.arange(n)
                    days_array = idx % 22 # 22 trading days ~ 1 month
                    df['days_to_major_event'] = days_array.astype(int)
                    df['is_event_week'] = 0 # ETFs less binary than stocks
                    df['event_iv_boost'] = 1.0 # Less vol crush for ETFs
        
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
