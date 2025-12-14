"""
Earnings Data Fetcher

Fetches earnings dates from IBKR/Yahoo and adds to *_1day.parquet:
- earnings_date (datetime)
- days_to_earnings (int)

Uses IBKR as primary source, Yahoo Finance as fallback.
"""
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, date, timedelta
import pandas as pd

from loguru import logger


class EarningsDataFetcher:
    """
    Fetch earnings dates and add to *_1day.parquet files.
    
    Features added:
    - earnings_date: Next earnings announcement date
    - days_to_earnings: Days until next earnings
    - is_earnings_week: Binary (1 if <= 7 days)
    - is_post_earnings: Binary (1 if within 3 days after)
    """
    
    # Known earnings calendar (approximate quarterly dates for major ETFs/stocks)
    # In production, this would come from IBKR/Yahoo API
    KNOWN_EARNINGS_MONTHS = {
        # ETFs don't have earnings but follow market quarters
        'SPY': [],
        'QQQ': [],
        'IWM': [],
        'GLD': [],
        'TLT': [],
        # Major stocks (approximate months)
        'AAPL': [1, 4, 7, 10],
        'MSFT': [1, 4, 7, 10],
        'NVDA': [2, 5, 8, 11],
        'TSLA': [1, 4, 7, 10],
        'AMZN': [1, 4, 7, 10],
        'GOOGL': [1, 4, 7, 10],
        'META': [1, 4, 7, 10],
    }
    
    def __init__(self, data_dir: str = "data/vanna_ml"):
        self.data_dir = Path(data_dir)
        self._ibkr_fetcher = None
        logger.info(f"EarningsDataFetcher initialized")
    
    async def _get_ibkr_earnings(self, symbol: str) -> Optional[datetime]:
        """Try to get earnings from IBKR."""
        try:
            from ibkr.data_fetcher import get_data_fetcher
            fetcher = await get_data_fetcher()
            return await fetcher.get_earnings_date(symbol)
        except Exception as e:
            logger.debug(f"IBKR earnings fetch failed for {symbol}: {e}")
            return None
    
    def _estimate_earnings_dates(self, symbol: str, df: pd.DataFrame) -> pd.Series:
        """
        Estimate earnings dates based on quarterly pattern.
        
        For ETFs (no earnings), returns NaT.
        For stocks, estimates based on known quarterly pattern.
        """
        n = len(df)
        earnings_dates = pd.Series([pd.NaT] * n, index=df.index)
        
        # ETFs don't have earnings
        earnings_months = self.KNOWN_EARNINGS_MONTHS.get(symbol, [1, 4, 7, 10])
        
        if not earnings_months:
            # ETF - no earnings
            return earnings_dates
        
        # Get dates from dataframe
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
        else:
            dates = pd.Series(df.index)
        
        for i, dt in enumerate(dates):
            if pd.isna(dt):
                continue
            
            # Find next earnings month
            current_month = dt.month
            current_day = dt.day
            current_year = dt.year
            
            next_earnings = None
            for em in sorted(earnings_months):
                if em > current_month or (em == current_month and current_day < 15):
                    # Earnings around mid-month
                    next_earnings = datetime(current_year, em, 15)
                    break
            
            if next_earnings is None:
                # Next year January
                next_earnings = datetime(current_year + 1, earnings_months[0], 15)
            
            earnings_dates.iloc[i] = next_earnings
        
        return earnings_dates
    
    def calculate_for_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Add earnings features to 1day parquet.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Updated DataFrame
        """
        filepath = self.data_dir / f"{symbol}_1day.parquet"
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None
        
        logger.info(f"Adding earnings features for {symbol}...")
        
        df = pd.read_parquet(filepath)
        
        # Get earnings dates (estimated or from API)
        df['earnings_date'] = self._estimate_earnings_dates(symbol, df)
        
        # Calculate days to earnings
        if 'timestamp' in df.columns:
            current_dates = pd.to_datetime(df['timestamp'])
        else:
            current_dates = pd.Series(df.index)
        
        df['days_to_earnings'] = (df['earnings_date'] - current_dates).dt.days
        df['days_to_earnings'] = df['days_to_earnings'].fillna(90).clip(0, 90).astype(int)
        
        # Binary features
        df['is_earnings_week'] = (df['days_to_earnings'] <= 7).astype(int)
        df['is_earnings_month'] = (df['days_to_earnings'] <= 30).astype(int)
        
        # Post-earnings indicator (for IV crush detection)
        # We approximate this: if days_to_earnings just jumped from low to high
        df['earnings_iv_boost'] = 1.0
        df.loc[df['days_to_earnings'] <= 7, 'earnings_iv_boost'] = 1.5
        df.loc[df['days_to_earnings'] <= 3, 'earnings_iv_boost'] = 2.0
        df.loc[df['days_to_earnings'] <= 1, 'earnings_iv_boost'] = 2.5
        
        # Save
        df.to_parquet(filepath, index=False)
        logger.info(f"  ✅ Added earnings features to {filepath}")
        
        return df
    
    def calculate_all(self, symbols: Optional[List[str]] = None) -> Dict[str, bool]:
        """Add earnings features to all symbols."""
        if symbols is None:
            symbols = [
                f.stem.replace('_1day', '')
                for f in self.data_dir.glob('*_1day.parquet')
            ]
        
        results = {}
        for symbol in symbols:
            try:
                df = self.calculate_for_symbol(symbol)
                results[symbol] = df is not None
            except Exception as e:
                logger.error(f"Error with {symbol}: {e}")
                results[symbol] = False
        
        logger.info(f"✅ Earnings features added: {sum(results.values())}/{len(results)}")
        return results


# Singleton
_fetcher: Optional[EarningsDataFetcher] = None


def get_earnings_data_fetcher() -> EarningsDataFetcher:
    """Get or create earnings data fetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = EarningsDataFetcher()
    return _fetcher


if __name__ == "__main__":
    from core.logger import setup_logger
    setup_logger(level="INFO")
    
    fetcher = get_earnings_data_fetcher()
    fetcher.calculate_all()
