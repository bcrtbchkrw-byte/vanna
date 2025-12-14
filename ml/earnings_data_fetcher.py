"""
Major Events Calculator

Calculates days_to_major_event for each symbol:
- SPY, QQQ: Mega-cap earnings (AAPL, MSFT, NVDA, AMZN, GOOGL)
- TLT, GLD, IWM: FOMC meetings and CPI releases

FOMC/CPI dates are fetched dynamically from Fed/BLS websites.
See: ml/economic_calendar.py
"""
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

from loguru import logger


class MajorEventsCalculator:
    """
    Calculate days_to_major_event for ETF parquet files.
    
    Events tracked:
    - Mega-cap earnings (AAPL, MSFT, NVDA, AMZN, GOOGL)
    - FOMC meetings (fetched dynamically)
    - CPI releases (fetched dynamically)
    
    Features added:
    - days_to_major_event
    - major_event_type: 'EARNINGS'|'FOMC'|'CPI'|'NONE'
    - is_event_week: Binary
    - event_iv_boost: IV multiplier hint
    """
    
    # Mega-cap earnings months (approximate)
    MEGA_CAP_EARNINGS = {
        'AAPL': [1, 4, 7, 10],
        'MSFT': [1, 4, 7, 10],
        'NVDA': [2, 5, 8, 11],
        'AMZN': [2, 4, 7, 10],
        'GOOGL': [1, 4, 7, 10],
    }
    
    # Which events affect which symbols
    SYMBOL_EVENTS = {
        'SPY': ['EARNINGS'],
        'QQQ': ['EARNINGS'],
        'IWM': ['FOMC', 'CPI', 'EARNINGS'],
        'TLT': ['FOMC', 'CPI'],
        'GLD': ['FOMC', 'CPI'],
    }
    
    def __init__(self, data_dir: str = "data/vanna_ml"):
        self.data_dir = Path(data_dir)
        
        # Fetch calendar dates dynamically
        from ml.economic_calendar import get_economic_calendar
        self._calendar = get_economic_calendar()
        
        # Cache dates
        self._fomc_dates = self._calendar.get_cached_dates('FOMC')
        self._cpi_dates = self._calendar.get_cached_dates('CPI')
        
        logger.info(f"MajorEventsCalculator initialized (FOMC: {len(self._fomc_dates)}, CPI: {len(self._cpi_dates)} dates)")
    
    def _get_mega_cap_earnings_dates(self, year: int) -> List[date]:
        """Generate approximate mega-cap earnings dates for a year."""
        dates = []
        for symbol, months in self.MEGA_CAP_EARNINGS.items():
            for month in months:
                # Approximate: 3rd week of month
                earnings_date = date(year, month, 20)
                dates.append(earnings_date)
        return sorted(set(dates))
    
    def _find_next_event(
        self, 
        current_date: date, 
        symbol: str
    ) -> tuple:
        """
        Find next major event for a symbol.
        
        Returns:
            (days_to_event, event_type)
        """
        event_types = self.SYMBOL_EVENTS.get(symbol, ['FOMC', 'CPI'])
        
        min_days = 999
        next_event_type = None
        
        # Check FOMC (using dynamically fetched dates)
        if 'FOMC' in event_types:
            for fomc_date in self._fomc_dates:
                if fomc_date > current_date:
                    days = (fomc_date - current_date).days
                    if days < min_days:
                        min_days = days
                        next_event_type = 'FOMC'
                    break
        
        # Check CPI (using dynamically fetched dates)
        if 'CPI' in event_types:
            for cpi_date in self._cpi_dates:
                if cpi_date > current_date:
                    days = (cpi_date - current_date).days
                    if days < min_days:
                        min_days = days
                        next_event_type = 'CPI'
                    break
        
        # Check mega-cap earnings
        if 'EARNINGS' in event_types:
            for year in [current_date.year, current_date.year + 1]:
                for earnings_date in self._get_mega_cap_earnings_dates(year):
                    if earnings_date > current_date:
                        days = (earnings_date - current_date).days
                        if days < min_days:
                            min_days = days
                            next_event_type = 'EARNINGS'
                        break
                if next_event_type == 'EARNINGS':
                    break
        
        return min(min_days, 90), next_event_type
    
    def calculate_for_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Add major event features to 1day parquet.
        
        Args:
            symbol: ETF symbol
            
        Returns:
            Updated DataFrame
        """
        filepath = self.data_dir / f"{symbol}_1day.parquet"
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None
        
        logger.info(f"Calculating major events for {symbol}...")
        
        df = pd.read_parquet(filepath)
        
        # Get dates
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp']).dt.date
        else:
            dates = pd.Series([date(2020, 1, 1)] * len(df))
        
        # Calculate for each row
        days_to_event = []
        event_types = []
        
        for current_date in dates:
            days, event_type = self._find_next_event(current_date, symbol)
            days_to_event.append(days)
            event_types.append(event_type or 'NONE')
        
        df['days_to_major_event'] = days_to_event
        df['major_event_type'] = event_types
        df['is_event_week'] = (df['days_to_major_event'] <= 7).astype(int)
        df['is_event_day'] = (df['days_to_major_event'] <= 1).astype(int)
        
        # IV boost based on proximity to event
        df['event_iv_boost'] = 1.0
        df.loc[df['days_to_major_event'] <= 7, 'event_iv_boost'] = 1.3
        df.loc[df['days_to_major_event'] <= 3, 'event_iv_boost'] = 1.5
        df.loc[df['days_to_major_event'] <= 1, 'event_iv_boost'] = 2.0
        
        # Post-event (potential vol crush)
        # Note: This is harder to calculate without knowing if event passed
        
        # Remove old earnings columns if they exist
        old_cols = ['earnings_date', 'days_to_earnings', 'is_earnings_week', 
                    'is_earnings_month', 'earnings_iv_boost']
        df = df.drop(columns=[c for c in old_cols if c in df.columns], errors='ignore')
        
        # Save
        df.to_parquet(filepath, index=False)
        logger.info(f"  ✅ {symbol}: Added major event features (next: {event_types[0] if event_types else 'NONE'})")
        
        return df
    
    def calculate_all(self, symbols: Optional[List[str]] = None) -> Dict[str, bool]:
        """Add major event features to all symbols."""
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
        
        logger.info(f"✅ Major events added: {sum(results.values())}/{len(results)}")
        return results


# Singleton
_calculator: Optional[MajorEventsCalculator] = None


def get_major_events_calculator() -> MajorEventsCalculator:
    """Get or create major events calculator."""
    global _calculator
    if _calculator is None:
        _calculator = MajorEventsCalculator()
    return _calculator


if __name__ == "__main__":
    from core.logger import setup_logger
    setup_logger(level="INFO")
    
    calc = get_major_events_calculator()
    calc.calculate_all()
