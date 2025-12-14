"""
Major Events Calculator

Calculates days_to_major_event for each symbol:
- SPY, QQQ: Mega-cap earnings (AAPL, MSFT, NVDA, AMZN, GOOGL)
- TLT, GLD, IWM: FOMC meetings and CPI releases

These events cause volatility spikes and are critical for options trading.
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
    - FOMC meetings (8 per year)
    - CPI releases (12 per year)
    
    Features added:
    - days_to_major_event: Days until next major event
    - event_type: 'EARNINGS'|'FOMC'|'CPI'|None
    - is_event_week: Binary (1 if <= 7 days)
    - event_iv_boost: IV multiplier hint
    """
    
    # 2024-2025 FOMC meeting dates (approximate - typically mid-week)
    FOMC_DATES = [
        # 2024
        date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),
        date(2024, 6, 12), date(2024, 7, 31), date(2024, 9, 18),
        date(2024, 11, 7), date(2024, 12, 18),
        # 2025
        date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
        date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
        date(2025, 11, 5), date(2025, 12, 17),
        # 2026
        date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6),
    ]
    
    # CPI release dates (typically 2nd week of month)
    # Generate for 2024-2026
    CPI_DATES = [
        # 2024
        date(2024, 1, 11), date(2024, 2, 13), date(2024, 3, 12),
        date(2024, 4, 10), date(2024, 5, 15), date(2024, 6, 12),
        date(2024, 7, 11), date(2024, 8, 14), date(2024, 9, 11),
        date(2024, 10, 10), date(2024, 11, 13), date(2024, 12, 11),
        # 2025
        date(2025, 1, 15), date(2025, 2, 12), date(2025, 3, 12),
        date(2025, 4, 10), date(2025, 5, 13), date(2025, 6, 11),
        date(2025, 7, 10), date(2025, 8, 13), date(2025, 9, 10),
        date(2025, 10, 10), date(2025, 11, 12), date(2025, 12, 10),
    ]
    
    # Mega-cap earnings months (approximate - typically late Jan, Apr, Jul, Oct)
    # These affect SPY/QQQ significantly
    MEGA_CAP_EARNINGS = {
        'AAPL': [1, 4, 7, 10],   # ~last week of earnings month
        'MSFT': [1, 4, 7, 10],
        'NVDA': [2, 5, 8, 11],  # Slightly offset
        'AMZN': [2, 4, 7, 10],
        'GOOGL': [1, 4, 7, 10],
    }
    
    # Which events affect which symbols
    SYMBOL_EVENTS = {
        'SPY': ['EARNINGS'],  # Big 5 earnings
        'QQQ': ['EARNINGS'],  # Big 5 earnings (even more weight)
        'IWM': ['FOMC', 'CPI', 'EARNINGS'],  # All events
        'TLT': ['FOMC', 'CPI'],  # Bond ETF = Fed sensitive
        'GLD': ['FOMC', 'CPI'],  # Gold = inflation/Fed sensitive
    }
    
    def __init__(self, data_dir: str = "data/vanna_ml"):
        self.data_dir = Path(data_dir)
        logger.info("MajorEventsCalculator initialized")
    
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
        
        # Check FOMC
        if 'FOMC' in event_types:
            for fomc_date in self.FOMC_DATES:
                if fomc_date > current_date:
                    days = (fomc_date - current_date).days
                    if days < min_days:
                        min_days = days
                        next_event_type = 'FOMC'
                    break
        
        # Check CPI
        if 'CPI' in event_types:
            for cpi_date in self.CPI_DATES:
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
