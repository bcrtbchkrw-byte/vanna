"""
Yahoo Finance Earnings Fetcher

Fetches earnings dates for individual stocks from Yahoo Finance.
Used by DailyOptionsScreener to filter stocks with upcoming earnings.

No API key required - uses public Yahoo Finance data.
"""
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List
from pathlib import Path
import json

from loguru import logger

# Optional: yfinance for better API
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not installed. Run: pip install yfinance")


class YahooEarningsFetcher:
    """
    Fetch earnings dates from Yahoo Finance.
    
    Methods:
    - get_next_earnings(symbol) -> Optional[date]
    - get_days_to_earnings(symbol) -> int
    - batch_fetch(symbols) -> Dict[str, date]
    
    Caches results to reduce API calls.
    """
    
    CACHE_FILE = "data/yahoo_earnings_cache.json"
    CACHE_TTL_HOURS = 12  # Refresh every 12 hours
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.cache_file = self.data_dir / "yahoo_earnings_cache.json"
        self._cache: Dict[str, dict] = {}
        
        self._load_cache()
        logger.info(f"YahooEarningsFetcher initialized (yfinance: {HAS_YFINANCE})")
    
    def _load_cache(self):
        """Load cached earnings dates."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self._cache = json.load(f)
            except Exception as e:
                logger.debug(f"Could not load earnings cache: {e}")
                self._cache = {}
    
    def _save_cache(self):
        """Save earnings cache to file."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self._cache, f, indent=2, default=str)
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached earnings is still valid."""
        if symbol not in self._cache:
            return False
        
        cached = self._cache[symbol]
        cached_time = datetime.fromisoformat(cached.get('fetched_at', '2000-01-01'))
        age_hours = (datetime.now() - cached_time).total_seconds() / 3600
        
        return age_hours < self.CACHE_TTL_HOURS
    
    def get_next_earnings(self, symbol: str) -> Optional[date]:
        """
        Get next earnings date for a symbol.
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            
        Returns:
            Next earnings date or None if unknown
        """
        # Check cache first
        if self._is_cache_valid(symbol):
            earnings_str = self._cache[symbol].get('earnings_date')
            if earnings_str:
                return date.fromisoformat(earnings_str)
            return None
        
        # Fetch from Yahoo
        earnings_date = self._fetch_from_yahoo(symbol)
        
        # Cache result
        self._cache[symbol] = {
            'earnings_date': earnings_date.isoformat() if earnings_date else None,
            'fetched_at': datetime.now().isoformat()
        }
        self._save_cache()
        
        return earnings_date
    
    def get_dividend_yield(self, symbol: str) -> float:
        """
        Get annual dividend yield for a symbol (decimal, e.g. 0.015).
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Dividend yield (0.0 if none or unknown)
        """
        # Check cache validity
        if self._is_cache_valid(symbol):
            cached_yield = self._cache[symbol].get('dividend_yield')
            if cached_yield is not None:
                return float(cached_yield)
        
        # Fetch from Yahoo
        div_yield = 0.0
        if HAS_YFINANCE:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                # Try dividendYield (forward) then trailingAnnualDividendYield
                val = info.get('dividendYield')
                if val is None:
                    val = info.get('trailingAnnualDividendYield')
                
                if val is not None:
                    div_yield = float(val)
                    
            except Exception as e:
                logger.debug(f"{symbol}: Yahoo dividend fetch error: {e}")
        
        # Update cache (preserve existing data)
        if symbol not in self._cache:
            self._cache[symbol] = {}
        
        self._cache[symbol]['dividend_yield'] = div_yield
        self._cache[symbol]['fetched_at'] = datetime.now().isoformat()
        self._save_cache()
        
        return div_yield

    def _fetch_from_yahoo(self, symbol: str) -> Optional[date]:
        """Fetch earnings date from Yahoo Finance."""
        if not HAS_YFINANCE:
            return self._estimate_earnings(symbol)
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get earnings date from calendar
            calendar = ticker.calendar
            
            if calendar is not None and not calendar.empty:
                # calendar is a DataFrame with earnings date
                if 'Earnings Date' in calendar.index:
                    earnings_dates = calendar.loc['Earnings Date']
                    if len(earnings_dates) > 0:
                        # Get the first (next) earnings date
                        next_earnings = earnings_dates.iloc[0]
                        if hasattr(next_earnings, 'date'):
                            return next_earnings.date()
                        elif isinstance(next_earnings, str):
                            return datetime.strptime(next_earnings, '%Y-%m-%d').date()
            
            # Fallback: check earnings_dates attribute
            earnings_dates = getattr(ticker, 'earnings_dates', None)
            if earnings_dates is not None and not earnings_dates.empty:
                future_dates = [d for d in earnings_dates.index if d.date() > date.today()]
                if future_dates:
                    return min(future_dates).date()
            
            logger.debug(f"{symbol}: No earnings date found in Yahoo")
            return self._estimate_earnings(symbol)
            
        except Exception as e:
            logger.debug(f"{symbol}: Yahoo fetch error: {e}")
            return self._estimate_earnings(symbol)
    
    def _estimate_earnings(self, symbol: str) -> Optional[date]:
        """
        Estimate earnings based on quarterly pattern.
        
        Most companies report:
        - Q4/FY: January
        - Q1: April
        - Q2: July
        - Q3: October
        """
        today = date.today()
        month = today.month
        
        # Find next earnings month
        earnings_months = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
        
        for em in earnings_months:
            if em > month:
                # Earnings in ~3rd week of that month
                return date(today.year, em, 20)
        
        # Next year January
        return date(today.year + 1, 1, 20)
    
    def get_days_to_earnings(self, symbol: str) -> int:
        """
        Get days until next earnings.
        
        Returns:
            Days to earnings (0-90+), or 45 if unknown
        """
        earnings_date = self.get_next_earnings(symbol)
        
        if earnings_date is None:
            return 45  # Default: middle of quarter
        
        days = (earnings_date - date.today()).days
        return max(0, min(days, 90))
    
    def batch_fetch(self, symbols: List[str]) -> Dict[str, int]:
        """
        Fetch days to earnings for multiple symbols.
        
        Args:
            symbols: List of stock tickers
            
        Returns:
            Dict of symbol -> days_to_earnings
        """
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.get_days_to_earnings(symbol)
            except Exception as e:
                logger.debug(f"{symbol}: Error fetching earnings: {e}")
                results[symbol] = 45  # Default
        
        logger.info(f"Fetched earnings for {len(results)} symbols")
        return results
    
    def filter_no_earnings(
        self, 
        symbols: List[str], 
        blackout_days: int = 7
    ) -> List[str]:
        """
        Filter stocks that DON'T have earnings soon.
        
        Used by screener to avoid earnings risk.
        
        Args:
            symbols: List of stock tickers
            blackout_days: Days before earnings to exclude
            
        Returns:
            List of symbols without imminent earnings
        """
        safe_symbols = []
        
        for symbol in symbols:
            days = self.get_days_to_earnings(symbol)
            if days > blackout_days:
                safe_symbols.append(symbol)
            else:
                logger.debug(f"{symbol}: Excluded (earnings in {days} days)")
        
        excluded = len(symbols) - len(safe_symbols)
        if excluded > 0:
            logger.info(f"Excluded {excluded} stocks with earnings within {blackout_days} days")
        
        return safe_symbols


# Singleton
_fetcher: Optional[YahooEarningsFetcher] = None


def get_yahoo_earnings_fetcher() -> YahooEarningsFetcher:
    """Get or create Yahoo earnings fetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = YahooEarningsFetcher()
    return _fetcher


if __name__ == "__main__":
    from core.logger import setup_logger
    setup_logger(level="INFO")
    
    fetcher = get_yahoo_earnings_fetcher()
    
    # Test a few stocks
    test_symbols = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD']
    
    for symbol in test_symbols:
        days = fetcher.get_days_to_earnings(symbol)
        earnings = fetcher.get_next_earnings(symbol)
        print(f"{symbol}: {days} days to earnings ({earnings})")
