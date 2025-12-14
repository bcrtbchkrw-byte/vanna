"""
Economic Calendar Fetcher

Fetches FOMC and CPI dates from APIs, with JSON file fallback.
Updates are cached and saved to a local JSON file.

Sources:
1. FRED API (Federal Reserve Economic Data)
2. Investing.com calendar (web scrape fallback)
3. Local JSON file (offline fallback)
"""
from pathlib import Path
from typing import List, Optional, Dict
from datetime import date, datetime
import json

from loguru import logger


class EconomicCalendarFetcher:
    """
    Fetch economic calendar events (FOMC, CPI) from APIs with caching.
    
    Priority:
    1. Try FRED API (if API key available)
    2. Try web scrape (investing.com)
    3. Fallback to local JSON file (updated manually)
    """
    
    CACHE_FILE = "data/economic_calendar.json"
    CACHE_MAX_AGE_DAYS = 7  # Re-fetch if older than 7 days
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.cache_file = self.data_dir / "economic_calendar.json"
        self._cache: Dict = {}
        
        self._load_cache()
        logger.info("EconomicCalendarFetcher initialized")
    
    def _load_cache(self):
        """Load cached calendar from JSON file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded calendar cache: {len(self._cache.get('fomc', []))} FOMC, {len(self._cache.get('cpi', []))} CPI dates")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self._cache = {}
    
    def _save_cache(self):
        """Save calendar to JSON file."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._cache['last_updated'] = datetime.now().isoformat()
        
        with open(self.cache_file, 'w') as f:
            json.dump(self._cache, f, indent=2, default=str)
        
        logger.info(f"Saved calendar cache to {self.cache_file}")
    
    def _is_cache_stale(self) -> bool:
        """Check if cache needs refresh."""
        last_updated = self._cache.get('last_updated')
        if not last_updated:
            return True
        
        try:
            updated_dt = datetime.fromisoformat(last_updated)
            age_days = (datetime.now() - updated_dt).days
            return age_days > self.CACHE_MAX_AGE_DAYS
        except:
            return True
    
    async def fetch_fomc_dates(self) -> List[date]:
        """
        Fetch FOMC meeting dates.
        
        Returns list of meeting dates (announcements at 2 PM ET).
        """
        # Try API fetch first
        dates = await self._fetch_fomc_from_api()
        
        if dates:
            self._cache['fomc'] = [d.isoformat() for d in dates]
            self._save_cache()
            return dates
        
        # Fallback to cache
        if 'fomc' in self._cache:
            return [date.fromisoformat(d) for d in self._cache['fomc']]
        
        # Ultimate fallback - return known dates
        logger.warning("Using hardcoded FOMC dates fallback")
        return self._get_fallback_fomc_dates()
    
    async def fetch_cpi_dates(self) -> List[date]:
        """
        Fetch CPI release dates.
        
        Returns list of release dates (typically 8:30 AM ET).
        """
        dates = await self._fetch_cpi_from_api()
        
        if dates:
            self._cache['cpi'] = [d.isoformat() for d in dates]
            self._save_cache()
            return dates
        
        if 'cpi' in self._cache:
            return [date.fromisoformat(d) for d in self._cache['cpi']]
        
        logger.warning("Using hardcoded CPI dates fallback")
        return self._get_fallback_cpi_dates()
    
    async def _fetch_fomc_from_api(self) -> Optional[List[date]]:
        """Try to fetch FOMC dates from Federal Reserve website."""
        try:
            import aiohttp
            
            # Federal Reserve calendar page
            url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status != 200:
                        return None
                    
                    html = await resp.text()
                    
                    # Parse dates from HTML (simplified - look for patterns like "January 28-29")
                    import re
                    
                    dates = []
                    # Pattern: Month Day-Day, Year or Month Day, Year
                    patterns = [
                        r'(\w+)\s+(\d{1,2})-\d{1,2},?\s*(\d{4})',  # January 28-29, 2025
                        r'(\w+)\s+(\d{1,2}),?\s*(\d{4})',  # January 29, 2025
                    ]
                    
                    month_map = {
                        'January': 1, 'February': 2, 'March': 3, 'April': 4,
                        'May': 5, 'June': 6, 'July': 7, 'August': 8,
                        'September': 9, 'October': 10, 'November': 11, 'December': 12
                    }
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, html)
                        for match in matches:
                            month_name, day, year = match
                            if month_name in month_map:
                                try:
                                    d = date(int(year), month_map[month_name], int(day))
                                    if d not in dates and d > date.today() - timedelta(days=365):
                                        dates.append(d)
                                except:
                                    pass
                    
                    if dates:
                        dates.sort()
                        logger.info(f"Fetched {len(dates)} FOMC dates from Fed website")
                        return dates
                    
        except Exception as e:
            logger.debug(f"Fed website fetch failed: {e}")
        
        return None
    
    async def _fetch_cpi_from_api(self) -> Optional[List[date]]:
        """Try to fetch CPI dates from BLS website."""
        try:
            import aiohttp
            
            # BLS schedule page
            url = "https://www.bls.gov/schedule/news_release/cpi.htm"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status != 200:
                        return None
                    
                    html = await resp.text()
                    
                    import re
                    
                    # Pattern: YYYY-MM-DD or Month Day, YYYY
                    dates = []
                    
                    # Look for date patterns
                    date_pattern = r'(\d{4})-(\d{2})-(\d{2})'
                    matches = re.findall(date_pattern, html)
                    
                    for match in matches:
                        try:
                            d = date(int(match[0]), int(match[1]), int(match[2]))
                            if d not in dates and d > date.today() - timedelta(days=365):
                                dates.append(d)
                        except:
                            pass
                    
                    if dates:
                        dates.sort()
                        logger.info(f"Fetched {len(dates)} CPI dates from BLS website")
                        return dates
                        
        except Exception as e:
            logger.debug(f"BLS website fetch failed: {e}")
        
        return None
    
    def _get_fallback_fomc_dates(self) -> List[date]:
        """Hardcoded fallback FOMC dates (keep updated!)."""
        return [
            # 2025
            date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
            date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
            date(2025, 11, 5), date(2025, 12, 17),
            # 2026
            date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6),
            date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
            date(2026, 11, 4), date(2026, 12, 16),
        ]
    
    def _get_fallback_cpi_dates(self) -> List[date]:
        """Hardcoded fallback CPI dates (keep updated!)."""
        return [
            # 2025 (typically 2nd week of month)
            date(2025, 1, 15), date(2025, 2, 12), date(2025, 3, 12),
            date(2025, 4, 10), date(2025, 5, 13), date(2025, 6, 11),
            date(2025, 7, 10), date(2025, 8, 13), date(2025, 9, 10),
            date(2025, 10, 10), date(2025, 11, 12), date(2025, 12, 10),
            # 2026
            date(2026, 1, 14), date(2026, 2, 11), date(2026, 3, 11),
        ]
    
    def get_cached_dates(self, event_type: str) -> List[date]:
        """Get dates from cache synchronously (for non-async contexts)."""
        if event_type.upper() == 'FOMC' and 'fomc' in self._cache:
            return [date.fromisoformat(d) for d in self._cache['fomc']]
        elif event_type.upper() == 'CPI' and 'cpi' in self._cache:
            return [date.fromisoformat(d) for d in self._cache['cpi']]
        
        # Return fallback
        if event_type.upper() == 'FOMC':
            return self._get_fallback_fomc_dates()
        else:
            return self._get_fallback_cpi_dates()
    
    async def refresh_calendar(self) -> Dict[str, int]:
        """Refresh all calendar dates from APIs."""
        results = {}
        
        fomc = await self.fetch_fomc_dates()
        results['fomc'] = len(fomc)
        
        cpi = await self.fetch_cpi_dates()
        results['cpi'] = len(cpi)
        
        logger.info(f"Calendar refreshed: {results}")
        return results


# Need timedelta
from datetime import timedelta


# Singleton
_fetcher: Optional[EconomicCalendarFetcher] = None


def get_economic_calendar() -> EconomicCalendarFetcher:
    """Get or create economic calendar fetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = EconomicCalendarFetcher()
    return _fetcher


if __name__ == "__main__":
    import asyncio
    from core.logger import setup_logger
    setup_logger(level="INFO")
    
    async def main():
        cal = get_economic_calendar()
        await cal.refresh_calendar()
        
        print("\nFOMC dates:")
        for d in cal.get_cached_dates('FOMC')[:5]:
            print(f"  {d}")
        
        print("\nCPI dates:")
        for d in cal.get_cached_dates('CPI')[:5]:
            print(f"  {d}")
    
    asyncio.run(main())
