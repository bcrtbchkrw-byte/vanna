#!/usr/bin/env python3
"""
rates_provider.py

POSKYTOVATEL SAZEB - Žádný hardcode!

SOFR (r): Federal Reserve Bank of New York API
Dividend Yield (q): 
    - Primárně: ThetaData API (pokud běží)
    - Fallback: Yahoo Finance (zdarma, vždy dostupné)
    - Cache: Vše uloženo na disk

POUŽITÍ:
    from rates_provider import RatesProvider
    
    provider = RatesProvider()
    provider.prefetch_all(['SPY', 'QQQ', 'AAPL'])  # Stáhne vše
    
    r = provider.get_sofr('2023-06-15')
    q = provider.get_dividend_yield('SPY', '2023-06-15', stock_price=450.0)
"""

import requests
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
import json
from typing import Optional, Union, Dict, List
from io import StringIO

# Try to import yfinance (optional)
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class RatesProvider:
    """Provider for SOFR and dividend yields - no hardcoding."""
    
    SOFR_API = "https://markets.newyorkfed.org/api/rates/secured/sofr/last/{days}.json"
    
    def __init__(self, cache_dir: str = "rates_cache",
                 thetadata_host: str = "localhost", 
                 thetadata_port: int = 25503):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.thetadata_url = f"http://{thetadata_host}:{thetadata_port}/v3"
        
        self._sofr: Dict[str, float] = {}
        self._dividends: Dict[str, Dict[str, float]] = {}
        self._prices: Dict[str, Dict[str, float]] = {}
        
        self._load_cache()
    
    def _load_cache(self):
        for name, attr in [('sofr', '_sofr'), ('dividends', '_dividends'), ('prices', '_prices')]:
            path = self.cache_dir / f"{name}.json"
            if path.exists():
                with open(path) as f:
                    setattr(self, attr, json.load(f))
    
    def _save_cache(self):
        for name, attr in [('sofr', '_sofr'), ('dividends', '_dividends'), ('prices', '_prices')]:
            with open(self.cache_dir / f"{name}.json", 'w') as f:
                json.dump(getattr(self, attr), f)
    
    # === SOFR ===
    
    def fetch_sofr_history(self, days: int = 1000) -> int:
        """Fetch SOFR from NY Fed. Returns count of rates fetched."""
        print(f"Fetching SOFR from NY Fed...")
        
        # NY Fed API has limit, try different approach - fetch all available
        # Using the "all" endpoint instead of "last/N"
        url = "https://markets.newyorkfed.org/api/rates/secured/sofr/search.json"
        
        try:
            # Try to get all data with date range
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
            
            params = {
                'startDate': start_date,
                'endDate': end_date,
                'type': 'rate'
            }
            
            resp = requests.get(url, params=params, timeout=60)
            
            if resp.status_code != 200:
                # Fallback to smaller batch
                print(f"  Search endpoint failed ({resp.status_code}), trying last/1000...")
                return self._fetch_sofr_batch(1000)
            
            data = resp.json()
            
            if 'refRates' not in data:
                print("  Invalid response, trying batch method...")
                return self._fetch_sofr_batch(1000)
            
            for item in data['refRates']:
                dt = item['effectiveDate']
                rate = float(item['percentRate']) / 100
                self._sofr[dt] = rate
            
            self._save_cache()
            
            if self._sofr:
                dates = sorted(self._sofr.keys())
                print(f"  ✓ {len(self._sofr)} rates: {dates[0]} to {dates[-1]}")
                print(f"  ✓ Latest: {self._sofr[dates[-1]]:.4f} ({self._sofr[dates[-1]]*100:.2f}%)")
            
            return len(self._sofr)
            
        except Exception as e:
            print(f"  Error: {e}, trying batch method...")
            return self._fetch_sofr_batch(1000)
    
    def _fetch_sofr_batch(self, days: int) -> int:
        """Fetch SOFR in smaller batches."""
        url = f"https://markets.newyorkfed.org/api/rates/secured/sofr/last/{days}.json"
        
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        
        data = resp.json()
        if 'refRates' not in data:
            raise ValueError("Invalid NY Fed response")
        
        for item in data['refRates']:
            dt = item['effectiveDate']
            rate = float(item['percentRate']) / 100
            self._sofr[dt] = rate
        
        self._save_cache()
        
        dates = sorted(self._sofr.keys())
        print(f"  ✓ {len(self._sofr)} rates: {dates[0]} to {dates[-1]}")
        print(f"  ✓ Latest: {self._sofr[dates[-1]]:.4f} ({self._sofr[dates[-1]]*100:.2f}%)")
        
        return len(self._sofr)
    
    def get_sofr(self, dt: Optional[Union[str, date, datetime]] = None) -> float:
        """Get SOFR for date. Uses closest previous if exact not available."""
        if not self._sofr:
            raise ValueError("No SOFR data. Call fetch_sofr_history() first.")
        
        target = self._normalize_date(dt)
        
        if target in self._sofr:
            return self._sofr[target]
        
        # Find closest previous
        dates = sorted(self._sofr.keys())
        closest = dates[0]
        for d in dates:
            if d <= target:
                closest = d
            else:
                break
        
        return self._sofr[closest]
    
    # === DIVIDENDS ===
    
    def fetch_dividend_history(self, symbol: str, years: int = 10) -> int:
        """
        Fetch dividend history. Tries ThetaData first, then Yahoo Finance.
        Returns count of records fetched.
        """
        # Try ThetaData first
        count = self._fetch_dividends_thetadata(symbol, years)
        
        if count > 0:
            return count
        
        # Fallback to Yahoo Finance
        return self._fetch_dividends_yahoo(symbol, years)
    
    def _fetch_dividends_thetadata(self, symbol: str, years: int) -> int:
        """Fetch dividends from ThetaData."""
        print(f"Fetching dividends for {symbol} from ThetaData...")
        
        start = (datetime.now() - timedelta(days=years*365)).strftime('%Y%m%d')
        end = datetime.now().strftime('%Y%m%d')
        
        try:
            resp = requests.get(
                f"{self.thetadata_url}/stock/history/dividend",
                params={'symbol': symbol, 'start_date': start, 'end_date': end},
                timeout=30
            )
            
            if resp.status_code != 200:
                print(f"  ⚠ ThetaData error: {resp.status_code}")
                return 0
            
            df = pd.read_csv(StringIO(resp.text))
            if len(df) == 0:
                print(f"  No dividends from ThetaData")
                return 0
            
            # Normalize columns
            date_col = 'ex_date' if 'ex_date' in df.columns else 'date'
            amt_col = 'amount' if 'amount' in df.columns else 'dividend'
            
            if symbol not in self._dividends:
                self._dividends[symbol] = {}
            
            for _, row in df.iterrows():
                dt = pd.to_datetime(row[date_col]).strftime('%Y-%m-%d')
                self._dividends[symbol][dt] = float(row[amt_col])
            
            self._save_cache()
            print(f"  ✓ {len(self._dividends[symbol])} dividend records (ThetaData)")
            
            return len(self._dividends[symbol])
            
        except Exception as e:
            print(f"  ThetaData failed: {e}")
            return 0
    
    def _fetch_dividends_yahoo(self, symbol: str, years: int) -> int:
        """Fetch dividends from Yahoo Finance (free, no API key)."""
        print(f"Fetching dividends for {symbol} from Yahoo Finance...")
        
        if not HAS_YFINANCE:
            print("  ⚠ yfinance not installed. Run: pip install yfinance")
            return 0
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get dividend history
            divs = ticker.dividends
            
            if divs is None or len(divs) == 0:
                print(f"  No dividends found (stock may not pay dividends)")
                return 0
            
            # Filter to requested years - fix timezone issue
            start_date = datetime.now() - timedelta(days=years*365)
            
            # Convert index to timezone-naive for comparison
            if divs.index.tz is not None:
                divs_naive = divs.copy()
                divs_naive.index = divs_naive.index.tz_localize(None)
            else:
                divs_naive = divs
            
            divs_filtered = divs_naive[divs_naive.index >= start_date]
            
            if symbol not in self._dividends:
                self._dividends[symbol] = {}
            
            for dt, amount in divs_filtered.items():
                date_str = dt.strftime('%Y-%m-%d')
                self._dividends[symbol][date_str] = float(amount)
            
            self._save_cache()
            print(f"  ✓ {len(self._dividends[symbol])} dividend records (Yahoo)")
            
            return len(self._dividends[symbol])
            
        except Exception as e:
            print(f"  Yahoo Finance failed: {e}")
            return 0
    
    def fetch_eod_prices(self, symbol: str, years: int = 10) -> int:
        """
        Fetch EOD prices. Tries ThetaData first, then Yahoo Finance.
        Returns count of records.
        """
        # Try ThetaData first
        count = self._fetch_eod_thetadata(symbol, years)
        
        if count > 0:
            return count
        
        # Fallback to Yahoo Finance
        return self._fetch_eod_yahoo(symbol, years)
    
    def _fetch_eod_thetadata(self, symbol: str, years: int) -> int:
        """Fetch EOD prices from ThetaData."""
        print(f"Fetching EOD prices for {symbol} from ThetaData...")
        
        start = (datetime.now() - timedelta(days=years*365)).strftime('%Y%m%d')
        end = datetime.now().strftime('%Y%m%d')
        
        try:
            resp = requests.get(
                f"{self.thetadata_url}/stock/history/eod",
                params={'symbol': symbol, 'start_date': start, 'end_date': end},
                timeout=60
            )
            
            if resp.status_code != 200:
                print(f"  ⚠ ThetaData error: {resp.status_code}")
                return 0
            
            df = pd.read_csv(StringIO(resp.text))
            if len(df) == 0:
                print(f"  No EOD data from ThetaData")
                return 0
            
            date_col = 'date' if 'date' in df.columns else df.columns[0]
            price_col = 'close' if 'close' in df.columns else 'last'
            
            if symbol not in self._prices:
                self._prices[symbol] = {}
            
            for _, row in df.iterrows():
                dt = pd.to_datetime(row[date_col]).strftime('%Y-%m-%d')
                self._prices[symbol][dt] = float(row[price_col])
            
            self._save_cache()
            print(f"  ✓ {len(self._prices[symbol])} EOD prices (ThetaData)")
            
            return len(self._prices[symbol])
            
        except Exception as e:
            print(f"  ThetaData failed: {e}")
            return 0
    
    def _fetch_eod_yahoo(self, symbol: str, years: int) -> int:
        """Fetch EOD prices from Yahoo Finance (free)."""
        print(f"Fetching EOD prices for {symbol} from Yahoo Finance...")
        
        if not HAS_YFINANCE:
            print("  ⚠ yfinance not installed. Run: pip install yfinance")
            return 0
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            start_date = datetime.now() - timedelta(days=years*365)
            hist = ticker.history(start=start_date, end=datetime.now())
            
            if hist is None or len(hist) == 0:
                print(f"  No price data from Yahoo")
                return 0
            
            if symbol not in self._prices:
                self._prices[symbol] = {}
            
            for dt, row in hist.iterrows():
                date_str = dt.strftime('%Y-%m-%d')
                self._prices[symbol][date_str] = float(row['Close'])
            
            self._save_cache()
            print(f"  ✓ {len(self._prices[symbol])} EOD prices (Yahoo)")
            
            return len(self._prices[symbol])
            
        except Exception as e:
            print(f"  Yahoo Finance failed: {e}")
            return 0
    
    def get_dividend_yield(self, symbol: str,
                           dt: Optional[Union[str, date, datetime]] = None,
                           stock_price: Optional[float] = None) -> float:
        """Get dividend yield = TTM dividends / price."""
        target = self._normalize_date(dt)
        target_date = datetime.strptime(target, '%Y-%m-%d').date()
        
        # TTM dividends
        if symbol not in self._dividends:
            return 0.0
        
        one_year_ago = (target_date - timedelta(days=365)).strftime('%Y-%m-%d')
        ttm = sum(amt for dt, amt in self._dividends[symbol].items() 
                  if one_year_ago < dt <= target)
        
        if ttm == 0:
            return 0.0
        
        # Get price
        if stock_price is None:
            stock_price = self._get_price(symbol, target)
        
        if stock_price is None or stock_price <= 0:
            return 0.0
        
        return ttm / stock_price
    
    def _get_price(self, symbol: str, target: str) -> Optional[float]:
        """Get price for date from cache."""
        if symbol not in self._prices:
            return None
        
        if target in self._prices[symbol]:
            return self._prices[symbol][target]
        
        # Find closest previous
        dates = sorted(self._prices[symbol].keys())
        closest = None
        for d in dates:
            if d <= target:
                closest = d
        
        return self._prices[symbol].get(closest)
    
    # === COMBINED ===
    
    def get_rates(self, symbol: str,
                  dt: Optional[Union[str, date, datetime]] = None,
                  stock_price: Optional[float] = None) -> Dict[str, float]:
        """Get both r (SOFR) and q (dividend yield)."""
        return {
            'r': self.get_sofr(dt),
            'q': self.get_dividend_yield(symbol, dt, stock_price)
        }
    
    def prefetch_all(self, symbols: List[str], years: int = 10):
        """Fetch all data for symbols."""
        print("="*60)
        print("PREFETCHING ALL RATES DATA")
        print("="*60)
        
        self.fetch_sofr_history()
        
        for symbol in symbols:
            print()
            self.fetch_dividend_history(symbol, years)
            self.fetch_eod_prices(symbol, years)
        
        print("\n✅ Prefetch complete!")
    
    def _normalize_date(self, dt) -> str:
        """Convert date to YYYY-MM-DD string."""
        if dt is None:
            return date.today().strftime('%Y-%m-%d')
        if isinstance(dt, str):
            return dt
        if isinstance(dt, datetime):
            return dt.strftime('%Y-%m-%d')
        return dt.strftime('%Y-%m-%d')
    
    def print_status(self):
        """Print cache status."""
        print("\n" + "="*50)
        print("RATES CACHE STATUS")
        print("="*50)
        
        if self._sofr:
            dates = sorted(self._sofr.keys())
            print(f"\nSOFR: {len(self._sofr)} rates")
            print(f"  Range: {dates[0]} to {dates[-1]}")
            print(f"  Latest: {self._sofr[dates[-1]]*100:.2f}%")
        else:
            print("\nSOFR: No data")
        
        print(f"\nDividends: {list(self._dividends.keys())}")
        for sym, divs in self._dividends.items():
            print(f"  {sym}: {len(divs)} records")
        
        print(f"\nPrices: {list(self._prices.keys())}")
        for sym, prices in self._prices.items():
            print(f"  {sym}: {len(prices)} days")


if __name__ == "__main__":
    print("Testing RatesProvider...")
    print(f"yfinance available: {HAS_YFINANCE}")
    
    provider = RatesProvider()
    
    # Test SOFR (works without ThetaData)
    try:
        provider.fetch_sofr_history(100)
        print(f"\nToday's SOFR: {provider.get_sofr():.4f}")
        print(f"SOFR 2023-06-15: {provider.get_sofr('2023-06-15'):.4f}")
    except Exception as e:
        print(f"SOFR test failed: {e}")
    
    # Test dividends (ThetaData -> Yahoo fallback)
    try:
        provider.fetch_dividend_history('SPY')
        provider.fetch_eod_prices('SPY')
        q = provider.get_dividend_yield('SPY')
        print(f"\nSPY dividend yield: {q:.4f} ({q*100:.2f}%)")
    except Exception as e:
        print(f"Dividend test failed: {e}")
    
    provider.print_status()
    
    print("\n" + "="*50)
    print("DATA SOURCE PRIORITY:")
    print("  1. ThetaData (if Terminal running)")
    print("  2. Yahoo Finance (free fallback)")
    print("  3. Cached data (always available)")
    print("="*50)