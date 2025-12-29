"""
ThetaData V3 Historical Options Data Downloader

Downloads historical Greeks (all 3 orders) and Open Interest from ThetaData V3 REST API.
Requires Theta Terminal running locally on port 25503.

V3 API Features:
- All Greeks including 2nd order (vanna, charm, vomma) and 3rd order (speed, zomma, color, ultima)
- Wildcard support for expiration and strike (*) 
- NDJSON format for efficient streaming

Setup:
1. Install Theta Terminal from https://thetadata.net
2. Login with your V3 subscription credentials
3. Terminal runs on http://localhost:25503

Usage:
    python -m ml.thetadata_downloader --symbols SPY AAPL --days 365
"""
import asyncio
import aiohttp
import pandas as pd
import json
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Any
from io import StringIO

from core.logger import get_logger
from ml.symbols import TRAINING_SYMBOLS

logger = get_logger()


class ThetaDataDownloaderV3:
    """
    Downloads historical options data from ThetaData V3 API.
    
    Features:
    - All Greeks (1st, 2nd, 3rd order)
    - Historical Open Interest
    - Efficient wildcard queries for all strikes/expirations
    """
    
    BASE_URL = "http://localhost:25503/v3"
    DATA_DIR = Path("data/thetadata")
    
    # Rate limiting
    MAX_CONCURRENT = 5
    REQUEST_DELAY = 0.3  # seconds between requests
    
    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def check_connection(self) -> bool:
        """Check if Theta Terminal V3 is running."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.BASE_URL}/option/list/symbols", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    logger.info("✅ Theta Terminal V3 connected (port 25503)")
                    return True
                else:
                    logger.error(f"Theta Terminal returned status {resp.status}")
                    return False
        except aiohttp.ClientConnectorError:
            logger.error("❌ Cannot connect to Theta Terminal V3")
            logger.error("   Make sure Theta Terminal is running on http://localhost:25503")
            return False
    
    async def get_expirations(self, symbol: str) -> List[str]:
        """Get available expirations for a symbol."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/option/list/expirations?symbol={symbol}"
        
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                
                data = await resp.json()
                # V3 format: {"expirations": ["2024-01-19", "2024-01-26", ...]}
                return data.get("expirations", [])
        except Exception as e:
            logger.warning(f"Failed to get expirations for {symbol}: {e}")
            return []
    
    async def download_all_greeks_for_date(
        self,
        symbol: str,
        date_str: str,
        expiration: str = "*",
        interval: str = "15m"
    ) -> Optional[pd.DataFrame]:
        """
        Download ALL Greeks (1st, 2nd, 3rd order) for a symbol on a specific date.
        
        Uses wildcard (*) for expiration and strike to get all contracts.
        
        Args:
            symbol: Underlying symbol (e.g., "SPY")
            date_str: Date in YYYYMMDD format
            expiration: Specific expiration or "*" for all
            interval: Time interval (1m, 5m, 10m, 15m, 30m, 1h)
            
        Returns:
            DataFrame with all Greeks data
        """
        async with self.semaphore:
            session = await self._get_session()
            
            # V3 endpoint for ALL Greeks (includes 1st, 2nd, 3rd order)
            url = (
                f"{self.BASE_URL}/option/history/greeks/all"
                f"?symbol={symbol}"
                f"&expiration={expiration}"
                f"&strike=*"  # All strikes
                f"&date={date_str}"
                f"&interval={interval}"
                f"&format=ndjson"  # Efficient streaming format
            )
            
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        if "no data" not in text.lower():
                            logger.debug(f"No Greeks data for {symbol} on {date_str}")
                        return None
                    
                    # Parse NDJSON
                    text = await resp.text()
                    rows = []
                    for line in text.strip().split('\n'):
                        if line:
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                    
                    if not rows:
                        return None
                    
                    df = pd.DataFrame(rows)
                    df['date'] = date_str
                    df['symbol'] = symbol
                    
                    return df
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {symbol} Greeks on {date_str}")
                return None
            except Exception as e:
                logger.error(f"Error downloading Greeks: {e}")
                return None
            finally:
                await asyncio.sleep(self.REQUEST_DELAY)
    
    async def download_open_interest_for_date(
        self,
        symbol: str,
        date_str: str,
        expiration: str = "*"
    ) -> Optional[pd.DataFrame]:
        """
        Download Open Interest for all contracts on a specific date.
        
        Args:
            symbol: Underlying symbol
            date_str: Date in YYYYMMDD format
            expiration: Specific expiration or "*" for all
            
        Returns:
            DataFrame with OI data
        """
        async with self.semaphore:
            session = await self._get_session()
            
            url = (
                f"{self.BASE_URL}/option/history/open_interest"
                f"?symbol={symbol}"
                f"&expiration={expiration}"
                f"&strike=*"
                f"&date={date_str}"
                f"&format=ndjson"
            )
            
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status != 200:
                        return None
                    
                    text = await resp.text()
                    rows = []
                    for line in text.strip().split('\n'):
                        if line:
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                    
                    if not rows:
                        return None
                    
                    df = pd.DataFrame(rows)
                    df['date'] = date_str
                    df['symbol_root'] = symbol
                    
                    return df
                    
            except Exception as e:
                logger.debug(f"No OI data for {symbol} on {date_str}")
                return None
            finally:
                await asyncio.sleep(self.REQUEST_DELAY)
    
    async def download_symbol_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        save_daily: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Download full options history for a symbol.
        
        Args:
            symbol: Root symbol (e.g., "SPY")
            start_date: Start date
            end_date: End date
            save_daily: Save each day separately (for large datasets)
            
        Returns:
            Dict with "greeks" and "oi" DataFrames
        """
        logger.info(f"📊 Downloading {symbol} from {start_date} to {end_date}...")
        
        # Generate trading days (skip weekends)
        trading_days = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Monday=0, Friday=4
                trading_days.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        
        logger.info(f"   {len(trading_days)} trading days to process")
        
        all_greeks = []
        all_oi = []
        
        for i, day in enumerate(trading_days):
            if (i + 1) % 20 == 0:
                logger.info(f"   Progress: {i+1}/{len(trading_days)} days...")
            
            # Download Greeks for this day
            greeks_df = await self.download_all_greeks_for_date(symbol, day)
            if greeks_df is not None and len(greeks_df) > 0:
                all_greeks.append(greeks_df)
            
            # Download OI for this day
            oi_df = await self.download_open_interest_for_date(symbol, day)
            if oi_df is not None and len(oi_df) > 0:
                all_oi.append(oi_df)
        
        result = {}
        
        # Combine and save Greeks
        if all_greeks:
            greeks_combined = pd.concat(all_greeks, ignore_index=True)
            result["greeks"] = greeks_combined
            
            path = self.DATA_DIR / f"{symbol}_greeks_all.parquet"
            greeks_combined.to_parquet(path, index=False, compression='snappy')
            logger.info(f"   ✅ Saved {len(greeks_combined):,} Greeks rows to {path.name}")
        else:
            logger.warning(f"   ⚠️ No Greeks data for {symbol}")
        
        # Combine and save OI
        if all_oi:
            oi_combined = pd.concat(all_oi, ignore_index=True)
            result["oi"] = oi_combined
            
            path = self.DATA_DIR / f"{symbol}_oi.parquet"
            oi_combined.to_parquet(path, index=False, compression='snappy')
            logger.info(f"   ✅ Saved {len(oi_combined):,} OI rows to {path.name}")
        else:
            logger.warning(f"   ⚠️ No OI data for {symbol}")
        
        return result
    
    async def download_all_symbols(
        self,
        symbols: List[str] = None,
        days_back: int = 365
    ) -> Dict[str, Dict]:
        """Download history for all training symbols."""
        symbols = symbols or TRAINING_SYMBOLS
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info("=" * 60)
        logger.info(f"📊 ThetaData V3 Bulk Download")
        logger.info(f"   Symbols: {len(symbols)} - {symbols}")
        logger.info(f"   Period: {start_date} to {end_date} ({days_back} days)")
        logger.info("=" * 60)
        
        # Check connection first
        if not await self.check_connection():
            logger.error("Cannot proceed without Theta Terminal")
            return {}
        
        results = {}
        
        for idx, symbol in enumerate(symbols):
            logger.info(f"\n[{idx+1}/{len(symbols)}] Processing {symbol}...")
            
            try:
                result = await self.download_symbol_history(symbol, start_date, end_date)
                results[symbol] = {
                    "greeks_count": len(result.get("greeks", [])) if "greeks" in result else 0,
                    "oi_count": len(result.get("oi", [])) if "oi" in result else 0,
                }
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        
        await self.close()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📋 Download Summary")
        logger.info("=" * 60)
        
        total_greeks = 0
        total_oi = 0
        
        for symbol, data in results.items():
            if "error" in data:
                logger.warning(f"   ❌ {symbol}: {data['error']}")
            else:
                greeks = data.get("greeks_count", 0)
                oi = data.get("oi_count", 0)
                total_greeks += greeks
                total_oi += oi
                logger.info(f"   ✅ {symbol}: {greeks:,} Greeks, {oi:,} OI")
        
        logger.info("-" * 60)
        logger.info(f"   TOTAL: {total_greeks:,} Greeks, {total_oi:,} OI records")
        logger.info(f"   Saved to: {self.DATA_DIR}")
        logger.info("=" * 60)
        
        return results
    
    def list_downloaded_files(self) -> List[Path]:
        """List all downloaded parquet files."""
        return list(self.DATA_DIR.glob("*.parquet"))


async def download_thetadata_v3(symbols: List[str] = None, days: int = 365):
    """Convenience function for downloading."""
    downloader = ThetaDataDownloaderV3()
    return await downloader.download_all_symbols(symbols, days)


# CLI
if __name__ == "__main__":
    import argparse
    from core.logger import setup_logger
    
    parser = argparse.ArgumentParser(description="Download ThetaData V3 historical options")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to download (default: all training symbols)")
    parser.add_argument("--days", type=int, default=365, help="Days of history (default: 365)")
    
    args = parser.parse_args()
    
    setup_logger(level="INFO")
    
    print("=" * 60)
    print("📊 ThetaData V3 Downloader")
    print("=" * 60)
    print(f"Symbols: {args.symbols or 'ALL TRAINING SYMBOLS'}")
    print(f"Days: {args.days}")
    print("=" * 60)
    
    asyncio.run(download_thetadata_v3(args.symbols, args.days))
