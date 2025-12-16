"""
Data Maintenance Script

Automated maintenance for ML training data:
- Checks for gaps in historical data
- Patches missing data from IBKR
- Runs monthly on 1st day of month

Usage:
    python -m automation.data_maintenance
"""
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

from core.logger import get_logger, setup_logger
from ml.vanna_data_pipeline import get_vanna_pipeline
from ml.data_storage import get_data_storage

logger = get_logger()


class DataMaintenanceManager:
    """
    Manages data integrity and gap patching.
    
    Features:
    - Detects gaps in timestamp sequences (> 1 minute)
    - Downloads missing data from IBKR
    - Validates and merges patched data
    - Global lock to prevent concurrent downloads
    """
    
    # Single Source of Truth: import from ml.symbols
    from ml.symbols import TRAINING_SYMBOLS as SYMBOLS
    MAX_GAP_MINUTES = 1  # Gaps larger than this are flagged
    
    # Global lock to prevent double downloads
    _download_lock: Optional[asyncio.Lock] = None
    _downloading: bool = False
    
    def __init__(self):
        self.pipeline = get_vanna_pipeline()
        self.storage = get_data_storage()
        
        # Initialize lock if not exists
        if DataMaintenanceManager._download_lock is None:
            DataMaintenanceManager._download_lock = asyncio.Lock()
        
        logger.info("DataMaintenanceManager initialized")
    
    async def check_historical_data_exists(self) -> Dict[str, bool]:
        """
        Check if historical data exists for all symbols.
        
        Returns:
            Dict of symbol -> has_data (True only if BOTH 1min and 1day exist)
        """
        results = {}
        
        for symbol in self.SYMBOLS:
            # Check BOTH 1min and 1day parquet files
            df_1min = self.storage.load_historical_parquet(symbol, '1min')
            df_1day = self.storage.load_historical_parquet(symbol, '1day')
            
            has_1min = df_1min is not None and len(df_1min) > 1000
            has_1day = df_1day is not None and len(df_1day) > 100
            
            if has_1min and has_1day:
                results[symbol] = True
                logger.info(f"✅ {symbol}: {len(df_1min):,} 1min bars, {len(df_1day):,} daily bars")
            else:
                results[symbol] = False
                missing = []
                if not has_1min:
                    missing.append("1min")
                if not has_1day:
                    missing.append("1day")
                logger.warning(f"❌ {symbol}: Missing {', '.join(missing)} data")
        
        return results
    
    async def ensure_historical_data(self) -> bool:
        """
        Ensure historical data exists, download if missing.
        
        Called on startup - checks and downloads missing data.
        Uses lock to prevent concurrent downloads.
        
        Returns:
            True if all data is available
        """
        # Check if already downloading (non-blocking check)
        if DataMaintenanceManager._downloading:
            logger.info("⏸️ Download already in progress, skipping...")
            return True
        
        # Acquire lock for downloading
        async with DataMaintenanceManager._download_lock:
            # Double check after acquiring lock
            if DataMaintenanceManager._downloading:
                logger.info("⏸️ Download already in progress, skipping...")
                return True
            
            DataMaintenanceManager._downloading = True
            
            try:
                logger.info("=" * 60)
                logger.info("🔍 Checking historical data...")
                logger.info("=" * 60)
                
                existing = await self.check_historical_data_exists()
                missing_symbols = [s for s, exists in existing.items() if not exists]
                
                if not missing_symbols:
                    logger.info("✅ All historical data present")
                    return True
                
                logger.info(f"📥 Downloading missing data for: {', '.join(missing_symbols)}")
                logger.info("⚠️ This will take a while with 20s delay per request...")
                
                for symbol in missing_symbols:
                    try:
                        # Download 550 days of 1-min data
                        await self.pipeline.process_historical_data(symbol, days=550, save=True)
                        
                        # Download 10 years of daily data (with VIX)
                        daily_df = await self.pipeline.fetch_historical_daily(symbol, years=10)
                        if daily_df is not None:
                            # Add VIX data for regime classification
                            daily_df = await self.pipeline.fetch_vix_daily(daily_df)
                            daily_df = self.pipeline.feature_eng.process_all_features(daily_df)
                            self.storage.save_historical_parquet(daily_df, symbol, '1day')
                        
                        logger.info(f"✅ Downloaded historical data for {symbol}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to download {symbol}: {e}")
                
                return True
                
            finally:
                DataMaintenanceManager._downloading = False
    
    def find_gaps(
        self,
        df: pd.DataFrame,
        max_gap_minutes: int = 1
    ) -> List[Dict]:
        """
        Find timestamp gaps in data.
        
        Args:
            df: DataFrame with 'timestamp' column
            max_gap_minutes: Maximum allowed gap (default 1 min)
            
        Returns:
            List of gap info dicts with start, end, gap_minutes
        """
        if df is None or len(df) < 2:
            return []
        
        # Ensure sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        gaps = []
        
        for i in range(1, len(df)):
            prev_ts = pd.to_datetime(df.loc[i-1, 'timestamp'])
            curr_ts = pd.to_datetime(df.loc[i, 'timestamp'])
            
            gap_minutes = (curr_ts - prev_ts).total_seconds() / 60
            
            # During market hours (9:30 - 16:00), gaps > 1 min are suspicious
            # Overnight gaps are expected
            if gap_minutes > max_gap_minutes:
                # Check if it's an overnight gap (market closed)
                is_overnight = (
                    prev_ts.hour >= 16 or  # After market close
                    curr_ts.hour < 9 or    # Before market open
                    (prev_ts.hour >= 16 and curr_ts.date() > prev_ts.date())  # Next day
                )
                
                # Weekend gap
                is_weekend = (
                    prev_ts.weekday() == 4 and curr_ts.weekday() == 0  # Fri -> Mon
                )
                
                if not is_overnight and not is_weekend:
                    gaps.append({
                        'start': prev_ts,
                        'end': curr_ts,
                        'gap_minutes': gap_minutes,
                        'index_start': i - 1,
                        'index_end': i
                    })
        
        return gaps
    
    def find_daily_gaps(self, df: pd.DataFrame, max_gap_days: int = 3) -> List[Dict]:
        """
        Find gaps in daily data.
        
        Args:
            df: DataFrame with 'timestamp' column (daily data)
            max_gap_days: Maximum allowed gap (default 3 days for weekends)
            
        Returns:
            List of gap info dicts with start, end, gap_days
        """
        if df is None or len(df) < 2:
            return []
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        gaps = []
        
        for i in range(1, len(df)):
            prev_ts = pd.to_datetime(df.loc[i-1, 'timestamp'])
            curr_ts = pd.to_datetime(df.loc[i, 'timestamp'])
            
            gap_days = (curr_ts - prev_ts).days
            
            # More than 3 days gap is suspicious (weekends = 2-3 days)
            if gap_days > max_gap_days:
                gaps.append({
                    'start': prev_ts,
                    'end': curr_ts,
                    'gap_days': gap_days
                })
        
        return gaps
    
    async def update_daily_data(self, symbol: str) -> bool:
        """
        Update daily data for a symbol by fetching latest bars.
        
        Args:
            symbol: Symbol to update
            
        Returns:
            True if successful
        """
        try:
            # Load existing daily data
            existing_df = self.storage.load_historical_parquet(symbol, '1day')
            
            if existing_df is None or len(existing_df) == 0:
                # No existing data, download full history
                logger.info(f"   Downloading full 10-year daily data for {symbol}...")
                daily_df = await self.pipeline.fetch_historical_daily(symbol, years=10)
                if daily_df is not None:
                    daily_df = await self.pipeline.fetch_vix_daily(daily_df)
                    daily_df = self.pipeline.feature_eng.process_all_features(daily_df)
                    self.storage.save_historical_parquet(daily_df, symbol, '1day')
                    return True
                return False
            
            # Find last date in existing data
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
            last_date = existing_df['timestamp'].max()
            
            # Calculate days to fetch
            from datetime import datetime, timedelta
            days_missing = (datetime.now() - last_date).days
            
            if days_missing <= 1:
                logger.info(f"   ✅ {symbol} daily data is up to date")
                return True
            
            logger.info(f"   Fetching {days_missing} days of daily data for {symbol}...")
            
            # Fetch recent daily data (last 30 days to be safe)
            new_df = await self.pipeline.fetch_historical_daily(symbol, years=1)
            
            if new_df is None or len(new_df) == 0:
                return False
            
            # Merge with existing
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
            merged_df = pd.concat([existing_df, new_df], ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset=['timestamp'], keep='last')
            merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
            
            # Re-apply VIX and features
            merged_df = await self.pipeline.fetch_vix_daily(merged_df)
            merged_df = self.pipeline.feature_eng.process_all_features(merged_df)
            
            # Save
            self.storage.save_historical_parquet(merged_df, symbol, '1day')
            logger.info(f"   ✅ Updated {symbol} daily data: {len(merged_df):,} bars")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Failed to update daily data for {symbol}: {e}")
            return False
    
    async def patch_gaps(
        self,
        symbol: str,
        gaps: List[Dict]
    ) -> int:
        """
        Patch data gaps by downloading from IBKR.
        
        Args:
            symbol: Symbol to patch
            gaps: List of gap info dicts
            
        Returns:
            Number of gaps patched
        """
        if not gaps:
            return 0
        
        patched = 0
        
        for gap in gaps:
            try:
                start = gap['start']
                end = gap['end']
                
                logger.info(f"🔧 Patching {symbol} gap: {start} to {end} ({gap['gap_minutes']:.0f} min)")
                
                # Calculate days to fetch
                gap_days = (end - start).days + 1
                
                # Fetch missing data
                conn = await self.pipeline._get_connection()
                if not conn or not conn.is_connected:
                    logger.error("IBKR not connected, cannot patch")
                    continue
                
                ib = conn.ib
                
                from ib_insync import Stock
                contract = Stock(symbol, 'SMART', 'USD')
                await ib.qualifyContractsAsync(contract)
                
                # Fetch bars for the gap period
                bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end + timedelta(hours=1),
                    durationStr=f"{max(1, gap_days)} D",
                    barSizeSetting='1 min',
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1
                )
                
                if bars:
                    # Filter to only gap period
                    patch_df = pd.DataFrame([{
                        'timestamp': bar.date,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                        'symbol': symbol
                    } for bar in bars])
                    
                    patch_df['timestamp'] = pd.to_datetime(patch_df['timestamp'])
                    patch_df = patch_df[
                        (patch_df['timestamp'] > start) & 
                        (patch_df['timestamp'] < end)
                    ]
                    
                    if len(patch_df) > 0:
                        # Merge with existing data
                        existing_df = self.storage.load_historical_parquet(symbol, '1min')
                        if existing_df is not None:
                            merged = pd.concat([existing_df, patch_df], ignore_index=True)
                            merged = merged.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                            
                            # Re-apply features
                            merged = self.pipeline.feature_eng.process_all_features(merged)
                            
                            self.storage.save_historical_parquet(merged, symbol, '1min')
                            patched += 1
                            
                            logger.info(f"✅ Patched {len(patch_df)} bars for {symbol}")
                
                await asyncio.sleep(3)  # Rate limiting (IBKR pacing)
                
            except Exception as e:
                logger.error(f"Error patching gap for {symbol}: {e}")
        
        return patched
    
    async def run_maintenance(self) -> Dict[str, any]:
        """
        Run full data maintenance.
        
        Checks all symbols for gaps in 1min AND 1day data, patches them.
        
        Returns:
            Summary of maintenance results
        """
        logger.info("=" * 60)
        logger.info("🔧 Starting Data Maintenance")
        logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info("=" * 60)
        
        results = {
            'symbols_checked': 0,
            'total_gaps_found': 0,
            'gaps_patched': 0,
            'daily_updated': 0,
            'errors': []
        }
        
        for symbol in self.SYMBOLS:
            try:
                logger.info(f"\n📊 Checking {symbol}...")
                results['symbols_checked'] += 1
                
                # ============================================
                # 1. Check 1-MINUTE data
                # ============================================
                df_1min = self.storage.load_historical_parquet(symbol, '1min')
                
                if df_1min is None or len(df_1min) == 0:
                    logger.warning(f"   No 1min data for {symbol}, downloading full history...")
                    await self.pipeline.process_historical_data(symbol, days=550, save=True)
                else:
                    # Find gaps in 1min
                    gaps = self.find_gaps(df_1min)
                    results['total_gaps_found'] += len(gaps)
                    
                    if gaps:
                        logger.warning(f"   Found {len(gaps)} gaps in 1min data")
                        patched = await self.patch_gaps(symbol, gaps)
                        results['gaps_patched'] += patched
                    else:
                        logger.info(f"   ✅ 1min: No gaps ({len(df_1min):,} bars)")
                
                # ============================================
                # 2. Check DAILY data
                # ============================================
                df_1day = self.storage.load_historical_parquet(symbol, '1day')
                
                if df_1day is None or len(df_1day) == 0:
                    logger.warning(f"   No 1day data for {symbol}, downloading...")
                    if await self.update_daily_data(symbol):
                        results['daily_updated'] += 1
                else:
                    # Check if daily data is up to date
                    df_1day['timestamp'] = pd.to_datetime(df_1day['timestamp'])
                    last_date = df_1day['timestamp'].max()
                    days_old = (datetime.now() - last_date).days
                    
                    if days_old > 1:
                        logger.info(f"   Daily data is {days_old} days old, updating...")
                        if await self.update_daily_data(symbol):
                            results['daily_updated'] += 1
                    else:
                        logger.info(f"   ✅ 1day: Up to date ({len(df_1day):,} bars)")
                
            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")
                results['errors'].append(f"{symbol}: {str(e)}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📋 Maintenance Summary")
        logger.info("=" * 60)
        logger.info(f"   Symbols checked: {results['symbols_checked']}")
        logger.info(f"   1min gaps found: {results['total_gaps_found']}")
        logger.info(f"   1min gaps patched: {results['gaps_patched']}")
        logger.info(f"   1day data updated: {results['daily_updated']}")
        if results['errors']:
            logger.warning(f"   Errors: {len(results['errors'])}")
        logger.info("=" * 60)
        
        return results
    
    async def merge_live_to_parquet(self) -> Dict[str, int]:
        """
        Merge live data from SQLite live_bars into parquet files.
        
        Steps:
        1. Load new data from live_bars table (SQLite)
        2. Load existing data from *_1min.parquet
        3. Merge and deduplicate by timestamp
        4. Save updated *_1min.parquet
        5. Recalculate Greeks for *_1min_vanna.parquet
        
        Returns:
            Dict with merge statistics per symbol
        """
        import sqlite3
        from ml.vectorized_greeks import VectorizedGreeksCalculator
        
        logger.info("=" * 60)
        logger.info("🔄 Merging Live Data to Parquet")
        logger.info("=" * 60)
        
        results = {}
        db_path = self.storage.db_path
        
        for symbol in self.SYMBOLS:
            try:
                # 1. Load live data from SQLite
                with sqlite3.connect(db_path) as conn:
                    query = """
                        SELECT timestamp, symbol, open, high, low, close, volume,
                               vix, vix3m, vix_ratio, regime, sin_time, cos_time,
                               options_iv_atm, options_volume, options_put_call_ratio
                        FROM live_bars 
                        WHERE symbol = ? AND timeframe = '1min'
                        ORDER BY timestamp
                    """
                    live_df = pd.read_sql_query(query, conn, params=[symbol])
                
                if live_df.empty:
                    logger.info(f"⏭️ {symbol}: No live data to merge")
                    results[symbol] = 0
                    continue
                
                live_df['timestamp'] = pd.to_datetime(live_df['timestamp'])
                logger.info(f"📊 {symbol}: Found {len(live_df):,} live bars")
                
                # 2. Load existing parquet
                parquet_path = self.storage.data_dir / f"{symbol}_1min.parquet"
                
                if parquet_path.exists():
                    existing_df = pd.read_parquet(parquet_path)
                    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                    logger.info(f"   Existing: {len(existing_df):,} bars")
                else:
                    existing_df = pd.DataFrame()
                    logger.info(f"   No existing parquet, creating new")
                
                # 3. Merge and deduplicate
                if not existing_df.empty:
                    # Get union of all columns (preserve all data)
                    all_cols = list(set(existing_df.columns) | set(live_df.columns))
                    
                    # Add missing columns with NaN
                    for col in all_cols:
                        if col not in existing_df.columns:
                            existing_df[col] = np.nan
                        if col not in live_df.columns:
                            live_df[col] = np.nan
                    
                    merged_df = pd.concat([existing_df, live_df], ignore_index=True)
                else:
                    merged_df = live_df
                
                # Remove duplicates by timestamp and LOG count
                pre_dedup_count = len(merged_df)
                merged_df = merged_df.drop_duplicates(subset=['timestamp'], keep='last')
                duplicates_removed = pre_dedup_count - len(merged_df)
                if duplicates_removed > 0:
                    logger.info(f"   🗑️ Removed {duplicates_removed} duplicate timestamps")
                
                merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
                
                # Re-apply feature engineering to fill any missing options data
                # This ensures options_iv_atm, options_put_call_ratio, options_volume_norm
                # are estimated from VIX if they're NaN
                if 'options_iv_atm' in merged_df.columns and merged_df['options_iv_atm'].isnull().any():
                    merged_df = self.pipeline.feature_eng.add_options_features(merged_df)
                    logger.info(f"   📊 Re-estimated options features from VIX")
                
                # Validate timestamp continuity (smoke test)
                if len(merged_df) > 1:
                    gaps = self.find_gaps(merged_df, max_gap_minutes=5)  # 5 min tolerance
                    if gaps:
                        logger.warning(f"   ⚠️ Found {len(gaps)} gaps in merged data (>5min)")
                
                new_bars = len(merged_df) - len(existing_df) if not existing_df.empty else len(merged_df)
                logger.info(f"   After merge: {len(merged_df):,} bars (+{new_bars} new)")
                
                # 4. Save updated parquet
                merged_df.to_parquet(parquet_path, index=False, compression='snappy')
                logger.info(f"   ✅ Saved {parquet_path.name}")
                
                results[symbol] = new_bars
                
            except Exception as e:
                logger.error(f"❌ Error merging {symbol}: {e}")
                results[symbol] = -1
        
        # 5. Recalculate Greeks for all updated files
        await self._recalculate_vanna_parquet()
        
        # 6. Aggregate 1min to daily for _1day.parquet
        daily_results = await self._aggregate_to_daily()
        
        total_1min = sum(v for v in results.values() if v > 0)
        total_daily = sum(v for v in daily_results.values() if v > 0)
        
        logger.info("=" * 60)
        logger.info(f"Merge complete: {total_1min} new 1min bars, {total_daily} days updated")
        logger.info("=" * 60)
        
        return results
    
    async def _recalculate_vanna_parquet(self):
        """Recalculate Greeks for all *_1min.parquet files."""
        from ml.vectorized_greeks import VectorizedGreeksCalculator
        
        logger.info("📐 Recalculating Greeks...")
        calculator = VectorizedGreeksCalculator()
        
        for symbol in self.SYMBOLS:
            input_path = self.storage.data_dir / f"{symbol}_1min.parquet"
            output_path = self.storage.data_dir / f"{symbol}_1min_vanna.parquet"
            
            if not input_path.exists():
                continue
            
            try:
                calculator.process_parquet_file(str(input_path), str(output_path))
                logger.info(f"   ✅ {symbol}_1min_vanna.parquet updated")
            except Exception as e:
                logger.error(f"   ❌ Error calculating Greeks for {symbol}: {e}")
    
    async def _aggregate_to_daily(self) -> Dict[str, int]:
        """
        Aggregate 1min bars to daily OHLCV and update _1day.parquet.
        
        Steps:
        1. Load *_1min.parquet
        2. Aggregate to daily OHLCV (open=first, high=max, low=min, close=last, volume=sum)
        3. Load existing *_1day.parquet
        4. Merge new daily bars (deduplicate by date)
        5. Save updated *_1day.parquet
        
        Returns:
            Dict with new daily bars count per symbol
        """
        logger.info("📅 Aggregating 1min → Daily...")
        
        results = {}
        
        for symbol in self.SYMBOLS:
            try:
                # 1. Load 1min data
                min_path = self.storage.data_dir / f"{symbol}_1min.parquet"
                if not min_path.exists():
                    logger.info(f"   ⏭️ {symbol}: No 1min data")
                    results[symbol] = 0
                    continue
                
                min_df = pd.read_parquet(min_path)
                min_df['timestamp'] = pd.to_datetime(min_df['timestamp'])
                min_df['date'] = min_df['timestamp'].dt.date
                
                # 2. Aggregate to daily OHLCV
                daily_agg = min_df.groupby('date').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).reset_index()
                
                daily_agg['symbol'] = symbol
                daily_agg['timestamp'] = pd.to_datetime(daily_agg['date'])
                
                # Add VIX if available (use daily close)
                if 'vix' in min_df.columns:
                    vix_daily = min_df.groupby('date')['vix'].last().reset_index()
                    daily_agg = daily_agg.merge(vix_daily, on='date', how='left')
                
                # 3. Load existing daily data
                day_path = self.storage.data_dir / f"{symbol}_1day.parquet"
                
                if day_path.exists():
                    existing_df = pd.read_parquet(day_path)
                    existing_df['date'] = pd.to_datetime(existing_df['timestamp']).dt.date
                    
                    # 4. Merge and deduplicate by date
                    # Keep new data where date overlaps (fresher)
                    existing_dates = set(existing_df['date'])
                    new_dates = set(daily_agg['date'])
                    overlap = existing_dates & new_dates
                    
                    if overlap:
                        existing_df = existing_df[~existing_df['date'].isin(overlap)]
                    
                    merged_df = pd.concat([existing_df, daily_agg], ignore_index=True)
                else:
                    merged_df = daily_agg
                
                # Drop temp date column
                if 'date' in merged_df.columns:
                    merged_df = merged_df.drop(columns=['date'])
                
                merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
                
                # 5. Save
                new_days = len(daily_agg)
                merged_df.to_parquet(day_path, index=False, compression='snappy')
                
                logger.info(f"   ✅ {symbol}: {new_days} days aggregated, total {len(merged_df)} days")
                results[symbol] = new_days
                
            except Exception as e:
                logger.error(f"   ❌ Error aggregating daily for {symbol}: {e}")
                results[symbol] = -1
        
        return results
    
    def should_run_monthly_maintenance(self) -> bool:
        """Check if today is maintenance day (1st of month)."""
        return datetime.now().day == 1
    
    def should_run_saturday_merge(self) -> bool:
        """Check if today is Saturday (weekday 5 = Saturday)."""
        return datetime.now().weekday() == 5
    
    async def run_saturday_merge(self) -> Dict[str, int]:
        """
        Run Saturday merge job.
        
        Called every Saturday to merge accumulated live data
        from the trading week into parquet files.
        
        Returns:
            Merge results per symbol
        """
        if not self.should_run_saturday_merge():
            logger.info("⏭️ Not Saturday, skipping merge")
            return {}
        
        logger.info("🗓️ Saturday detected - running weekly data merge")
        return await self.merge_live_to_parquet()


# Singleton
_maintenance_manager: Optional[DataMaintenanceManager] = None


def get_maintenance_manager() -> DataMaintenanceManager:
    """Get or create maintenance manager."""
    global _maintenance_manager
    if _maintenance_manager is None:
        _maintenance_manager = DataMaintenanceManager()
    return _maintenance_manager


async def run_startup_check():
    """Run on application startup."""
    manager = get_maintenance_manager()
    await manager.ensure_historical_data()


async def run_monthly_maintenance():
    """Run monthly maintenance (call on 1st of each month)."""
    manager = get_maintenance_manager()
    return await manager.run_maintenance()


# CLI Entry point
if __name__ == "__main__":
    setup_logger(level="INFO")
    logger = get_logger()
    
    async def main():
        manager = get_maintenance_manager()
        
        # Check for existing data
        logger.info("Checking historical data...")
        await manager.ensure_historical_data()
        
        # Run maintenance
        logger.info("Running maintenance...")
        await manager.run_maintenance()
    
    asyncio.run(main())
