"""
Options Open Interest (OI) Data Collector

Collects daily OI data from IBKR and stores in SQLite.
IBKR only provides LIVE OI data, so we collect and store daily.

Usage:
    # Collect OI for all training symbols
    await collect_daily_oi()
    
    # Load OI data for training
    df = load_oi_data(start_date, end_date)
"""
import asyncio
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional
import sqlite3

import pandas as pd

from core.logger import get_logger
from ibkr.data_fetcher import get_data_fetcher
from ml.symbols import TRAINING_SYMBOLS

logger = get_logger()


class OICollector:
    """
    Collects and stores Options Open Interest data daily.
    
    IBKR only provides live OI, not historical.
    We collect once per day (after market close) and store in SQLite.
    """
    
    DB_PATH = "data/vanna_ml.db"
    TABLE_NAME = "options_oi"
    
    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or self.DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.fetcher = get_data_fetcher()
        
        self._ensure_table()
    
    def _ensure_table(self):
        """Create options_oi table if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS options_oi (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    symbol TEXT NOT NULL,
                    total_call_oi INTEGER DEFAULT 0,
                    total_put_oi INTEGER DEFAULT 0,
                    put_call_oi_ratio REAL DEFAULT 1.0,
                    iv_atm REAL DEFAULT 0.2,
                    put_call_volume_ratio REAL DEFAULT 0.8,
                    total_volume INTEGER DEFAULT 0,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, symbol)
                )
            """)
            
            # Create index for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_oi_date_symbol 
                ON options_oi(date, symbol)
            """)
            
            conn.commit()
            logger.info(f"✅ OI table ensured in {self.db_path}")
    
    async def collect_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Collect OI data for a single symbol.
        
        Returns:
            Dict with OI data or None if failed
        """
        try:
            data = await self.fetcher.get_options_market_data(symbol)
            
            if not data:
                logger.warning(f"No OI data for {symbol}")
                return None
            
            return {
                'symbol': symbol,
                'date': date.today(),
                'total_call_oi': int(data.get('total_call_oi', 0)),
                'total_put_oi': int(data.get('total_put_oi', 0)),
                'put_call_oi_ratio': float(data.get('put_call_oi_ratio', 1.0)),
                'iv_atm': float(data.get('iv_atm', 0.2)),
                'put_call_volume_ratio': float(data.get('put_call_ratio', 0.8)),
                'total_volume': int(data.get('total_volume', 0)),
            }
            
        except Exception as e:
            logger.error(f"Error collecting OI for {symbol}: {e}")
            return None
    
    def save_oi_data(self, data: Dict) -> bool:
        """Save single OI record to SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO options_oi 
                    (date, symbol, total_call_oi, total_put_oi, put_call_oi_ratio,
                     iv_atm, put_call_volume_ratio, total_volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['date'],
                    data['symbol'],
                    data['total_call_oi'],
                    data['total_put_oi'],
                    data['put_call_oi_ratio'],
                    data['iv_atm'],
                    data['put_call_volume_ratio'],
                    data['total_volume'],
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving OI data: {e}")
            return False
    
    async def collect_all(self, symbols: List[str] = None) -> Dict[str, bool]:
        """
        Collect OI data for all symbols.
        
        Args:
            symbols: List of symbols (default: TRAINING_SYMBOLS)
            
        Returns:
            Dict of symbol -> success status
        """
        symbols = symbols or TRAINING_SYMBOLS
        results = {}
        
        logger.info(f"📊 Collecting OI data for {len(symbols)} symbols...")
        
        for symbol in symbols:
            data = await self.collect_symbol(symbol)
            
            if data:
                success = self.save_oi_data(data)
                results[symbol] = success
                
                if success:
                    logger.info(
                        f"   ✅ {symbol}: Call OI={data['total_call_oi']:,}, "
                        f"Put OI={data['total_put_oi']:,}, "
                        f"P/C={data['put_call_oi_ratio']:.2f}"
                    )
            else:
                results[symbol] = False
            
            # Rate limiting - IBKR pacing
            await asyncio.sleep(2)
        
        success_count = sum(results.values())
        logger.info(f"📊 OI collection complete: {success_count}/{len(symbols)} symbols")
        
        return results
    
    def load_oi_data(
        self,
        start_date: date = None,
        end_date: date = None,
        symbols: List[str] = None
    ) -> pd.DataFrame:
        """
        Load OI data from SQLite.
        
        Args:
            start_date: Start date filter
            end_date: End date filter  
            symbols: List of symbols to load
            
        Returns:
            DataFrame with OI data
        """
        query = f"SELECT * FROM {self.TABLE_NAME} WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        if symbols:
            placeholders = ','.join('?' * len(symbols))
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)
        
        query += " ORDER BY date, symbol"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"Loaded {len(df)} OI records")
        return df
    
    def get_latest_oi(self, symbol: str) -> Optional[Dict]:
        """Get most recent OI data for a symbol."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM options_oi 
                WHERE symbol = ? 
                ORDER BY date DESC LIMIT 1
            """, (symbol,))
            
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
        
        return None
    
    def merge_oi_with_training_data(
        self, 
        df: pd.DataFrame, 
        symbol: str
    ) -> pd.DataFrame:
        """
        Merge OI data with training DataFrame using forward-fill.
        
        Strategy for missing OI:
        1. Load all available OI for this symbol
        2. Create daily OI lookup
        3. For each row, find OI from that day or closest previous day
        4. If no OI exists at all, use neutral defaults
        
        Args:
            df: Training DataFrame with 'timestamp' column
            symbol: Symbol to merge OI for
            
        Returns:
            DataFrame with OI columns added
        """
        # Ensure timestamp is datetime
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Load all OI data for this symbol
        oi_df = self.load_oi_data(symbols=[symbol])
        
        # Default OI values (used when no OI data exists)
        OI_DEFAULTS = {
            'total_call_oi': 0,
            'total_put_oi': 0,
            'put_call_oi_ratio': 1.0,  # Neutral
            'oi_available': False,  # Flag for "real vs default"
        }
        
        if oi_df.empty:
            # No OI data collected yet - use defaults
            logger.warning(f"No OI data for {symbol}, using defaults")
            for col, val in OI_DEFAULTS.items():
                df[col] = val
            df.drop(columns=['date'], inplace=True)
            return df
        
        # Create date -> OI lookup with forward-fill
        oi_df['date'] = pd.to_datetime(oi_df['date']).dt.date
        oi_df = oi_df.sort_values('date')
        
        # Create lookup dict
        oi_lookup = {}
        last_oi = OI_DEFAULTS.copy()
        
        # Generate all dates from first OI to today
        all_dates = pd.date_range(
            start=oi_df['date'].min(),
            end=pd.Timestamp.today().date(),
            freq='D'
        ).date
        
        for d in all_dates:
            # Check if we have OI for this date
            day_oi = oi_df[oi_df['date'] == d]
            
            if not day_oi.empty:
                row = day_oi.iloc[0]
                last_oi = {
                    'total_call_oi': int(row['total_call_oi']),
                    'total_put_oi': int(row['total_put_oi']),
                    'put_call_oi_ratio': float(row['put_call_oi_ratio']),
                    'oi_available': True,
                }
            
            oi_lookup[d] = last_oi
        
        # Apply OI to training data
        def get_oi_for_date(d):
            if d in oi_lookup:
                return oi_lookup[d]
            else:
                # Date before OI collection started
                return OI_DEFAULTS
        
        # Map OI to each row
        oi_series = df['date'].apply(get_oi_for_date)
        
        for col in ['total_call_oi', 'total_put_oi', 'put_call_oi_ratio', 'oi_available']:
            df[col] = oi_series.apply(lambda x: x[col])
        
        # Cleanup
        df.drop(columns=['date'], inplace=True)
        
        # Log stats
        oi_available_pct = df['oi_available'].mean() * 100
        logger.info(f"   OI merged for {symbol}: {oi_available_pct:.1f}% with real OI data")
        
        return df


# Singleton
_oi_collector: Optional[OICollector] = None


def get_oi_collector() -> OICollector:
    """Get global OI collector instance."""
    global _oi_collector
    if _oi_collector is None:
        _oi_collector = OICollector()
    return _oi_collector


async def collect_daily_oi(symbols: List[str] = None) -> Dict[str, bool]:
    """
    Convenience function to collect daily OI.
    
    Call this after market close (16:30+ ET).
    """
    collector = get_oi_collector()
    return await collector.collect_all(symbols)


# CLI
if __name__ == "__main__":
    from core.logger import setup_logger
    
    setup_logger(level="INFO")
    
    print("=" * 60)
    print("📊 Options OI Collector")
    print("=" * 60)
    
    asyncio.run(collect_daily_oi())
