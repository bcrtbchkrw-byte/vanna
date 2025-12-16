"""
Vanna Data Storage

Handles data persistence for ML training:
- SQLite for metadata and live data
- Parquet for large historical datasets
"""
from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

from core.logger import get_logger

logger = get_logger()


class VannaDataStorage:
    """
    Data storage for Vanna ML pipeline.
    
    Uses:
    - Parquet: Large historical datasets (efficient for columnar data)
    - SQLite: Live data, metadata, quick queries
    """
    
    # Single Source of Truth: import from ml.symbols
    from ml.symbols import TRAINING_SYMBOLS as SYMBOLS
    TIMEFRAMES = ['1min', '1day']
    
    def __init__(
        self,
        data_dir: str = "data/vanna_ml",
        db_path: str = "data/vanna.db"
    ):
        """
        Initialize storage.
        
        Args:
            data_dir: Directory for Parquet files
            db_path: Path to SQLite database
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"VannaDataStorage initialized: {self.data_dir}, {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Historical bars table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_bars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    vix REAL,
                    vix3m REAL,
                    vix_ratio REAL,
                    regime INTEGER,
                    sin_time REAL,
                    cos_time REAL,
                    options_iv_atm REAL,
                    options_volume INTEGER,
                    options_put_call_ratio REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, symbol, timeframe)
                )
            """)
            
            # Live bars table (same schema)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS live_bars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    vix REAL,
                    vix3m REAL,
                    vix_ratio REAL,
                    regime INTEGER,
                    sin_time REAL,
                    cos_time REAL,
                    options_iv_atm REAL,
                    options_volume INTEGER,
                    options_put_call_ratio REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, symbol, timeframe)
                )
            """)
            
            # Vanna calculations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vanna_calculations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    strike REAL,
                    expiry DATE,
                    option_type TEXT,
                    vanna REAL,
                    delta REAL,
                    gamma REAL,
                    vega REAL,
                    charm REAL,
                    volga REAL,
                    underlying_price REAL,
                    iv REAL,
                    risk_free_rate REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_historical_symbol_ts 
                ON historical_bars(symbol, timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_live_symbol_ts 
                ON live_bars(symbol, timestamp)
            """)
            
            conn.commit()
            logger.debug("Database tables initialized")
    
    def save_historical_parquet(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = '1min',
        subdir: str = ''
    ) -> Path:
        """
        Save historical data to Parquet file.
        
        Args:
            df: DataFrame with OHLCV + features
            symbol: Symbol name
            timeframe: '1min' or '1day'
            subdir: Optional subdirectory (e.g. 'lite')
            
        Returns:
            Path to saved file
        """
        filename = f"{symbol}_{timeframe}.parquet"
        
        if subdir:
            target_dir = self.data_dir / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
            filepath = target_dir / filename
        else:
            filepath = self.data_dir / filename
        
        # Ensure proper dtypes
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df.to_parquet(filepath, index=False, compression='snappy')
        logger.info(f"Saved {len(df)} bars to {filepath}")
        
        return filepath
    
    def load_historical_parquet(
        self,
        symbol: str,
        timeframe: str = '1min',
        subdir: str = ''
    ) -> Optional[pd.DataFrame]:
        """
        Load historical data from Parquet.
        
        Args:
            symbol: Symbol name
            timeframe: '1min' or '1day'
            subdir: Optional subdirectory (e.g. 'lite')
            
        Returns:
            DataFrame or None if not found
        """
        filename = f"{symbol}_{timeframe}.parquet"
        
        if subdir:
            filepath = self.data_dir / subdir / filename
        else:
            filepath = self.data_dir / filename
        
        if not filepath.exists():
            # If not found in specific subdir, try main dir as fallback (only if subdir was requested)
            # This allows seamless transition if data was moved
            if subdir and (self.data_dir / filename).exists():
                 logger.debug(f"File not in {subdir}, checking main dir...")
                 filepath = self.data_dir / filename
            else:
                logger.warning(f"Parquet file not found: {filepath}")
                return None
        
        df = pd.read_parquet(filepath)
        logger.debug(f"Loaded {len(df)} bars from {filepath}")
        
        return df
    
    def save_live_bar(self, bar_data: Dict[str, Any]) -> bool:
        """
        Save single live bar to SQLite.
        
        Args:
            bar_data: Dict with bar data
            
        Returns:
            Success flag
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO live_bars (
                        timestamp, symbol, timeframe,
                        open, high, low, close, volume,
                        vix, vix3m, vix_ratio, regime,
                        sin_time, cos_time,
                        options_iv_atm, options_volume, options_put_call_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    bar_data.get('timestamp'),
                    bar_data.get('symbol'),
                    bar_data.get('timeframe', '1min'),
                    bar_data.get('open'),
                    bar_data.get('high'),
                    bar_data.get('low'),
                    bar_data.get('close'),
                    bar_data.get('volume'),
                    bar_data.get('vix'),
                    bar_data.get('vix3m'),
                    bar_data.get('vix_ratio'),
                    bar_data.get('regime'),
                    bar_data.get('sin_time'),
                    bar_data.get('cos_time'),
                    bar_data.get('options_iv_atm'),
                    bar_data.get('options_volume'),
                    bar_data.get('options_put_call_ratio')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving live bar: {e}")
            return False
    
    def save_vanna_calculation(self, vanna_data: Dict[str, Any]) -> bool:
        """
        Save Vanna calculation result.
        
        Args:
            vanna_data: Dict with Greeks data
            
        Returns:
            Success flag
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO vanna_calculations (
                        timestamp, symbol, strike, expiry, option_type,
                        vanna, delta, gamma, vega, charm, volga,
                        underlying_price, iv, risk_free_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    vanna_data.get('timestamp'),
                    vanna_data.get('symbol'),
                    vanna_data.get('strike'),
                    vanna_data.get('expiry'),
                    vanna_data.get('option_type'),
                    vanna_data.get('vanna'),
                    vanna_data.get('delta'),
                    vanna_data.get('gamma'),
                    vanna_data.get('vega'),
                    vanna_data.get('charm'),
                    vanna_data.get('volga'),
                    vanna_data.get('underlying_price'),
                    vanna_data.get('iv'),
                    vanna_data.get('risk_free_rate')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving Vanna calculation: {e}")
            return False
    
    def get_training_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = '1min',
        source: str = 'parquet'
    ) -> pd.DataFrame:
        """
        Get combined training data for ML.
        
        Args:
            symbols: List of symbols (default: all)
            start_date: Start date filter
            end_date: End date filter
            timeframe: '1min' or '1day'
            source: 'parquet' or 'sqlite'
            
        Returns:
            Combined DataFrame
        """
        symbols = symbols or self.SYMBOLS
        dfs = []
        
        for symbol in symbols:
            if source == 'parquet':
                df = self.load_historical_parquet(symbol, timeframe)
            else:
                df = self._load_from_sqlite(symbol, timeframe, start_date, end_date)
            
            if df is not None and len(df) > 0:
                df['symbol'] = symbol
                dfs.append(df)
        
        if not dfs:
            logger.warning("No training data found")
            return pd.DataFrame()
        
        combined = pd.concat(dfs, ignore_index=True)
        
        # Apply date filters
        if 'timestamp' in combined.columns:
            combined['timestamp'] = pd.to_datetime(combined['timestamp'])
            if start_date:
                combined = combined[combined['timestamp'] >= start_date]
            if end_date:
                combined = combined[combined['timestamp'] <= end_date]
        
        logger.info(f"Training data: {len(combined)} rows, {len(symbols)} symbols")
        return combined
    
    def _load_from_sqlite(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Load data from SQLite."""
        query = """
            SELECT * FROM historical_bars 
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY timestamp"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_live_data(
        self,
        symbol: str,
        lookback_minutes: int = 60
    ) -> pd.DataFrame:
        """
        Get recent live data.
        
        Args:
            symbol: Symbol name
            lookback_minutes: How many minutes back
            
        Returns:
            DataFrame with live bars
        """
        cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM live_bars 
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp
            """
            return pd.read_sql_query(query, conn, params=[symbol, cutoff.isoformat()])
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'parquet_files': [],
            'sqlite_tables': {}
        }
        
        # Parquet files
        for symbol in self.SYMBOLS:
            for tf in self.TIMEFRAMES:
                filepath = self.data_dir / f"{symbol}_{tf}.parquet"
                if filepath.exists():
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    stats['parquet_files'].append({
                        'file': filepath.name,
                        'size_mb': round(size_mb, 2)
                    })
        
        # SQLite tables
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for table in ['historical_bars', 'live_bars', 'vanna_calculations']:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats['sqlite_tables'][table] = count
        
        return stats


# Singleton
_data_storage: Optional[VannaDataStorage] = None


def get_data_storage(
    data_dir: str = "data/vanna_ml",
    db_path: str = "data/vanna.db"
) -> VannaDataStorage:
    """Get or create singleton data storage."""
    global _data_storage
    if _data_storage is None:
        _data_storage = VannaDataStorage(data_dir, db_path)
    return _data_storage
