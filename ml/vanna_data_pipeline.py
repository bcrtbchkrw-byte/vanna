"""
Vanna Data Pipeline

Main pipeline for ML/NN/RL training data:
- Fetches historical data from IBKR (1-min for 550 days, daily for 10 years)
- Records live data every minute
- Computes all features including Vanna and other Greeks
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import time

import pandas as pd
import numpy as np

from core.logger import get_logger
from ibkr.connection import get_ibkr_connection
from ibkr.data_fetcher import get_data_fetcher

from ml.vanna_calculator import get_vanna_calculator
from ml.vanna_feature_engineering import get_feature_engineering
from ml.data_storage import get_data_storage

logger = get_logger()


class VannaDataPipeline:
    """
    Comprehensive data pipeline for Vanna ML training.
    
    Fetches and processes:
    - SPY, QQQ, IWM, GLD, TLT (main assets)
    - VIX, VIX3M (volatility indices)
    - Options chains for Greeks calculation
    
    Data timeframes:
    - 1-minute bars: 550 days of history
    - Daily bars: 10 years of history
    - Live: Every minute during market hours
    """
    
    # Import from Single Source of Truth
    from ml.symbols import TRAINING_SYMBOLS as SYMBOLS, VIX_SYMBOLS
    
    # IBKR limits
    MAX_BARS_PER_REQUEST = 1000  # Safe limit to avoid pacing
    REQUEST_DELAY = 20  # Seconds between requests (IBKR pacing - aggressive limit)
    
    def __init__(
        self,
        data_dir: str = "data/vanna_ml",
        db_path: str = "data/vanna.db"
    ):
        """
        Initialize pipeline.
        
        Args:
            data_dir: Directory for historical data files
            db_path: Path to SQLite database
        """
        self.connection = None  # Will be initialized on first use
        self.data_fetcher = get_data_fetcher()
        self.vanna_calc = get_vanna_calculator()
        self.feature_eng = get_feature_engineering()
        self.storage = get_data_storage(data_dir, db_path)
        
        # VIX fallback cache (last known values)
        self._last_vix: float = 18.0  # Default VIX
        self._last_vix3m: float = 19.0  # Default VIX3M
        
        self._running = False
        logger.info("VannaDataPipeline initialized")
    
    async def _get_connection(self):
        """Get IBKR connection (lazy async initialization)."""
        if self.connection is None:
            self.connection = await get_ibkr_connection()
        return self.connection
    
    async def fetch_historical_intraday(
        self,
        symbol: str,
        days: int = 550,
        bar_size: str = '1 min'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch intraday historical data from IBKR.
        
        Fetches in batches due to IBKR limits.
        550 days × 390 bars/day = ~214,500 bars
        
        Args:
            symbol: Ticker symbol
            days: Number of days of history
            bar_size: Bar size (default '1 min')
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {days} days of {bar_size} data for {symbol}...")
        
        all_bars = []
        conn = await self._get_connection()
        ib = conn.ib if conn.is_connected else None
        
        if not ib or not ib.isConnected():
            logger.error("IBKR not connected")
            return None
        
        from ib_insync import Stock, Index
        
        # Create contract
        if symbol in ['VIX', 'VIX3M']:
            contract = Index(symbol, 'CBOE', 'USD')
        else:
            contract = Stock(symbol, 'SMART', 'USD')
        
        contracts = await ib.qualifyContractsAsync(contract)
        if not contracts:
            logger.error(f"Could not qualify contract for {symbol}")
            return None
        
        contract = contracts[0]
        
        # Fetch in chunks (IBKR allows max ~60 days per request for 1-min data)
        chunk_days = 5  # Small chunks to avoid IBKR pacing violations
        end_date = datetime.now()
        
        for i in range(0, days, chunk_days):
            chunk_end = end_date - timedelta(days=i)
            duration = f"{min(chunk_days, days - i)} D"
            
            try:
                bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=chunk_end,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow='TRADES' if symbol not in ['VIX', 'VIX3M'] else 'TRADES',
                    useRTH=True,  # Regular trading hours only
                    formatDate=1
                )
                
                if bars:
                    for bar in bars:
                        all_bars.append({
                            'timestamp': bar.date,
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume,
                            'symbol': symbol
                        })
                    
                    logger.debug(f"Fetched {len(bars)} bars for {symbol} ending {chunk_end.date()}")
                
                # Rate limiting
                await asyncio.sleep(self.REQUEST_DELAY)
                
            except Exception as e:
                logger.error(f"Error fetching chunk {i}: {e}")
                await asyncio.sleep(5)  # Longer delay on error
        
        if not all_bars:
            logger.warning(f"No bars fetched for {symbol}")
            return None
        
        df = pd.DataFrame(all_bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        logger.info(f"Fetched {len(df)} total bars for {symbol}")
        return df
    
    async def fetch_historical_daily(
        self,
        symbol: str,
        years: int = 10
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily historical data from IBKR.
        
        Args:
            symbol: Ticker symbol
            years: Number of years of history
            
        Returns:
            DataFrame with daily OHLCV
        """
        logger.info(f"Fetching {years} years of daily data for {symbol}...")
        
        conn = await self._get_connection()
        ib = conn.ib if conn.is_connected else None
        
        if not ib or not ib.isConnected():
            logger.error("IBKR not connected")
            return None
        
        from ib_insync import Stock, Index
        
        # Create contract
        if symbol in ['VIX', 'VIX3M']:
            contract = Index(symbol, 'CBOE', 'USD')
        else:
            contract = Stock(symbol, 'SMART', 'USD')
        
        contracts = await ib.qualifyContractsAsync(contract)
        if not contracts:
            logger.error(f"Could not qualify contract for {symbol}")
            return None
        
        contract = contracts[0]
        
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',  # Now
                durationStr=f"{years} Y",
                barSizeSetting='1 day',
                whatToShow='TRADES' if symbol not in ['VIX', 'VIX3M'] else 'TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if not bars:
                logger.warning(f"No daily bars for {symbol}")
                return None
            
            df = pd.DataFrame([{
                'timestamp': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'symbol': symbol
            } for bar in bars])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timeframe'] = '1day'
            
            logger.info(f"Fetched {len(df)} daily bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching daily data: {e}")
            return None
    
    async def fetch_vix_data(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add VIX and VIX3M data to DataFrame.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with VIX columns added
        """
        # Fetch VIX data for the same period
        vix_df = await self.fetch_historical_intraday('VIX', days=550)
        
        if vix_df is not None:
            vix_df = vix_df[['timestamp', 'close']].rename(columns={'close': 'vix'})
            df = df.merge(vix_df, on='timestamp', how='left')
        else:
            df['vix'] = np.nan
        
        # VIX3M
        vix3m_df = await self.fetch_historical_intraday('VIX3M', days=550)
        
        if vix3m_df is not None:
            vix3m_df = vix3m_df[['timestamp', 'close']].rename(columns={'close': 'vix3m'})
            df = df.merge(vix3m_df, on='timestamp', how='left')
        else:
            df['vix3m'] = np.nan
        
        # Forward fill missing values
        df['vix'] = df['vix'].ffill()
        df['vix3m'] = df['vix3m'].ffill()
        
        return df
    
    async def fetch_vix_daily(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add VIX and VIX3M data to daily DataFrame.
        
        Args:
            df: DataFrame with timestamp column (daily data)
            
        Returns:
            DataFrame with VIX columns added
        """
        # Fetch daily VIX data
        vix_df = await self.fetch_historical_daily('VIX', years=10)
        
        if vix_df is not None:
            # Normalize timestamp to date for merging
            vix_df['date'] = pd.to_datetime(vix_df['timestamp']).dt.date
            vix_df = vix_df[['date', 'close']].rename(columns={'close': 'vix'})
            
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            df = df.merge(vix_df, on='date', how='left')
            df = df.drop(columns=['date'])
        
        # Ensure 'vix' column exists even if merge failed or data missing from IBKR
        if 'vix' not in df.columns:
            logger.warning("VIX column missing after merge, using default 18.0")
            df['vix'] = 18.0  # Safe default
        else:
            # Handle NaN values explicitly
            if df['vix'].isnull().all():
                 df['vix'] = 18.0

        
        # VIX3M
        vix3m_df = await self.fetch_historical_daily('VIX3M', years=10)
        
        if vix3m_df is not None:
            vix3m_df['date'] = pd.to_datetime(vix3m_df['timestamp']).dt.date
            vix3m_df = vix3m_df[['date', 'close']].rename(columns={'close': 'vix3m'})
            
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            df = df.merge(vix3m_df, on='date', how='left')
            df = df.drop(columns=['date'])
        else:
            df['vix3m'] = np.nan
        
        # Forward fill missing values
        df['vix'] = df['vix'].ffill()
        df['vix3m'] = df['vix3m'].ffill()
        
        logger.info(f"Added VIX data to daily DataFrame ({len(df)} rows)")
        return df
    
    async def process_historical_data(
        self,
        symbol: str,
        days: int = 550,
        save: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch, process, and save historical data with all features.
        
        Args:
            symbol: Ticker symbol
            days: Days of history (intraday)
            save: Whether to save to storage
            
        Returns:
            Processed DataFrame
        """
        # Fetch raw data
        df = await self.fetch_historical_intraday(symbol, days)
        
        if df is None or len(df) == 0:
            return None
        
        # Add VIX data
        df = await self.fetch_vix_data(df)
        
        # Add all features
        df = self.feature_eng.process_all_features(df)
        
        # Add timeframe
        df['timeframe'] = '1min'
        
        if save:
            self.storage.save_historical_parquet(df, symbol, '1min')
        
        logger.info(f"Processed {len(df)} bars for {symbol} with {len(df.columns)} features")
        return df
    
    async def fetch_all_historical(self, days: int = 550, years: int = 10):
        """
        Fetch historical data for all symbols.
        
        Args:
            days: Days for intraday data
            years: Years for daily data
        """
        logger.info(f"Fetching historical data: {days} days (1min), {years} years (daily)")
        
        for symbol in self.SYMBOLS:
            # Intraday
            await self.process_historical_data(symbol, days, save=True)
            
            # Daily (with VIX data for regime classification)
            daily_df = await self.fetch_historical_daily(symbol, years)
            if daily_df is not None:
                # Add VIX data to daily DataFrame (fixes "No 'vix' column" warning)
                daily_df = await self.fetch_vix_daily(daily_df)
                daily_df = self.feature_eng.process_all_features(daily_df)
                self.storage.save_historical_parquet(daily_df, symbol, '1day')
        
        logger.info("Historical data fetch complete")
    
    async def record_live_bar(self) -> Dict[str, bool]:
        """
        Record current market data for all symbols.
        
        Returns:
            Dict of symbol -> success
        """
        results = {}
        timestamp = datetime.now()
        
        # Get VIX values first (with fallback to cached values)
        vix = await self._get_vix_value()
        vix3m = await self._get_vix3m_value()
        
        # Use cached values if IBKR returns None
        if vix is None:
            vix = self._last_vix
            logger.warning(f"VIX unavailable, using cached value: {vix:.2f}")
        else:
            self._last_vix = vix  # Update cache
        
        if vix3m is None:
            # Estimate from VIX if unavailable (typical contango ~5%)
            vix3m = vix * 1.05 if vix else self._last_vix3m
            logger.debug(f"VIX3M unavailable, using estimate: {vix3m:.2f}")
        else:
            self._last_vix3m = vix3m  # Update cache
        
        vix_ratio = vix / vix3m if vix3m and vix3m > 0 else 1.0
        
        # Compute time features
        minute_of_day = (timestamp.hour - 9) * 60 + timestamp.minute - 30
        minute_of_day = max(0, min(389, minute_of_day))
        sin_time = np.sin(2 * np.pi * minute_of_day / 390)
        cos_time = np.cos(2 * np.pi * minute_of_day / 390)
        
        # Determine regime
        if vix and vix < 15:
            regime = 0
        elif vix and vix >= 25:
            regime = 2
        else:
            regime = 1
        
        for symbol in self.SYMBOLS:
            try:
                # Get current quote
                quote = await self.data_fetcher.get_stock_quote(symbol)
                
                if quote:
                    bar_data = {
                        'timestamp': timestamp.isoformat(),
                        'symbol': symbol,
                        'timeframe': '1min',
                        'open': quote.get('last'),
                        'high': quote.get('last'),
                        'low': quote.get('last'),
                        'close': quote.get('last'),
                        'volume': quote.get('volume', 0),
                        'vix': vix,
                        'vix3m': vix3m,
                        'vix_ratio': vix_ratio,
                        'regime': regime,
                        'sin_time': sin_time,
                        'cos_time': cos_time,
                        # NOTE: Options data requires separate IBKR call (get_options_chain)
                        # Using 0.0 as default; will be enriched during Saturday pipeline
                        'options_iv_atm': 0.0,
                        'options_volume': 0.0,
                        'options_put_call_ratio': 0.0
                    }
                    
                    results[symbol] = self.storage.save_live_bar(bar_data)
                else:
                    results[symbol] = False
                    
            except Exception as e:
                logger.error(f"Error recording live bar for {symbol}: {e}")
                results[symbol] = False
        
        logger.debug(f"Recorded live bars: {results}")
        return results
    
    async def _get_vix_value(self) -> Optional[float]:
        """Get current VIX value."""
        try:
            return await self.data_fetcher.get_vix()
        except Exception as e:
            logger.debug(f"Could not get VIX: {e}")
            return None
    
    async def _get_vix3m_value(self) -> Optional[float]:
        """Get current VIX3M value."""
        try:
            conn = await self._get_connection()
            ib = conn.ib if conn and conn.is_connected else None
            if not ib or not ib.isConnected():
                return None
            
            from ib_insync import Index
            contract = Index('VIX3M', 'CBOE', 'USD')
            contracts = await ib.qualifyContractsAsync(contract)
            
            if contracts:
                ticker = ib.reqMktData(contracts[0], '', False, False)
                await asyncio.sleep(3)  # Wait for VIX3M data
                value = ticker.last if ticker.last else ticker.close
                ib.cancelMktData(contracts[0])
                return value
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not get VIX3M: {e}")
            return None
    
    async def start_live_recording(self, interval_seconds: int = 60):
        """
        Start live data recording loop.
        
        Args:
            interval_seconds: Recording interval (default 60s)
        """
        self._running = True
        logger.info(f"Starting live recording (every {interval_seconds}s)...")
        
        while self._running:
            try:
                # Check if market is open
                now = datetime.now()
                # Simple market hours check (EST/EDT)
                if now.hour >= 9 and now.hour < 16:
                    await self.record_live_bar()
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in live recording loop: {e}")
                await asyncio.sleep(5)
    
    def stop_live_recording(self):
        """Stop live recording loop."""
        self._running = False
        logger.info("Live recording stopped")
    
    def get_training_data(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = '1min'
    ) -> pd.DataFrame:
        """
        Get combined training data.
        
        Args:
            symbols: List of symbols (default: all)
            timeframe: '1min' or '1day'
            
        Returns:
            Combined DataFrame with all features
        """
        return self.storage.get_training_data(symbols, timeframe=timeframe)


# Singleton
_pipeline: Optional[VannaDataPipeline] = None


def get_vanna_pipeline() -> VannaDataPipeline:
    """Get or create singleton pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = VannaDataPipeline()
    return _pipeline
