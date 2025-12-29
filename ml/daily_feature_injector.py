"""
Daily Feature Injector

Injects features from *_1day.parquet into *_1min.parquet.

CRITICAL: NO LOOK-AHEAD BIAS!
- Each 1-min bar only sees YESTERDAY's daily features
- We shift daily data by +1 day before merging
"""
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

from loguru import logger


class DailyFeatureInjector:
    """
    Inject daily features into 1-min data WITHOUT look-ahead bias.
    
    For each 1-min bar at time T:
    → Uses daily features from PREVIOUS CLOSED day (T-1 or earlier)
    → NEVER uses same-day or future data
    
    Features injected (with 'day_' prefix):
    - day_sma_200, day_sma_50
    - day_rsi_14
    - day_atr_14, day_atr_pct
    - day_bb_position
    - day_macd_hist
    - day_above_sma200
    - days_to_earnings, is_earnings_week
    """
    
    # Columns to inject from daily to 1-min
    DAILY_FEATURES = [
        'sma_200', 'sma_50', 'sma_20',
        'price_vs_sma200', 'price_vs_sma50',
        'rsi_14',
        'atr_14', 'atr_pct',
        'bb_position', 'bb_upper', 'bb_lower',
        'macd', 'macd_signal', 'macd_hist',
        'above_sma200', 'above_sma50',
        'sma_50_200_ratio',
        # Major events (FOMC/CPI for TLT/GLD, mega-cap earnings for SPY/QQQ)
        'days_to_major_event', 'is_event_week', 'is_event_day', 'event_iv_boost',
    ]
    
    def __init__(self, data_dir: str = "data/enriched"):
        self.data_dir = Path(data_dir)
        logger.info(f"DailyFeatureInjector initialized")
    
    def inject_for_symbol(self, symbol: str, target_suffix: str = "1min_vanna") -> Optional[pd.DataFrame]:
        """
        Inject daily features into 1-min data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'SPY')
            target_suffix: Which 1min file to update ('1min', '1min_vanna', '1min_rl')
            
        Returns:
            Updated DataFrame or None
        """
        daily_path = self.data_dir / f"{symbol}_1day.parquet"
        min_path = self.data_dir / f"{symbol}_{target_suffix}.parquet"
        
        if not daily_path.exists():
            logger.warning(f"Daily file not found: {daily_path}")
            return None
        
        if not min_path.exists():
            logger.warning(f"1-min file not found: {min_path}")
            return None
        
        logger.info(f"Injecting daily features into {symbol}_{target_suffix}...")
        
        # Load data
        df_daily = pd.read_parquet(daily_path)
        df_min = pd.read_parquet(min_path)
        
        # Prepare daily data
        # 1. Extract date from timestamp (strip timezone to avoid tz-aware vs tz-naive issues)
        if 'timestamp' in df_daily.columns:
            ts = pd.to_datetime(df_daily['timestamp'])
            # Strip timezone if present
            if ts.dt.tz is not None:
                ts = ts.dt.tz_localize(None)
            df_daily['trade_date'] = ts.dt.date
        else:
            # Assume index is date
            df_daily['trade_date'] = pd.to_datetime(df_daily.index).date
        
        # 2. CRITICAL: Shift by 1 day to avoid look-ahead bias!
        # Each row now represents "end of day" data available for NEXT day's trading
        df_daily_shifted = df_daily.copy()
        df_daily_shifted['trade_date'] = df_daily_shifted['trade_date'].shift(-1)
        
        # Drop last row (no next day)
        df_daily_shifted = df_daily_shifted.dropna(subset=['trade_date'])
        
        # 3. Select only features we want to inject
        available_features = [f for f in self.DAILY_FEATURES if f in df_daily_shifted.columns]
        
        if not available_features:
            logger.warning(f"No daily features found to inject for {symbol}")
            return df_min
        
        # Prepare injection columns with 'day_' prefix
        inject_cols = ['trade_date'] + available_features
        df_inject = df_daily_shifted[inject_cols].copy()
        
        # Rename with 'day_' prefix
        rename_map = {f: f'day_{f}' for f in available_features}
        df_inject = df_inject.rename(columns=rename_map)
        
        # 4. Extract trade_date from 1-min data (strip timezone)
        if 'timestamp' in df_min.columns:
            ts_min = pd.to_datetime(df_min['timestamp'])
            if ts_min.dt.tz is not None:
                ts_min = ts_min.dt.tz_localize(None)
            df_min['trade_date'] = ts_min.dt.date
        elif 'datetime' in df_min.columns:
            ts_min = pd.to_datetime(df_min['datetime'])
            if ts_min.dt.tz is not None:
                ts_min = ts_min.dt.tz_localize(None)
            df_min['trade_date'] = ts_min.dt.date
        else:
            logger.error(f"No timestamp column in 1-min data for {symbol}")
            return None
        
        # 5. Remove existing day_ columns if any (re-injection)
        existing_day_cols = [c for c in df_min.columns if c.startswith('day_')]
        if existing_day_cols:
            logger.debug(f"  Removing {len(existing_day_cols)} existing day_ columns")
            df_min = df_min.drop(columns=existing_day_cols)
        
        n_before = len(df_min.columns)
        
        # 6. Merge on trade_date
        df_merged = df_min.merge(
            df_inject,
            on='trade_date',
            how='left'
        )
        
        # 7. Fill NaN with sensible defaults
        for col in df_merged.columns:
            if col.startswith('day_'):
                if 'sma' in col or 'atr' in col:
                    # Fill with forward-fill then neutral
                    df_merged[col] = df_merged[col].ffill().fillna(0)
                elif 'rsi' in col:
                    df_merged[col] = df_merged[col].ffill().fillna(50)
                elif 'bb_position' in col:
                    df_merged[col] = df_merged[col].ffill().fillna(0.5)
                elif 'above' in col or 'is_event' in col:
                    df_merged[col] = df_merged[col].ffill().fillna(0).astype(int)
                elif 'days_to_major_event' in col:
                    df_merged[col] = df_merged[col].ffill().fillna(30).astype(int)
                elif 'event_iv_boost' in col:
                    df_merged[col] = df_merged[col].ffill().fillna(1.0)
                else:
                    df_merged[col] = df_merged[col].ffill().fillna(0)
        
        # Drop helper column
        df_merged = df_merged.drop(columns=['trade_date'], errors='ignore')
        
        n_after = len(df_merged.columns)
        
        # 8. Save
        df_merged.to_parquet(min_path, index=False)
        
        logger.info(f"  ✅ {symbol}: Injected {n_after - n_before} daily features ({n_after} total columns)")
        
        return df_merged
    
    def inject_all(
        self, 
        symbols: Optional[List[str]] = None,
        target_suffix: str = "1min_vanna"
    ) -> Dict[str, bool]:
        """
        Inject daily features into all symbols.
        
        Args:
            symbols: List of symbols or None to auto-detect
            target_suffix: Which 1min file type to update
            
        Returns:
            Dict of results per symbol
        """
        if symbols is None:
            symbols = [
                f.stem.replace('_1day', '')
                for f in self.data_dir.glob('*_1day.parquet')
            ]
        
        logger.info(f"Injecting daily features into {len(symbols)} symbols...")
        
        results = {}
        for symbol in symbols:
            try:
                df = self.inject_for_symbol(symbol, target_suffix)
                results[symbol] = df is not None
            except Exception as e:
                logger.error(f"Error injecting {symbol}: {e}")
                results[symbol] = False
        
        success = sum(results.values())
        logger.info(f"✅ Daily injection complete: {success}/{len(results)} symbols")
        
        return results


# Singleton
_injector: Optional[DailyFeatureInjector] = None


def get_daily_feature_injector() -> DailyFeatureInjector:
    """Get or create daily feature injector."""
    global _injector
    if _injector is None:
        _injector = DailyFeatureInjector()
    return _injector


if __name__ == "__main__":
    from core.logger import setup_logger
    setup_logger(level="INFO")
    
    injector = get_daily_feature_injector()
    injector.inject_all()
