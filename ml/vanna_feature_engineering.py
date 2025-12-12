"""
Vanna Feature Engineering

Extracts features for ML/NN/RL training:
- Time features (sin/cos encoding)
- VIX features (VIX, VIX3M, ratio, term structure)
- Market regime labels
- Options aggregate features
"""
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core.logger import get_logger

logger = get_logger()


class VannaFeatureEngineering:
    """
    Feature engineering for Vanna ML pipeline.
    
    Features extracted:
    - Time encoding (cyclical sin/cos)
    - VIX metrics (level, ratio, change)
    - Market regime (volatility-based)
    - Options data (IV, volume, P/C ratio)
    """
    
    # Market hours: 9:30 - 16:00 = 390 minutes
    MINUTES_PER_DAY = 390
    
    # Regime thresholds
    VIX_LOW_THRESHOLD = 15.0
    VIX_HIGH_THRESHOLD = 25.0
    
    def __init__(self):
        logger.info("VannaFeatureEngineering initialized")
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical time features using sin/cos encoding.
        
        Encodes:
        - Minute of day (0-389 → sin/cos)
        - Day of week (0-4 → sin/cos)
        - Day of year (0-251 → sin/cos for ~252 trading days)
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            DataFrame with added time features
        """
        df = df.copy()
        
        if 'timestamp' not in df.columns:
            logger.warning("No 'timestamp' column found for time features")
            return df
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Minute of day (0-389)
        df['minute_of_day'] = (
            (df['timestamp'].dt.hour - 9) * 60 + 
            df['timestamp'].dt.minute - 30
        ).clip(0, self.MINUTES_PER_DAY - 1)
        
        # Sin/Cos encoding for minute
        df['sin_time'] = np.sin(2 * np.pi * df['minute_of_day'] / self.MINUTES_PER_DAY)
        df['cos_time'] = np.cos(2 * np.pi * df['minute_of_day'] / self.MINUTES_PER_DAY)
        
        # Day of week (0=Monday, 4=Friday)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
        df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
        
        # Day of year (approximate trading day)
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 252)
        df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 252)
        
        # Clean up intermediate columns
        df = df.drop(columns=['minute_of_day', 'day_of_week', 'day_of_year'], errors='ignore')
        
        logger.debug(f"Added time features: sin_time, cos_time, sin_dow, cos_dow, sin_doy, cos_doy")
        return df
    
    def add_vix_features(
        self,
        df: pd.DataFrame,
        vix_col: str = 'vix',
        vix3m_col: str = 'vix3m'
    ) -> pd.DataFrame:
        """
        Add VIX-based features.
        
        Features:
        - vix_ratio: VIX/VIX3M (contango/backwardation indicator)
        - vix_change_1d: 1-day VIX change
        - vix_change_5d: 5-day VIX change
        - vix_percentile: Rolling percentile (where is VIX vs recent history)
        
        Args:
            df: DataFrame with VIX columns
            vix_col: Column name for VIX
            vix3m_col: Column name for VIX3M
            
        Returns:
            DataFrame with VIX features
        """
        df = df.copy()
        
        # VIX ratio (term structure)
        if vix_col in df.columns and vix3m_col in df.columns:
            # Avoid division by zero
            df['vix_ratio'] = df[vix_col] / df[vix3m_col].replace(0, np.nan)
            df['vix_ratio'] = df['vix_ratio'].fillna(1.0)
            
            # Contango = VIX < VIX3M = ratio < 1
            # Backwardation = VIX > VIX3M = ratio > 1
            df['vix_in_contango'] = (df['vix_ratio'] < 1).astype(int)
        
        if vix_col in df.columns:
            # VIX changes
            df['vix_change_1d'] = df[vix_col].pct_change(periods=1).fillna(0)
            df['vix_change_5d'] = df[vix_col].pct_change(periods=5).fillna(0)
            
            # Rolling percentile (20-day window)
            def rolling_percentile(x):
                return pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
            
            df['vix_percentile'] = df[vix_col].rolling(window=20, min_periods=1).apply(
                rolling_percentile, raw=False
            )
            
            # VIX z-score (standardized)
            df['vix_zscore'] = (
                (df[vix_col] - df[vix_col].rolling(20).mean()) / 
                df[vix_col].rolling(20).std().replace(0, 1)
            ).fillna(0)
        
        logger.debug("Added VIX features: vix_ratio, vix_change_*, vix_percentile, vix_zscore")
        return df
    
    def add_regime_labels(
        self,
        df: pd.DataFrame,
        vix_col: str = 'vix'
    ) -> pd.DataFrame:
        """
        Add market regime labels based on VIX levels.
        
        Regimes:
        - 0: Low volatility (VIX < 15)
        - 1: Normal volatility (15 <= VIX < 25)
        - 2: High volatility (VIX >= 25)
        
        Args:
            df: DataFrame with VIX column
            vix_col: Column name for VIX
            
        Returns:
            DataFrame with regime labels
        """
        df = df.copy()
        
        if vix_col not in df.columns:
            logger.warning(f"No '{vix_col}' column for regime classification")
            df['regime'] = 1  # Default to normal
            return df
        
        # Classify regime
        conditions = [
            df[vix_col] < self.VIX_LOW_THRESHOLD,
            (df[vix_col] >= self.VIX_LOW_THRESHOLD) & (df[vix_col] < self.VIX_HIGH_THRESHOLD),
            df[vix_col] >= self.VIX_HIGH_THRESHOLD
        ]
        choices = [0, 1, 2]
        
        df['regime'] = np.select(conditions, choices, default=1)
        
        # Regime labels for readability
        df['regime_label'] = df['regime'].map({
            0: 'low_vol',
            1: 'normal',
            2: 'high_vol'
        })
        
        logger.debug(f"Added regime labels: {df['regime'].value_counts().to_dict()}")
        return df
    
    def add_options_features(
        self,
        df: pd.DataFrame,
        options_data: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Add options-derived features.
        
        Features:
        - options_iv_atm: At-the-money implied volatility
        - options_volume: Total options volume
        - options_put_call_ratio: Put/Call ratio
        - options_skew: Put-Call skew (25-delta)
        
        Args:
            df: DataFrame to add features to
            options_data: Dict with options aggregates per timestamp
            
        Returns:
            DataFrame with options features
        """
        df = df.copy()
        
        if options_data is None:
            # Placeholder columns
            df['options_iv_atm'] = np.nan
            df['options_volume'] = np.nan
            df['options_put_call_ratio'] = np.nan
            logger.debug("No options data provided, added placeholder columns")
            return df
        
        # Map options data to DataFrame
        # Assuming options_data is {timestamp: {iv_atm, volume, pc_ratio}}
        for col in ['options_iv_atm', 'options_volume', 'options_put_call_ratio']:
            df[col] = df['timestamp'].map(
                lambda ts: options_data.get(str(ts), {}).get(col.replace('options_', ''))
            )
        
        return df
    
    def add_price_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Add price-based technical features.
        
        Features:
        - return_1m: 1-minute return
        - return_5m: 5-minute return
        - volatility_20: 20-bar realized volatility
        - momentum_20: 20-bar momentum
        
        Args:
            df: DataFrame with price column
            price_col: Column name for price
            
        Returns:
            DataFrame with price features
        """
        df = df.copy()
        
        if price_col not in df.columns:
            logger.warning(f"No '{price_col}' column for price features")
            return df
        
        # Returns
        df['return_1m'] = df[price_col].pct_change(1).fillna(0)
        df['return_5m'] = df[price_col].pct_change(5).fillna(0)
        
        # Realized volatility (annualized)
        df['volatility_20'] = df['return_1m'].rolling(20).std() * np.sqrt(252 * 390)
        df['volatility_20'] = df['volatility_20'].fillna(0.2)  # Default 20%
        
        # Momentum
        df['momentum_20'] = df[price_col].pct_change(20).fillna(0)
        
        # High-Low range
        if 'high' in df.columns and 'low' in df.columns:
            df['range_pct'] = (df['high'] - df['low']) / df[price_col]
        
        logger.debug("Added price features: return_*, volatility_20, momentum_20")
        return df
    
    def process_all_features(
        self,
        df: pd.DataFrame,
        options_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Apply all feature engineering in sequence.
        
        Args:
            df: Raw DataFrame with OHLCV + VIX data
            options_data: Optional options aggregates
            
        Returns:
            DataFrame with all features
        """
        df = self.add_time_features(df)
        df = self.add_vix_features(df)
        df = self.add_regime_labels(df)
        df = self.add_options_features(df, options_data)
        df = self.add_price_features(df)
        
        logger.info(f"Feature engineering complete: {len(df.columns)} columns")
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names (for ML input)."""
        return [
            # Time features
            'sin_time', 'cos_time', 'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy',
            # VIX features
            'vix', 'vix3m', 'vix_ratio', 'vix_change_1d', 'vix_change_5d',
            'vix_percentile', 'vix_zscore', 'vix_in_contango',
            # Regime
            'regime',
            # Options
            'options_iv_atm', 'options_volume', 'options_put_call_ratio',
            # Price
            'return_1m', 'return_5m', 'volatility_20', 'momentum_20', 'range_pct'
        ]


# Singleton
_feature_engineering: Optional[VannaFeatureEngineering] = None


def get_feature_engineering() -> VannaFeatureEngineering:
    """Get or create singleton feature engineering instance."""
    global _feature_engineering
    if _feature_engineering is None:
        _feature_engineering = VannaFeatureEngineering()
    return _feature_engineering
