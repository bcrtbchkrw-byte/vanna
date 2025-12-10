"""
Feature Engineering Module

Transforms raw market data into ML-ready features.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from loguru import logger

class FeatureEngineer:
    """
    Computes technical indicators and features.
    """
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators on a DataFrame of Open/High/Low/Close.
        """
        if df.empty or len(df) < 20: # Need history
            return df
            
        df = df.copy()
        
        # 1. Price Changes
        df['returns'] = df['close'].pct_change()
        
        # 2. Volatility (Historical) - 20 period
        df['hist_vol'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # 3. RSI (14)
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # 4. Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # 5. Distance from SMA
        df['dist_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        # Drop NaN
        df.dropna(inplace=True)
        return df

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def extract_inference_features(self, market_data: Dict[str, Any]) -> List[float]:
        """
        Extract features from a single point of data (for realtime inference).
        Note: Real ML models need the full window history.
        
        For Phase 9, we mock this input vector structure.
        Features: [VIX, IV_Rank, RSI_Dummy]
        """
        return [
            market_data.get('vix', 0),
            market_data.get('iv_rank', 0),
            market_data.get('rsi', 50) # Assuming pre-calculated or passed
        ]
