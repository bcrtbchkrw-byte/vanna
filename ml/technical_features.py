#!/usr/bin/env python3
"""
technical_features.py

KOMPLETNÍ FEATURE SET pro ML/LSTM/PPO

Kategorie features:
1. TREND (SMA, EMA, trend direction)
2. MOMENTUM (RSI, MACD, Stochastic)
3. VOLATILITY (ATR, Bollinger Bands, Keltner)
4. VOLUME (OBV, VWAP, Volume ratios)
5. SUPPORT/RESISTANCE (Pivot Points, S/R levels)
6. MARKET REGIME (VIX ratios, term structure)
7. OPTIONS SPECIFIC (IV rank, Put/Call ratio, skew)
8. TIME FEATURES (hour, day of week, month)
9. EARNINGS (days to/since, historical move)
10. MACRO (FOMC days, CPI, GDP)

Usage:
    from technical_features import FeatureEngineer
    
    engineer = FeatureEngineer()
    df = engineer.add_all_features(df, symbol='SPY')
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Kompletní feature engineering pro trading ML.
    
    Přidává 50+ features rozdělených do kategorií.
    """
    
    def __init__(self):
        self.feature_names: List[str] = []
    
    def add_all_features(self, df: pd.DataFrame, 
                         symbol: str = None) -> pd.DataFrame:
        """
        Přidá VŠECHNY features.
        
        Args:
            df: DataFrame s OHLCV daty
            symbol: Ticker (pro earnings, atd.)
            
        Returns:
            DataFrame s 50+ novými sloupci
        """
        df = df.copy()
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")
                return df
        
        logger.info(f"Adding features for {symbol or 'unknown'}, {len(df):,} rows")
        
        # 1. TREND FEATURES
        df = self._add_trend_features(df)
        
        # 2. MOMENTUM FEATURES
        df = self._add_momentum_features(df)
        
        # 3. VOLATILITY FEATURES
        df = self._add_volatility_features(df)
        
        # 4. VOLUME FEATURES
        df = self._add_volume_features(df)
        
        # 5. SUPPORT/RESISTANCE
        df = self._add_support_resistance(df)
        
        # 6. MARKET REGIME
        df = self._add_regime_features(df)
        
        # 7. TIME FEATURES
        df = self._add_time_features(df)
        
        # 8. PRICE ACTION
        df = self._add_price_action_features(df)
        
        # 9. DERIVED RATIOS
        df = self._add_derived_ratios(df)
        
        # Count features added
        new_features = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'date', 'symbol']]
        logger.info(f"Added {len(new_features)} features")
        
        return df
    
    # =========================================================================
    # 1. TREND FEATURES
    # =========================================================================
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trend indikátory.
        
        Features:
        - sma_10, sma_20, sma_50, sma_100, sma_200
        - ema_9, ema_21, ema_50
        - price_vs_sma200 (% nad/pod SMA200)
        - sma_50_200_cross (Golden/Death cross signal)
        - trend_strength (ADX)
        - trend_direction (-1, 0, 1)
        """
        close = df['close']
        
        # Simple Moving Averages
        for period in [10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = close.rolling(window=period, min_periods=1).mean()
        
        # Exponential Moving Averages
        for period in [9, 21, 50]:
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # Price vs SMAs (% distance)
        df['price_vs_sma20'] = (close - df['sma_20']) / df['sma_20'] * 100
        df['price_vs_sma50'] = (close - df['sma_50']) / df['sma_50'] * 100
        df['price_vs_sma200'] = (close - df['sma_200']) / df['sma_200'] * 100
        
        # SMA Crosses
        df['sma_20_50_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        df['sma_50_200_cross'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)  # Golden/Death
        
        # Trend Direction (simple)
        df['trend_direction'] = np.where(
            (close > df['sma_20']) & (df['sma_20'] > df['sma_50']), 1,
            np.where(
                (close < df['sma_20']) & (df['sma_20'] < df['sma_50']), -1, 0
            )
        )
        
        # ADX (Trend Strength)
        df = self._calculate_adx(df, period=14)
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=1).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=1).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(window=period, min_periods=1).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        return df
    
    # =========================================================================
    # 2. MOMENTUM FEATURES
    # =========================================================================
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum indikátory.
        
        Features:
        - rsi_14, rsi_7
        - macd, macd_signal, macd_histogram
        - stoch_k, stoch_d
        - roc_10, roc_20 (Rate of Change)
        - momentum_10
        - williams_r
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        for period in [7, 14]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI zones (normalized)
        df['rsi_zone'] = np.where(df['rsi_14'] > 70, 1,    # Overbought
                          np.where(df['rsi_14'] < 30, -1,  # Oversold
                          0))                               # Neutral
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        # Stochastic
        period = 14
        lowest_low = low.rolling(window=period, min_periods=1).min()
        highest_high = high.rolling(window=period, min_periods=1).max()
        df['stoch_k'] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=1).mean()
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (close - close.shift(period)) / close.shift(period) * 100
        
        # Momentum
        df['momentum_10'] = close - close.shift(10)
        df['momentum_20'] = close - close.shift(20)
        
        # Williams %R
        df['williams_r'] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        
        # CCI (Commodity Channel Index)
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=20, min_periods=1).mean()
        mad = typical_price.rolling(window=20, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
        
        return df
    
    # =========================================================================
    # 3. VOLATILITY FEATURES
    # =========================================================================
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility indikátory.
        
        Features:
        - atr_14, atr_20
        - atr_percent (ATR jako % ceny)
        - bb_upper, bb_lower, bb_middle, bb_width, bb_position
        - keltner_upper, keltner_lower
        - historical_volatility_20
        - intraday_range
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR (Average True Range)
        for period in [14, 20]:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[f'atr_{period}'] = tr.rolling(window=period, min_periods=1).mean()
        
        # ATR as percentage of price
        df['atr_percent'] = df['atr_14'] / close * 100
        
        # Bollinger Bands
        period = 20
        std_dev = 2
        sma = close.rolling(window=period, min_periods=1).mean()
        std = close.rolling(window=period, min_periods=1).std()
        df['bb_upper'] = sma + (std * std_dev)
        df['bb_lower'] = sma - (std * std_dev)
        df['bb_middle'] = sma
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Keltner Channels
        ema = close.ewm(span=20, adjust=False).mean()
        df['keltner_upper'] = ema + (df['atr_14'] * 2)
        df['keltner_lower'] = ema - (df['atr_14'] * 2)
        
        # Historical Volatility
        returns = close.pct_change()
        df['volatility_20'] = returns.rolling(window=20, min_periods=1).std() * np.sqrt(252) * 100
        df['volatility_50'] = returns.rolling(window=50, min_periods=1).std() * np.sqrt(252) * 100
        
        # Intraday Range
        df['intraday_range'] = (high - low) / close * 100
        df['intraday_range_avg'] = df['intraday_range'].rolling(window=20, min_periods=1).mean()
        
        # Volatility Ratio (current vs average)
        df['volatility_ratio'] = df['volatility_20'] / (df['volatility_50'] + 1e-10)
        
        return df
    
    # =========================================================================
    # 4. VOLUME FEATURES
    # =========================================================================
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume indikátory.
        
        Features:
        - volume_sma_20, volume_ratio
        - obv (On-Balance Volume)
        - vwap
        - volume_price_trend
        - accumulation_distribution
        """
        if 'volume' not in df.columns:
            logger.warning("No volume column, skipping volume features")
            return df
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Volume SMA and Ratio
        df['volume_sma_20'] = volume.rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = volume / (df['volume_sma_20'] + 1e-10)
        
        # Volume spike detection
        df['volume_spike'] = np.where(df['volume_ratio'] > 2, 1, 0)
        
        # OBV (On-Balance Volume)
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        df['obv_sma'] = df['obv'].rolling(window=20, min_periods=1).mean()
        df['obv_trend'] = np.where(df['obv'] > df['obv_sma'], 1, -1)
        
        # VWAP (simplified - daily reset would need date grouping)
        typical_price = (high + low + close) / 3
        df['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()
        df['price_vs_vwap'] = (close - df['vwap']) / df['vwap'] * 100
        
        # Accumulation/Distribution
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        df['ad_line'] = (clv * volume).cumsum()
        
        # Money Flow Index
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=14, min_periods=1).sum()
        negative_mf = negative_flow.rolling(window=14, min_periods=1).sum()
        
        df['mfi'] = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        
        return df
    
    # =========================================================================
    # 5. SUPPORT/RESISTANCE
    # =========================================================================
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Support/Resistance levels.
        
        Features:
        - pivot, r1, r2, r3, s1, s2, s3
        - distance_to_resistance, distance_to_support
        - at_support, at_resistance (binary)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Classic Pivot Points (using previous period)
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        
        df['pivot'] = (prev_high + prev_low + prev_close) / 3
        df['r1'] = 2 * df['pivot'] - prev_low
        df['s1'] = 2 * df['pivot'] - prev_high
        df['r2'] = df['pivot'] + (prev_high - prev_low)
        df['s2'] = df['pivot'] - (prev_high - prev_low)
        df['r3'] = prev_high + 2 * (df['pivot'] - prev_low)
        df['s3'] = prev_low - 2 * (prev_high - df['pivot'])
        
        # Distance to S/R (%)
        df['dist_to_r1'] = (df['r1'] - close) / close * 100
        df['dist_to_s1'] = (close - df['s1']) / close * 100
        
        # At support/resistance (within 0.5%)
        df['at_resistance'] = np.where(abs(df['dist_to_r1']) < 0.5, 1, 0)
        df['at_support'] = np.where(abs(df['dist_to_s1']) < 0.5, 1, 0)
        
        # Rolling High/Low (support/resistance zones)
        df['rolling_high_20'] = high.rolling(window=20, min_periods=1).max()
        df['rolling_low_20'] = low.rolling(window=20, min_periods=1).min()
        df['rolling_high_50'] = high.rolling(window=50, min_periods=1).max()
        df['rolling_low_50'] = low.rolling(window=50, min_periods=1).min()
        
        # Position in range
        df['position_in_range'] = (close - df['rolling_low_20']) / (df['rolling_high_20'] - df['rolling_low_20'] + 1e-10)
        
        return df
    
    # =========================================================================
    # 6. MARKET REGIME
    # =========================================================================
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Market regime features.
        
        Features:
        - vix_level (pokud je VIX v datech)
        - vix_percentile
        - regime_score (-1 to 1)
        """
        # VIX features (pokud jsou v datech)
        if 'vix' in df.columns:
            df['vix_sma_20'] = df['vix'].rolling(window=20, min_periods=1).mean()
            df['vix_ratio'] = df['vix'] / df['vix_sma_20']
            
            # VIX percentile (rolling 252 days)
            df['vix_percentile'] = df['vix'].rolling(window=252, min_periods=20).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x) * 100, raw=False
            )
            
            # VIX regime
            df['vix_regime'] = np.where(df['vix'] < 15, 0,      # Low vol
                               np.where(df['vix'] < 20, 1,      # Normal
                               np.where(df['vix'] < 25, 2,      # Elevated
                               np.where(df['vix'] < 35, 3, 4)))) # High/Crisis
        
        # Market regime from price action
        close = df['close']
        returns = close.pct_change()
        
        # Trend regime
        df['trend_regime'] = np.where(
            (close > df.get('sma_50', close)) & (close > df.get('sma_200', close)), 1,  # Bull
            np.where(
                (close < df.get('sma_50', close)) & (close < df.get('sma_200', close)), -1,  # Bear
                0  # Neutral
            )
        )
        
        # Volatility regime
        if 'volatility_20' in df.columns:
            vol_median = df['volatility_20'].rolling(window=252, min_periods=20).median()
            df['vol_regime'] = np.where(df['volatility_20'] > vol_median * 1.5, 1,  # High vol
                               np.where(df['volatility_20'] < vol_median * 0.5, -1,  # Low vol
                               0))
        
        return df
    
    # =========================================================================
    # 7. TIME FEATURES
    # =========================================================================
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Time-based features.
        
        Features:
        - hour, minute (pro intraday)
        - day_of_week (0=Monday)
        - day_of_month
        - month
        - is_month_end, is_month_start
        - is_quarter_end
        - days_to_opex (options expiration)
        """
        # Get timestamp
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            ts = pd.to_datetime(df['date'])
        else:
            logger.warning("No timestamp/date column for time features")
            return df
        
        # Basic time features
        df['hour'] = ts.dt.hour
        df['minute'] = ts.dt.minute
        df['day_of_week'] = ts.dt.dayofweek  # 0=Monday
        df['day_of_month'] = ts.dt.day
        df['month'] = ts.dt.month
        df['quarter'] = ts.dt.quarter
        
        # Cyclical encoding (lepší pro ML)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Calendar effects
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        
        # Options expiration (3rd Friday)
        def is_opex_week(d):
            # Third Friday is between 15th and 21st
            if d.day >= 15 and d.day <= 21 and d.weekday() == 4:
                return 1
            if d.day >= 12 and d.day <= 18:  # Week of opex
                return 1
            return 0
        
        df['is_opex_week'] = ts.apply(is_opex_week)
        
        # Trading session
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        df['is_power_hour'] = ((df['hour'] == 15) | ((df['hour'] == 9) & (df['minute'] < 60))).astype(int)
        
        return df
    
    # =========================================================================
    # 8. PRICE ACTION
    # =========================================================================
    
    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Price action features.
        
        Features:
        - candle patterns
        - gaps
        - consecutive moves
        """
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Basic candle features
        df['body'] = close - open_price
        df['body_pct'] = df['body'] / open_price * 100
        df['upper_shadow'] = high - pd.concat([open_price, close], axis=1).max(axis=1)
        df['lower_shadow'] = pd.concat([open_price, close], axis=1).min(axis=1) - low
        df['is_bullish'] = (close > open_price).astype(int)
        
        # Gap
        df['gap'] = open_price - close.shift(1)
        df['gap_pct'] = df['gap'] / close.shift(1) * 100
        df['gap_up'] = (df['gap_pct'] > 0.5).astype(int)
        df['gap_down'] = (df['gap_pct'] < -0.5).astype(int)
        
        # Consecutive moves
        df['is_up'] = (close > close.shift(1)).astype(int)
        df['consecutive_up'] = df['is_up'].groupby((df['is_up'] != df['is_up'].shift()).cumsum()).cumsum()
        df['consecutive_down'] = (1 - df['is_up']).groupby(((1 - df['is_up']) != (1 - df['is_up']).shift()).cumsum()).cumsum()
        
        # Returns
        df['return_1'] = close.pct_change(1) * 100
        df['return_5'] = close.pct_change(5) * 100
        df['return_10'] = close.pct_change(10) * 100
        df['return_20'] = close.pct_change(20) * 100
        
        # Drawdown
        rolling_max = close.rolling(window=252, min_periods=1).max()
        df['drawdown'] = (close - rolling_max) / rolling_max * 100
        
        return df
    
    # =========================================================================
    # 9. DERIVED RATIOS
    # =========================================================================
    
    def _add_derived_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derived ratios combining multiple features.
        
        Features:
        - trend_momentum_score
        - risk_reward_ratio
        - composite_score
        """
        # Trend-Momentum Score (-1 to 1)
        trend_score = 0
        if 'trend_direction' in df.columns:
            trend_score += df['trend_direction'] * 0.3
        if 'macd_cross' in df.columns:
            trend_score += df['macd_cross'] * 0.2
        if 'rsi_14' in df.columns:
            trend_score += ((df['rsi_14'] - 50) / 50) * 0.2
        if 'adx' in df.columns:
            trend_score += (df['adx'] / 100) * 0.3
        
        df['trend_momentum_score'] = trend_score
        
        # Volatility-adjusted strength
        if 'atr_percent' in df.columns and 'return_1' in df.columns:
            df['vol_adjusted_return'] = df['return_1'] / (df['atr_percent'] + 1e-10)
        
        # Composite regime score
        regime_score = 0
        if 'vix_regime' in df.columns:
            regime_score += df['vix_regime'] * 0.4
        if 'trend_regime' in df.columns:
            regime_score += df['trend_regime'] * 0.3
        if 'vol_regime' in df.columns:
            regime_score += df['vol_regime'] * 0.3
        
        df['composite_regime'] = regime_score
        
        return df


# =============================================================================
# FEATURE LIST SUMMARY
# =============================================================================

FEATURE_CATEGORIES = {
    'TREND': [
        'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
        'ema_9', 'ema_21', 'ema_50',
        'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
        'sma_20_50_cross', 'sma_50_200_cross',
        'trend_direction', 'adx', 'plus_di', 'minus_di'
    ],
    'MOMENTUM': [
        'rsi_7', 'rsi_14', 'rsi_zone',
        'macd', 'macd_signal', 'macd_histogram', 'macd_cross',
        'stoch_k', 'stoch_d',
        'roc_5', 'roc_10', 'roc_20',
        'momentum_10', 'momentum_20',
        'williams_r', 'cci'
    ],
    'VOLATILITY': [
        'atr_14', 'atr_20', 'atr_percent',
        'bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position',
        'keltner_upper', 'keltner_lower',
        'volatility_20', 'volatility_50', 'volatility_ratio',
        'intraday_range', 'intraday_range_avg'
    ],
    'VOLUME': [
        'volume_sma_20', 'volume_ratio', 'volume_spike',
        'obv', 'obv_sma', 'obv_trend',
        'vwap', 'price_vs_vwap',
        'ad_line', 'mfi'
    ],
    'SUPPORT_RESISTANCE': [
        'pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3',
        'dist_to_r1', 'dist_to_s1',
        'at_resistance', 'at_support',
        'rolling_high_20', 'rolling_low_20',
        'rolling_high_50', 'rolling_low_50',
        'position_in_range'
    ],
    'REGIME': [
        'vix_sma_20', 'vix_ratio', 'vix_percentile', 'vix_regime',
        'trend_regime', 'vol_regime'
    ],
    'TIME': [
        'hour', 'minute', 'day_of_week', 'day_of_month', 'month', 'quarter',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'is_monday', 'is_friday', 'is_month_start', 'is_month_end',
        'is_opex_week', 'is_market_open', 'is_power_hour'
    ],
    'PRICE_ACTION': [
        'body', 'body_pct', 'upper_shadow', 'lower_shadow', 'is_bullish',
        'gap', 'gap_pct', 'gap_up', 'gap_down',
        'consecutive_up', 'consecutive_down',
        'return_1', 'return_5', 'return_10', 'return_20',
        'drawdown'
    ],
    'COMPOSITE': [
        'trend_momentum_score', 'vol_adjusted_return', 'composite_regime'
    ]
}


def get_feature_list() -> List[str]:
    """Get flat list of all features."""
    features = []
    for category, feats in FEATURE_CATEGORIES.items():
        features.extend(feats)
    return features


def print_feature_summary():
    """Print summary of all features."""
    print("=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    
    total = 0
    for category, features in FEATURE_CATEGORIES.items():
        print(f"\n{category} ({len(features)} features):")
        for f in features:
            print(f"  - {f}")
        total += len(features)
    
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total} features")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print_feature_summary()
    
    # Test
    print("\n" + "=" * 60)
    print("TEST: Generate features for sample data")
    print("=" * 60)
    
    # Create sample OHLCV data
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1H')
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close + np.random.randn(n) * 0.2,
        'high': close + abs(np.random.randn(n) * 0.5),
        'low': close - abs(np.random.randn(n) * 0.5),
        'close': close,
        'volume': np.random.randint(1000000, 5000000, n),
        'vix': 15 + np.random.randn(n) * 3
    })
    
    # Add features
    engineer = FeatureEngineer()
    df_features = engineer.add_all_features(df, symbol='TEST')
    
    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"After features: {len(df_features.columns)}")
    print(f"Features added: {len(df_features.columns) - len(df.columns)}")
    
    print("\nSample output (first 5 rows, selected columns):")
    sample_cols = ['close', 'sma_200', 'price_vs_sma200', 'rsi_14', 'macd', 'atr_percent', 'bb_position']
    print(df_features[sample_cols].head())
