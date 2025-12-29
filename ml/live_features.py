#!/usr/bin/env python3
"""
live_features.py

LIVE FEATURE CALCULATOR

Problém: Při tréninku máme celou historii, při live máme jen aktuální bar.
Řešení: Rolling buffer který udržuje potřebnou historii.

KRITICKÉ: Features MUSÍ být IDENTICKÉ s training features!

Architektura:
┌─────────────────────────────────────────────────────────────┐
│                    LiveFeatureCalculator                     │
├─────────────────────────────────────────────────────────────┤
│  RollingBuffer (500 barů)                                   │
│  ├── Stačí pro SMA200 + buffer                             │
│  ├── Automaticky odstraňuje staré bary                      │
│  └── FIFO (First In, First Out)                            │
│                                                             │
│  Incremental Calculations                                   │
│  ├── RSI: Udržuje avg_gain, avg_loss                       │
│  ├── EMA: Udržuje předchozí EMA hodnotu                    │
│  └── MACD: Udržuje EMA12, EMA26, Signal                    │
│                                                             │
│  Precomputed Lookups                                        │
│  ├── VIX percentiles (historické)                          │
│  ├── Earnings calendar                                      │
│  └── Support/Resistance levels                             │
└─────────────────────────────────────────────────────────────┘

Usage:
    # Initialize once
    calculator = LiveFeatureCalculator(symbol='SPY')
    
    # On each new bar from IBKR:
    features = calculator.update(bar)
    
    # features je dict s IDENTICKÝMI názvy jako training
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Deque
from collections import deque
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Bar:
    """OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    vix: float = None  # Optional - from separate feed


@dataclass 
class IncrementalState:
    """State pro incremental kalkulace."""
    # RSI
    rsi_avg_gain: float = 0.0
    rsi_avg_loss: float = 0.0
    rsi_prev_close: float = None
    
    # EMAs
    ema_9: float = None
    ema_12: float = None
    ema_21: float = None
    ema_26: float = None
    ema_50: float = None
    
    # MACD
    macd_signal: float = None
    
    # OBV
    obv: float = 0.0
    
    # Counters
    bars_processed: int = 0


# =============================================================================
# ROLLING BUFFER
# =============================================================================

class RollingBuffer:
    """
    Rolling buffer pro historické bary.
    
    Udržuje posledních N barů pro kalkulace vyžadující historii.
    """
    
    def __init__(self, maxlen: int = 500):
        """
        Args:
            maxlen: Maximální počet barů (500 stačí pro SMA200 + buffer)
        """
        self.maxlen = maxlen
        self._bars: Deque[Bar] = deque(maxlen=maxlen)
        
        # Pre-allocated arrays pro rychlé výpočty
        self._closes: Deque[float] = deque(maxlen=maxlen)
        self._highs: Deque[float] = deque(maxlen=maxlen)
        self._lows: Deque[float] = deque(maxlen=maxlen)
        self._volumes: Deque[int] = deque(maxlen=maxlen)
        self._vix: Deque[float] = deque(maxlen=maxlen)
    
    def append(self, bar: Bar):
        """Přidá nový bar."""
        self._bars.append(bar)
        self._closes.append(bar.close)
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._volumes.append(bar.volume)
        if bar.vix is not None:
            self._vix.append(bar.vix)
    
    def __len__(self) -> int:
        return len(self._bars)
    
    @property
    def closes(self) -> np.ndarray:
        return np.array(self._closes)
    
    @property
    def highs(self) -> np.ndarray:
        return np.array(self._highs)
    
    @property
    def lows(self) -> np.ndarray:
        return np.array(self._lows)
    
    @property
    def volumes(self) -> np.ndarray:
        return np.array(self._volumes)
    
    @property
    def vix_values(self) -> np.ndarray:
        return np.array(self._vix) if self._vix else np.array([])
    
    @property
    def last_bar(self) -> Optional[Bar]:
        return self._bars[-1] if self._bars else None
    
    @property
    def prev_bar(self) -> Optional[Bar]:
        return self._bars[-2] if len(self._bars) >= 2 else None
    
    def get_last_n(self, n: int) -> np.ndarray:
        """Vrátí posledních N closes."""
        closes = self.closes
        return closes[-n:] if len(closes) >= n else closes
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame (pro debugging)."""
        return pd.DataFrame({
            'timestamp': [b.timestamp for b in self._bars],
            'open': [b.open for b in self._bars],
            'high': list(self._highs),
            'low': list(self._lows),
            'close': list(self._closes),
            'volume': list(self._volumes),
        })


# =============================================================================
# LIVE FEATURE CALCULATOR
# =============================================================================

class LiveFeatureCalculator:
    """
    Počítá features IDENTICKÉ s training features.
    
    Optimalizováno pro real-time:
    - Incremental kalkulace kde možné
    - Rolling buffer pro historii
    - Precomputed lookups pro statické hodnoty
    """
    
    # Požadovaná historie pro každou feature kategorii
    REQUIRED_HISTORY = {
        'sma_200': 200,
        'sma_100': 100,
        'sma_50': 50,
        'volatility_50': 50,
        'vix_percentile': 252,  # 1 rok pro percentil
    }
    
    MIN_HISTORY = 200  # Minimum pro spolehlivé features
    WARMUP_BARS = 250  # Doporučený warmup
    
    def __init__(self, 
                 symbol: str,
                 buffer_size: int = 500,
                 preload_history: pd.DataFrame = None):
        """
        Args:
            symbol: Ticker symbol
            buffer_size: Velikost rolling bufferu
            preload_history: Historická data pro inicializaci
        """
        self.symbol = symbol
        self.buffer = RollingBuffer(maxlen=buffer_size)
        self.state = IncrementalState()
        
        # Precomputed lookups
        self._vix_historical: Optional[np.ndarray] = None
        self._earnings_dates: List[date] = []
        self._support_levels: List[float] = []
        self._resistance_levels: List[float] = []
        
        # Preload history if provided
        if preload_history is not None:
            self._preload(preload_history)
        
        logger.info(f"LiveFeatureCalculator initialized for {symbol}")
    
    # =========================================================================
    # WARMUP METHODS - KRITICKÉ PRO SPRÁVNÉ FEATURES!
    # =========================================================================
    
    def warmup_from_ibkr(self, ib_client, bars_needed: int = 300) -> bool:
        """
        Warmup z IBKR historických dat.
        
        DOPORUČENÁ METODA - IBKR poskytuje data zdarma!
        
        Args:
            ib_client: ib_insync IB instance
            bars_needed: Kolik barů stáhnout (default 300 > 200 minimum)
            
        Returns:
            True pokud warmup úspěšný
        """
        from datetime import datetime, timedelta
        
        logger.info(f"Warming up {self.symbol} from IBKR ({bars_needed} bars)...")
        
        try:
            from ib_insync import Stock, util
            
            # Create contract
            contract = Stock(self.symbol, 'SMART', 'USD')
            
            # Request historical data
            # Duration: "2 D" = 2 days of 1-min bars ≈ 780 bars
            bars = ib_client.reqHistoricalData(
                contract,
                endDateTime='',  # Now
                durationStr='3 D',  # 3 days = ~1170 bars
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=True,  # Regular trading hours only
                formatDate=1
            )
            
            if not bars:
                logger.error(f"No historical data returned for {self.symbol}")
                return False
            
            logger.info(f"Got {len(bars)} bars from IBKR")
            
            # Convert to our Bar format and feed to buffer
            for ib_bar in bars[-bars_needed:]:  # Take last N bars
                bar = Bar(
                    timestamp=ib_bar.date,
                    open=ib_bar.open,
                    high=ib_bar.high,
                    low=ib_bar.low,
                    close=ib_bar.close,
                    volume=ib_bar.volume
                )
                self.update(bar, return_features=False)
            
            logger.info(f"✅ Warmup complete: {len(self.buffer)} bars in buffer")
            return self.is_ready()
            
        except Exception as e:
            logger.error(f"IBKR warmup failed: {e}")
            return False
    
    def warmup_from_parquet(self, parquet_path: str = None, 
                            bars_needed: int = 300) -> bool:
        """
        Warmup z uložených parquet dat.
        
        Používá data z tréninku - VŽDY dostupná!
        
        Args:
            parquet_path: Cesta k parquet souboru (None = auto-detect)
            bars_needed: Kolik barů načíst
            
        Returns:
            True pokud warmup úspěšný
        """
        from pathlib import Path
        
        logger.info(f"Warming up {self.symbol} from parquet ({bars_needed} bars)...")
        
        # Auto-detect path
        if parquet_path is None:
            data_dir = Path("data/enriched")
            candidates = [
                data_dir / f"{self.symbol}_1min_features.parquet",
                data_dir / f"{self.symbol}_1min_vanna.parquet",
                data_dir / f"{self.symbol}_1min.parquet",
            ]
            
            for candidate in candidates:
                if candidate.exists():
                    parquet_path = candidate
                    break
            
            if parquet_path is None:
                logger.error(f"No parquet file found for {self.symbol}")
                return False
        
        try:
            df = pd.read_parquet(parquet_path)
            
            # Sort by timestamp and take last N bars
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            df = df.tail(bars_needed)
            
            logger.info(f"Loading {len(df)} bars from {parquet_path}")
            
            # Feed to buffer
            for _, row in df.iterrows():
                bar = Bar(
                    timestamp=row.get('timestamp', datetime.now()),
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row.get('volume', 0),
                    vix=row.get('vix')
                )
                self.update(bar, return_features=False)
            
            logger.info(f"✅ Warmup complete: {len(self.buffer)} bars in buffer")
            return self.is_ready()
            
        except Exception as e:
            logger.error(f"Parquet warmup failed: {e}")
            return False
    
    def warmup_from_df(self, df: pd.DataFrame) -> bool:
        """
        Warmup z DataFrame.
        
        Args:
            df: DataFrame s OHLCV daty
            
        Returns:
            True pokud warmup úspěšný
        """
        logger.info(f"Warming up {self.symbol} from DataFrame ({len(df)} rows)...")
        
        # Take last 500 rows max
        df = df.tail(500)
        
        for _, row in df.iterrows():
            bar = Bar(
                timestamp=row.get('timestamp', datetime.now()),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row.get('volume', 0),
                vix=row.get('vix')
            )
            self.update(bar, return_features=False)
        
        logger.info(f"✅ Warmup complete: {len(self.buffer)} bars")
        return self.is_ready()
    
    def get_warmup_status(self) -> Dict[str, Any]:
        """
        Vrací detailní status warmupu.
        
        KRITICKÉ: Neobchoduj dokud is_ready() == True!
        """
        bars = len(self.buffer)
        needed = self.MIN_HISTORY
        
        return {
            'is_ready': bars >= needed,
            'bars_in_buffer': bars,
            'bars_needed': needed,
            'progress_percent': min(100, bars / needed * 100),
            'missing_bars': max(0, needed - bars),
            
            # Feature availability
            'features_available': {
                'sma_10': bars >= 10,
                'sma_20': bars >= 20,
                'sma_50': bars >= 50,
                'sma_100': bars >= 100,
                'sma_200': bars >= 200,  # KRITICKÉ!
                'rsi_14': bars >= 15,
                'macd': bars >= 26,
                'bollinger': bars >= 20,
                'atr': bars >= 14,
                'vix_percentile': bars >= 20 and self._vix_historical is not None,
            },
            
            # Warnings
            'warnings': self._get_warmup_warnings(bars)
        }
    
    def _get_warmup_warnings(self, bars: int) -> List[str]:
        """Generuje varování pro nedostatečný warmup."""
        warnings = []
        
        if bars < 200:
            warnings.append(f"⚠️ SMA200 NEDOSTUPNÉ! Potřeba {200 - bars} barů.")
        
        if bars < 50:
            warnings.append(f"⚠️ SMA50 NEDOSTUPNÉ! Potřeba {50 - bars} barů.")
        
        if bars < self.MIN_HISTORY:
            warnings.append(f"🚨 NEOBCHODUJ! Warmup {bars}/{self.MIN_HISTORY} ({bars/self.MIN_HISTORY*100:.0f}%)")
        
        if self._vix_historical is None:
            warnings.append("⚠️ VIX historie nenastavena - vix_percentile bude nepřesný")
        
        if not self._earnings_dates:
            warnings.append("⚠️ Earnings kalendář nenastaven")
        
        return warnings
    
    def _preload(self, df: pd.DataFrame):
        """Preload historical data into buffer."""
        logger.info(f"Preloading {len(df)} historical bars...")
        
        for _, row in df.iterrows():
            bar = Bar(
                timestamp=row.get('timestamp', datetime.now()),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row.get('volume', 0),
                vix=row.get('vix')
            )
            self.update(bar, return_features=False)
        
        logger.info(f"Preloaded {len(self.buffer)} bars, ready for live")
    
    def set_vix_history(self, vix_values: np.ndarray):
        """Set historical VIX pro percentil kalkulaci."""
        self._vix_historical = vix_values
    
    def set_earnings_calendar(self, dates: List[date]):
        """Set earnings dates."""
        self._earnings_dates = sorted(dates)
    
    def update(self, bar: Bar, return_features: bool = True) -> Optional[Dict[str, float]]:
        """
        Update s novým barem a vrať features.
        
        Args:
            bar: Nový OHLCV bar
            return_features: Vrátit features dict?
            
        Returns:
            Dict s features IDENTICKÝMI jako training
        """
        # Add to buffer
        self.buffer.append(bar)
        
        # Update incremental state
        self._update_incremental(bar)
        
        self.state.bars_processed += 1
        
        if not return_features:
            return None
        
        # Check minimum history
        if len(self.buffer) < self.MIN_HISTORY:
            logger.debug(f"Warming up: {len(self.buffer)}/{self.MIN_HISTORY} bars")
            return None
        
        # Calculate all features
        return self._calculate_all_features(bar)
    
    def _update_incremental(self, bar: Bar):
        """Update incremental kalkulace."""
        close = bar.close
        
        # RSI incremental update
        if self.state.rsi_prev_close is not None:
            delta = close - self.state.rsi_prev_close
            gain = max(0, delta)
            loss = max(0, -delta)
            
            # Smoothed average
            alpha = 1 / 14  # RSI period
            self.state.rsi_avg_gain = (1 - alpha) * self.state.rsi_avg_gain + alpha * gain
            self.state.rsi_avg_loss = (1 - alpha) * self.state.rsi_avg_loss + alpha * loss
        
        self.state.rsi_prev_close = close
        
        # EMA incremental updates
        def update_ema(prev_ema, close, period):
            if prev_ema is None:
                return close
            alpha = 2 / (period + 1)
            return alpha * close + (1 - alpha) * prev_ema
        
        self.state.ema_9 = update_ema(self.state.ema_9, close, 9)
        self.state.ema_12 = update_ema(self.state.ema_12, close, 12)
        self.state.ema_21 = update_ema(self.state.ema_21, close, 21)
        self.state.ema_26 = update_ema(self.state.ema_26, close, 26)
        self.state.ema_50 = update_ema(self.state.ema_50, close, 50)
        
        # MACD signal
        if self.state.ema_12 is not None and self.state.ema_26 is not None:
            macd = self.state.ema_12 - self.state.ema_26
            self.state.macd_signal = update_ema(self.state.macd_signal, macd, 9)
        
        # OBV
        if self.buffer.prev_bar:
            if close > self.buffer.prev_bar.close:
                self.state.obv += bar.volume
            elif close < self.buffer.prev_bar.close:
                self.state.obv -= bar.volume
    
    def _calculate_all_features(self, bar: Bar) -> Dict[str, float]:
        """
        Vypočítá VŠECHNY features.
        
        KRITICKÉ: Názvy a kalkulace MUSÍ být IDENTICKÉ s technical_features.py!
        """
        features = {}
        
        closes = self.buffer.closes
        highs = self.buffer.highs
        lows = self.buffer.lows
        volumes = self.buffer.volumes
        
        close = bar.close
        high = bar.high
        low = bar.low
        open_price = bar.open
        
        # =====================================================================
        # 1. TREND FEATURES
        # =====================================================================
        
        # SMAs
        for period in [10, 20, 50, 100, 200]:
            if len(closes) >= period:
                features[f'sma_{period}'] = np.mean(closes[-period:])
            else:
                features[f'sma_{period}'] = np.mean(closes)
        
        # EMAs (from incremental state)
        features['ema_9'] = self.state.ema_9 or close
        features['ema_21'] = self.state.ema_21 or close
        features['ema_50'] = self.state.ema_50 or close
        
        # Price vs SMAs
        features['price_vs_sma20'] = (close - features['sma_20']) / features['sma_20'] * 100
        features['price_vs_sma50'] = (close - features['sma_50']) / features['sma_50'] * 100
        features['price_vs_sma200'] = (close - features['sma_200']) / features['sma_200'] * 100
        
        # SMA crosses
        features['sma_20_50_cross'] = 1 if features['sma_20'] > features['sma_50'] else -1
        features['sma_50_200_cross'] = 1 if features['sma_50'] > features['sma_200'] else -1
        
        # Trend direction
        if close > features['sma_20'] and features['sma_20'] > features['sma_50']:
            features['trend_direction'] = 1
        elif close < features['sma_20'] and features['sma_20'] < features['sma_50']:
            features['trend_direction'] = -1
        else:
            features['trend_direction'] = 0
        
        # ADX (simplified)
        features['adx'], features['plus_di'], features['minus_di'] = self._calculate_adx_full(highs, lows, closes)
        
        # =====================================================================
        # 2. MOMENTUM FEATURES
        # =====================================================================
        
        # RSI (from incremental)
        if self.state.rsi_avg_loss > 0:
            rs = self.state.rsi_avg_gain / self.state.rsi_avg_loss
            features['rsi_14'] = 100 - (100 / (1 + rs))
        else:
            features['rsi_14'] = 50
        
        features['rsi_7'] = self._calculate_rsi_fast(closes, 7)
        features['rsi_zone'] = 1 if features['rsi_14'] > 70 else (-1 if features['rsi_14'] < 30 else 0)
        
        # MACD
        features['macd'] = (self.state.ema_12 or close) - (self.state.ema_26 or close)
        features['macd_signal'] = self.state.macd_signal or features['macd']
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        features['macd_cross'] = 1 if features['macd'] > features['macd_signal'] else -1
        
        # Stochastic
        if len(closes) >= 14:
            lowest_low = np.min(lows[-14:])
            highest_high = np.max(highs[-14:])
            features['stoch_k'] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        else:
            features['stoch_k'] = 50
        features['stoch_d'] = features['stoch_k']  # Simplified
        
        # ROC
        for period in [5, 10, 20]:
            if len(closes) > period:
                features[f'roc_{period}'] = (close - closes[-period-1]) / closes[-period-1] * 100
            else:
                features[f'roc_{period}'] = 0
        
        # Momentum
        features['momentum_10'] = close - closes[-11] if len(closes) > 10 else 0
        features['momentum_20'] = close - closes[-21] if len(closes) > 20 else 0
        
        # Williams %R
        if len(closes) >= 14:
            highest_high = np.max(highs[-14:])
            lowest_low = np.min(lows[-14:])
            features['williams_r'] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        else:
            features['williams_r'] = -50
        
        # CCI
        features['cci'] = self._calculate_cci_fast(highs, lows, closes)
        
        # =====================================================================
        # 3. VOLATILITY FEATURES
        # =====================================================================
        
        # ATR
        features['atr_14'] = self._calculate_atr_fast(highs, lows, closes, 14)
        features['atr_20'] = self._calculate_atr_fast(highs, lows, closes, 20)
        features['atr_percent'] = features['atr_14'] / close * 100
        
        # Bollinger Bands
        if len(closes) >= 20:
            sma = np.mean(closes[-20:])
            std = np.std(closes[-20:])
            features['bb_upper'] = sma + 2 * std
            features['bb_lower'] = sma - 2 * std
            features['bb_middle'] = sma
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma * 100
            features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
        else:
            features['bb_upper'] = close * 1.02
            features['bb_lower'] = close * 0.98
            features['bb_middle'] = close
            features['bb_width'] = 4
            features['bb_position'] = 0.5
        
        # Historical volatility
        if len(closes) >= 21:
            returns = np.diff(closes[-21:]) / closes[-21:-1]
            features['volatility_20'] = np.std(returns) * np.sqrt(252) * 100
        else:
            features['volatility_20'] = 20
        
        if len(closes) >= 51:
            returns = np.diff(closes[-51:]) / closes[-51:-1]
            features['volatility_50'] = np.std(returns) * np.sqrt(252) * 100
        else:
            features['volatility_50'] = features['volatility_20']
        
        features['volatility_ratio'] = features['volatility_20'] / (features['volatility_50'] + 1e-10)
        
        # Intraday range
        features['intraday_range'] = (high - low) / close * 100
        if len(closes) >= 20:
            ranges = (highs[-20:] - lows[-20:]) / closes[-20:] * 100
            features['intraday_range_avg'] = np.mean(ranges)
        else:
            features['intraday_range_avg'] = features['intraday_range']
        
        # Keltner Channels
        ema_20 = np.mean(closes[-20:]) if len(closes) >= 20 else close
        features['keltner_upper'] = ema_20 + (features['atr_14'] * 2)
        features['keltner_lower'] = ema_20 - (features['atr_14'] * 2)
        
        # =====================================================================
        # 4. VOLUME FEATURES
        # =====================================================================
        
        if len(volumes) > 0 and volumes[-1] > 0:
            # Volume SMA
            if len(volumes) >= 20:
                features['volume_sma_20'] = np.mean(volumes[-20:])
            else:
                features['volume_sma_20'] = np.mean(volumes)
            
            features['volume_ratio'] = bar.volume / (features['volume_sma_20'] + 1e-10)
            features['volume_spike'] = 1 if features['volume_ratio'] > 2 else 0
            
            # OBV
            features['obv'] = self.state.obv
            if len(closes) >= 20:
                obv_sma = np.mean([self.state.obv] * 20)  # Simplified
                features['obv_sma'] = obv_sma
            else:
                features['obv_sma'] = self.state.obv
            features['obv_trend'] = 1 if self.state.obv > features['obv_sma'] else -1
            
            # Accumulation/Distribution
            clv = ((close - low) - (high - close)) / (high - low + 1e-10)
            features['ad_line'] = clv * bar.volume
            
            # VWAP (simplified)
            features['vwap'] = close  # Would need proper daily calc
            features['price_vs_vwap'] = 0
            
            # MFI
            features['mfi'] = self._calculate_mfi_fast(highs, lows, closes, volumes)
        else:
            features['volume_sma_20'] = 0
            features['volume_ratio'] = 1
            features['volume_spike'] = 0
            features['obv'] = 0
            features['obv_sma'] = 0
            features['obv_trend'] = 0
            features['ad_line'] = 0
            features['vwap'] = close
            features['price_vs_vwap'] = 0
            features['mfi'] = 50
        
        # =====================================================================
        # 5. SUPPORT/RESISTANCE
        # =====================================================================
        
        if len(closes) >= 2:
            prev_high = highs[-2]
            prev_low = lows[-2]
            prev_close = closes[-2]
            
            features['pivot'] = (prev_high + prev_low + prev_close) / 3
            features['r1'] = 2 * features['pivot'] - prev_low
            features['s1'] = 2 * features['pivot'] - prev_high
            features['r2'] = features['pivot'] + (prev_high - prev_low)
            features['s2'] = features['pivot'] - (prev_high - prev_low)
            features['r3'] = prev_high + 2 * (features['pivot'] - prev_low)
            features['s3'] = prev_low - 2 * (prev_high - features['pivot'])
            
            features['dist_to_r1'] = (features['r1'] - close) / close * 100
            features['dist_to_s1'] = (close - features['s1']) / close * 100
            features['at_resistance'] = 1 if abs(features['dist_to_r1']) < 0.5 else 0
            features['at_support'] = 1 if abs(features['dist_to_s1']) < 0.5 else 0
        else:
            features['pivot'] = close
            features['r1'] = close * 1.01
            features['s1'] = close * 0.99
            features['dist_to_r1'] = 1
            features['dist_to_s1'] = 1
            features['at_resistance'] = 0
            features['at_support'] = 0
        
        # Rolling high/low
        if len(highs) >= 20:
            features['rolling_high_20'] = np.max(highs[-20:])
            features['rolling_low_20'] = np.min(lows[-20:])
            features['position_in_range'] = (close - features['rolling_low_20']) / (features['rolling_high_20'] - features['rolling_low_20'] + 1e-10)
        else:
            features['rolling_high_20'] = high
            features['rolling_low_20'] = low
            features['position_in_range'] = 0.5
        
        if len(highs) >= 50:
            features['rolling_high_50'] = np.max(highs[-50:])
            features['rolling_low_50'] = np.min(lows[-50:])
        else:
            features['rolling_high_50'] = features['rolling_high_20']
            features['rolling_low_50'] = features['rolling_low_20']
        
        # =====================================================================
        # 6. REGIME FEATURES
        # =====================================================================
        
        if bar.vix is not None:
            features['vix'] = bar.vix
            
            vix_values = self.buffer.vix_values
            if len(vix_values) >= 20:
                features['vix_sma_20'] = np.mean(vix_values[-20:])
                features['vix_ratio'] = bar.vix / features['vix_sma_20']
            else:
                features['vix_sma_20'] = bar.vix
                features['vix_ratio'] = 1.0
            
            # VIX percentile
            if self._vix_historical is not None and len(self._vix_historical) > 0:
                features['vix_percentile'] = np.sum(bar.vix > self._vix_historical) / len(self._vix_historical) * 100
            else:
                features['vix_percentile'] = 50
            
            # VIX regime
            vix = bar.vix
            if vix < 15:
                features['vix_regime'] = 0
            elif vix < 20:
                features['vix_regime'] = 1
            elif vix < 25:
                features['vix_regime'] = 2
            elif vix < 35:
                features['vix_regime'] = 3
            else:
                features['vix_regime'] = 4
        else:
            features['vix'] = 18
            features['vix_sma_20'] = 18
            features['vix_ratio'] = 1.0
            features['vix_percentile'] = 50
            features['vix_regime'] = 1
        
        features['trend_regime'] = features['trend_direction']
        features['vol_regime'] = 0
        
        # =====================================================================
        # 7. TIME FEATURES
        # =====================================================================
        
        ts = bar.timestamp
        features['hour'] = ts.hour
        features['minute'] = ts.minute
        features['day_of_week'] = ts.weekday()
        features['day_of_month'] = ts.day
        features['month'] = ts.month
        features['quarter'] = (ts.month - 1) // 3 + 1
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24)
        features['dow_sin'] = np.sin(2 * np.pi * ts.weekday() / 7)
        features['dow_cos'] = np.cos(2 * np.pi * ts.weekday() / 7)
        features['month_sin'] = np.sin(2 * np.pi * ts.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * ts.month / 12)
        
        features['is_monday'] = 1 if ts.weekday() == 0 else 0
        features['is_friday'] = 1 if ts.weekday() == 4 else 0
        features['is_month_start'] = 1 if ts.day <= 3 else 0
        features['is_month_end'] = 1 if ts.day >= 28 else 0
        features['is_market_open'] = 1 if 9 <= ts.hour < 16 else 0
        features['is_power_hour'] = 1 if ts.hour == 15 else 0
        
        # OPEX week (3rd Friday)
        is_opex = 0
        if 15 <= ts.day <= 21 and ts.weekday() == 4:
            is_opex = 1
        elif 12 <= ts.day <= 18:
            is_opex = 1
        features['is_opex_week'] = is_opex
        
        # =====================================================================
        # 8. PRICE ACTION
        # =====================================================================
        
        features['body'] = close - open_price
        features['body_pct'] = features['body'] / open_price * 100
        features['upper_shadow'] = high - max(open_price, close)
        features['lower_shadow'] = min(open_price, close) - low
        features['is_bullish'] = 1 if close > open_price else 0
        
        # Consecutive moves
        if len(closes) >= 2:
            is_up = 1 if close > closes[-2] else 0
            features['consecutive_up'] = is_up  # Simplified
            features['consecutive_down'] = 1 - is_up
        else:
            features['consecutive_up'] = 0
            features['consecutive_down'] = 0
        
        # Gap
        if len(closes) >= 2:
            features['gap'] = open_price - closes[-2]
            features['gap_pct'] = features['gap'] / closes[-2] * 100
            features['gap_up'] = 1 if features['gap_pct'] > 0.5 else 0
            features['gap_down'] = 1 if features['gap_pct'] < -0.5 else 0
        else:
            features['gap'] = 0
            features['gap_pct'] = 0
            features['gap_up'] = 0
            features['consecutive_up'] = 0
            features['consecutive_down'] = 0
            features['gap_down'] = 0
        
        # Returns
        for period in [1, 5, 10, 20]:
            if len(closes) > period:
                features[f'return_{period}'] = (close - closes[-period-1]) / closes[-period-1] * 100
            else:
                features[f'return_{period}'] = 0
        
        # ROC (separate from momentum loop above)
        features['roc_5'] = features.get('roc_5', 0)
        features['roc_10'] = features.get('roc_10', 0)
        features['roc_20'] = features.get('roc_20', 0)
        
        # Drawdown
        if len(closes) >= 20:
            rolling_max = np.max(closes[-252:]) if len(closes) >= 252 else np.max(closes)
            features['drawdown'] = (close - rolling_max) / rolling_max * 100
        else:
            features['drawdown'] = 0
        
        # =====================================================================
        # 9. EARNINGS FEATURES
        # =====================================================================
        
        if self._earnings_dates:
            current_date = ts.date()
            
            # Days to next earnings
            days_to = 90
            for earn_date in self._earnings_dates:
                if earn_date >= current_date:
                    days_to = (earn_date - current_date).days
                    break
            
            # Days since last earnings
            days_since = 90
            for earn_date in reversed(self._earnings_dates):
                if earn_date < current_date:
                    days_since = (current_date - earn_date).days
                    break
            
            features['days_to_earnings'] = days_to
            features['days_since_earnings'] = days_since
            features['earnings_proximity'] = max(0, 1 - abs(days_to) / 30)
            features['is_earnings_week'] = 1 if 0 <= days_to <= 5 else 0
            features['is_earnings_day'] = 1 if days_to == 0 else 0
        else:
            features['days_to_earnings'] = 90
            features['days_since_earnings'] = 90
            features['earnings_proximity'] = 0
            features['is_earnings_week'] = 0
            features['is_earnings_day'] = 0
        
        # =====================================================================
        # 10. COMPOSITE FEATURES
        # =====================================================================
        
        # Trend-Momentum Score
        trend_score = (
            features['trend_direction'] * 0.3 +
            features['macd_cross'] * 0.2 +
            (features['rsi_14'] - 50) / 50 * 0.2 +
            min(features['adx'], 100) / 100 * 0.3
        )
        features['trend_momentum_score'] = trend_score
        
        # Vol-adjusted return
        if features.get('atr_percent', 0) > 0:
            features['vol_adjusted_return'] = features.get('return_1', 0) / features['atr_percent']
        else:
            features['vol_adjusted_return'] = 0
        
        # Composite regime
        features['composite_regime'] = (
            features['vix_regime'] * 0.4 +
            features['trend_regime'] * 0.3 +
            features['vol_regime'] * 0.3
        )
        
        return features
    
    # =========================================================================
    # FAST CALCULATION HELPERS
    # =========================================================================
    
    def _calculate_rsi_fast(self, closes: np.ndarray, period: int) -> float:
        """Fast RSI calculation."""
        if len(closes) < period + 1:
            return 50
        
        deltas = np.diff(closes[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr_fast(self, highs: np.ndarray, lows: np.ndarray, 
                           closes: np.ndarray, period: int) -> float:
        """Fast ATR calculation."""
        if len(closes) < period + 1:
            return (highs[-1] - lows[-1]) if len(highs) > 0 else 0
        
        tr_list = []
        for i in range(-period, 0):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr_list.append(max(tr1, tr2, tr3))
        
        return np.mean(tr_list)
    
    def _calculate_adx_fast(self, highs: np.ndarray, lows: np.ndarray,
                           closes: np.ndarray, period: int = 14) -> float:
        """Fast ADX approximation."""
        if len(closes) < period + 1:
            return 25  # Default neutral
        
        # Simplified ADX
        price_range = np.max(highs[-period:]) - np.min(lows[-period:])
        volatility = np.std(closes[-period:])
        
        if volatility == 0:
            return 25
        
        # ADX approximation based on range/volatility
        adx_approx = min(100, (price_range / closes[-1] * 100) * 10)
        return adx_approx
    
    def _calculate_adx_full(self, highs: np.ndarray, lows: np.ndarray,
                           closes: np.ndarray, period: int = 14) -> tuple:
        """Calculate ADX with +DI and -DI for full consistency."""
        if len(closes) < period + 1:
            return 25, 50, 50  # Default neutral values
        
        # True Range
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []
        
        for i in range(-period, 0):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr_list.append(max(tr1, tr2, tr3))
            
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            plus_dm = up_move if (up_move > down_move and up_move > 0) else 0
            minus_dm = down_move if (down_move > up_move and down_move > 0) else 0
            
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)
        
        atr = np.mean(tr_list)
        plus_di = 100 * np.mean(plus_dm_list) / (atr + 1e-10)
        minus_di = 100 * np.mean(minus_dm_list) / (atr + 1e-10)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx  # Simplified - would need smoothing for true ADX
        
        return adx, plus_di, minus_di
    
    def _calculate_cci_fast(self, highs: np.ndarray, lows: np.ndarray,
                           closes: np.ndarray, period: int = 20) -> float:
        """Fast CCI calculation."""
        if len(closes) < period:
            return 0
        
        typical_prices = (highs[-period:] + lows[-period:] + closes[-period:]) / 3
        sma = np.mean(typical_prices)
        mad = np.mean(np.abs(typical_prices - sma))
        
        if mad == 0:
            return 0
        
        return (typical_prices[-1] - sma) / (0.015 * mad)
    
    def _calculate_mfi_fast(self, highs: np.ndarray, lows: np.ndarray,
                           closes: np.ndarray, volumes: np.ndarray,
                           period: int = 14) -> float:
        """Fast MFI calculation."""
        if len(closes) < period + 1:
            return 50
        
        typical_prices = (highs[-period-1:] + lows[-period-1:] + closes[-period-1:]) / 3
        raw_money_flow = typical_prices * volumes[-period-1:]
        
        positive_flow = 0
        negative_flow = 0
        
        for i in range(1, len(typical_prices)):
            if typical_prices[i] > typical_prices[i-1]:
                positive_flow += raw_money_flow[i]
            else:
                negative_flow += raw_money_flow[i]
        
        if negative_flow == 0:
            return 100
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    # =========================================================================
    # STATUS & DEBUGGING
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get calculator status."""
        return {
            'symbol': self.symbol,
            'bars_in_buffer': len(self.buffer),
            'bars_processed': self.state.bars_processed,
            'is_warmed_up': len(self.buffer) >= self.MIN_HISTORY,
            'warmup_progress': f"{len(self.buffer)}/{self.MIN_HISTORY}",
            'has_vix_history': self._vix_historical is not None,
            'has_earnings': len(self._earnings_dates) > 0,
        }
    
    def is_ready(self) -> bool:
        """Check if calculator is warmed up and ready."""
        return len(self.buffer) >= self.MIN_HISTORY


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_live_calculator(symbol: str,
                           preload_days: int = 300) -> LiveFeatureCalculator:
    """
    Vytvoří LiveFeatureCalculator s preloaded historií.
    
    Args:
        symbol: Ticker symbol
        preload_days: Kolik dní historie načíst
        
    Returns:
        Inicializovaný calculator
    """
    logger.info(f"Creating LiveFeatureCalculator for {symbol}")
    
    # Load historical data
    from pathlib import Path
    data_dir = Path("data/enriched")
    
    history_df = None
    
    # Try to load from parquet
    parquet_file = data_dir / f"{symbol}_1min_vanna.parquet"
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
        # Get last N days
        if 'timestamp' in df.columns:
            cutoff = datetime.now() - timedelta(days=preload_days)
            df = df[df['timestamp'] >= cutoff]
        history_df = df.tail(500)  # Max 500 bars
        logger.info(f"Loaded {len(history_df)} historical bars from {parquet_file}")
    
    # Create calculator
    calc = LiveFeatureCalculator(
        symbol=symbol,
        preload_history=history_df
    )
    
    # Load earnings calendar
    try:
        from earnings_features import EarningsCalendar
        calendar = EarningsCalendar()
        earnings = calendar.get_earnings_history(symbol)
        calc.set_earnings_calendar([e.date for e in earnings])
        logger.info(f"Loaded {len(earnings)} earnings dates")
    except Exception as e:
        logger.warning(f"Could not load earnings calendar: {e}")
    
    return calc


# =============================================================================
# CLI / TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("LIVE FEATURE CALCULATOR TEST")
    print("=" * 60)
    
    # Create calculator
    calc = LiveFeatureCalculator(symbol='SPY')
    
    # Show warmup status BEFORE warmup
    print("\n📊 WARMUP STATUS (before):")
    status = calc.get_warmup_status()
    print(f"   Ready: {status['is_ready']}")
    print(f"   Progress: {status['progress_percent']:.0f}%")
    print(f"   Warnings:")
    for w in status['warnings']:
        print(f"      {w}")
    
    # Simulate warmup with random data
    import random
    
    price = 590.0
    
    print("\n🔄 Warming up with 250 bars...")
    for i in range(250):
        price += random.gauss(0, 0.5)
        bar = Bar(
            timestamp=datetime.now() - timedelta(minutes=250-i),
            open=price - random.uniform(0, 0.3),
            high=price + random.uniform(0, 0.5),
            low=price - random.uniform(0, 0.5),
            close=price,
            volume=random.randint(100000, 500000),
            vix=18 + random.gauss(0, 1)
        )
        calc.update(bar, return_features=(i >= 249))
    
    # Show warmup status AFTER warmup
    print("\n📊 WARMUP STATUS (after):")
    status = calc.get_warmup_status()
    print(f"   Ready: {status['is_ready']} ✅")
    print(f"   Progress: {status['progress_percent']:.0f}%")
    print(f"   Bars: {status['bars_in_buffer']}/{status['bars_needed']}")
    print("\n   Feature availability:")
    for feat, available in status['features_available'].items():
        emoji = "✅" if available else "❌"
        print(f"      {emoji} {feat}")
    
    # Get features for last bar
    features = calc.update(bar)
    
    if features:
        print(f"\n📈 FEATURES CALCULATED: {len(features)}")
        print("\nSample features:")
        sample_keys = ['sma_200', 'price_vs_sma200', 'rsi_14', 'macd', 
                       'atr_percent', 'bb_position', 'vix_regime', 'trend_direction']
        for key in sample_keys:
            val = features.get(key, 'N/A')
            if isinstance(val, float):
                print(f"   {key}: {val:.4f}")
            else:
                print(f"   {key}: {val}")
    
    # ==========================================================================
    # USAGE EXAMPLES
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("📚 USAGE EXAMPLES")
    print("=" * 60)
    
    print("""
# ═══════════════════════════════════════════════════════════════════════
# PŘÍKLAD 1: Warmup z IBKR (DOPORUČENO)
# ═══════════════════════════════════════════════════════════════════════

from ib_insync import IB
from live_features import LiveFeatureCalculator

# Connect to IBKR
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Create calculator and warmup
calc = LiveFeatureCalculator(symbol='SPY')
calc.warmup_from_ibkr(ib, bars_needed=300)

# Check if ready
if not calc.is_ready():
    print("NOT READY - waiting for more data!")
    
# Start live trading
def on_bar_update(bar):
    features = calc.update(bar)
    if features:
        prediction = model.predict(features)
        # ... trade logic ...

# ═══════════════════════════════════════════════════════════════════════
# PŘÍKLAD 2: Warmup z parquet (offline start)
# ═══════════════════════════════════════════════════════════════════════

calc = LiveFeatureCalculator(symbol='AAPL')
calc.warmup_from_parquet()  # Auto-detects file

print(calc.get_warmup_status())

# ═══════════════════════════════════════════════════════════════════════
# PŘÍKLAD 3: Warmup z DataFrame
# ═══════════════════════════════════════════════════════════════════════

import pandas as pd

df = pd.read_parquet('data/enriched/SPY_1min.parquet')
df = df.tail(500)  # Last 500 bars

calc = LiveFeatureCalculator(symbol='SPY')
calc.warmup_from_df(df)
""")
