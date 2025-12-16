"""
Live Feature Builder

Single Source of Truth for constructing features during live trading.
Ensures consistency with TradingEnv.MARKET_FEATURES used in RL training.

Features are built in EXACTLY the same order as TradingEnv expects.
"""
from typing import Dict, Optional, List, Any
from datetime import datetime, date
import math
import numpy as np

from core.logger import get_logger

logger = get_logger()


class LiveFeatureBuilder:
    """
    Builds live features matching TradingEnv exactly.
    
    Single Source of Truth: Import MARKET_FEATURES from TradingEnv
    to ensure consistency between training and inference.
    """
    
    # Import feature list from TradingEnv - Single Source of Truth
    # These MUST match rl/trading_env.py MARKET_FEATURES exactly!
    MARKET_FEATURES = [
        # Time (6)
        'sin_time', 'cos_time', 'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy',
        # VIX (8)
        'vix_ratio', 'vix_in_contango', 'vix_change_1d', 'vix_change_5d',
        'vix_percentile', 'vix_zscore', 'vix_norm', 'vix3m_norm',
        # Regime (1)
        'regime',
        # Options (3)
        'options_iv_atm', 'options_put_call_ratio', 'options_volume_norm',
        # Price (5)
        'return_1m', 'return_5m', 'volatility_20', 'momentum_20', 'range_pct',
        # Greeks (7)
        'delta', 'gamma', 'theta', 'vega', 'vanna', 'charm', 'volga',
        # ML Regime outputs (4)
        'regime_ml', 'regime_adj_position', 'regime_adj_delta', 'regime_adj_dte',
        # ML DTE outputs (2)
        'dte_confidence', 'optimal_dte_norm',
        # ML Trade outputs (1)
        'trade_prob',
        # Binary signals (5)
        'signal_high_prob', 'signal_low_vol', 'signal_crisis',
        'signal_contango', 'signal_backwardation',
        # Major event features (4)
        'days_to_major_event', 'is_event_week', 'is_event_day', 'event_iv_boost',
        # Daily features (21)
        'day_sma_200', 'day_sma_50', 'day_sma_20',
        'day_price_vs_sma200', 'day_price_vs_sma50',
        'day_rsi_14',
        'day_atr_14', 'day_atr_pct',
        'day_bb_position', 'day_bb_upper', 'day_bb_lower',  # Bollinger Bands
        'day_macd', 'day_macd_signal', 'day_macd_hist',  # MACD complete
        'day_above_sma200', 'day_above_sma50',
        'day_sma_50_200_ratio',
        'day_days_to_major_event', 'day_is_event_week', 'day_is_event_day', 'day_event_iv_boost',
    ]
    
    POSITION_FEATURES = [
        'pnl_pct', 'days_held', 'position_flag', 'capital_ratio',
        'trade_count', 'bid_ask_spread', 'market_open'
    ]
    
    N_MARKET_FEATURES = 67
    N_POSITION_FEATURES = 7
    N_FEATURES = 74
    
    def __init__(self):
        # History tracking for derived features
        self._vix_history: List[float] = []
        self._price_history: Dict[str, List[float]] = {}
        
        # Lazy-loaded components
        self._regime_classifier = None
        self._dte_optimizer = None
        self._trade_predictor = None
        self._vanna_calculator = None
        self._events_calculator = None
        
        logger.info(f"LiveFeatureBuilder initialized (expecting {self.N_FEATURES} features)")
    
    def _lazy_init_components(self):
        """Lazy initialize ML components."""
        if self._regime_classifier is None:
            from ml.regime_classifier import get_regime_classifier
            self._regime_classifier = get_regime_classifier()
        
        if self._dte_optimizer is None:
            from ml.dte_optimizer import get_dte_optimizer
            self._dte_optimizer = get_dte_optimizer()
        
        if self._trade_predictor is None:
            from ml.trade_success_predictor import get_trade_success_predictor
            self._trade_predictor = get_trade_success_predictor()
        
        if self._vanna_calculator is None:
            from ml.vanna_calculator import get_vanna_calculator
            self._vanna_calculator = get_vanna_calculator()
        
        if self._events_calculator is None:
            try:
                from ml.earnings_data_fetcher import get_major_events_calculator
                self._events_calculator = get_major_events_calculator()
            except Exception as e:
                logger.warning(f"Could not load MajorEventsCalculator: {e}")
    
    def build_market_features(
        self,
        symbol: str,
        price: float,
        vix: float,
        vix3m: Optional[float] = None,
        quote: Optional[Dict] = None,
        options_data: Optional[Dict] = None,
        daily_features: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Build 63 market features matching TradingEnv.MARKET_FEATURES exactly.
        
        Args:
            symbol: Stock symbol
            price: Current price
            vix: Current VIX
            vix3m: VIX3M (optional, estimated if not provided)
            quote: Quote dict with high, low, etc.
            options_data: Dict with put_call_ratio, volume, iv_atm
            daily_features: Dict with daily technicals (SMA, RSI, etc.)
            
        Returns:
            Dict with exactly 63 market features in correct order
        """
        self._lazy_init_components()
        
        now = datetime.now()
        quote = quote or {}
        options_data = options_data or {}
        daily_features = daily_features or {}
        
        # Estimate VIX3M if not provided
        if vix3m is None:
            vix3m = vix * 1.05  # Typical slight contango
        
        # ================================================================
        # TIME FEATURES (6)
        # ================================================================
        minutes_of_day = now.hour * 60 + now.minute
        day_of_week = now.weekday()
        day_of_year = now.timetuple().tm_yday
        
        sin_time = math.sin(2 * math.pi * minutes_of_day / 1440)
        cos_time = math.cos(2 * math.pi * minutes_of_day / 1440)
        sin_dow = math.sin(2 * math.pi * day_of_week / 7)
        cos_dow = math.cos(2 * math.pi * day_of_week / 7)
        sin_doy = math.sin(2 * math.pi * day_of_year / 365)
        cos_doy = math.cos(2 * math.pi * day_of_year / 365)
        
        # ================================================================
        # VIX FEATURES (8)
        # ================================================================
        vix_norm = vix / 100.0
        vix3m_norm = vix3m / 100.0
        vix_ratio = vix / vix3m if vix3m > 0 else 1.0
        vix_in_contango = 1 if vix_ratio < 1 else 0
        vix_percentile = min(vix / 50.0, 1.0)
        vix_zscore = (vix - 20) / 10
        
        # VIX changes from history
        vix_change_1d = 0.0
        vix_change_5d = 0.0
        if len(self._vix_history) >= 1:
            vix_change_1d = (vix - self._vix_history[-1]) / self._vix_history[-1] if self._vix_history[-1] > 0 else 0
        if len(self._vix_history) >= 5:
            vix_change_5d = (vix - self._vix_history[-5]) / self._vix_history[-5] if self._vix_history[-5] > 0 else 0
        
        # Update VIX history
        self._vix_history.append(vix)
        if len(self._vix_history) > 10:
            self._vix_history = self._vix_history[-10:]
        
        # ================================================================
        # REGIME (1) - Using RegimeClassifier
        # ================================================================
        regime_result = self._regime_classifier.classify_by_vix(vix)
        regime = regime_result.regime
        
        # ================================================================
        # OPTIONS DATA (3) - From IBKR options chain or estimates
        # ================================================================
        # Use real data if available, otherwise estimate
        options_iv_atm = options_data.get('iv_atm', vix_norm)
        options_put_call_ratio = options_data.get('put_call_ratio', self._estimate_put_call_ratio(vix))
        options_volume_norm = options_data.get('volume_norm', 0.5)
        
        # ================================================================
        # PRICE FEATURES (5)
        # ================================================================
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        
        prices = self._price_history[symbol]
        return_1m = (price - prices[-1]) / prices[-1] if len(prices) >= 1 and prices[-1] > 0 else 0.0
        return_5m = (price - prices[-5]) / prices[-5] if len(prices) >= 5 and prices[-5] > 0 else 0.0
        
        # Volatility and momentum
        if len(prices) >= 20:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] > 0]
            volatility_20 = float(np.std(returns[-20:])) if len(returns) >= 20 else vix / 100 * 0.05
            momentum_20 = (price - prices[-20]) / prices[-20] if prices[-20] > 0 else 0.0
        else:
            volatility_20 = vix / 100 * 0.05
            momentum_20 = 0.0
        
        range_pct = (quote.get('high', price) - quote.get('low', price)) / price if price > 0 else 0.02
        
        # Update price history
        prices.append(price)
        if len(prices) > 30:
            self._price_history[symbol] = prices[-30:]
        
        # ================================================================
        # GREEKS (7) - Using VannaCalculator
        # ================================================================
        try:
            S = price
            K = price  # ATM
            T = 30 / 365  # ~30 DTE
            sigma = vix / 100
            
            greeks = self._vanna_calculator.calculate_all_greeks(
                S=S, K=K, T=T, sigma=sigma, option_type='put'
            )
            
            delta = greeks.delta
            gamma = greeks.gamma
            theta = greeks.theta
            vega = greeks.vega
            vanna = greeks.vanna
            charm = greeks.charm
            volga = greeks.volga
            
        except Exception as e:
            logger.debug(f"Greeks calculation failed: {e}")
            delta, gamma, theta, vega = -0.5, 0.02, -0.03, 0.15
            vanna, charm, volga = 0.01, 0.001, 0.05
        
        # ================================================================
        # ML REGIME OUTPUTS (4) - Using RegimeClassifier
        # ================================================================
        regime_ml = regime
        adj = self._regime_classifier.get_strategy_adjustment(regime)
        regime_adj_position = adj.get('position_size', 1.0)
        regime_adj_delta = adj.get('delta_target', 1.0)
        regime_adj_dte = adj.get('dte_adjustment', 1.0)
        
        # ================================================================
        # ML DTE OUTPUTS (2) - Using DTEOptimizer
        # ================================================================
        dte_result = self._dte_optimizer.get_optimal_dte(vix, vix3m)
        dte_confidence = dte_result.confidence
        optimal_dte_norm = dte_result.dte / 60.0  # Normalize to 0-1
        
        # ================================================================
        # ML TRADE OUTPUTS (1) - Using TradeSuccessPredictor
        # ================================================================
        try:
            trade_features = {
                'vix': vix,
                'vix_ratio': vix_ratio,
                'regime': regime_ml,
                'sin_time': sin_time,
                'cos_time': cos_time,
                'delta': delta,
                'vanna': vanna,
                'volga': volga,
            }
            trade_prob = self._trade_predictor.predict(trade_features)
        except Exception as e:
            logger.debug(f"Trade prediction failed: {e}")
            trade_prob = 0.5
        
        # ================================================================
        # BINARY SIGNALS (5)
        # ================================================================
        signal_high_prob = 1 if trade_prob > 0.6 else 0
        signal_low_vol = 1 if vix < 15 else 0
        signal_crisis = 1 if vix > 30 else 0
        signal_contango = vix_in_contango
        signal_backwardation = 1 - vix_in_contango
        
        # ================================================================
        # MAJOR EVENT FEATURES (4) - Using MajorEventsCalculator
        # ================================================================
        event_features = self._get_event_features(symbol)
        days_to_major_event = event_features['days_to_major_event']
        is_event_week = event_features['is_event_week']
        is_event_day = event_features['is_event_day']
        event_iv_boost = event_features['event_iv_boost']
        
        # ================================================================
        # DAILY FEATURES (17) - From daily_features dict
        # ================================================================
        day_sma_200 = daily_features.get('day_sma_200', 1.0)
        day_sma_50 = daily_features.get('day_sma_50', 1.0)
        day_sma_20 = daily_features.get('day_sma_20', 1.0)
        day_price_vs_sma200 = daily_features.get('day_price_vs_sma200', 0.0)
        day_price_vs_sma50 = daily_features.get('day_price_vs_sma50', 0.0)
        day_rsi_14 = daily_features.get('day_rsi_14', 0.5)
        day_atr_14 = daily_features.get('day_atr_14', 0.02)
        day_atr_pct = daily_features.get('day_atr_pct', 0.02)
        day_bb_position = daily_features.get('day_bb_position', 0.5)
        day_bb_upper = daily_features.get('day_bb_upper', price * 1.02)  # 2% above price
        day_bb_lower = daily_features.get('day_bb_lower', price * 0.98)  # 2% below price
        day_macd = daily_features.get('day_macd', 0.0)
        day_macd_signal = daily_features.get('day_macd_signal', 0.0)
        day_macd_hist = daily_features.get('day_macd_hist', 0.0)
        day_above_sma200 = daily_features.get('day_above_sma200', 1)
        day_above_sma50 = daily_features.get('day_above_sma50', 1)
        day_sma_50_200_ratio = daily_features.get('day_sma_50_200_ratio', 1.0)
        day_days_to_major_event = days_to_major_event
        day_is_event_week = is_event_week
        day_is_event_day = daily_features.get('day_is_event_day', is_event_day)
        day_event_iv_boost = event_iv_boost
        
        # ================================================================
        # BUILD FEATURE DICT IN CORRECT ORDER
        # ================================================================
        features = {
            # Time (6)
            'sin_time': sin_time,
            'cos_time': cos_time,
            'sin_dow': sin_dow,
            'cos_dow': cos_dow,
            'sin_doy': sin_doy,
            'cos_doy': cos_doy,
            # VIX (8)
            'vix_ratio': vix_ratio,
            'vix_in_contango': vix_in_contango,
            'vix_change_1d': vix_change_1d,
            'vix_change_5d': vix_change_5d,
            'vix_percentile': vix_percentile,
            'vix_zscore': vix_zscore,
            'vix_norm': vix_norm,
            'vix3m_norm': vix3m_norm,
            # Regime (1)
            'regime': regime,
            # Options (3)
            'options_iv_atm': options_iv_atm,
            'options_put_call_ratio': options_put_call_ratio,
            'options_volume_norm': options_volume_norm,
            # Price (5)
            'return_1m': return_1m,
            'return_5m': return_5m,
            'volatility_20': volatility_20,
            'momentum_20': momentum_20,
            'range_pct': range_pct,
            # Greeks (7)
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'vanna': vanna,
            'charm': charm,
            'volga': volga,
            # ML Regime (4)
            'regime_ml': regime_ml,
            'regime_adj_position': regime_adj_position,
            'regime_adj_delta': regime_adj_delta,
            'regime_adj_dte': regime_adj_dte,
            # ML DTE (2)
            'dte_confidence': dte_confidence,
            'optimal_dte_norm': optimal_dte_norm,
            # ML Trade (1)
            'trade_prob': trade_prob,
            # Binary signals (5)
            'signal_high_prob': signal_high_prob,
            'signal_low_vol': signal_low_vol,
            'signal_crisis': signal_crisis,
            'signal_contango': signal_contango,
            'signal_backwardation': signal_backwardation,
            # Events (4)
            'days_to_major_event': days_to_major_event,
            'is_event_week': is_event_week,
            'is_event_day': is_event_day,
            'event_iv_boost': event_iv_boost,
            # Daily (21)
            'day_sma_200': day_sma_200,
            'day_sma_50': day_sma_50,
            'day_sma_20': day_sma_20,
            'day_price_vs_sma200': day_price_vs_sma200,
            'day_price_vs_sma50': day_price_vs_sma50,
            'day_rsi_14': day_rsi_14,
            'day_atr_14': day_atr_14,
            'day_atr_pct': day_atr_pct,
            'day_bb_position': day_bb_position,
            'day_bb_upper': day_bb_upper,
            'day_bb_lower': day_bb_lower,
            'day_macd': day_macd,
            'day_macd_signal': day_macd_signal,
            'day_macd_hist': day_macd_hist,
            'day_above_sma200': day_above_sma200,
            'day_above_sma50': day_above_sma50,
            'day_sma_50_200_ratio': day_sma_50_200_ratio,
            'day_days_to_major_event': day_days_to_major_event,
            'day_is_event_week': day_is_event_week,
            'day_is_event_day': day_is_event_day,
            'day_event_iv_boost': day_event_iv_boost,
        }
        
        # Validate feature count
        assert len(features) == self.N_MARKET_FEATURES, \
            f"Expected {self.N_MARKET_FEATURES} market features, got {len(features)}"
        
        return features
    
    def build_position_features(
        self,
        has_position: bool,
        pnl_pct: float = 0.0,
        days_held: float = 0.0,
        capital_ratio: float = 1.0,
        trade_count: int = 0,
        bid_ask_spread: float = 0.02,
        market_open: bool = True,
    ) -> Dict[str, float]:
        """
        Build 7 position features.
        
        Returns:
            Dict with exactly 7 position features in correct order
        """
        features = {
            'pnl_pct': pnl_pct,
            'days_held': days_held,
            'position_flag': 1.0 if has_position else 0.0,
            'capital_ratio': capital_ratio,
            'trade_count': min(trade_count / 10, 1.0),
            'bid_ask_spread': bid_ask_spread,
            'market_open': 1.0 if market_open else 0.0,
        }
        
        assert len(features) == self.N_POSITION_FEATURES, \
            f"Expected {self.N_POSITION_FEATURES} position features, got {len(features)}"
        
        return features
    
    def build_all_features(
        self,
        symbol: str,
        price: float,
        vix: float,
        vix3m: Optional[float] = None,
        quote: Optional[Dict] = None,
        options_data: Optional[Dict] = None,
        daily_features: Optional[Dict] = None,
        has_position: bool = False,
        pnl_pct: float = 0.0,
        days_held: float = 0.0,
        capital_ratio: float = 1.0,
        trade_count: int = 0,
        bid_ask_spread: float = 0.02,
        market_open: bool = True,
    ) -> Dict[str, float]:
        """
        Build all 70 features (63 market + 7 position).
        
        Returns:
            Dict with exactly 70 features in correct order
        """
        market = self.build_market_features(
            symbol=symbol,
            price=price,
            vix=vix,
            vix3m=vix3m,
            quote=quote,
            options_data=options_data,
            daily_features=daily_features,
        )
        
        position = self.build_position_features(
            has_position=has_position,
            pnl_pct=pnl_pct,
            days_held=days_held,
            capital_ratio=capital_ratio,
            trade_count=trade_count,
            bid_ask_spread=bid_ask_spread,
            market_open=market_open,
        )
        
        # Combine in correct order
        features = {**market, **position}
        
        assert len(features) == self.N_FEATURES, \
            f"Expected {self.N_FEATURES} total features, got {len(features)}"
        
        return features
    
    def _get_event_features(self, symbol: str) -> Dict[str, float]:
        """Get major event features using MajorEventsCalculator."""
        defaults = {
            'days_to_major_event': 30,
            'is_event_week': 0,
            'is_event_day': 0,
            'event_iv_boost': 1.0,
        }
        
        if self._events_calculator is None:
            return defaults
        
        try:
            today = date.today()
            days, event_type = self._events_calculator._find_next_event(today, symbol)
            
            is_event_week = 1 if days <= 7 else 0
            is_event_day = 1 if days <= 1 else 0
            
            # IV boost based on proximity
            if days <= 1:
                event_iv_boost = 2.0
            elif days <= 3:
                event_iv_boost = 1.5
            elif days <= 7:
                event_iv_boost = 1.3
            else:
                event_iv_boost = 1.0
            
            return {
                'days_to_major_event': min(days, 90),
                'is_event_week': is_event_week,
                'is_event_day': is_event_day,
                'event_iv_boost': event_iv_boost,
            }
            
        except Exception as e:
            logger.debug(f"Event features calculation failed: {e}")
            return defaults
    
    def _estimate_put_call_ratio(self, vix: float) -> float:
        """
        Estimate put/call ratio from VIX.
        
        Higher VIX = more puts traded (fear) → higher ratio
        Lower VIX = more calls traded (greed) → lower ratio
        
        Typical range: 0.6-1.2
        """
        # Base ratio around 0.8
        base = 0.8
        
        # Adjust based on VIX
        if vix < 15:
            # Low fear, more calls
            return base - 0.1
        elif vix < 20:
            return base
        elif vix < 25:
            return base + 0.1
        elif vix < 30:
            return base + 0.2
        else:
            # High fear, way more puts
            return base + 0.4
    
    def validate_features(self, features: Dict[str, float]) -> bool:
        """
        Validate that features match expected format.
        
        Returns:
            True if valid, False otherwise
        """
        expected = self.MARKET_FEATURES + self.POSITION_FEATURES
        
        if len(features) != len(expected):
            logger.error(f"Feature count mismatch: {len(features)} vs {len(expected)}")
            return False
        
        for i, name in enumerate(expected):
            if name not in features:
                logger.error(f"Missing feature: {name}")
                return False
        
        return True


# Singleton
_builder: Optional[LiveFeatureBuilder] = None


def get_live_feature_builder() -> LiveFeatureBuilder:
    """Get or create live feature builder singleton."""
    global _builder
    if _builder is None:
        _builder = LiveFeatureBuilder()
    return _builder
