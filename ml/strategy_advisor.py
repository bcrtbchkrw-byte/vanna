"""
Strategy Advisor - XGBoost-based Strategy Selection

Part of Multi-Model Trading Architecture.
Recommends optimal strategy based on market conditions.

Outputs:
- regime: Market regime (0-4)
- strategy: Recommended strategy (CALL, PUT, SPREAD, etc.)
- side: BUY or SELL recommendation
- confidence: Prediction confidence (0-1)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.logger import get_logger

logger = get_logger()


class MarketRegime(Enum):
    """Market regime classification."""
    CALM = 0        # VIX < 15, low vol, trend market
    NORMAL = 1      # VIX 15-20, normal conditions
    ELEVATED = 2    # VIX 20-25, increased uncertainty
    HIGH_VOL = 3    # VIX 25-35, high volatility
    CRISIS = 4      # VIX > 35, market panic


class Strategy(Enum):
    """Available trading strategies."""
    CASH = "CASH"               # No trade, wait
    BUY_CALL = "BUY_CALL"       # Long call
    BUY_PUT = "BUY_PUT"         # Long put
    SELL_PUT = "SELL_PUT"       # Short put (theta harvest)
    SELL_CALL = "SELL_CALL"     # Covered call / short call
    CALL_SPREAD = "CALL_SPREAD" # Bull call spread
    PUT_SPREAD = "PUT_SPREAD"   # Bear put spread
    IRON_CONDOR = "IRON_CONDOR" # Neutral, sell premium
    STRADDLE = "STRADDLE"       # Long volatility


@dataclass
class StrategyRecommendation:
    """Strategy advisor output."""
    regime: MarketRegime
    strategy: Strategy
    side: str  # BUY or SELL 
    dte_bucket: int  # 0=0DTE, 1=WEEKLY, 2=MONTHLY, 3=LEAPS
    win_probability: float
    confidence: float
    reasoning: str


class StrategyAdvisor:
    """
    XGBoost-enhanced strategy selection.
    
    Uses combination of:
    1. Rule-based regime detection
    2. XGBoost win probability
    3. Market condition analysis
    """
    
    # Strategy selection matrix based on regime and trend
    STRATEGY_MATRIX = {
        # (regime, trend) -> [(strategy, weight), ...]
        (MarketRegime.CALM, "BULLISH"): [
            (Strategy.BUY_CALL, 0.4),
            (Strategy.SELL_PUT, 0.4),
            (Strategy.CALL_SPREAD, 0.2),
        ],
        (MarketRegime.CALM, "BEARISH"): [
            (Strategy.BUY_PUT, 0.3),
            (Strategy.PUT_SPREAD, 0.3),
            (Strategy.CASH, 0.4),  # Low vol = bad for puts
        ],
        (MarketRegime.CALM, "NEUTRAL"): [
            (Strategy.IRON_CONDOR, 0.5),
            (Strategy.SELL_PUT, 0.3),
            (Strategy.CASH, 0.2),
        ],
        (MarketRegime.NORMAL, "BULLISH"): [
            (Strategy.BUY_CALL, 0.3),
            (Strategy.CALL_SPREAD, 0.4),
            (Strategy.SELL_PUT, 0.3),
        ],
        (MarketRegime.NORMAL, "BEARISH"): [
            (Strategy.BUY_PUT, 0.4),
            (Strategy.PUT_SPREAD, 0.4),
            (Strategy.CASH, 0.2),
        ],
        (MarketRegime.NORMAL, "NEUTRAL"): [
            (Strategy.IRON_CONDOR, 0.4),
            (Strategy.SELL_PUT, 0.3),
            (Strategy.CASH, 0.3),
        ],
        (MarketRegime.ELEVATED, "BULLISH"): [
            (Strategy.CALL_SPREAD, 0.5),  # Defined risk
            (Strategy.BUY_CALL, 0.2),
            (Strategy.CASH, 0.3),
        ],
        (MarketRegime.ELEVATED, "BEARISH"): [
            (Strategy.PUT_SPREAD, 0.4),
            (Strategy.BUY_PUT, 0.3),
            (Strategy.CASH, 0.3),
        ],
        (MarketRegime.ELEVATED, "NEUTRAL"): [
            (Strategy.CASH, 0.5),
            (Strategy.IRON_CONDOR, 0.3),
            (Strategy.PUT_SPREAD, 0.2),
        ],
        (MarketRegime.HIGH_VOL, "BULLISH"): [
            (Strategy.CALL_SPREAD, 0.4),
            (Strategy.CASH, 0.4),
            (Strategy.BUY_CALL, 0.2),
        ],
        (MarketRegime.HIGH_VOL, "BEARISH"): [
            (Strategy.BUY_PUT, 0.4),
            (Strategy.PUT_SPREAD, 0.3),
            (Strategy.CASH, 0.3),
        ],
        (MarketRegime.HIGH_VOL, "NEUTRAL"): [
            (Strategy.STRADDLE, 0.3),  # Buy vol when high
            (Strategy.CASH, 0.5),
            (Strategy.PUT_SPREAD, 0.2),
        ],
        (MarketRegime.CRISIS, "BULLISH"): [
            (Strategy.CASH, 0.7),
            (Strategy.CALL_SPREAD, 0.3),
        ],
        (MarketRegime.CRISIS, "BEARISH"): [
            (Strategy.BUY_PUT, 0.5),  # Capitalize on panic
            (Strategy.CASH, 0.5),
        ],
        (MarketRegime.CRISIS, "NEUTRAL"): [
            (Strategy.CASH, 0.8),
            (Strategy.BUY_PUT, 0.2),  # Hedge
        ],
    }
    
    # DTE bucket selection based on regime
    # 0=0DTE (0), 1=WEEKLY (1-14), 2=MONTHLY (15-45), 3=LEAPS (46+)
    DTE_RULES = {
        MarketRegime.CALM: 1,     # WEEKLY - faster theta in calm
        MarketRegime.NORMAL: 2,   # MONTHLY - 80% WR sweet spot
        MarketRegime.ELEVATED: 2, # MONTHLY - need time buffer
        MarketRegime.HIGH_VOL: 2, # MONTHLY - avoid gamma risk
        MarketRegime.CRISIS: 3,   # LEAPS - maximum safety
    }
    
    def __init__(self, xgb_predictor=None, load_lstm: bool = False):
        """
        Initialize advisor.
        
        Args:
            xgb_predictor: Optional TradeSuccessPredictor for win probability
            load_lstm: Whether to load LSTM regime model (optional)
        """
        self.xgb_predictor = xgb_predictor
        self._lstm_model = None
        
        # Try to load XGBoost if not provided
        if self.xgb_predictor is None:
            try:
                from ml.trade_success_predictor import get_trade_success_predictor
                self.xgb_predictor = get_trade_success_predictor()
                logger.info("StrategyAdvisor loaded XGBoost predictor")
            except Exception as e:
                logger.warning(f"Could not load XGBoost: {e}")
        
        # Optionally load LSTM regime model
        if load_lstm:
            self._load_lstm_model()
    
    def classify_regime(self, vix: float, vix3m: float = None, 
                        features: Dict = None, use_lstm: bool = False) -> MarketRegime:
        """
        Classify market regime.
        
        Primary: VIX-based rules (fast, reliable, always work)
        Optional: LSTM if model loaded and features provided
        
        Args:
            vix: Current VIX value
            vix3m: VIX 3-month (optional, for term structure)
            features: Dict of features for LSTM (optional)
            use_lstm: Whether to attempt LSTM prediction
            
        Returns:
            MarketRegime
        """
        # LSTM attempt (if enabled and model available)
        if use_lstm and features is not None and self._lstm_model is not None:
            try:
                lstm_regime = self._classify_with_lstm(features)
                if lstm_regime is not None:
                    return lstm_regime
            except Exception as e:
                logger.debug(f"LSTM classification failed, using VIX rules: {e}")
        
        # VIX-based rules - ALWAYS reliable
        if vix < 15:
            return MarketRegime.CALM
        elif vix < 20:
            return MarketRegime.NORMAL
        elif vix < 25:
            return MarketRegime.ELEVATED
        elif vix < 35:
            return MarketRegime.HIGH_VOL
        else:
            return MarketRegime.CRISIS
    
    def _classify_with_lstm(self, features: Dict) -> Optional[MarketRegime]:
        """
        Optional LSTM-based classification.
        
        Returns None if classification fails (triggers VIX fallback).
        """
        if self._lstm_model is None:
            return None
            
        try:
            import numpy as np
            
            # Build feature sequence
            feature_names = [
                'vix_ratio', 'vix_change_1d', 'vix_zscore',
                'return_1m', 'return_5m', 'volatility_20', 'momentum_20',
                'volume_ratio', 'high_low_range', 'price_acceleration',
                'vanna', 'volga'
            ]
            
            # Extract features
            seq = np.array([[features.get(f, 0.0) for f in feature_names]])
            
            # Need proper sequence shape for LSTM
            if len(seq.shape) == 2:
                seq = seq.reshape(1, 1, -1)  # (1, 1, features)
            
            # Predict
            proba = self._lstm_model.predict(seq, verbose=0)[0]
            regime_idx = int(np.argmax(proba))
            
            regime_map = {
                0: MarketRegime.CALM,
                1: MarketRegime.NORMAL,
                2: MarketRegime.ELEVATED,
                3: MarketRegime.HIGH_VOL,
                4: MarketRegime.CRISIS,
            }
            
            return regime_map.get(regime_idx, MarketRegime.NORMAL)
            
        except Exception as e:
            logger.debug(f"LSTM error: {e}")
            return None
    
    def _load_lstm_model(self):
        """Load LSTM model if available."""
        from pathlib import Path
        model_path = Path("data/models/regime_classifier/regime_model.keras")
        
        if model_path.exists():
            try:
                import tensorflow as tf
                self._lstm_model = tf.keras.models.load_model(str(model_path))
                logger.info(f"Loaded LSTM regime model from {model_path}")
            except Exception as e:
                logger.debug(f"Could not load LSTM model: {e}")
                self._lstm_model = None
        else:
            self._lstm_model = None
    
    def detect_trend(self, features: Dict[str, Any]) -> str:
        """Detect market trend from features."""
        # Use SMA comparison if available
        close = features.get('close', 0)
        sma50 = features.get('sma50', close)
        sma200 = features.get('sma200', close)
        rsi = features.get('rsi', 50)
        
        # Trend scoring
        score = 0
        
        if close > sma50:
            score += 1
        if close > sma200:
            score += 1
        if sma50 > sma200:
            score += 1
        if rsi > 50:
            score += 0.5
        if rsi > 70:
            score -= 0.5  # Overbought
        if rsi < 30:
            score += 0.5  # Oversold = potential reversal up
            
        if score >= 2:
            return "BULLISH"
        elif score <= 0:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def get_win_probability(self, features: Dict[str, Any]) -> float:
        """Get XGBoost win probability."""
        if self.xgb_predictor is None:
            return 0.5  # Default neutral
        
        try:
            return self.xgb_predictor.predict(features)
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
            return 0.5
    
    def recommend(self, features: Dict[str, Any]) -> StrategyRecommendation:
        """
        Get strategy recommendation.
        
        Args:
            features: Market features dict with:
                - vix, vix3m
                - close, sma50, sma200, rsi
                - iv, delta, gamma
                - etc.
        
        Returns:
            StrategyRecommendation with full recommendation
        """
        # 1. Classify regime
        vix = features.get('vix', 18)
        vix3m = features.get('vix3m', vix)
        regime = self.classify_regime(vix, vix3m)
        
        # 2. Detect trend
        trend = self.detect_trend(features)
        
        # 3. Get XGBoost probability
        win_prob = self.get_win_probability(features)
        
        # 4. Select strategy from matrix
        key = (regime, trend)
        if key not in self.STRATEGY_MATRIX:
            key = (regime, "NEUTRAL")
        
        strategies = self.STRATEGY_MATRIX.get(key, [(Strategy.CASH, 1.0)])
        
        # Weighted selection based on win probability
        if win_prob > 0.6:
            # High confidence - take more directional trades
            selected_idx = 0  # First (most aggressive)
        elif win_prob < 0.4:
            # Low confidence - prefer CASH
            selected_idx = len(strategies) - 1
        else:
            # Medium - middle option
            selected_idx = len(strategies) // 2
        
        selected_strategy, base_weight = strategies[min(selected_idx, len(strategies)-1)]
        
        # 5. Determine side
        if selected_strategy in [Strategy.BUY_CALL, Strategy.CALL_SPREAD, Strategy.STRADDLE]:
            side = "BUY"
        elif selected_strategy in [Strategy.SELL_PUT, Strategy.SELL_CALL, Strategy.IRON_CONDOR]:
            side = "SELL"
        elif selected_strategy in [Strategy.BUY_PUT, Strategy.PUT_SPREAD]:
            side = "BUY"
        else:
            side = "HOLD"
        
        # 6. Select DTE
        dte_bucket = self.DTE_RULES.get(regime, 1)
        
        # 0DTE only in CALM regime with high win prob
        if regime == MarketRegime.CALM and win_prob > 0.65:
            dte_bucket = 0  # 0DTE allowed
        
        # 7. Calculate confidence
        confidence = base_weight * (0.5 + win_prob * 0.5)
        
        # Boost confidence if trend and regime align
        if regime == MarketRegime.CALM and trend == "BULLISH":
            confidence *= 1.1
        elif regime == MarketRegime.CRISIS and trend == "BEARISH":
            confidence *= 1.1  # Crisis + bearish = put buying makes sense
        
        confidence = min(confidence, 0.95)
        
        # 8. Generate reasoning
        reasoning = self._generate_reasoning(
            regime, trend, selected_strategy, win_prob, vix
        )
        
        return StrategyRecommendation(
            regime=regime,
            strategy=selected_strategy,
            side=side,
            dte_bucket=dte_bucket,
            win_probability=win_prob,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _generate_reasoning(self, regime, trend, strategy, win_prob, vix) -> str:
        """Generate human-readable reasoning."""
        parts = []
        
        # Regime
        regime_text = {
            MarketRegime.CALM: f"Calm market (VIX={vix:.1f})",
            MarketRegime.NORMAL: f"Normal conditions (VIX={vix:.1f})",
            MarketRegime.ELEVATED: f"Elevated uncertainty (VIX={vix:.1f})",
            MarketRegime.HIGH_VOL: f"High volatility (VIX={vix:.1f})",
            MarketRegime.CRISIS: f"CRISIS MODE (VIX={vix:.1f})",
        }
        parts.append(regime_text.get(regime, "Unknown regime"))
        
        # Trend
        parts.append(f"Trend: {trend}")
        
        # Win probability
        if win_prob > 0.6:
            parts.append(f"High win probability ({win_prob:.0%})")
        elif win_prob < 0.4:
            parts.append(f"Low win probability ({win_prob:.0%}) - caution")
        
        # Strategy rationale
        strategy_reason = {
            Strategy.BUY_CALL: "Bullish directional play",
            Strategy.BUY_PUT: "Bearish protection or directional",
            Strategy.SELL_PUT: "Theta harvest, bullish bias",
            Strategy.CALL_SPREAD: "Defined risk bullish",
            Strategy.PUT_SPREAD: "Defined risk bearish",
            Strategy.IRON_CONDOR: "Neutral, collect premium",
            Strategy.STRADDLE: "Expecting big move either way",
            Strategy.CASH: "No clear edge, stay out",
        }
        parts.append(f"→ {strategy.value}: {strategy_reason.get(strategy, '')}")
        
        return " | ".join(parts)
    
    def recommend_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add strategy recommendations to DataFrame.
        
        Adds columns:
        - strategy_rec: Strategy name
        - strategy_side: BUY/SELL
        - strategy_dte: DTE bucket
        - strategy_conf: Confidence
        """
        results = []
        
        for idx, row in df.iterrows():
            features = row.to_dict()
            rec = self.recommend(features)
            results.append({
                'strategy_rec': rec.strategy.value,
                'strategy_side': rec.side,
                'strategy_dte': rec.dte_bucket,
                'strategy_conf': rec.confidence,
                'regime': rec.regime.value,
            })
        
        rec_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), rec_df], axis=1)


# Singleton
_advisor: Optional[StrategyAdvisor] = None

def get_strategy_advisor() -> StrategyAdvisor:
    """Get singleton StrategyAdvisor."""
    global _advisor
    if _advisor is None:
        _advisor = StrategyAdvisor()
    return _advisor


# Quick test
if __name__ == "__main__":
    from core.logger import setup_logger
    setup_logger(level="INFO")
    
    advisor = StrategyAdvisor()
    
    # Test cases
    test_cases = [
        {"vix": 12, "close": 100, "sma50": 98, "sma200": 95, "rsi": 60},  # Calm, bullish
        {"vix": 22, "close": 100, "sma50": 102, "sma200": 105, "rsi": 40}, # Elevated, bearish
        {"vix": 40, "close": 100, "sma50": 100, "sma200": 100, "rsi": 50}, # Crisis, neutral
    ]
    
    for features in test_cases:
        rec = advisor.recommend(features)
        print(f"\n{rec.reasoning}")
        print(f"  Strategy: {rec.strategy.value} | Side: {rec.side} | DTE: {rec.dte_bucket}")
        print(f"  Confidence: {rec.confidence:.2f}")
