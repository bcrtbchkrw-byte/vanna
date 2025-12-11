"""
Market Regime Classifier for Vanna Trading Bot.

Classifies market conditions into distinct regimes to help select
appropriate trading strategies. Uses technical indicators and VIX
to determine if market is:
- TRENDING_UP: Strong bullish momentum
- TRENDING_DOWN: Strong bearish momentum  
- RANGE_BOUND: Sideways price action
- HIGH_VOLATILITY: Elevated VIX, unstable conditions
"""
from dataclasses import dataclass

from core.logger import get_logger


@dataclass
class RegimeState:
    """Current market regime state."""
    regime: str
    confidence: float
    vix_level: float
    trend_direction: str
    volatility_percentile: float


class RegimeClassifier:
    """
    Classifies market regime based on multiple technical factors.
    
    Uses:
    - VIX level and percentile
    - SMA crossovers (20/50/200)
    - Price momentum
    - Historical volatility
    """
    
    # Regime constants
    REGIME_TRENDING_UP = "TRENDING_UP"
    REGIME_TRENDING_DOWN = "TRENDING_DOWN"
    REGIME_RANGE_BOUND = "RANGE_BOUND"
    REGIME_HIGH_VOL = "HIGH_VOLATILITY"
    
    # Thresholds
    VIX_HIGH_THRESHOLD = 25.0
    VIX_PANIC_THRESHOLD = 35.0
    TREND_THRESHOLD = 0.02  # 2% above/below SMA
    MOMENTUM_THRESHOLD = 0.03  # 3% momentum
    
    def __init__(self) -> None:
        self.logger = get_logger()
        self._current_regime: RegimeState | None = None
    
    def classify_regime(
        self,
        current_price: float,
        sma_20: float,
        sma_50: float,
        sma_200: float,
        vix: float,
        historical_volatility: float,
        price_momentum: float | None = None
    ) -> RegimeState:
        """
        Classify current market regime.
        
        Args:
            current_price: Current underlying price
            sma_20: 20-day simple moving average
            sma_50: 50-day simple moving average
            sma_200: 200-day simple moving average
            vix: Current VIX level
            historical_volatility: 30-day historical volatility
            price_momentum: Optional 20-day price change %
            
        Returns:
            RegimeState with classification and confidence
        """
        # Calculate price position relative to SMAs
        above_sma_20 = current_price > sma_20
        above_sma_50 = current_price > sma_50
        above_sma_200 = current_price > sma_200
        
        # SMA alignment score (-3 to +3)
        sma_alignment = sum([
            1 if above_sma_20 else -1,
            1 if above_sma_50 else -1,
            1 if above_sma_200 else -1
        ])
        
        # Trend strength (distance from SMA50)
        trend_strength = abs(current_price - sma_50) / sma_50
        
        # Momentum calculation
        if price_momentum is None:
            price_momentum = (current_price - sma_20) / sma_20
        
        # VIX analysis
        vix_percentile = min(vix / 50.0, 1.0)  # Normalize to 0-1
        
        # Determine regime
        confidence = 0.0
        
        # Check for HIGH VOLATILITY regime first
        if vix >= self.VIX_PANIC_THRESHOLD:
            regime = self.REGIME_HIGH_VOL
            confidence = 0.95
            trend_dir = "UNSTABLE"
        elif vix >= self.VIX_HIGH_THRESHOLD:
            regime = self.REGIME_HIGH_VOL
            confidence = 0.75 + (vix - self.VIX_HIGH_THRESHOLD) / 20
            trend_dir = "ELEVATED"
        # Check for strong trends
        elif sma_alignment >= 2 and trend_strength > self.TREND_THRESHOLD:
            regime = self.REGIME_TRENDING_UP
            confidence = min(0.6 + trend_strength * 5, 0.95)
            trend_dir = "BULLISH"
        elif sma_alignment <= -2 and trend_strength > self.TREND_THRESHOLD:
            regime = self.REGIME_TRENDING_DOWN
            confidence = min(0.6 + trend_strength * 5, 0.95)
            trend_dir = "BEARISH"
        # Default to range-bound
        else:
            regime = self.REGIME_RANGE_BOUND
            confidence = 0.6 + (1 - trend_strength) * 0.3
            trend_dir = "NEUTRAL"
        
        state = RegimeState(
            regime=regime,
            confidence=confidence,
            vix_level=vix,
            trend_direction=trend_dir,
            volatility_percentile=vix_percentile
        )
        
        self._current_regime = state
        self.logger.info(
            f"Regime: {regime} (confidence: {confidence:.0%}, VIX: {vix:.1f})"
        )
        
        return state
    
    def get_current_regime(self) -> RegimeState | None:
        """Get the last classified regime."""
        return self._current_regime
    
    def get_strategy_weights(self, regime: str) -> dict[str, float]:
        """
        Get recommended strategy weights for a given regime.
        
        Returns weights between 0-1 for each strategy type.
        Higher weight = more suitable for this regime.
        """
        weights: dict[str, float] = {
            "credit_spread": 0.0,
            "iron_condor": 0.0,
            "debit_spread": 0.0,
            "calendar": 0.0,
            "jade_lizard": 0.0,
            "pmcc": 0.0
        }
        
        if regime == self.REGIME_HIGH_VOL:
            # High IV = premium selling favorable
            weights["credit_spread"] = 0.8
            weights["iron_condor"] = 0.3  # Risky in high vol
            weights["jade_lizard"] = 0.7
        elif regime == self.REGIME_TRENDING_UP:
            # Bullish strategies
            weights["credit_spread"] = 0.7  # Bull put spreads
            weights["pmcc"] = 0.8  # Poor man's covered call
            weights["jade_lizard"] = 0.6
        elif regime == self.REGIME_TRENDING_DOWN:
            # Bearish strategies
            weights["credit_spread"] = 0.7  # Bear call spreads
            weights["debit_spread"] = 0.6
        elif regime == self.REGIME_RANGE_BOUND:
            # Neutral strategies
            weights["iron_condor"] = 0.9
            weights["calendar"] = 0.7
            weights["credit_spread"] = 0.5
        
        return weights
    
    def is_trading_safe(self) -> bool:
        """Check if current regime allows trading."""
        if self._current_regime is None:
            return True  # No regime data, allow trading
        
        # Block trading in extreme volatility
        if self._current_regime.vix_level >= self.VIX_PANIC_THRESHOLD:
            return False
        
        return True


# Singleton
_classifier: RegimeClassifier | None = None


def get_regime_classifier() -> RegimeClassifier:
    """Get global regime classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = RegimeClassifier()
    return _classifier
