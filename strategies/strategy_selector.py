"""
Strategy Selector for Vanna Trading Bot.

Dynamically selects optimal trading strategies based on:
- Market regime (from RegimeClassifier)
- VIX level
- IV skew
- Available capital
"""
from typing import Any

from core.logger import get_logger
from ml.regime_classifier import RegimeClassifier, get_regime_classifier


class StrategySelector:
    """
    Intelligent strategy selector based on market conditions.
    
    Maps market regimes to optimal strategies and provides
    weighted recommendations for the current environment.
    """
    
    # Strategy priority by regime (numeric keys match RegimeClassifier)
    # 0 = Low Vol, 1 = Normal, 2 = Elevated, 3 = High Vol, 4 = Crisis
    REGIME_STRATEGIES: dict[int, list[str]] = {
        0: [  # Low Vol - directional plays
            "POOR_MANS_COVERED_CALL",
            "BULL_PUT_SPREAD",
            "CALENDAR_SPREAD"
        ],
        1: [  # Normal - balanced
            "BULL_PUT_SPREAD",
            "IRON_CONDOR",
            "JADE_LIZARD"
        ],
        2: [  # Elevated - premium selling
            "IRON_CONDOR",
            "IRON_BUTTERFLY",
            "JADE_LIZARD"
        ],
        3: [  # High Vol - defensive premium selling
            "JADE_LIZARD",
            "BULL_PUT_SPREAD",
            "BEAR_CALL_SPREAD"
        ],
        4: [  # Crisis - minimal or avoid
            "BEAR_CALL_SPREAD",
            "PUT_DEBIT_SPREAD"
        ]
    }
    
    def __init__(self) -> None:
        self.logger = get_logger()
        self._regime_classifier = get_regime_classifier()
    
    def select_strategies(
        self,
        vix: float,
        current_price: float,
        sma_20: float,
        sma_50: float,
        sma_200: float,
        available_capital: float,
        iv_rank: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Select optimal strategies for current conditions.
        
        Args:
            vix: Current VIX level
            current_price: Current underlying price
            sma_20: 20-day SMA
            sma_50: 50-day SMA
            sma_200: 200-day SMA
            available_capital: Available trading capital
            iv_rank: Optional IV rank percentile
            
        Returns:
            List of strategy recommendations with weights
        """
        # Classify current regime
        regime_state = self._regime_classifier.classify_regime(
            current_price=current_price,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            vix=vix,
            historical_volatility=0.0  # Will use VIX as proxy
        )
        
        regime = regime_state.regime
        self.logger.info(f"Current regime: {regime} (confidence: {regime_state.confidence:.0%})")
        
        # Get strategies for this regime
        strategies = self.REGIME_STRATEGIES.get(regime, ["BULL_PUT_SPREAD"])
        
        # Build recommendations
        recommendations = []
        
        for idx, strategy in enumerate(strategies):
            # Base weight from priority order
            weight = 1.0 - (idx * 0.2)
            
            # Adjust weight based on conditions
            weight = self._adjust_weight(
                strategy=strategy,
                weight=weight,
                vix=vix,
                iv_rank=iv_rank,
                available_capital=available_capital
            )
            
            if weight > 0.3:  # Only include meaningful recommendations
                recommendations.append({
                    "strategy": strategy,
                    "weight": weight,
                    "regime": regime,
                    "suitable_for_capital": self._check_capital_fit(
                        strategy, available_capital
                    )
                })
        
        # Sort by weight
        recommendations.sort(key=lambda x: x.get("weight", 0.0), reverse=True)  # type: ignore[arg-type, return-value]

        
        return recommendations
    
    def _adjust_weight(
        self,
        strategy: str,
        weight: float,
        vix: float,
        iv_rank: float | None,
        available_capital: float
    ) -> float:
        """Adjust strategy weight based on current conditions."""
        # High VIX boosts premium selling strategies
        if vix > 20 and strategy in ["BULL_PUT_SPREAD", "BEAR_CALL_SPREAD", "JADE_LIZARD"]:
            weight += 0.1
        
        # High IV rank boosts all selling strategies
        if iv_rank and iv_rank > 50:
            if "SPREAD" in strategy or strategy == "JADE_LIZARD":
                weight += 0.1
        
        # Low IV rank favors buying strategies
        if iv_rank and iv_rank < 25:
            if "DEBIT" in strategy or strategy == "POOR_MANS_COVERED_CALL":
                weight += 0.1
        
        return min(weight, 1.0)
    
    def _check_capital_fit(self, strategy: str, available_capital: float) -> bool:
        """Check if strategy fits available capital."""
        # Rough capital requirements
        min_capital = {
            "BULL_PUT_SPREAD": 500,
            "BEAR_CALL_SPREAD": 500,
            "IRON_CONDOR": 1000,
            "IRON_BUTTERFLY": 1000,
            "JADE_LIZARD": 800,
            "POOR_MANS_COVERED_CALL": 2000,  # LEAPS are expensive
            "PUT_DEBIT_SPREAD": 300,
            "CALENDAR_SPREAD": 500
        }
        
        required = min_capital.get(strategy, 500)
        return available_capital >= required
    
    def get_primary_strategy(
        self,
        vix: float,
        current_price: float,
        sma_20: float,
        sma_50: float,
        sma_200: float,
        available_capital: float
    ) -> str:
        """Get single best strategy for current conditions."""
        recommendations = self.select_strategies(
            vix=vix,
            current_price=current_price,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            available_capital=available_capital
        )
        
        if recommendations:
            return str(recommendations[0]["strategy"])

        
        return "BULL_PUT_SPREAD"  # Default fallback


# Singleton
_selector: StrategySelector | None = None


def get_strategy_selector() -> StrategySelector:
    """Get global strategy selector instance."""
    global _selector
    if _selector is None:
        _selector = StrategySelector()
    return _selector
