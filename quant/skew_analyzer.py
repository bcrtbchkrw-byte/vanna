"""
Volatility Skew Analyzer for Vanna Trading Bot.

Analyzes put/call IV skew patterns to identify:
- Normal skew (puts more expensive - fear premium)
- Reverse skew (unusual market conditions)
- Smile patterns (both wings elevated)

Used for strategy selection and risk assessment.
"""
from dataclasses import dataclass
from typing import Any

from core.logger import get_logger


@dataclass
class SkewMetrics:
    """Skew analysis metrics."""
    skew_slope: float  # Negative = put skew, Positive = call skew
    put_wing_iv: float  # Average IV of OTM puts
    call_wing_iv: float  # Average IV of OTM calls
    atm_iv: float
    skew_percentile: float  # Historical percentile (if available)
    pattern: str  # "PUT_SKEW", "CALL_SKEW", "SMILE", "SMIRK"


class SkewAnalyzer:
    """
    Analyzes implied volatility skew patterns.
    
    Key concepts:
    - Put skew: OTM puts have higher IV than OTM calls (normal in equities)
    - Call skew: OTM calls higher IV (unusual, bullish sentiment)
    - Smile: Both wings elevated relative to ATM
    """
    
    def __init__(self) -> None:
        self.logger = get_logger()
    
    def analyze_skew(
        self,
        option_chain: list[dict[str, Any]],
        underlying_price: float,
        target_dte: int = 30
    ) -> SkewMetrics:
        """
        Analyze skew for a given expiration.
        
        Args:
            option_chain: Option chain with iv, strike, right (P/C)
            underlying_price: Current stock price
            target_dte: Target days to expiry for analysis
            
        Returns:
            SkewMetrics with detailed skew analysis
        """
        # Filter to target DTE range
        target_options = [
            opt for opt in option_chain
            if abs(opt.get("dte", 0) - target_dte) <= 7
        ]
        
        if len(target_options) < 5:
            self.logger.warning("Insufficient options for skew analysis")
            return SkewMetrics(
                skew_slope=0,
                put_wing_iv=0,
                call_wing_iv=0,
                atm_iv=0,
                skew_percentile=50,
                pattern="UNKNOWN"
            )
        
        # Categorize by moneyness
        atm_range = (underlying_price * 0.98, underlying_price * 1.02)
        
        atm_options = []
        otm_puts = []
        otm_calls = []
        
        for opt in target_options:
            strike = opt.get("strike", 0)
            iv = opt.get("iv", opt.get("impliedVolatility", 0))
            right = opt.get("right", "")
            
            if iv <= 0:
                continue
            
            if atm_range[0] <= strike <= atm_range[1]:
                atm_options.append(iv)
            elif right == "P" and strike < underlying_price * 0.95:
                otm_puts.append({"strike": strike, "iv": iv})
            elif right == "C" and strike > underlying_price * 1.05:
                otm_calls.append({"strike": strike, "iv": iv})
        
        # Calculate averages
        atm_iv = sum(atm_options) / len(atm_options) if atm_options else 0
        put_wing_iv = sum(p["iv"] for p in otm_puts) / len(otm_puts) if otm_puts else 0
        call_wing_iv = sum(c["iv"] for c in otm_calls) / len(otm_calls) if otm_calls else 0
        
        # Calculate skew slope (put IV - call IV) / atm IV
        if atm_iv > 0:
            skew_slope = (put_wing_iv - call_wing_iv) / atm_iv
        else:
            skew_slope = 0
        
        # Determine pattern
        pattern = self._determine_pattern(put_wing_iv, call_wing_iv, atm_iv)
        
        metrics = SkewMetrics(
            skew_slope=skew_slope,
            put_wing_iv=put_wing_iv,
            call_wing_iv=call_wing_iv,
            atm_iv=atm_iv,
            skew_percentile=50,  # Would need historical data
            pattern=pattern
        )
        
        self.logger.info(
            f"Skew analysis: {pattern}, slope={skew_slope:.3f}, "
            f"put_wing={put_wing_iv:.2%}, atm={atm_iv:.2%}, call_wing={call_wing_iv:.2%}"
        )
        
        return metrics
    
    def _determine_pattern(
        self,
        put_wing_iv: float,
        call_wing_iv: float,
        atm_iv: float
    ) -> str:
        """Determine the skew pattern."""
        if atm_iv == 0:
            return "UNKNOWN"
        
        put_elevated = put_wing_iv > atm_iv * 1.1
        call_elevated = call_wing_iv > atm_iv * 1.1
        
        if put_elevated and call_elevated:
            return "SMILE"
        elif put_elevated and not call_elevated:
            return "PUT_SKEW"
        elif call_elevated and not put_elevated:
            return "CALL_SKEW"
        elif put_wing_iv > call_wing_iv:
            return "SMIRK"
        else:
            return "NORMAL"
    
    def get_strategy_recommendation(self, skew_metrics: SkewMetrics) -> dict[str, Any]:
        """
        Get strategy recommendations based on skew.
        
        Args:
            skew_metrics: Output from analyze_skew
            
        Returns:
            Dict with recommended strategies and reasoning
        """
        recommendations: dict[str, Any] = {
            "strategies": [],
            "reasoning": "",
            "avoid": []
        }
        
        if skew_metrics.pattern == "PUT_SKEW":
            # Elevated put IV = good to sell puts
            recommendations["strategies"] = ["BULL_PUT_SPREAD", "JADE_LIZARD"]
            recommendations["reasoning"] = "Elevated put IV offers premium selling opportunity"
            recommendations["avoid"] = ["PUT_DEBIT_SPREAD"]
        
        elif skew_metrics.pattern == "CALL_SKEW":
            # Unusual - calls expensive, maybe sell calls
            recommendations["strategies"] = ["BEAR_CALL_SPREAD"]
            recommendations["reasoning"] = "Elevated call IV - unusual bullish sentiment"
            recommendations["avoid"] = ["POOR_MANS_COVERED_CALL"]
        
        elif skew_metrics.pattern == "SMILE":
            # Both wings elevated - iron condor might be good
            recommendations["strategies"] = ["IRON_CONDOR", "STRANGLE_SELL"]
            recommendations["reasoning"] = "Both wings elevated - sell both sides"
        
        else:
            # Normal market
            recommendations["strategies"] = ["BULL_PUT_SPREAD", "IRON_CONDOR"]
            recommendations["reasoning"] = "Normal skew - standard strategies"
        
        return recommendations


# Singleton
_skew_analyzer: SkewAnalyzer | None = None


def get_skew_analyzer() -> SkewAnalyzer:
    """Get global skew analyzer instance."""
    global _skew_analyzer
    if _skew_analyzer is None:
        _skew_analyzer = SkewAnalyzer()
    return _skew_analyzer
