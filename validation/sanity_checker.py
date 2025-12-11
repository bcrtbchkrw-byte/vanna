"""
AI Sanity Checker for Vanna Trading Bot.

Uses LLM (Gemini) to provide a second opinion on trade setups.
Acts as a final validation layer before order execution to catch
potential issues that rule-based systems might miss.
"""
from typing import Any

from core.logger import get_logger


class SanityCheckResult:
    """Result of sanity check analysis."""
    
    def __init__(
        self,
        passed: bool,
        confidence: float,
        reason: str,
        concerns: list[str] | None = None
    ) -> None:
        self.passed = passed
        self.confidence = confidence
        self.reason = reason
        self.concerns = concerns or []


class SanityChecker:
    """
    LLM-based sanity checker for trade validation.
    
    Provides a second opinion on trade setups by asking the AI
    to evaluate the trade from multiple risk perspectives.
    """
    
    def __init__(self) -> None:
        self.logger = get_logger()
        self._ai_client: Any = None
    
    def _get_ai_client(self) -> Any:
        """Lazy-load the AI client."""
        if self._ai_client is None:
            try:
                from ai.gemini_client import get_gemini_client
                self._ai_client = get_gemini_client()
            except Exception as e:
                self.logger.error(f"Failed to load AI client: {e}")
        return self._ai_client
    
    async def verify_setup(
        self,
        symbol: str,
        strategy: str,
        entry_price: float,
        max_loss: float,
        days_to_expiry: int,
        vix: float,
        delta: float,
        confidence_score: float,
        market_regime: str
    ) -> SanityCheckResult:
        """
        Verify a trade setup using AI analysis.
        
        Args:
            symbol: Trading symbol
            strategy: Strategy type (e.g., "BULL_PUT")
            entry_price: Credit/premium received
            max_loss: Maximum possible loss
            days_to_expiry: Days until expiration
            vix: Current VIX level
            delta: Short leg delta
            confidence_score: Gatekeeper confidence
            market_regime: Current market regime
            
        Returns:
            SanityCheckResult with pass/fail and reasoning
        """
        ai_client = self._get_ai_client()
        
        if ai_client is None:
            # Fallback to rule-based check if AI unavailable
            return self._rule_based_check(
                max_loss=max_loss,
                days_to_expiry=days_to_expiry,
                vix=vix,
                delta=delta,
                confidence_score=confidence_score
            )
        
        # Construct prompt for AI
        prompt = f"""Analyze this options trade setup for potential issues:

Trade Details:
- Symbol: {symbol}
- Strategy: {strategy}
- Entry Credit: ${entry_price:.2f}
- Max Loss: ${max_loss:.2f}
- Days to Expiry: {days_to_expiry}
- VIX: {vix:.1f}
- Delta: {delta:.2f}
- ML Confidence: {confidence_score:.0%}
- Market Regime: {market_regime}

Evaluate for:
1. Risk/Reward ratio appropriateness
2. Market timing concerns
3. Greeks alignment with strategy
4. Any red flags or concerns

Respond in this exact JSON format:
{{"passed": true/false, "confidence": 0.0-1.0, "reason": "brief", "concerns": []}}"""


        try:
            response = await ai_client.analyze(prompt)
            
            # Parse response
            import json
            result_data = json.loads(response)
            
            return SanityCheckResult(
                passed=result_data.get("passed", True),
                confidence=result_data.get("confidence", 0.5),
                reason=result_data.get("reason", "No response"),
                concerns=result_data.get("concerns", [])
            )
        except Exception as e:
            self.logger.error(f"AI sanity check failed: {e}")
            return self._rule_based_check(
                max_loss=max_loss,
                days_to_expiry=days_to_expiry,
                vix=vix,
                delta=delta,
                confidence_score=confidence_score
            )
    
    def _rule_based_check(
        self,
        max_loss: float,
        days_to_expiry: int,
        vix: float,
        delta: float,
        confidence_score: float
    ) -> SanityCheckResult:
        """Fallback rule-based sanity check."""
        concerns = []
        passed = True
        
        # Check max loss
        if max_loss > 500:
            concerns.append(f"High max loss: ${max_loss:.0f}")
        
        # Check DTE
        if days_to_expiry < 7:
            concerns.append(f"Very short DTE: {days_to_expiry} days")
            passed = False
        elif days_to_expiry > 60:
            concerns.append(f"Long DTE: {days_to_expiry} days")
        
        # Check VIX
        if vix > 35:
            concerns.append(f"Extreme VIX: {vix:.1f}")
            passed = False
        
        # Check delta
        if abs(delta) > 0.35:
            concerns.append(f"High delta: {delta:.2f}")
        
        # Check confidence
        if confidence_score < 0.5:
            concerns.append(f"Low ML confidence: {confidence_score:.0%}")
            passed = False
        
        reason = "Passed rule-based checks" if passed else "Failed rule-based checks"
        if concerns:
            reason += f" ({len(concerns)} concerns)"
        
        return SanityCheckResult(
            passed=passed,
            confidence=0.7 if passed else 0.3,
            reason=reason,
            concerns=concerns
        )


# Singleton
_checker: SanityChecker | None = None


def get_sanity_checker() -> SanityChecker:
    """Get global sanity checker instance."""
    global _checker
    if _checker is None:
        _checker = SanityChecker()
    return _checker
