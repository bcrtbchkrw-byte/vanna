"""
Poor Man's Covered Call (PMCC) Strategy for Vanna Trading Bot.

Replaces stock ownership with a deep ITM LEAPS call, then sells
short-term OTM calls against it for income.

Best for: Bullish outlook with limited capital.
Benefit: Capital efficient alternative to covered calls.
"""
from dataclasses import dataclass

from core.logger import get_logger


@dataclass
class PMCCCandidate:
    """Poor Man's Covered Call candidate."""
    symbol: str
    leaps_expiry: str
    leaps_strike: float
    leaps_cost: float
    short_call_expiry: str
    short_call_strike: float
    short_credit: float
    max_profit: float
    max_loss: float
    breakeven: float
    capital_efficiency: float  # Comparison to actual covered call


class PoorMansCoveredCall:
    """
    Poor Man's Covered Call (Diagonal Spread) implementation.
    
    Structure:
    - Buy 1 deep ITM LEAPS call (>0.70 delta, >6 months out)
    - Sell 1 short-term OTM call (0.20-0.30 delta, <45 days)
    
    Goal: Generate income from short calls while benefiting from
    stock appreciation with less capital than owning shares.
    """
    
    def __init__(self) -> None:
        self.logger = get_logger()

    
    def find_candidates(
        self,
        symbol: str,
        current_price: float,
        option_chain: list[dict],
        min_leaps_dte: int = 180,
        max_short_dte: int = 45,
        target_leaps_delta: float = 0.75,
        target_short_delta: float = 0.25
    ) -> list[PMCCCandidate]:
        """
        Find PMCC candidates from option chain.
        
        Args:
            symbol: Underlying symbol
            current_price: Current stock price
            option_chain: List of option contracts
            min_leaps_dte: Minimum DTE for LEAPS (default 6 months)
            max_short_dte: Maximum DTE for short call
            target_leaps_delta: Target delta for LEAPS (deep ITM)
            target_short_delta: Target delta for short call (OTM)
            
        Returns:
            List of PMCCCandidate sorted by capital efficiency
        """
        candidates = []
        
        # Separate LEAPS and short-term options
        leaps_options = [
            opt for opt in option_chain
            if opt.get("dte", 0) >= min_leaps_dte
            and opt.get("right") == "C"
        ]
        
        short_options = [
            opt for opt in option_chain
            if opt.get("dte", 0) <= max_short_dte
            and opt.get("dte", 0) >= 14  # Minimum 2 weeks
            and opt.get("right") == "C"
        ]
        
        if not leaps_options or not short_options:
            self.logger.warning("Insufficient options for PMCC")
            return []
        
        # Find best LEAPS (deep ITM)
        leaps_calls = [
            opt for opt in leaps_options
            if opt.get("strike", 0) < current_price * 0.85  # At least 15% ITM
        ]
        leaps_calls.sort(key=lambda x: abs(x.get("delta", 0) - target_leaps_delta))
        
        if not leaps_calls:
            self.logger.warning("No suitable LEAPS found")
            return []
        
        leaps = leaps_calls[0]
        leaps_expiry = leaps.get("expiry", "")
        
        # Find short call (OTM with target delta)
        short_calls = [
            opt for opt in short_options
            if opt.get("strike", 0) > current_price
        ]
        short_calls.sort(key=lambda x: abs(x.get("delta", 0) - target_short_delta))
        
        if not short_calls:
            return []
        
        short = short_calls[0]
        short_expiry = short.get("expiry", "")
        
        # Calculate costs
        leaps_cost = (leaps.get("ask", 0) + leaps.get("bid", 0)) / 2 * 100
        short_credit = (short.get("bid", 0) + short.get("ask", 0)) / 2 * 100
        
        # Max profit: short strike - LEAPS strike + short credit - LEAPS cost
        max_profit = (
            (short.get("strike", 0) - leaps.get("strike", 0)) * 100 
            + short_credit 
            - leaps_cost
        )
        
        # Max loss: LEAPS cost (if stock goes to 0)
        max_loss = leaps_cost - short_credit
        
        # Breakeven
        breakeven = leaps.get("strike", 0) + (leaps_cost - short_credit) / 100
        
        # Capital efficiency vs covered call
        # Covered call would cost current_price * 100
        covered_call_cost = current_price * 100
        capital_efficiency = covered_call_cost / leaps_cost if leaps_cost > 0 else 0
        
        candidate = PMCCCandidate(
            symbol=symbol,
            leaps_expiry=leaps_expiry,
            leaps_strike=leaps.get("strike", 0),
            leaps_cost=leaps_cost,
            short_call_expiry=short_expiry,
            short_call_strike=short.get("strike", 0),
            short_credit=short_credit,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven=breakeven,
            capital_efficiency=capital_efficiency
        )
        
        candidates.append(candidate)
        
        self.logger.info(
            f"PMCC: {symbol} "
            f"LEAPS {leaps.get('strike')} ({leaps_expiry}) / "
            f"Short {short.get('strike')} ({short_expiry}) "
            f"Efficiency: {capital_efficiency:.1f}x"
        )
        
        return candidates


# Singleton
_pmcc: PoorMansCoveredCall | None = None


def get_pmcc_strategy() -> PoorMansCoveredCall:
    """Get global PMCC strategy instance."""
    global _pmcc
    if _pmcc is None:
        _pmcc = PoorMansCoveredCall()
    return _pmcc
