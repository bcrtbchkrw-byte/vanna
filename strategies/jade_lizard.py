"""
Jade Lizard Strategy for Vanna Trading Bot.

Combines:
- Short Put (naked or spread)
- Bear Call Spread

Best for: Neutral to slightly bullish outlook with high IV.
Benefit: No upside risk if structured correctly.
"""
from dataclasses import dataclass

from core.logger import get_logger


@dataclass
class JadeLizardCandidate:
    """Jade Lizard trade candidate."""
    symbol: str
    expiry: str
    short_put_strike: float
    short_call_strike: float
    long_call_strike: float
    total_credit: float
    max_loss_downside: float
    max_loss_upside: float  # 0 if properly structured
    breakeven: float
    roi_pct: float


class JadeLizard:
    """
    Jade Lizard strategy implementation.
    
    Structure:
    - Sell 1 OTM Put
    - Sell 1 OTM Call (higher strike)
    - Buy 1 further OTM Call (defines max upside loss)
    
    Goal: Collect premium with no upside risk if credit > call spread width.
    """
    
    def __init__(self) -> None:
        self.logger = get_logger()

    
    def find_candidates(
        self,
        symbol: str,
        current_price: float,
        option_chain: list[dict],
        target_days_to_expiry: int = 30,
        target_put_delta: float = -0.20,
        target_call_delta: float = 0.30,
        call_spread_width: float = 5.0
    ) -> list[JadeLizardCandidate]:
        """
        Find Jade Lizard candidates from option chain.
        
        Args:
            symbol: Underlying symbol
            current_price: Current stock price
            option_chain: List of option contracts
            target_days_to_expiry: Target DTE
            target_put_delta: Target delta for short put (negative)
            target_call_delta: Target delta for short call
            call_spread_width: Width of call spread
            
        Returns:
            List of JadeLizardCandidate sorted by ROI
        """
        candidates = []
        
        # Filter to target expiry
        expiry_options = [
            opt for opt in option_chain 
            if opt.get("dte", 0) >= target_days_to_expiry - 7
            and opt.get("dte", 0) <= target_days_to_expiry + 7
        ]
        
        if not expiry_options:
            self.logger.warning(f"No options found near {target_days_to_expiry} DTE")
            return []
        
        # Get unique expiry
        expiry = expiry_options[0].get("expiry", "")
        
        # Find short put (OTM put with target delta)
        puts = [
            opt for opt in expiry_options 
            if opt.get("right") == "P" and opt.get("strike", 0) < current_price
        ]
        puts.sort(key=lambda x: abs(x.get("delta", 0) - target_put_delta))
        
        if not puts:
            return []
        
        short_put = puts[0]
        
        # Find short call (OTM call with target delta)
        calls = [
            opt for opt in expiry_options
            if opt.get("right") == "C" and opt.get("strike", 0) > current_price
        ]
        calls.sort(key=lambda x: abs(x.get("delta", 0) - target_call_delta))
        
        if not calls:
            return []
        
        short_call = calls[0]
        
        # Find long call (further OTM)
        long_call_strike = short_call.get("strike", 0) + call_spread_width
        long_call_options = [
            opt for opt in calls
            if abs(opt.get("strike", 0) - long_call_strike) < 1
        ]
        
        if not long_call_options:
            # Use next available strike
            higher_calls = [
                opt for opt in calls 
                if opt.get("strike", 0) > short_call.get("strike", 0)
            ]
            if not higher_calls:
                return []
            long_call_options = [higher_calls[0]]
        
        long_call = long_call_options[0]
        
        # Calculate credits
        put_credit = (short_put.get("bid", 0) + short_put.get("ask", 0)) / 2
        call_credit = (short_call.get("bid", 0) + short_call.get("ask", 0)) / 2
        call_debit = (long_call.get("bid", 0) + long_call.get("ask", 0)) / 2
        
        total_credit = put_credit + call_credit - call_debit
        
        # Max losses
        actual_call_width = long_call.get("strike", 0) - short_call.get("strike", 0)
        max_loss_upside = max(0, actual_call_width - (call_credit - call_debit))
        max_loss_downside = short_put.get("strike", 0) * 100 - total_credit * 100
        
        # Breakeven
        breakeven = short_put.get("strike", 0) - total_credit
        
        # ROI
        roi_pct = (total_credit / max_loss_downside * 10000) if max_loss_downside > 0 else 0
        
        candidate = JadeLizardCandidate(
            symbol=symbol,
            expiry=expiry,
            short_put_strike=short_put.get("strike", 0),
            short_call_strike=short_call.get("strike", 0),
            long_call_strike=long_call.get("strike", 0),
            total_credit=total_credit,
            max_loss_downside=max_loss_downside,
            max_loss_upside=max_loss_upside,
            breakeven=breakeven,
            roi_pct=roi_pct
        )
        
        candidates.append(candidate)
        
        self.logger.info(
            f"Jade Lizard: {symbol} "
            f"Put {short_put.get('strike')}/Call {short_call.get('strike')}-{long_call.get('strike')} "
            f"Credit: ${total_credit:.2f}, ROI: {roi_pct:.1f}%"
        )
        
        return candidates


# Singleton
_jade_lizard: JadeLizard | None = None


def get_jade_lizard() -> JadeLizard:
    """Get global Jade Lizard strategy instance."""
    global _jade_lizard
    if _jade_lizard is None:
        _jade_lizard = JadeLizard()
    return _jade_lizard
