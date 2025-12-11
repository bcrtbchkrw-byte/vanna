"""
Max Pain Validator for Vanna Trading Bot.

Calculates and validates option strikes against max pain levels
to avoid trades near pinning zones where price tends to settle
at expiration due to market makers hedging.
"""
from core.logger import get_logger


class MaxPainResult:
    """Result of max pain calculation."""
    
    def __init__(
        self,
        max_pain_strike: float,
        current_price: float,
        distance_pct: float,
        is_safe: bool
    ) -> None:
        self.max_pain_strike = max_pain_strike
        self.current_price = current_price
        self.distance_pct = distance_pct
        self.is_safe = is_safe


class MaxPainValidator:
    """
    Validates trade strikes against max pain levels.
    
    Max pain is the strike price where the most options expire
    worthless, causing maximum pain to option holders. Prices
    tend to gravitate toward this level near expiration.
    """
    
    # Safety threshold - don't trade within this % of max pain
    SAFETY_THRESHOLD_PCT = 3.0
    
    def __init__(self) -> None:
        self.logger = get_logger()
    
    def calculate_max_pain(
        self,
        option_chain: list[dict[str, float]]
    ) -> float | None:
        """
        Calculate max pain from option chain data.
        
        Args:
            option_chain: List of dicts with keys:
                - strike: Strike price
                - call_oi: Call open interest
                - put_oi: Put open interest
                
        Returns:
            Max pain strike or None if insufficient data
        """
        if not option_chain:
            return None
        
        # Get unique strikes
        strikes = sorted(set(opt["strike"] for opt in option_chain))
        
        if len(strikes) < 3:
            return None
        
        # Calculate total pain at each strike
        pain_by_strike: dict[float, float] = {}
        
        for test_strike in strikes:
            total_pain = 0.0
            
            for opt in option_chain:
                strike = opt["strike"]
                call_oi = opt.get("call_oi", 0)
                put_oi = opt.get("put_oi", 0)
                
                # Call pain: if price settles below strike, calls expire worthless
                # Put pain: if price settles above strike, puts expire worthless
                if test_strike < strike:
                    # All calls at this strike expire worthless
                    total_pain += call_oi * (strike - test_strike) * 100
                else:
                    # All puts at this strike expire worthless
                    total_pain += put_oi * (test_strike - strike) * 100
            
            pain_by_strike[test_strike] = total_pain
        
        # Find strike with minimum pain (max pain for market makers)
        max_pain_strike = min(pain_by_strike.keys(), key=lambda k: pain_by_strike[k])
        
        self.logger.info(f"Max pain calculated: ${max_pain_strike:.2f}")
        
        return max_pain_strike
    
    def is_strike_safe(
        self,
        strike: float,
        max_pain_strike: float,
        current_price: float
    ) -> tuple[bool, float]:
        """
        Check if a strike is safely away from max pain.
        
        Args:
            strike: Strike to validate
            max_pain_strike: Calculated max pain strike
            current_price: Current underlying price
            
        Returns:
            Tuple of (is_safe, distance_pct)
        """
        # Distance from max pain as percentage
        distance_pct = abs(strike - max_pain_strike) / max_pain_strike * 100
        
        is_safe = distance_pct >= self.SAFETY_THRESHOLD_PCT
        
        if not is_safe:
            self.logger.warning(
                f"Strike {strike} is only {distance_pct:.1f}% from "
                f"max pain {max_pain_strike} (min: {self.SAFETY_THRESHOLD_PCT}%)"
            )
        
        return is_safe, distance_pct
    
    def validate_trade(
        self,
        short_strike: float,
        long_strike: float,
        option_chain: list[dict[str, float]],
        current_price: float
    ) -> MaxPainResult:
        """
        Validate a spread trade against max pain.
        
        Args:
            short_strike: Short leg strike
            long_strike: Long leg strike
            option_chain: Option chain data
            current_price: Current underlying price
            
        Returns:
            MaxPainResult with validation details
        """
        max_pain = self.calculate_max_pain(option_chain)
        
        if max_pain is None:
            # No data, assume safe
            return MaxPainResult(
                max_pain_strike=0,
                current_price=current_price,
                distance_pct=100,
                is_safe=True
            )
        
        # Check both legs
        short_safe, short_dist = self.is_strike_safe(
            short_strike, max_pain, current_price
        )
        long_safe, long_dist = self.is_strike_safe(
            long_strike, max_pain, current_price
        )
        
        # Trade is safe if both legs are safe
        is_safe = short_safe and long_safe
        min_distance = min(short_dist, long_dist)
        
        return MaxPainResult(
            max_pain_strike=max_pain,
            current_price=current_price,
            distance_pct=min_distance,
            is_safe=is_safe
        )


# Singleton
_validator: MaxPainValidator | None = None


def get_max_pain_validator() -> MaxPainValidator:
    """Get global max pain validator instance."""
    global _validator
    if _validator is None:
        _validator = MaxPainValidator()
    return _validator
