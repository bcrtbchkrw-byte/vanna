"""
Probability of Touch Calculator for Vanna Trading Bot.

Calculates the probability that an underlying price will touch
a given strike price before expiration. Uses statistical models
based on implied volatility and time to expiry.
"""
import math
from typing import NamedTuple

from core.logger import get_logger


class ProbOfTouchResult(NamedTuple):
    """Result of probability of touch calculation."""
    probability: float
    days_to_expiry: int
    strike: float
    current_price: float
    direction: str  # "UP" or "DOWN"


class ProbabilityOfTouch:
    """
    Calculates probability of price touching a strike before expiry.
    
    Uses the reflection principle from option pricing theory:
    P(touch) ≈ 2 * N(d2) for out-of-the-money options
    
    Where d2 = ln(S/K) / (σ * √T)
    """
    
    TRADING_DAYS_PER_YEAR = 252
    
    def __init__(self) -> None:
        self.logger = get_logger()
    
    def _norm_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def calculate(
        self,
        current_price: float,
        strike: float,
        days_to_expiry: int,
        implied_volatility: float,
        risk_free_rate: float = 0.05
    ) -> ProbOfTouchResult:
        """
        Calculate probability of touching the strike.
        
        Args:
            current_price: Current underlying price
            strike: Strike price to calculate probability for
            days_to_expiry: Days until expiration
            implied_volatility: Annualized IV as decimal (e.g., 0.25 for 25%)
            risk_free_rate: Risk-free rate as decimal
            
        Returns:
            ProbOfTouchResult with probability and metadata
        """
        if days_to_expiry <= 0:
            # Already expired
            prob = 1.0 if current_price >= strike else 0.0
            direction = "UP" if strike > current_price else "DOWN"
            return ProbOfTouchResult(
                probability=prob,
                days_to_expiry=0,
                strike=strike,
                current_price=current_price,
                direction=direction
            )
        
        # Time to expiry in years
        T = days_to_expiry / self.TRADING_DAYS_PER_YEAR
        
        # Determine direction
        direction = "UP" if strike > current_price else "DOWN"
        
        # Calculate d2 (from Black-Scholes)
        sigma_sqrt_t = implied_volatility * math.sqrt(T)
        
        if sigma_sqrt_t < 0.0001:
            # Very low vol or very short time
            prob = 1.0 if abs(strike - current_price) / current_price < 0.01 else 0.0
        else:
            # Log ratio
            log_ratio = math.log(current_price / strike)
            
            # d2 calculation
            d2 = (log_ratio + (risk_free_rate - 0.5 * implied_volatility**2) * T) / sigma_sqrt_t
            
            # Probability of touch using reflection principle
            # For OTM options: P(touch) ≈ 2 * N(-|d2|)
            prob = 2 * self._norm_cdf(-abs(d2))
            
            # Adjust for ATM options
            if abs(strike - current_price) / current_price < 0.01:
                prob = max(prob, 0.5)  # ATM has at least 50% chance
        
        # Clamp probability
        prob = max(0.0, min(1.0, prob))
        
        self.logger.debug(
            f"PoT: {current_price:.2f} -> {strike:.2f} ({direction}), "
            f"DTE={days_to_expiry}, IV={implied_volatility:.0%} -> {prob:.1%}"
        )
        
        return ProbOfTouchResult(
            probability=prob,
            days_to_expiry=days_to_expiry,
            strike=strike,
            current_price=current_price,
            direction=direction
        )
    
    def is_strike_safe(
        self,
        current_price: float,
        strike: float,
        days_to_expiry: int,
        implied_volatility: float,
        max_probability: float = 0.30
    ) -> tuple[bool, float]:
        """
        Check if a strike has acceptable probability of touch.
        
        Args:
            current_price: Current underlying price
            strike: Strike to evaluate
            days_to_expiry: Days until expiry
            implied_volatility: IV as decimal
            max_probability: Maximum acceptable touch probability
            
        Returns:
            Tuple of (is_safe, probability)
        """
        result = self.calculate(
            current_price=current_price,
            strike=strike,
            days_to_expiry=days_to_expiry,
            implied_volatility=implied_volatility
        )
        
        is_safe = result.probability <= max_probability
        
        if not is_safe:
            self.logger.warning(
                f"Strike {strike} has {result.probability:.0%} PoT "
                f"(max: {max_probability:.0%})"
            )
        
        return is_safe, result.probability
    
    def find_safe_strike(
        self,
        current_price: float,
        strikes: list[float],
        days_to_expiry: int,
        implied_volatility: float,
        direction: str,
        max_probability: float = 0.30
    ) -> float | None:
        """
        Find the closest strike with acceptable touch probability.
        
        Args:
            current_price: Current underlying price
            strikes: Available strikes to choose from
            days_to_expiry: Days until expiry
            implied_volatility: IV as decimal
            direction: "PUT" for below price, "CALL" for above
            max_probability: Maximum acceptable touch probability
            
        Returns:
            Best strike or None if none found
        """
        # Sort strikes appropriately
        if direction.upper() == "PUT":
            # For puts, want strikes below current price, sorted descending
            valid_strikes = sorted(
                [s for s in strikes if s < current_price],
                reverse=True
            )
        else:
            # For calls, want strikes above current price, sorted ascending
            valid_strikes = sorted(
                [s for s in strikes if s > current_price]
            )
        
        for strike in valid_strikes:
            is_safe, prob = self.is_strike_safe(
                current_price=current_price,
                strike=strike,
                days_to_expiry=days_to_expiry,
                implied_volatility=implied_volatility,
                max_probability=max_probability
            )
            if is_safe:
                self.logger.info(
                    f"Found safe {direction} strike: {strike} (PoT: {prob:.0%})"
                )
                return strike
        
        self.logger.warning(f"No safe {direction} strike found within threshold")
        return None


# Singleton
_pot_calculator: ProbabilityOfTouch | None = None


def get_probability_of_touch() -> ProbabilityOfTouch:
    """Get global probability of touch calculator."""
    global _pot_calculator
    if _pot_calculator is None:
        _pot_calculator = ProbabilityOfTouch()
    return _pot_calculator
