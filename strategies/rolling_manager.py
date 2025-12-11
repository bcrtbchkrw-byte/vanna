"""
Rolling Manager for Vanna Trading Bot.

Handles position defense by rolling options that are under pressure.
Implements automated logic for:
- Rolling up/down untested sides
- Rolling out in time for more credit
- Early exit decisions
"""
from dataclasses import dataclass

from core.logger import get_logger


@dataclass
class RollRecommendation:
    """Recommendation for rolling a position."""
    should_roll: bool
    roll_type: str  # "UP", "DOWN", "OUT", "CLOSE"
    reason: str
    new_strike: float | None
    new_expiry: str | None
    expected_credit: float
    urgency: str  # "LOW", "MEDIUM", "HIGH"


class RollingManager:
    """
    Manages position rolling and defense.
    
    Monitors positions and recommends rolling when:
    - Short options are being tested (price approaching strike)
    - Delta exceeds thresholds
    - Time decay is no longer favorable
    """
    
    # Thresholds
    DELTA_DANGER_THRESHOLD = 0.40  # Roll when delta exceeds this
    TESTED_THRESHOLD_PCT = 2.0  # Roll when price within 2% of strike
    MIN_DAYS_TO_ROLL = 7  # Don't roll with less than 7 DTE
    
    def __init__(self) -> None:
        self.logger = get_logger()
    
    def should_roll(
        self,
        position_type: str,  # "PUT" or "CALL"
        short_strike: float,
        current_price: float,
        current_delta: float,
        days_to_expiry: int,
        current_pnl: float,
        entry_credit: float
    ) -> RollRecommendation:
        """
        Determine if a position should be rolled.
        
        Args:
            position_type: "PUT" or "CALL"
            short_strike: Strike of short option
            current_price: Current underlying price
            current_delta: Current delta of short option
            days_to_expiry: Days until expiration
            current_pnl: Current P&L of position
            entry_credit: Original credit received
            
        Returns:
            RollRecommendation with action to take
        """
        # Calculate distance from strike
        distance = abs(current_price - short_strike)
        distance_pct = distance / short_strike * 100
        
        # Check if being tested
        is_tested = False
        if position_type == "PUT" and current_price < short_strike:
            is_tested = distance_pct < self.TESTED_THRESHOLD_PCT
        elif position_type == "CALL" and current_price > short_strike:
            is_tested = distance_pct < self.TESTED_THRESHOLD_PCT
        
        # Check delta danger
        is_delta_danger = abs(current_delta) > self.DELTA_DANGER_THRESHOLD
        
        # Early roll logic
        if days_to_expiry <= self.MIN_DAYS_TO_ROLL:
            return self._handle_expiration_week(
                position_type, current_pnl, entry_credit
            )
        
        # Immediate action needed
        if is_tested and is_delta_danger:
            return RollRecommendation(
                should_roll=True,
                roll_type="OUT" if position_type == "PUT" else "OUT",
                reason=f"Price testing strike ({distance_pct:.1f}% away) with high delta ({current_delta:.2f})",
                new_strike=self._calculate_new_strike(position_type, short_strike, current_price),
                new_expiry=None,  # Will be calculated
                expected_credit=entry_credit * 0.3,  # Rough estimate
                urgency="HIGH"
            )
        
        # Warning zone
        if is_delta_danger:
            return RollRecommendation(
                should_roll=True,
                roll_type="DOWN" if position_type == "PUT" else "UP",
                reason=f"Delta ({current_delta:.2f}) exceeds threshold",
                new_strike=self._calculate_new_strike(position_type, short_strike, current_price),
                new_expiry=None,
                expected_credit=entry_credit * 0.2,
                urgency="MEDIUM"
            )
        
        # No action needed
        return RollRecommendation(
            should_roll=False,
            roll_type="HOLD",
            reason=f"Position healthy: {distance_pct:.1f}% from strike, delta {current_delta:.2f}",
            new_strike=None,
            new_expiry=None,
            expected_credit=0,
            urgency="LOW"
        )
    
    def _handle_expiration_week(
        self,
        position_type: str,
        current_pnl: float,
        entry_credit: float
    ) -> RollRecommendation:
        """Handle positions in expiration week."""
        profit_pct = current_pnl / (entry_credit * 100) if entry_credit > 0 else 0
        
        # If profitable, close
        if profit_pct >= 0.5:  # 50% profit
            return RollRecommendation(
                should_roll=False,
                roll_type="CLOSE",
                reason=f"Expiration week with {profit_pct:.0%} profit - close early",
                new_strike=None,
                new_expiry=None,
                expected_credit=current_pnl,
                urgency="MEDIUM"
            )
        
        # If losing, roll out
        return RollRecommendation(
            should_roll=True,
            roll_type="OUT",
            reason=f"Expiration week with only {profit_pct:.0%} profit - roll out",
            new_strike=None,
            new_expiry=None,
            expected_credit=entry_credit * 0.5,
            urgency="HIGH"
        )
    
    def _calculate_new_strike(
        self,
        position_type: str,
        current_strike: float,
        current_price: float
    ) -> float:
        """Calculate new strike for rolling."""
        # Roll away from current price
        if position_type == "PUT":
            # Roll down for puts
            return round(current_price * 0.95, 0)  # 5% OTM
        else:
            # Roll up for calls
            return round(current_price * 1.05, 0)  # 5% OTM
    
    def evaluate_spread_roll(
        self,
        short_strike: float,
        long_strike: float,
        current_price: float,
        short_delta: float,
        days_to_expiry: int,
        current_pnl: float,
        entry_credit: float
    ) -> RollRecommendation:
        """Evaluate a spread position for rolling."""
        # Determine if put or call spread
        if short_strike < long_strike:
            # Put spread (short is lower)
            position_type = "PUT"
        else:
            # Call spread (short is higher)
            position_type = "CALL"
        
        return self.should_roll(
            position_type=position_type,
            short_strike=short_strike,
            current_price=current_price,
            current_delta=short_delta,
            days_to_expiry=days_to_expiry,
            current_pnl=current_pnl,
            entry_credit=entry_credit
        )


# Singleton
_rolling_manager: RollingManager | None = None


def get_rolling_manager() -> RollingManager:
    """Get global rolling manager instance."""
    global _rolling_manager
    if _rolling_manager is None:
        _rolling_manager = RollingManager()
    return _rolling_manager
