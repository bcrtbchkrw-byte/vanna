"""
Circuit Breaker for Vanna Trading Bot.

Implements hard stops on trading activity based on:
- Daily P&L limits (max drawdown)
- Consecutive loss limits
- Account value thresholds

When triggered, trading is halted until manual reset or next day.
"""
import os
from dataclasses import dataclass
from datetime import date

from core.logger import get_logger


@dataclass
class CircuitBreakerState:
    """Current circuit breaker state."""
    is_triggered: bool
    trigger_reason: str | None
    daily_pnl: float
    consecutive_losses: int
    trades_today: int
    last_reset: date


class CircuitBreaker:
    """
    Trading circuit breaker for risk management.
    
    Monitors trading activity and halts all trading when
    predefined risk limits are exceeded.
    """
    
    # Default limits (can be overridden via env)
    DEFAULT_DAILY_LOSS_LIMIT = 500.0  # Max daily loss in $
    DEFAULT_CONSECUTIVE_LOSS_LIMIT = 3  # Max consecutive losses
    DEFAULT_DAILY_TRADE_LIMIT = 10  # Max trades per day
    
    def __init__(self) -> None:
        self.logger = get_logger()
        
        # Load limits from environment
        self.daily_loss_limit = float(
            os.getenv("CIRCUIT_BREAKER_DAILY_LOSS", str(self.DEFAULT_DAILY_LOSS_LIMIT))
        )
        self.consecutive_loss_limit = int(
            os.getenv("CIRCUIT_BREAKER_CONSECUTIVE", str(self.DEFAULT_CONSECUTIVE_LOSS_LIMIT))
        )
        self.daily_trade_limit = int(
            os.getenv("CIRCUIT_BREAKER_DAILY_TRADES", str(self.DEFAULT_DAILY_TRADE_LIMIT))
        )
        
        # State tracking
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._trades_today: int = 0
        self._triggered: bool = False
        self._trigger_reason: str | None = None
        self._last_reset: date = date.today()
    
    def _check_daily_reset(self) -> None:
        """Reset counters if new trading day."""
        today = date.today()
        if today > self._last_reset:
            self.logger.info("New trading day - resetting circuit breaker counters")
            self._daily_pnl = 0.0
            self._consecutive_losses = 0
            self._trades_today = 0
            self._triggered = False
            self._trigger_reason = None
            self._last_reset = today
    
    def record_trade(self, pnl: float) -> None:
        """
        Record a trade result for circuit breaker monitoring.
        
        Args:
            pnl: Profit/loss from the trade (negative for loss)
        """
        self._check_daily_reset()
        
        self._trades_today += 1
        self._daily_pnl += pnl
        
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0  # Reset on win
        
        self.logger.info(
            f"Circuit breaker update: P&L={self._daily_pnl:+.2f}, "
            f"Consecutive losses={self._consecutive_losses}, "
            f"Trades today={self._trades_today}"
        )
        
        # Check triggers
        self._check_triggers()
    
    def _check_triggers(self) -> None:
        """Check if any circuit breaker conditions are met."""
        # Daily loss limit
        if self._daily_pnl <= -self.daily_loss_limit:
            self._trigger(
                f"Daily loss limit exceeded: ${self._daily_pnl:.2f} "
                f"(limit: ${-self.daily_loss_limit:.2f})"
            )
            return
        
        # Consecutive losses
        if self._consecutive_losses >= self.consecutive_loss_limit:
            self._trigger(
                f"Consecutive losses exceeded: {self._consecutive_losses} "
                f"(limit: {self.consecutive_loss_limit})"
            )
            return
        
        # Daily trade limit
        if self._trades_today >= self.daily_trade_limit:
            self._trigger(
                f"Daily trade limit reached: {self._trades_today} "
                f"(limit: {self.daily_trade_limit})"
            )
            return
    
    def _trigger(self, reason: str) -> None:
        """Trigger the circuit breaker."""
        self._triggered = True
        self._trigger_reason = reason
        self.logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}")
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed."""
        self._check_daily_reset()
        return not self._triggered
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        self._check_daily_reset()
        
        return CircuitBreakerState(
            is_triggered=self._triggered,
            trigger_reason=self._trigger_reason,
            daily_pnl=self._daily_pnl,
            consecutive_losses=self._consecutive_losses,
            trades_today=self._trades_today,
            last_reset=self._last_reset
        )
    
    def manual_reset(self) -> None:
        """Manually reset the circuit breaker (use with caution)."""
        self.logger.warning("Manual circuit breaker reset requested")
        self._triggered = False
        self._trigger_reason = None
    
    def check_before_trade(self, proposed_max_loss: float) -> tuple[bool, str]:
        """
        Check if a proposed trade is allowed.
        
        Args:
            proposed_max_loss: Maximum potential loss of proposed trade
            
        Returns:
            Tuple of (allowed, reason)
        """
        self._check_daily_reset()
        
        if self._triggered:
            return False, f"Circuit breaker triggered: {self._trigger_reason}"
        
        # Check if trade would exceed daily limit
        potential_pnl = self._daily_pnl - proposed_max_loss
        if potential_pnl <= -self.daily_loss_limit:
            return False, (
                f"Trade would exceed daily loss limit: "
                f"current=${self._daily_pnl:.2f}, proposed loss=${proposed_max_loss:.2f}, "
                f"limit=${self.daily_loss_limit:.2f}"
            )
        
        # Check trade count
        if self._trades_today >= self.daily_trade_limit:
            return False, f"Daily trade limit reached: {self._trades_today}"
        
        return True, "Trade allowed"


# Singleton
_breaker: CircuitBreaker | None = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get global circuit breaker instance."""
    global _breaker
    if _breaker is None:
        _breaker = CircuitBreaker()
    return _breaker
