"""
Circuit Breaker Pattern

Prevents cascading failures when external services are down.
Opens after N consecutive failures, closes after timeout.
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core.logger import get_logger
from core.exceptions import CircuitBreakerOpenError

logger = get_logger()


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Simple circuit breaker for external service calls.
    
    Usage:
        cb = CircuitBreaker("ibkr", failure_threshold=3, recovery_timeout=60)
        
        async def call_service():
            cb.check()  # Raises if open
            try:
                result = await external_call()
                cb.record_success()
                return result
            except Exception as e:
                cb.record_failure()
                raise
    """
    
    service_name: str
    failure_threshold: int = 5      # Open after N consecutive failures
    recovery_timeout: float = 60.0  # Seconds before trying again
    
    # Internal state
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    last_failure_time: float = field(default=0)
    success_count: int = field(default=0)
    
    def check(self) -> None:
        """
        Check if circuit allows operation.
        
        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        if self.state == CircuitState.CLOSED:
            return
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout passed
            time_since_failure = time.time() - self.last_failure_time
            if time_since_failure >= self.recovery_timeout:
                # Move to half-open, allow one request
                self.state = CircuitState.HALF_OPEN
                logger.info(f"⚡ Circuit {self.service_name}: HALF_OPEN (testing)")
                return
            
            # Still open
            retry_after = self.recovery_timeout - time_since_failure
            raise CircuitBreakerOpenError(self.service_name, retry_after)
        
        # Half-open: allow request (will be tested)
        return
    
    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            # Success in half-open means service recovered
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            logger.info(f"✅ Circuit {self.service_name}: CLOSED (recovered)")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset failures on success
        
        self.success_count += 1
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open means service still down
            self.state = CircuitState.OPEN
            logger.warning(f"🔴 Circuit {self.service_name}: OPEN (still failing)")
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"🔴 Circuit {self.service_name}: OPEN after "
                    f"{self.failure_count} consecutive failures"
                )
    
    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"🔄 Circuit {self.service_name}: Manual RESET")
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN


# ============================================================
# Global Circuit Breakers for Services
# ============================================================

_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
) -> CircuitBreaker:
    """Get or create a circuit breaker for a service."""
    if service_name not in _circuit_breakers:
        _circuit_breakers[service_name] = CircuitBreaker(
            service_name=service_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
    return _circuit_breakers[service_name]


def get_ibkr_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for IBKR connection."""
    return get_circuit_breaker("ibkr", failure_threshold=3, recovery_timeout=30.0)


def get_gemini_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for Gemini API."""
    return get_circuit_breaker("gemini", failure_threshold=5, recovery_timeout=60.0)
