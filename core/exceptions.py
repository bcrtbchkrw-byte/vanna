"""
Vanna Custom Exceptions

Provides a clear exception hierarchy for better error handling.
Use specific exceptions instead of generic Exception.
"""


class VannaError(Exception):
    """Base exception for all Vanna errors."""
    pass


# ============================================================
# Order-related Exceptions
# ============================================================

class OrderError(VannaError):
    """Base exception for order-related errors."""
    pass


class OrderValidationError(OrderError):
    """Invalid order parameters (quantity, price, legs)."""
    pass


class OrderPlacementError(OrderError):
    """Failed to place order with broker."""
    pass


class InsufficientFundsError(OrderError):
    """Not enough buying power for order."""
    pass


# ============================================================
# Connection-related Exceptions
# ============================================================

class ConnectionError(VannaError):
    """Base exception for connection errors."""
    pass


class IBKRConnectionError(ConnectionError):
    """Failed to connect to IBKR Gateway."""
    pass


class IBKRDisconnectedError(ConnectionError):
    """Lost connection to IBKR during operation."""
    pass


class APIConnectionError(ConnectionError):
    """Failed to connect to external API (Gemini, etc)."""
    pass


# ============================================================
# Circuit Breaker Exceptions
# ============================================================

class CircuitBreakerError(VannaError):
    """Circuit breaker related errors."""
    pass


class CircuitBreakerOpenError(CircuitBreakerError):
    """Circuit breaker is open - service unavailable."""
    
    def __init__(self, service_name: str, retry_after_seconds: float = 0):
        self.service_name = service_name
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Circuit breaker open for {service_name}. "
            f"Retry after {retry_after_seconds:.0f}s"
        )


# ============================================================
# Validation Exceptions
# ============================================================

class ValidationError(VannaError):
    """Invalid input or configuration."""
    pass


class ConfigurationError(ValidationError):
    """Invalid or missing configuration."""
    pass


# ============================================================
# ML/AI Exceptions
# ============================================================

class MLError(VannaError):
    """Machine learning related errors."""
    pass


class ModelNotLoadedError(MLError):
    """ML model not loaded or unavailable."""
    pass


class PredictionError(MLError):
    """Error during prediction."""
    pass


# ============================================================
# Data Exceptions
# ============================================================

class DataError(VannaError):
    """Data-related errors."""
    pass


class DataNotFoundError(DataError):
    """Requested data not found."""
    pass


class DataValidationError(DataError):
    """Data failed validation checks."""
    pass
