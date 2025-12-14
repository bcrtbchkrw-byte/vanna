"""
Tests for core security modules.

Tests circuit breaker, exceptions, and audit logger.
"""
import pytest
import tempfile
import json
from pathlib import Path

from core.exceptions import (
    VannaError,
    OrderValidationError,
    OrderPlacementError,
    CircuitBreakerOpenError,
    IBKRDisconnectedError
)
from core.circuit_breaker import CircuitBreaker, CircuitState, get_ibkr_circuit_breaker
from core.audit_logger import AuditLogger


class TestExceptions:
    """Test custom exception hierarchy."""
    
    def test_vanna_error_is_base(self):
        """VannaError is base for all custom exceptions."""
        with pytest.raises(VannaError):
            raise OrderValidationError("test")
    
    def test_order_validation_error(self):
        """OrderValidationError contains message."""
        try:
            raise OrderValidationError("Invalid quantity")
        except OrderValidationError as e:
            assert "Invalid quantity" in str(e)
    
    def test_circuit_breaker_open_error(self):
        """CircuitBreakerOpenError contains service info."""
        err = CircuitBreakerOpenError("ibkr", retry_after_seconds=30)
        assert err.service_name == "ibkr"
        assert err.retry_after_seconds == 30
        assert "ibkr" in str(err)


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
    def test_initial_state_closed(self):
        """New circuit breaker starts closed."""
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert not cb.is_open
    
    def test_opens_after_threshold(self):
        """Opens after failure_threshold consecutive failures."""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        
        cb.record_failure()  # 3rd failure
        assert cb.state == CircuitState.OPEN
        assert cb.is_open
    
    def test_check_raises_when_open(self):
        """Check raises CircuitBreakerOpenError when open."""
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            cb.check()
        
        assert exc_info.value.service_name == "test"
    
    def test_success_resets_failure_count(self):
        """Success resets failure_count when closed."""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2
        
        cb.record_success()
        assert cb.failure_count == 0
    
    def test_reset_clears_state(self):
        """Manual reset clears all state."""
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_singleton_factory(self):
        """get_ibkr_circuit_breaker returns same instance."""
        cb1 = get_ibkr_circuit_breaker()
        cb2 = get_ibkr_circuit_breaker()
        assert cb1 is cb2


class TestAuditLogger:
    """Test audit logging."""
    
    def test_log_order_attempt(self, tmp_path):
        """Log order attempt creates entry."""
        audit = AuditLogger(str(tmp_path))
        
        corr_id = audit.log_order_attempt(
            symbol="SPY",
            action="SELL",
            quantity=1,
            price=1.50,
            strategy="BULL_PUT",
            order_type="SPREAD"
        )
        
        assert len(corr_id) == 8  # UUID[:8]
        
        # Check file was created
        files = list(tmp_path.glob("trades_*.jsonl"))
        assert len(files) == 1
        
        # Check entry content
        with open(files[0]) as f:
            entry = json.loads(f.readline())
        
        assert entry["event"] == "ORDER_ATTEMPT"
        assert entry["symbol"] == "SPY"
        assert entry["correlation_id"] == corr_id
    
    def test_log_order_result(self, tmp_path):
        """Log order result links to attempt."""
        audit = AuditLogger(str(tmp_path))
        
        corr_id = audit.log_order_attempt(
            symbol="SPY", action="SELL", quantity=1, 
            price=1.50, strategy="TEST"
        )
        
        audit.log_order_result(
            correlation_id=corr_id,
            success=True,
            order_id="12345"
        )
        
        # Check both entries
        files = list(tmp_path.glob("trades_*.jsonl"))
        with open(files[0]) as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        result = json.loads(lines[1])
        assert result["event"] == "ORDER_RESULT"
        assert result["correlation_id"] == corr_id
        assert result["success"] is True
    
    def test_log_validation_error(self, tmp_path):
        """Log validation errors."""
        audit = AuditLogger(str(tmp_path))
        
        audit.log_validation_error(
            symbol="SPY",
            error_type="SPREAD_ORDER",
            details="Quantity must be positive"
        )
        
        files = list(tmp_path.glob("trades_*.jsonl"))
        with open(files[0]) as f:
            entry = json.loads(f.readline())
        
        assert entry["event"] == "VALIDATION_ERROR"
        assert entry["error_type"] == "SPREAD_ORDER"


class TestOrderManagerValidation:
    """Test order manager input validation."""
    
    def test_empty_legs_raises(self):
        """Empty legs list raises OrderValidationError."""
        from execution.order_manager import OrderManager
        
        om = OrderManager()
        
        with pytest.raises(OrderValidationError) as exc_info:
            om._validate_spread_order([], 1, 1.0)
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_invalid_quantity_raises(self):
        """Non-positive quantity raises OrderValidationError."""
        from execution.order_manager import OrderManager
        from ib_insync import Contract
        
        om = OrderManager()
        legs = [{"contract": Contract(), "action": "BUY"}]
        
        with pytest.raises(OrderValidationError) as exc_info:
            om._validate_spread_order(legs, 0, 1.0)
        
        assert "quantity" in str(exc_info.value).lower()
    
    def test_excessive_quantity_raises(self):
        """Quantity over MAX_QUANTITY raises OrderValidationError."""
        from execution.order_manager import OrderManager, MAX_QUANTITY
        from ib_insync import Contract
        
        om = OrderManager()
        legs = [{"contract": Contract(), "action": "BUY"}]
        
        with pytest.raises(OrderValidationError) as exc_info:
            om._validate_spread_order(legs, MAX_QUANTITY + 1, 1.0)
        
        assert "exceeds" in str(exc_info.value).lower()
