"""Phase 14 Tests - Validation and Logic (Sanity, Max Pain, Circuit Breaker)."""
import asyncio

import pytest


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class TestSanityChecker:
    """Test AI sanity checker."""
    
    @pytest.mark.asyncio
    async def test_rule_based_check_pass(self):
        """Test rule-based sanity check with good setup."""
        from validation.sanity_checker import SanityChecker
        
        checker = SanityChecker()
        
        result = await checker.verify_setup(
            symbol="SPY",
            strategy="BULL_PUT",
            entry_price=1.50,
            max_loss=350,
            days_to_expiry=30,
            vix=18,
            delta=-0.15,
            confidence_score=0.75,
            market_regime="RANGE_BOUND"
        )
        
        assert result.passed is True
        assert result.confidence >= 0.5
    
    @pytest.mark.asyncio
    async def test_rule_based_check_fail(self):
        """Test rule-based sanity check with bad setup."""
        from validation.sanity_checker import SanityChecker
        
        checker = SanityChecker()
        
        result = await checker.verify_setup(
            symbol="SPY",
            strategy="BULL_PUT",
            entry_price=0.50,
            max_loss=500,
            days_to_expiry=5,  # Too short
            vix=40,  # Too high
            delta=-0.45,  # Too high
            confidence_score=0.3,  # Too low
            market_regime="HIGH_VOLATILITY"
        )
        
        assert result.passed is False
        assert len(result.concerns) > 0


class TestMaxPainValidator:
    """Test max pain validation."""
    
    def test_max_pain_calculation(self):
        """Test max pain calculation from option chain."""
        from validation.max_pain_validator import MaxPainValidator
        
        validator = MaxPainValidator()
        
        # Sample option chain
        chain = [
            {"strike": 440, "call_oi": 1000, "put_oi": 5000},
            {"strike": 445, "call_oi": 2000, "put_oi": 4000},
            {"strike": 450, "call_oi": 5000, "put_oi": 5000},  # Expected max pain
            {"strike": 455, "call_oi": 4000, "put_oi": 2000},
            {"strike": 460, "call_oi": 5000, "put_oi": 1000},
        ]
        
        max_pain = validator.calculate_max_pain(chain)
        
        assert max_pain is not None
        assert 440 <= max_pain <= 460
    
    def test_strike_safety(self):
        """Test strike safety evaluation."""
        from validation.max_pain_validator import get_max_pain_validator
        
        validator = get_max_pain_validator()
        
        # Strike far from max pain should be safe
        is_safe, distance = validator.is_strike_safe(
            strike=430,
            max_pain_strike=450,
            current_price=450
        )
        
        assert distance > 3.0  # 4.4% away
        assert is_safe is True
    
    def test_strike_unsafety(self):
        """Test unsafe strike detection."""
        from validation.max_pain_validator import MaxPainValidator
        
        validator = MaxPainValidator()
        
        # Strike very close to max pain
        is_safe, distance = validator.is_strike_safe(
            strike=449,
            max_pain_strike=450,
            current_price=450
        )
        
        assert distance < 3.0  # Only 0.2% away
        assert is_safe is False


class TestCircuitBreaker:
    """Test trading circuit breaker."""
    
    def test_initial_state(self):
        """Test circuit breaker initial state."""
        from validation.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker()
        
        state = breaker.get_state()
        
        assert state.is_triggered is False
        assert state.daily_pnl == 0.0
        assert state.consecutive_losses == 0
    
    def test_record_winning_trade(self):
        """Test recording a winning trade."""
        from validation.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker()
        breaker.record_trade(pnl=100.0)
        
        state = breaker.get_state()
        
        assert state.daily_pnl == 100.0
        assert state.trades_today == 1
        assert breaker.is_trading_allowed() is True
    
    def test_consecutive_loss_trigger(self):
        """Test circuit breaker triggers on consecutive losses."""
        from validation.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker()
        breaker.consecutive_loss_limit = 3  # Override for test
        
        breaker.record_trade(pnl=-50)
        breaker.record_trade(pnl=-50)
        breaker.record_trade(pnl=-50)
        
        assert breaker.is_trading_allowed() is False
        
        state = breaker.get_state()
        assert state.is_triggered is True
        assert "consecutive" in state.trigger_reason.lower()
    
    def test_daily_loss_trigger(self):
        """Test circuit breaker triggers on daily loss limit."""
        from validation.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker()
        breaker.daily_loss_limit = 200  # Override for test
        
        breaker.record_trade(pnl=-250)  # Exceeds limit
        
        assert breaker.is_trading_allowed() is False
        
        state = breaker.get_state()
        assert state.is_triggered is True
    
    def test_check_before_trade(self):
        """Test pre-trade validation."""
        from validation.circuit_breaker import get_circuit_breaker
        
        breaker = get_circuit_breaker()
        breaker._triggered = False  # Reset for test
        breaker._daily_pnl = 0
        breaker._trades_today = 0
        
        allowed, reason = breaker.check_before_trade(proposed_max_loss=100)
        
        assert allowed is True
        assert "allowed" in reason.lower()
    
    def test_manual_reset(self):
        """Test manual reset functionality."""
        from validation.circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker()
        breaker._trigger("Test trigger")
        
        assert breaker.is_trading_allowed() is False
        
        breaker.manual_reset()
        
        assert breaker.is_trading_allowed() is True
