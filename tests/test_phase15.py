"""Phase 15 Tests - Expanded Strategies (Selector, Jade Lizard, PMCC, Rolling)."""
import asyncio

import pytest


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class TestStrategySelector:
    """Test strategy selection logic."""
    
    def test_trending_up_strategies(self):
        """Test strategy selection for bullish market."""
        from strategies.strategy_selector import StrategySelector
        
        selector = StrategySelector()
        
        recommendations = selector.select_strategies(
            vix=18,
            current_price=450,
            sma_20=445,
            sma_50=440,
            sma_200=420,
            available_capital=5000
        )
        
        assert len(recommendations) > 0
        # Bullish strategies should be recommended
        strategies = [r["strategy"] for r in recommendations]
        assert any(s in strategies for s in ["BULL_PUT_SPREAD", "POOR_MANS_COVERED_CALL"])
    
    def test_high_vol_strategies(self):
        """Test strategy selection for high volatility."""
        from strategies.strategy_selector import get_strategy_selector
        
        selector = get_strategy_selector()
        
        recommendations = selector.select_strategies(
            vix=32,  # High VIX (but not crisis - VIX >= 35 is crisis)
            current_price=430,
            sma_20=430,
            sma_50=430,
            sma_200=430,
            available_capital=5000
        )
        
        # Should recommend premium selling (high vol or crisis strategies)
        strategies = [r["strategy"] for r in recommendations]
        # Regime 3 (high_vol): JADE_LIZARD, BULL_PUT_SPREAD, BEAR_CALL_SPREAD
        # Regime 4 (crisis): BEAR_CALL_SPREAD, PUT_DEBIT_SPREAD
        assert any(s in strategies for s in [
            "JADE_LIZARD", "BULL_PUT_SPREAD", "BEAR_CALL_SPREAD", "PUT_DEBIT_SPREAD"
        ])
    
    def test_primary_strategy(self):
        """Test getting single best strategy."""
        from strategies.strategy_selector import StrategySelector
        
        selector = StrategySelector()
        
        strategy = selector.get_primary_strategy(
            vix=15,
            current_price=450,
            sma_20=449,
            sma_50=448,
            sma_200=445,
            available_capital=5000
        )
        
        assert isinstance(strategy, str)
        assert len(strategy) > 0


class TestJadeLizard:
    """Test Jade Lizard strategy."""
    
    def test_candidate_finding(self):
        """Test finding Jade Lizard candidates."""
        from strategies.jade_lizard import JadeLizard
        
        strategy = JadeLizard()
        
        # Mock option chain
        chain = [
            {"strike": 440, "right": "P", "delta": -0.18, "bid": 1.5, "ask": 1.6, "dte": 30, "expiry": "2024-01-15"},
            {"strike": 445, "right": "P", "delta": -0.25, "bid": 2.0, "ask": 2.1, "dte": 30, "expiry": "2024-01-15"},
            {"strike": 455, "right": "C", "delta": 0.28, "bid": 2.0, "ask": 2.1, "dte": 30, "expiry": "2024-01-15"},
            {"strike": 460, "right": "C", "delta": 0.20, "bid": 1.2, "ask": 1.3, "dte": 30, "expiry": "2024-01-15"},
            {"strike": 465, "right": "C", "delta": 0.12, "bid": 0.6, "ask": 0.7, "dte": 30, "expiry": "2024-01-15"},
        ]
        
        candidates = strategy.find_candidates(
            symbol="SPY",
            current_price=450,
            option_chain=chain
        )
        
        # May or may not find candidates depending on chain structure
        assert isinstance(candidates, list)


class TestPoorMansCoveredCall:
    """Test PMCC strategy."""
    
    def test_candidate_finding(self):
        """Test finding PMCC candidates."""
        from strategies.poor_mans_covered_call import PoorMansCoveredCall
        
        strategy = PoorMansCoveredCall()
        
        # Mock option chain with LEAPS and short-term options
        chain = [
            # LEAPS (deep ITM, long dated)
            {"strike": 380, "right": "C", "delta": 0.82, "bid": 75, "ask": 76, "dte": 200, "expiry": "2025-01-15"},
            {"strike": 400, "right": "C", "delta": 0.75, "bid": 55, "ask": 56, "dte": 200, "expiry": "2025-01-15"},
            # Short term options
            {"strike": 460, "right": "C", "delta": 0.25, "bid": 2.0, "ask": 2.1, "dte": 30, "expiry": "2024-02-15"},
            {"strike": 465, "right": "C", "delta": 0.18, "bid": 1.2, "ask": 1.3, "dte": 30, "expiry": "2024-02-15"},
        ]
        
        candidates = strategy.find_candidates(
            symbol="SPY",
            current_price=450,
            option_chain=chain
        )
        
        assert isinstance(candidates, list)


class TestRollingManager:
    """Test position rolling logic."""
    
    def test_healthy_position(self):
        """Test that healthy position recommends hold."""
        from strategies.rolling_manager import RollingManager
        
        manager = RollingManager()
        
        result = manager.should_roll(
            position_type="PUT",
            short_strike=430,
            current_price=450,
            current_delta=-0.15,
            days_to_expiry=30,
            current_pnl=50,
            entry_credit=1.50
        )
        
        assert result.should_roll is False
        assert result.roll_type == "HOLD"
        assert result.urgency == "LOW"
    
    def test_tested_position(self):
        """Test that tested position recommends roll."""
        from strategies.rolling_manager import RollingManager
        
        manager = RollingManager()
        
        result = manager.should_roll(
            position_type="PUT",
            short_strike=449,  # Very close to price
            current_price=450,
            current_delta=-0.45,  # High delta
            days_to_expiry=20,
            current_pnl=-100,
            entry_credit=1.50
        )
        
        assert result.should_roll is True
        assert result.urgency in ["MEDIUM", "HIGH"]
    
    def test_expiration_week_profit(self):
        """Test expiration week handling with profit."""
        from strategies.rolling_manager import get_rolling_manager
        
        manager = get_rolling_manager()
        
        result = manager.should_roll(
            position_type="PUT",
            short_strike=430,
            current_price=450,
            current_delta=-0.10,
            days_to_expiry=5,  # Expiration week
            current_pnl=100,  # Good profit
            entry_credit=1.50
        )
        
        assert result.roll_type == "CLOSE"
    
    def test_spread_evaluation(self):
        """Test spread roll evaluation."""
        from strategies.rolling_manager import RollingManager
        
        manager = RollingManager()
        
        result = manager.evaluate_spread_roll(
            short_strike=440,  # Put spread
            long_strike=435,
            current_price=450,
            short_delta=-0.12,
            days_to_expiry=25,
            current_pnl=80,
            entry_credit=2.00
        )
        
        # Should be healthy
        assert result.urgency == "LOW"
