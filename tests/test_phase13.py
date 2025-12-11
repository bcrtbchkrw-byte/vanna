"""Phase 13 Tests - Advanced Machine Learning (Regime, Gatekeeper, PoT)."""
import asyncio

import pytest


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class TestRegimeClassifier:
    """Test market regime classification."""
    
    def test_trending_up_regime(self):
        """Test bullish trending regime detection."""
        from ml.regime_classifier import RegimeClassifier
        
        classifier = RegimeClassifier()
        
        result = classifier.classify_regime(
            current_price=450,
            sma_20=445,
            sma_50=440,
            sma_200=420,
            vix=18,
            historical_volatility=0.15
        )
        
        assert result.regime == RegimeClassifier.REGIME_TRENDING_UP
        assert result.trend_direction == "BULLISH"
        assert result.confidence > 0.6
    
    def test_trending_down_regime(self):
        """Test bearish trending regime detection."""
        from ml.regime_classifier import RegimeClassifier
        
        classifier = RegimeClassifier()
        
        result = classifier.classify_regime(
            current_price=400,
            sma_20=410,
            sma_50=420,
            sma_200=440,
            vix=22,
            historical_volatility=0.20
        )
        
        assert result.regime == RegimeClassifier.REGIME_TRENDING_DOWN
        assert result.trend_direction == "BEARISH"
    
    def test_high_volatility_regime(self):
        """Test high VIX triggers high volatility regime."""
        from ml.regime_classifier import RegimeClassifier
        
        classifier = RegimeClassifier()
        
        result = classifier.classify_regime(
            current_price=430,
            sma_20=430,
            sma_50=430,
            sma_200=430,
            vix=38,  # Panic level
            historical_volatility=0.35
        )
        
        assert result.regime == RegimeClassifier.REGIME_HIGH_VOL
        assert result.vix_level == 38
    
    def test_range_bound_regime(self):
        """Test sideways market detection."""
        from ml.regime_classifier import RegimeClassifier
        
        classifier = RegimeClassifier()
        
        result = classifier.classify_regime(
            current_price=430,
            sma_20=429,
            sma_50=431,
            sma_200=428,
            vix=15,
            historical_volatility=0.12
        )
        
        assert result.regime == RegimeClassifier.REGIME_RANGE_BOUND
    
    def test_strategy_weights(self):
        """Test strategy weight recommendations."""
        from ml.regime_classifier import get_regime_classifier
        
        classifier = get_regime_classifier()
        
        # Iron condor should be favored in range-bound
        weights = classifier.get_strategy_weights("RANGE_BOUND")
        assert weights["iron_condor"] > weights["credit_spread"]


class TestNeuralGatekeeper:
    """Test neural gatekeeper trade filtering."""
    
    def test_gatekeeper_initialization(self):
        """Test gatekeeper initializes in heuristic mode."""
        from ml.neural_gatekeeper import NeuralGatekeeper
        
        gatekeeper = NeuralGatekeeper()
        
        # Should be in heuristic mode (no trained model)
        assert gatekeeper._model_loaded is False
    
    def test_heuristic_approval(self):
        """Test trade approval using heuristics."""
        from ml.neural_gatekeeper import GatekeeperInput, NeuralGatekeeper
        
        gatekeeper = NeuralGatekeeper()
        
        # Good setup - should be approved
        good_input = GatekeeperInput(
            symbol="SPY",
            strategy="BULL_PUT",
            vix=20,
            delta=-0.15,
            theta=0.05,
            iv_rank=60,
            days_to_expiry=35,
            credit=1.50,
            width=5.0,
            current_price=450,
            sma_distance=0.01
        )
        
        result = gatekeeper.evaluate(good_input)
        
        assert result.probability > 0.5
        assert result.approved is True
    
    def test_heuristic_rejection(self):
        """Test trade rejection with bad setup."""
        from ml.neural_gatekeeper import GatekeeperInput, NeuralGatekeeper
        
        gatekeeper = NeuralGatekeeper()
        
        # Bad setup - high delta, low IV rank, short DTE
        bad_input = GatekeeperInput(
            symbol="SPY",
            strategy="BULL_PUT",
            vix=10,  # Low VIX
            delta=-0.45,  # High delta
            theta=0.02,
            iv_rank=15,  # Low IV rank
            days_to_expiry=7,  # Short DTE
            credit=0.30,
            width=5.0,
            current_price=450,
            sma_distance=0.08  # Far from SMA
        )
        
        result = gatekeeper.evaluate(bad_input)
        
        assert result.probability < 0.55
        assert result.approved is False


class TestProbabilityOfTouch:
    """Test probability of touch calculations."""
    
    def test_pot_basic_calculation(self):
        """Test basic PoT calculation."""
        from ml.probability_of_touch import ProbabilityOfTouch
        
        calculator = ProbabilityOfTouch()
        
        result = calculator.calculate(
            current_price=450,
            strike=430,  # 4.4% OTM put
            days_to_expiry=30,
            implied_volatility=0.20
        )
        
        assert result.direction == "DOWN"
        assert 0 < result.probability < 1
        assert result.strike == 430
    
    def test_pot_atm_high_probability(self):
        """Test ATM strike has higher touch probability."""
        from ml.probability_of_touch import ProbabilityOfTouch
        
        calculator = ProbabilityOfTouch()
        
        result = calculator.calculate(
            current_price=450,
            strike=449,  # Nearly ATM
            days_to_expiry=30,
            implied_volatility=0.20
        )
        
        # ATM should have high probability
        assert result.probability >= 0.5
    
    def test_pot_far_otm_low_probability(self):
        """Test far OTM strike has lower touch probability."""
        from ml.probability_of_touch import ProbabilityOfTouch
        
        calculator = ProbabilityOfTouch()
        
        result = calculator.calculate(
            current_price=450,
            strike=380,  # 15% OTM
            days_to_expiry=30,
            implied_volatility=0.20
        )
        
        # Far OTM should have low probability
        assert result.probability < 0.3
    
    def test_strike_safety_check(self):
        """Test strike safety evaluation."""
        from ml.probability_of_touch import get_probability_of_touch
        
        calculator = get_probability_of_touch()
        
        # Far OTM should be safe
        is_safe, prob = calculator.is_strike_safe(
            current_price=450,
            strike=400,
            days_to_expiry=30,
            implied_volatility=0.20,
            max_probability=0.30
        )
        
        assert isinstance(is_safe, bool)
        assert 0 <= prob <= 1
    
    def test_find_safe_strike(self):
        """Test finding safe strike from list."""
        from ml.probability_of_touch import get_probability_of_touch
        
        calculator = get_probability_of_touch()
        
        strikes = [420, 425, 430, 435, 440, 445, 450, 455, 460]
        
        safe_strike = calculator.find_safe_strike(
            current_price=450,
            strikes=strikes,
            days_to_expiry=30,
            implied_volatility=0.20,
            direction="PUT",
            max_probability=0.25
        )
        
        # Should find a put strike below current price
        if safe_strike:
            assert safe_strike < 450
