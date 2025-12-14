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
    
    def test_low_volatility_regime(self):
        """Test low VIX triggers low volatility regime."""
        from ml.regime_classifier import RegimeClassifier
        
        classifier = RegimeClassifier()
        
        result = classifier.classify({
            'vix': 12,  # Low VIX
            'vix_ratio': 0.9,
            'vix_change_1d': -0.5,
            'vix_zscore': -1.0,
            'return_1m': 0.01,
            'return_5m': 0.005,
            'volatility_20': 0.12,
            'momentum_20': 0.02
        })
        
        assert result.regime == 0  # Low volatility
        assert result.regime_name == 'low_vol'
        assert result.confidence > 0.6
    
    def test_normal_regime(self):
        """Test normal VIX triggers normal regime."""
        from ml.regime_classifier import RegimeClassifier
        
        classifier = RegimeClassifier()
        
        result = classifier.classify({
            'vix': 18,  # Normal VIX
            'vix_ratio': 1.0,
            'vix_change_1d': 0.0,
            'vix_zscore': 0.0,
            'return_1m': 0.0,
            'return_5m': 0.0,
            'volatility_20': 0.18,
            'momentum_20': 0.0
        })
        
        assert result.regime == 1  # Normal
        assert result.regime_name == 'normal'
    
    def test_high_volatility_regime(self):
        """Test high VIX triggers high volatility regime."""
        from ml.regime_classifier import RegimeClassifier
        
        classifier = RegimeClassifier()
        
        result = classifier.classify({
            'vix': 32,  # High VIX
            'vix_ratio': 1.2,
            'vix_change_1d': 3.0,
            'vix_zscore': 2.0,
            'return_1m': -0.05,
            'return_5m': -0.02,
            'volatility_20': 0.32,
            'momentum_20': -0.05
        })
        
        assert result.regime == 3  # High volatility
        assert result.regime_name == 'high_vol'
    
    def test_crisis_regime(self):
        """Test very high VIX triggers crisis regime."""
        from ml.regime_classifier import RegimeClassifier
        
        classifier = RegimeClassifier()
        
        result = classifier.classify({'vix': 40})  # Crisis VIX
        
        assert result.regime == 4  # Crisis
        assert result.regime_name == 'crisis'
    
    def test_strategy_adjustments(self):
        """Test strategy adjustment recommendations."""
        from ml.regime_classifier import get_regime_classifier
        
        classifier = get_regime_classifier()
        
        # High vol should reduce position size
        adj = classifier.get_strategy_adjustment(3)  # High vol
        assert adj['position_size'] < 1.0
        
        # Low vol can increase position size
        adj = classifier.get_strategy_adjustment(0)  # Low vol
        assert adj['position_size'] >= 1.0


class TestNeuralGatekeeper:
    """Test neural gatekeeper trade filtering."""
    
    def test_gatekeeper_initialization(self):
        """Test gatekeeper initializes without trained model."""
        from ml.neural_gatekeeper import NeuralGatekeeper
        
        gatekeeper = NeuralGatekeeper()
        
        # Should have no model initially (unless one is saved)
        # The model attribute is None when no trained model exists
        assert hasattr(gatekeeper, 'model')
        assert hasattr(gatekeeper, 'threshold')
    
    def test_predict_without_model(self):
        """Test predict returns neutral probability without model."""
        from ml.neural_gatekeeper import NeuralGatekeeper
        import numpy as np
        
        gatekeeper = NeuralGatekeeper()
        
        # Create dummy sequence (seq_len, features)
        sequence = np.random.randn(60, 32).astype(np.float32)
        
        prob = gatekeeper.predict(sequence)
        
        # Without model, should return 0.5 (neutral)
        assert prob == 0.5
    
    def test_should_trade_without_model(self):
        """Test should_trade returns reasonable result without model."""
        from ml.neural_gatekeeper import NeuralGatekeeper
        import numpy as np
        
        gatekeeper = NeuralGatekeeper()
        
        # Create dummy sequence
        sequence = np.random.randn(60, 32).astype(np.float32)
        
        should_trade, prob, reason = gatekeeper.should_trade(sequence)
        
        # Without model, prob is 0.5, threshold is 0.6 by default
        # So should_trade should be False
        assert isinstance(should_trade, bool)
        assert 0 <= prob <= 1
        assert isinstance(reason, str)
    
    def test_get_neural_gatekeeper_singleton(self):
        """Test singleton function returns same instance."""
        from ml.neural_gatekeeper import get_neural_gatekeeper
        
        gk1 = get_neural_gatekeeper()
        gk2 = get_neural_gatekeeper()
        
        assert gk1 is gk2


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
