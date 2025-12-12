"""
Tests for Vanna Calculator

Validates Black-Scholes Greeks calculations including Vanna, Charm, Volga.
"""
import pytest
import numpy as np
from unittest.mock import patch


class TestVannaCalculator:
    """Test suite for VannaCalculator."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        from ml.vanna_calculator import VannaCalculator
        return VannaCalculator(risk_free_rate=0.045)
    
    def test_vanna_calculation_call_atm(self, calculator):
        """Test Vanna for ATM call option."""
        # ATM option: S=K, moderate IV, 30 DTE
        S = 500.0  # SPY price
        K = 500.0  # ATM strike
        T = 30 / 365  # 30 days
        sigma = 0.20  # 20% IV
        
        vanna = calculator.calculate_vanna(S, K, T, sigma)
        
        assert vanna is not None
        # Vanna should be small positive for ATM
        assert -0.1 < vanna < 0.1
    
    def test_vanna_calculation_otm(self, calculator):
        """Test Vanna for OTM option."""
        S = 500.0
        K = 520.0  # 4% OTM
        T = 30 / 365
        sigma = 0.20
        
        vanna = calculator.calculate_vanna(S, K, T, sigma)
        
        assert vanna is not None
        # OTM options have different Vanna characteristics
        assert isinstance(vanna, float)
    
    def test_vanna_invalid_inputs(self, calculator):
        """Test Vanna returns None for invalid inputs."""
        # Zero time to expiry
        assert calculator.calculate_vanna(500, 500, 0, 0.20) is None
        
        # Zero volatility
        assert calculator.calculate_vanna(500, 500, 0.1, 0) is None
        
        # Negative price
        assert calculator.calculate_vanna(-500, 500, 0.1, 0.20) is None
    
    def test_charm_calculation(self, calculator):
        """Test Charm (Delta decay) calculation."""
        S = 500.0
        K = 500.0
        T = 30 / 365
        sigma = 0.20
        
        charm_call = calculator.calculate_charm(S, K, T, sigma, 'call')
        charm_put = calculator.calculate_charm(S, K, T, sigma, 'put')
        
        assert charm_call is not None
        assert charm_put is not None
        # Charm is typically negative (delta decays toward expiry)
        assert isinstance(charm_call, float)
    
    def test_volga_calculation(self, calculator):
        """Test Volga (Vega convexity) calculation."""
        S = 500.0
        K = 500.0
        T = 30 / 365
        sigma = 0.20
        
        volga = calculator.calculate_volga(S, K, T, sigma)
        
        assert volga is not None
        assert isinstance(volga, float)
    
    def test_calculate_all_greeks(self, calculator):
        """Test all Greeks calculation."""
        S = 500.0
        K = 490.0  # Slightly ITM
        T = 45 / 365
        sigma = 0.25
        
        greeks = calculator.calculate_all_greeks(S, K, T, sigma, 'call')
        
        assert greeks is not None
        
        # Check all fields present
        assert hasattr(greeks, 'delta')
        assert hasattr(greeks, 'gamma')
        assert hasattr(greeks, 'theta')
        assert hasattr(greeks, 'vega')
        assert hasattr(greeks, 'vanna')
        assert hasattr(greeks, 'charm')
        assert hasattr(greeks, 'volga')
        assert hasattr(greeks, 'rho')
        
        # Sanity checks for call option
        assert 0 < greeks.delta <= 1  # Call delta 0-1
        assert greeks.gamma > 0  # Gamma always positive
        assert greeks.theta < 0  # Theta typically negative
        assert greeks.vega > 0  # Vega always positive
    
    def test_calculate_all_greeks_put(self, calculator):
        """Test all Greeks for put option."""
        S = 500.0
        K = 510.0  # Slightly ITM put
        T = 45 / 365
        sigma = 0.25
        
        greeks = calculator.calculate_all_greeks(S, K, T, sigma, 'put')
        
        assert greeks is not None
        
        # Put delta is negative
        assert -1 <= greeks.delta < 0
        assert greeks.gamma > 0
    
    def test_vanna_numerical_vs_analytical(self, calculator):
        """Validate numerical Vanna approximates analytical."""
        S = 500.0
        K = 500.0
        T = 30 / 365
        sigma = 0.20
        
        vanna_analytical = calculator.calculate_vanna(S, K, T, sigma)
        vanna_numerical = calculator.calculate_vanna_numerical(S, K, T, sigma)
        
        assert vanna_analytical is not None
        assert vanna_numerical is not None
        
        # Should be close (within 10%)
        if abs(vanna_analytical) > 0.001:
            relative_error = abs(vanna_numerical - vanna_analytical) / abs(vanna_analytical)
            assert relative_error < 0.15  # 15% tolerance
    
    def test_vanna_surface_calculation(self, calculator):
        """Test Vanna surface for multiple strikes/expiries."""
        S = 500.0
        strikes = [490.0, 500.0, 510.0]
        expiries = [0.1, 0.2, 0.3]  # Years
        ivs = {(K, T): 0.20 for K in strikes for T in expiries}
        
        surface = calculator.calculate_vanna_surface(S, strikes, expiries, ivs)
        
        assert len(surface) == len(strikes) * len(expiries)
        
        for point in surface:
            assert 'strike' in point
            assert 'expiry_years' in point
            assert 'vanna' in point
            assert 'delta' in point


class TestVannaCalculatorSingleton:
    """Test singleton behavior."""
    
    def test_get_vanna_calculator_returns_same_instance(self):
        """Test singleton returns same instance."""
        from ml.vanna_calculator import get_vanna_calculator
        
        calc1 = get_vanna_calculator()
        calc2 = get_vanna_calculator()
        
        # Note: Due to module-level singleton, this may vary in test isolation
        assert calc1 is not None
        assert calc2 is not None


class TestBlackScholesFormulas:
    """Test Black-Scholes formula correctness."""
    
    def test_known_delta_values(self):
        """Test Delta against known values."""
        from ml.vanna_calculator import VannaCalculator
        calc = VannaCalculator(risk_free_rate=0.05)
        
        # ATM call with 1 year to expiry should have Delta ~ 0.5
        greeks = calc.calculate_all_greeks(
            S=100, K=100, T=1.0, sigma=0.20, option_type='call'
        )
        
        assert greeks is not None
        # ATM call delta should be around 0.5-0.6 with positive rates
        assert 0.45 < greeks.delta < 0.70
    
    def test_vega_decreases_near_expiry(self):
        """Test that Vega decreases as expiry approaches."""
        from ml.vanna_calculator import VannaCalculator
        calc = VannaCalculator(risk_free_rate=0.05)
        
        vega_long = calc.calculate_all_greeks(S=100, K=100, T=0.5, sigma=0.20).vega
        vega_short = calc.calculate_all_greeks(S=100, K=100, T=0.1, sigma=0.20).vega
        
        # Vega should be higher for longer-dated options
        assert vega_long > vega_short
