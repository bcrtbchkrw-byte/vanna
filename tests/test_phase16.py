"""Phase 16 Tests - Quantitative Core (IV Surface, Skew)."""
import asyncio

import pytest


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_option_chain():
    """Create a mock option chain with realistic IV data."""
    chain = []
    underlying_price = 450
    
    for dte in [15, 30, 45, 60]:
        for strike_offset in [-30, -20, -10, -5, 0, 5, 10, 20, 30]:
            strike = underlying_price + strike_offset
            moneyness = strike / underlying_price
            
            # Simulate put skew - lower strikes have higher IV
            base_iv = 0.20
            if moneyness < 0.95:
                iv = base_iv + (0.95 - moneyness) * 0.5  # Put skew
            elif moneyness > 1.05:
                iv = base_iv + (moneyness - 1.05) * 0.2  # Slight call elevation
            else:
                iv = base_iv
            
            # Puts
            chain.append({
                "strike": strike,
                "dte": dte,
                "iv": iv,
                "impliedVolatility": iv,
                "right": "P"
            })
            
            # Calls
            chain.append({
                "strike": strike,
                "dte": dte,
                "iv": iv,
                "impliedVolatility": iv,
                "right": "C"
            })
    
    return chain


class TestIVSurface:
    """Test IV Surface analysis."""
    
    def test_build_surface(self, mock_option_chain):
        """Test building IV surface from option chain."""
        from quant.iv_surface import IVSurface
        
        surface = IVSurface()
        success = surface.build_surface(mock_option_chain, underlying_price=450)
        
        assert success is True
        assert len(surface._surface_data) > 0
    
    def test_get_iv_at(self, mock_option_chain):
        """Test getting IV at specific strike/DTE."""
        from quant.iv_surface import IVSurface
        
        surface = IVSurface()
        surface.build_surface(mock_option_chain, underlying_price=450)
        
        # Get IV at a point that exists
        iv = surface.get_iv_at(strike=450, days_to_expiry=30)
        
        assert iv is not None
        assert 0 < iv < 1  # Reasonable IV range
    
    def test_term_structure_analysis(self, mock_option_chain):
        """Test term structure analysis."""
        from quant.iv_surface import IVSurface
        
        surface = IVSurface()
        surface.build_surface(mock_option_chain, underlying_price=450)
        
        term_structure = surface.analyze_term_structure()
        
        assert term_structure in ["CONTANGO", "BACKWARDATION", "FLAT", "UNKNOWN"]
    
    def test_full_analysis(self, mock_option_chain):
        """Test comprehensive surface analysis."""
        from quant.iv_surface import get_iv_surface
        
        surface = get_iv_surface()
        surface.build_surface(mock_option_chain, underlying_price=450)
        
        analysis = surface.get_analysis()
        
        assert analysis.atm_iv > 0
        assert analysis.skew_type in ["PUT_SKEW", "CALL_SKEW", "SMILE", "NORMAL"]
        assert 0 <= analysis.surface_quality <= 1


class TestSkewAnalyzer:
    """Test Skew Analyzer."""
    
    def test_analyze_skew(self, mock_option_chain):
        """Test basic skew analysis."""
        from quant.skew_analyzer import SkewAnalyzer
        
        analyzer = SkewAnalyzer()
        
        metrics = analyzer.analyze_skew(
            option_chain=mock_option_chain,
            underlying_price=450,
            target_dte=30
        )
        
        assert metrics.atm_iv > 0
        # Our mock data has put skew
        assert metrics.put_wing_iv >= metrics.atm_iv or metrics.put_wing_iv == 0
    
    def test_skew_pattern_detection(self, mock_option_chain):
        """Test skew pattern detection."""
        from quant.skew_analyzer import get_skew_analyzer
        
        analyzer = get_skew_analyzer()
        
        metrics = analyzer.analyze_skew(
            option_chain=mock_option_chain,
            underlying_price=450,
            target_dte=30
        )
        
        assert metrics.pattern in ["PUT_SKEW", "CALL_SKEW", "SMILE", "SMIRK", "NORMAL", "UNKNOWN"]
    
    def test_strategy_recommendations(self, mock_option_chain):
        """Test strategy recommendations from skew."""
        from quant.skew_analyzer import SkewAnalyzer
        
        analyzer = SkewAnalyzer()
        
        metrics = analyzer.analyze_skew(
            option_chain=mock_option_chain,
            underlying_price=450,
            target_dte=30
        )
        
        recommendations = analyzer.get_strategy_recommendation(metrics)
        
        assert "strategies" in recommendations
        assert "reasoning" in recommendations
        assert len(recommendations["strategies"]) > 0
