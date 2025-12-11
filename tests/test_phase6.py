import asyncio

import nest_asyncio
import pytest

nest_asyncio.apply()



from strategies.credit_spreads import VerticalCreditSpread  # noqa: E402
from strategies.iron_condor import IronCondor  # noqa: E402


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Mock Data
MOCK_MARKET_DATA = {
    'price': 100.0,
    'vix': 18.0,
    'iv_rank': 55.0,
    'trend': 'NEUTRAL'
}

# Mock Chain
MOCK_CHAIN = [
    {'strike': 90, 'right': 'P', 'delta': -0.10, 'bid': 0.30, 'ask': 0.40},
    {'strike': 91, 'right': 'P', 'delta': -0.12, 'bid': 0.60, 'ask': 0.70},
    {'strike': 92, 'right': 'P', 'delta': -0.14, 'bid': 0.70, 'ask': 0.80},
    {'strike': 93, 'right': 'P', 'delta': -0.16, 'bid': 0.80, 'ask': 0.90},
    {'strike': 94, 'right': 'P', 'delta': -0.18, 'bid': 0.90, 'ask': 1.00},
    {'strike': 95, 'right': 'P', 'delta': -0.20, 'bid': 1.00, 'ask': 1.10}, 
    
    {'strike': 105, 'right': 'C', 'delta': 0.20, 'bid': 1.00, 'ask': 1.10},
    {'strike': 106, 'right': 'C', 'delta': 0.18, 'bid': 0.90, 'ask': 1.00},
    {'strike': 107, 'right': 'C', 'delta': 0.16, 'bid': 0.80, 'ask': 0.90},
    {'strike': 110, 'right': 'C', 'delta': 0.10, 'bid': 0.30, 'ask': 0.40},
]

RISK_PROFILE = {'max_risk_per_trade': 500}

@pytest.mark.asyncio
async def test_vertical_spreads():
    print("\n--- Testing Vertical Credit Spreads ---")
    strategy = VerticalCreditSpread()
    
    # 1. Analyze Market
    signal = await strategy.analyze_market("TEST", MOCK_MARKET_DATA)
    print(f"Signal: {signal.direction} (Quality: {signal.setup_quality})")
    
    assert signal.setup_quality >= 5, "Low signal quality unexpected"
        
    # 2. Find Candidates
    candidates = await strategy.find_execution_candidates("TEST", MOCK_CHAIN, RISK_PROFILE)
    print(f"Found {len(candidates)} candidates")
    
    types = [c['strategy'] for c in candidates]
    assert 'BULL_PUT' in types, "Failed to find Bull Put Spread"
    assert 'BEAR_CALL' in types, "Failed to find Bear Call Spread"
        
    # Verify math for one candidate
    # Bull Put: Short 95 (Bid 1.00) / Long 90 (Ask 0.40) -> Credit 0.60
    c = candidates[0]
    if c['strategy'] == 'BULL_PUT':
        assert abs(c['net_credit'] - 0.60) <= 0.01, f"Credit Calcu Error: Got {c['net_credit']}, Exp 0.60"
            
    print("✅ Vertical Spreads Logic Valid")

@pytest.mark.asyncio
async def test_iron_condor():
    print("\n--- Testing Iron Condor ---")
    strategy = IronCondor()
    
    # 1. Analyze
    signal = await strategy.analyze_market("TEST", MOCK_MARKET_DATA)
    print(f"Signal: {signal.direction} (Quality: {signal.setup_quality})")
    
    assert "Neutral Trend" in signal.reasoning, "Signal reasoning missing Trend check"
        
    # 2. Find Candidates
    candidates = await strategy.find_execution_candidates("TEST", MOCK_CHAIN, RISK_PROFILE)
    print(f"Found {len(candidates)} IC candidates")
    
    assert len(candidates) > 0, "No IC candidates found"
        
    ic = candidates[0]
    print(f"  Iron Condor Credit: ${ic['net_credit']:.2f}, "
          f"Short Put: {ic['short_put_strike']}, Short Call: {ic['short_call_strike']}")
    
    # Expected: Short Put 94 (Credit 0.50) + Short Call 106 (Credit 0.50) = 1.00
    assert abs(ic['net_credit'] - 1.00) <= 0.05, f"IC Credit Error: Got {ic['net_credit']}, Exp 1.00"
         
    print("✅ Iron Condor Logic Valid")
