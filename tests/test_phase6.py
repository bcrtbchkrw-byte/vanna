"""
Phase 6 Test - Strategies

Tests:
- Analysis Signal Generation (VIX/Trend logic)
- Candidate Selection (Delta matching, Credit calculation)
- Vertical Credit Spreads (Bull Put / Bear Call)
- Iron Condor

Run: python tests/test_phase6.py
"""
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import setup_logger
from strategies.credit_spreads import VerticalCreditSpread
from strategies.iron_condor import IronCondor

# Mock Data
MOCK_MARKET_DATA = {
    'price': 100.0,
    'vix': 18.0,
    'iv_rank': 55.0,
    'trend': 'NEUTRAL'
}

# Mock Chain: Puts (90-99), Calls (101-110)
# Delta approx: 
# Put 95: Delta -0.20 (Perfect for Short Put)
# Put 90: Delta -0.10 (Perfect for Long Put)
# Call 105: Delta 0.20 (Perfect for Short Call)
# Call 110: Delta 0.10 (Perfect for Long Call)
MOCK_CHAIN = [
    {'strike': 90, 'right': 'P', 'delta': -0.10, 'bid': 0.30, 'ask': 0.40},
    {'strike': 91, 'right': 'P', 'delta': -0.12, 'bid': 0.60, 'ask': 0.70},
    {'strike': 92, 'right': 'P', 'delta': -0.14, 'bid': 0.70, 'ask': 0.80},
    {'strike': 93, 'right': 'P', 'delta': -0.16, 'bid': 0.80, 'ask': 0.90},
    {'strike': 94, 'right': 'P', 'delta': -0.18, 'bid': 0.90, 'ask': 1.00},
    {'strike': 95, 'right': 'P', 'delta': -0.20, 'bid': 1.00, 'ask': 1.10}, # Target Short
    
    {'strike': 105, 'right': 'C', 'delta': 0.20, 'bid': 1.00, 'ask': 1.10}, # Target Short
    {'strike': 106, 'right': 'C', 'delta': 0.18, 'bid': 0.90, 'ask': 1.00},
    {'strike': 107, 'right': 'C', 'delta': 0.16, 'bid': 0.80, 'ask': 0.90},
    {'strike': 110, 'right': 'C', 'delta': 0.10, 'bid': 0.30, 'ask': 0.40},
]

RISK_PROFILE = {'max_risk_per_trade': 500}

async def test_vertical_spreads():
    print("\n--- Testing Vertical Credit Spreads ---")
    strategy = VerticalCreditSpread()
    
    # 1. Analyze Market
    signal = await strategy.analyze_market("TEST", MOCK_MARKET_DATA)
    print(f"Signal: {signal.direction} (Quality: {signal.setup_quality})")
    
    if signal.setup_quality < 5:
        print("❌ Low signal quality unexpected")
        return False
        
    # 2. Find Candidates
    candidates = await strategy.find_execution_candidates("TEST", MOCK_CHAIN, RISK_PROFILE)
    print(f"Found {len(candidates)} candidates")
    
    for c in candidates:
        print(f"  {c['strategy']} Credit: ${c['net_credit']:.2f}, ROI: {c['roi_pct']:.1f}%")
        
    # Expect at least one Bull Put and one Bear Call
    types = [c['strategy'] for c in candidates]
    if 'BULL_PUT' not in types:
        print("❌ Failed to find Bull Put Spread")
        return False
    if 'BEAR_CALL' not in types:
        print("❌ Failed to find Bear Call Spread")
        return False
        
    # Verify math for one candidate
    # Bull Put: Short 95 (Bid 1.00) / Long 90 (Ask 0.40) -> Credit 0.60
    # Width 5.0. Max Loss = 500 - 60 = 440.
    c = candidates[0]
    if c['strategy'] == 'BULL_PUT':
        if abs(c['net_credit'] - 0.60) > 0.01:
            print(f"❌ Credit Calcu Error: Got {c['net_credit']}, Exp 0.60")
            return False
            
    print("✅ Vertical Spreads Logic Valid")
    return True

async def test_iron_condor():
    print("\n--- Testing Iron Condor ---")
    strategy = IronCondor()
    
    # 1. Analyze
    signal = await strategy.analyze_market("TEST", MOCK_MARKET_DATA)
    print(f"Signal: {signal.direction} (Quality: {signal.setup_quality})")
    
    if "Neutral Trend" not in signal.reasoning:
        print("❌ Signal reasoning missing Trend check")
        return False
        
    # 2. Find Candidates
    candidates = await strategy.find_execution_candidates("TEST", MOCK_CHAIN, RISK_PROFILE)
    print(f"Found {len(candidates)} IC candidates")
    
    if not candidates:
        print("❌ No IC candidates found")
        return False
        
    ic = candidates[0]
    print(f"  Iron Condor Credit: ${ic['net_credit']:.2f}, "
          f"Short Put: {ic['short_put_strike']}, Short Call: {ic['short_call_strike']}")
    
    # Expected: Short Put 94 (Credit 0.50) + Short Call 106 (Credit 0.50) = 1.00
    if abs(ic['net_credit'] - 1.00) > 0.05:
         print(f"❌ IC Credit Error: Got {ic['net_credit']}, Exp 1.00")
         return False
         
    print("✅ Iron Condor Logic Valid")
    return True

async def run_tests():
    setup_logger()
    
    results = []
    results.append(await test_vertical_spreads())
    results.append(await test_iron_condor())
    
    if all(results):
        print("\n✅ ALL PHASE 6 TESTS PASSED")
        return 0
    else:
        print("\n❌ TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(run_tests()))
