"""
Phase 5 Test - Risk Management

Tests:
- Position Sizing Logic (Fixed Risk, Allocation, BP)
- Greeks Validation (Delta, Theta, Vanna Stress Test)

Run: python tests/test_phase5.py
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from core.logger import setup_logger
from risk.position_sizer import get_position_sizer
from risk.greeks_validator import get_greeks_validator

def test_position_sizing():
    print("\n--- Testing Position Sizer ---")
    sizer = get_position_sizer()
    
    # Mock Account
    account_value = 10_000.0
    buying_power = 5_000.0
    
    # Scenario 1: Small Risk (Should allow multiple contracts)
    # Risk per contract: $50 (e.g. $0.50 wide spread, debit)
    # Strategy cap required: $50
    # Config Max Risk per Trade: $120 (from config or default)
    # Let's assume Config is defaults ($120)
    
    res1 = sizer.calculate_position_size(
        account_value=account_value,
        buying_power=buying_power,
        risk_per_contract=50.0,
        strategy_capital_per_contract=50.0
    )
    print(f"Scenario 1 (Risk $50): {res1['contracts']} contracts. Reason: {res1['reason']}")
    
    if res1['contracts'] < 1:
        print("❌ Scenario 1 Failed (Should allow contracts)")
        return False
        
    # Scenario 2: High Risk (Should be blocked or limited)
    # Risk per contract: $500
    # Max Risk Config: $120
    res2 = sizer.calculate_position_size(
        account_value=account_value,
        buying_power=buying_power,
        risk_per_contract=500.0,
        strategy_capital_per_contract=500.0
    )
    print(f"Scenario 2 (Risk $500): {res2['contracts']} contracts. Reason: {res2['reason']}")
    
    if res2['contracts'] != 0:
        print("❌ Scenario 2 Failed (Should be 0 contracts due to risk limit)")
        return False
        
    return True

def test_greeks_validation():
    print("\n--- Testing Greeks Validator ---")
    validator = get_greeks_validator()
    
    # 1. Delta Check
    leg_good = {'delta': -0.20, 'right': 'C'} # Short Call OTM
    leg_bad = {'delta': -0.45, 'right': 'C'}  # Short Call too ITM
    
    v1 = validator.validate_leg(leg_good, 'CREDIT')
    v2 = validator.validate_leg(leg_bad, 'CREDIT')
    
    if not v1['valid']:
        print(f"❌ Good Leg Rejected: {v1['reason']}")
        return False
    if v2['valid']:
        print(f"❌ Bad Leg Accepted: {v2}")
        return False
        
    print(f"✅ Delta Check Passed (Accepted -0.20, Rejected -0.45)")
    
    # 2. Vanna Stress Test
    # Net Delta: 0.10
    # Net Vanna: -0.10 (Negative vanna = delta decreases as IV rises, or becomes more negative)
    # If IV +5%: Delta change = -0.10 * 0.05 = -0.005. New Delta = 0.095. SAFE.
    
    greeks_safe = {'delta': 0.10, 'theta': 2.0, 'vanna': -0.10}
    res_safe = validator.validate_strategy_greeks(greeks_safe, iv_stress_test=5.0)
    
    if not res_safe['valid']:
        print(f"❌ Safe Vanna Rejected: {res_safe['reason']}")
        return False
        
    # Risky Vanna
    # Vanna: -10.0 (Huge). IV +5% (0.05) -> Delta change = -0.50.
    # New Delta = 0.10 - 0.50 = -0.40. (Threshold 0.40). Borderline/Fail.
    greeks_risky = {'delta': 0.10, 'theta': 2.0, 'vanna': -10.0}
    res_risky = validator.validate_strategy_greeks(greeks_risky, iv_stress_test=5.0)
    
    if res_risky['valid']:
        print(f"❌ Risky Vanna Accepted: {res_risky}")
        return False
        
    print(f"✅ Vanna Stress Test Passed (Detected risky delta expansion)")
    
    return True

async def run_tests():
    setup_logger()
    
    results = []
    results.append(test_position_sizing())
    results.append(test_greeks_validation())
    
    if all(results):
        print("\n✅ ALL PHASE 5 TESTS PASSED")
        return 0
    else:
        print(f"\n❌ TESTS FAILED")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(run_tests()))
