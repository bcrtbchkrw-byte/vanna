


from risk.greeks_validator import get_greeks_validator
from risk.position_sizer import get_position_sizer


def test_position_sizing():
    print("\n--- Testing Position Sizer ---")
    sizer = get_position_sizer()
    
    # Mock Account
    account_value = 10_000.0
    buying_power = 5_000.0
    
    # Scenario 1: Small Risk (Should allow multiple contracts)
    res1 = sizer.calculate_position_size(
        account_value=account_value,
        buying_power=buying_power,
        risk_per_contract=50.0,
        strategy_capital_per_contract=50.0
    )
    print(f"Scenario 1 (Risk $50): {res1['contracts']} contracts. Reason: {res1['reason']}")
    
    assert res1['contracts'] >= 1, "Scenario 1 Failed (Should allow contracts)"
        
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
    
    assert res2['contracts'] == 0, "Scenario 2 Failed (Should be 0 contracts due to risk limit)"

def test_greeks_validation():
    print("\n--- Testing Greeks Validator ---")
    validator = get_greeks_validator()
    
    # 1. Delta Check
    leg_good = {'delta': -0.20, 'right': 'C'} # Short Call OTM
    leg_bad = {'delta': -0.45, 'right': 'C'}  # Short Call too ITM
    
    v1 = validator.validate_leg(leg_good, 'CREDIT')
    v2 = validator.validate_leg(leg_bad, 'CREDIT')
    
    assert v1['valid'], f"Good Leg Rejected: {v1['reason']}"
    assert not v2['valid'], f"Bad Leg Accepted: {v2}"
        
    print("✅ Delta Check Passed (Accepted -0.20, Rejected -0.45)")
    
    # 2. Vanna Stress Test
    greeks_safe = {'delta': 0.10, 'theta': 2.0, 'vanna': -0.10}
    res_safe = validator.validate_strategy_greeks(greeks_safe, iv_stress_test=5.0)
    
    assert res_safe['valid'], f"Safe Vanna Rejected: {res_safe['reason']}"
        
    # Risky Vanna
    greeks_risky = {'delta': 0.10, 'theta': 2.0, 'vanna': -10.0}
    res_risky = validator.validate_strategy_greeks(greeks_risky, iv_stress_test=5.0)
    
    assert not res_risky['valid'], f"Risky Vanna Accepted: {res_risky}"
        
    print("✅ Vanna Stress Test Passed (Detected risky delta expansion)")
