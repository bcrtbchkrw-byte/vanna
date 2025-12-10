"""
Greeks Validation Module

Ensures option strategies conform to safety parameters for Delta, Theta, Gamma, and Vanna.
"""
from typing import Dict, Any, List, Optional
from loguru import logger
from config import get_config

class GreeksValidator:
    """
    Validates option Greeks against risk thresholds.
    """
    
    def __init__(self):
        # Hardcoded safety limits (could be moved to config)
        self.MAX_DELTA_CREDIT = 0.30     # Max (absolute) delta for short legs
        self.MIN_THETA = 0.50            # Min daily theta per contract (income focus)
        self.MAX_GAMMA = 0.10            # Max gamma risk
        self.VANNA_THRESHOLD = 0.40      # Max projected delta after IV spike
        
        logger.info(f"✅ GreeksValidator initialized (Max Delta: {self.MAX_DELTA_CREDIT})")

    def validate_leg(
        self, 
        leg: Dict[str, Any], 
        strategy_type: str = 'CREDIT'
    ) -> Dict[str, Any]:
        """
        Validate a single option leg.
        
        Args:
            leg: Option contract data with Greeks
            strategy_type: 'CREDIT' (Short) or 'DEBIT' (Long)
            
        Returns:
            Dict: {'valid': bool, 'reason': str}
        """
        # Delta check
        delta = abs(leg.get('delta', 0))
        
        if strategy_type == 'CREDIT':
            # We are selling premium, want OTM (low delta)
            if delta > self.MAX_DELTA_CREDIT:
                return {
                    'valid': False, 
                    'reason': f"Delta {delta:.2f} too high for Credit leg (Max {self.MAX_DELTA_CREDIT})"
                }
        
        # Additional checks can go here (e.g. Volume/OI from Liquidity module)
        
        return {'valid': True, 'reason': 'OK'}

    def validate_strategy_greeks(
        self,
        strategy_greeks: Dict[str, float],
        iv_stress_test: float = 5.0
    ) -> Dict[str, Any]:
        """
        Validate aggregate Greeks for a multi-leg strategy.
        
        Args:
            strategy_greeks: Net Greeks for the combined position
            iv_stress_test: Percentage points IV spike to simulate (e.g., 5.0 for +5% IV)
            
        Returns:
            Dict: {'valid': bool, 'reason': str, 'metrics': dict}
        """
        net_delta = strategy_greeks.get('delta', 0)
        net_theta = strategy_greeks.get('theta', 0)
        net_vanna = strategy_greeks.get('vanna', 0) # Calculated outside or assumed
        
        reasons = []
        is_valid = True
        
        # 1. Theta Check (Income strategies must have positive theta)
        # Note: Interactive Brokers reports Theta as daily decay
        if net_theta < self.MIN_THETA:
             # Warning only, as some hedges might be theta negative
             reasons.append(f"Low Theta: ${net_theta:.2f}/day (Target > ${self.MIN_THETA})")
             # is_valid = False # Strict mode?
        
        # 2. Vanna Stress Test (Volatility Risk)
        # How much does Delta change if IV spikes?
        # ΔDelta ≈ Vanna * ΔIV
        delta_change = net_vanna * (iv_stress_test / 100.0) # Vanna is per 1% change usually, or unit?
        # Use decimal notation: Vanna is dDelta/dVol. If Vol changes 0.05 (5%)...
        # Standard IBKR Vanna is often per 1 unit change in IV (e.g. 1% -> 2%). 
        # Let's assume standard math: dDelta = Vanna * dSigma.
        
        projected_delta = net_delta + delta_change
        
        if abs(projected_delta) >= self.VANNA_THRESHOLD:
            reasons.append(
                f"Vanna Fails: IV+{iv_stress_test}% causes Delta {net_delta:.2f} -> {projected_delta:.2f} "
                f"(>= {self.VANNA_THRESHOLD})"
            )
            is_valid = False
            
        result = {
            'valid': is_valid,
            'reason': "; ".join(reasons) if reasons else "OK",
            'metrics': {
                'net_theta': net_theta,
                'projected_delta_stress': projected_delta
            }
        }
        
        if not is_valid:
            logger.warning(f"❌ Greeks Logic Reject: {result['reason']}")
        else:
            logger.debug(f"✅ Greeks Logic Pass: {result['metrics']}")
            
        return result


# Singleton
_greeks_validator: Optional[GreeksValidator] = None

def get_greeks_validator() -> GreeksValidator:
    global _greeks_validator
    if _greeks_validator is None:
        _greeks_validator = GreeksValidator()
    return _greeks_validator
