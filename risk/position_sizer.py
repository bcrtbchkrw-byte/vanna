"""
Position Sizing Module

Determines safe trade sizes based on account equity, risk parameters,
and mathematical models (Fixed Risk, Kelly Criterion).
"""
from typing import Dict, Any, Optional
from loguru import logger
from config import get_config
import math

class PositionSizer:
    """
    Calculates appropriate position sizes to manage risk.
    """
    
    def __init__(self):
        self.config = get_config()
        self.max_risk_per_trade = self.config.trading.max_risk_per_trade
        self.max_allocation_percent = self.config.trading.max_allocation_percent
        self.min_buying_power_buffer = 1000.0 # Keep $1000 free
        
        logger.info(
            f"✅ PositionSizer initialized (Max Risk: ${self.max_risk_per_trade}, "
            f"Max Alloc: {self.max_allocation_percent}%)"
        )
        
    def calculate_position_size(
        self,
        account_value: float,
        buying_power: float,
        risk_per_contract: float,
        strategy_capital_per_contract: float,
        win_probability: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate recommended position size (number of contracts).
        
        Args:
            account_value: Total Net Liquidation Value
            buying_power: Available Buying Power
            risk_per_contract: Max loss per contract (e.g. spread width or debit paid)
            strategy_capital_per_contract: Margin required per contract
            win_probability: Estimated win rate (needed for Kelly)
            
        Returns:
            Dict with recommended contracts and reasoning
        """
        reasons = []
        
        # 1. Fixed Risk Check: e.g. Max $200 loss per trade
        # If risk_per_contract is $500, then max 0 contracts (too risky)
        if risk_per_contract > self.max_risk_per_trade:
            logger.warning(
                f"⚠️ Trade risk ${risk_per_contract} exceeds max limit ${self.max_risk_per_trade}"
            )
            return {
                'contracts': 0,
                'reason': f"Risk/Contract ${risk_per_contract:.0f} > Max Limit ${self.max_risk_per_trade:.0f}"
            }
        
        max_contracts_risk = int(self.max_risk_per_trade / risk_per_contract)
        reasons.append(f"Risk Limit: {max_contracts_risk} contracts (Max ${self.max_risk_per_trade})")
        
        # 2. Allocation Check: e.g. Max 25% of account
        max_capital = account_value * (self.max_allocation_percent / 100.0)
        max_contracts_alloc = int(max_capital / strategy_capital_per_contract)
        reasons.append(f"Alloc Limit: {max_contracts_alloc} contracts ({self.max_allocation_percent}%)")
        
        # 3. Buying Power Check: Must leave buffer
        available_bp = max(0, buying_power - self.min_buying_power_buffer)
        max_contracts_bp = int(available_bp / strategy_capital_per_contract)
        reasons.append(f"BP Limit: {max_contracts_bp} contracts (Available ${available_bp:.0f})")
        
        # 4. Final Calculation
        recommended = min(max_contracts_risk, max_contracts_alloc, max_contracts_bp)
        
        # 5. Kelly Criterion (Optional - Future Enhancement)
        # For now, we stick to conservative Fixed Risk
        # kelly_fraction = win_probability - ((1 - win_probability) / risk_reward_ratio)
        
        result = {
            'contracts': max(0, recommended),
            'max_contracts_risk': max_contracts_risk,
            'max_contracts_alloc': max_contracts_alloc,
            'max_contracts_bp': max_contracts_bp,
            'reason': "; ".join(reasons)
        }
        
        logger.info(f"📏 Position Sizing: Recommended {result['contracts']} contracts. ({result['reason']})")
        
        return result
    
    def validate_portfolio_allocation(
        self,
        new_trade_capital: float,
        current_used_capital: float,
        total_equity: float
    ) -> bool:
        """
        Check if adding this trade keeps total portfolio risk sane.
        Global limits (e.g. max 50% of account deployed total)
        """
        global_max_utilization = 0.50 # 50% max deployed
        
        projected_usage = current_used_capital + new_trade_capital
        usage_pct = projected_usage / total_equity
        
        if usage_pct > global_max_utilization:
            logger.warning(f"⚠️ Portfolio Over-allocation: Projected {usage_pct:.1%} > Max {global_max_utilization:.0%}")
            return False
            
        return True


# Singleton
_position_sizer: Optional[PositionSizer] = None

def get_position_sizer() -> PositionSizer:
    global _position_sizer
    if _position_sizer is None:
        _position_sizer = PositionSizer()
    return _position_sizer
