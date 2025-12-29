"""
Margin Manager - IBKR Margin Account Support

Calculates margin requirements for option strategies.
Tracks margin usage and ensures compliance with limits.

IBKR Portfolio Margin rules:
- Defined risk (spreads): margin = max_loss
- Undefined risk (naked): margin based on stress test

Standard Margin (Reg-T):
- Naked PUT: max(20% * underlying - OTM_amount, 10% * strike) * 100
- Naked CALL: max(20% * underlying - OTM_amount, 10% * underlying) * 100
- Credit Spread: width * 100 (margin), width * 100 - credit (BPR)

IMPORTANT: Premium (credit) does NOT add to margin requirement.
Credit REDUCES buying power reduction (BPR), not margin.
Maintenance margin = ~90% of initial (not 75%)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

from core.logger import get_logger

logger = get_logger()


class MarginType(Enum):
    """Type of margin account."""
    REG_T = "REG_T"           # Standard margin
    PORTFOLIO = "PORTFOLIO"    # Portfolio margin (lower requirements)


@dataclass
class MarginRequirement:
    """Margin requirement for a position."""
    initial_margin: float      # Required to open
    maintenance_margin: float  # Required to hold
    buying_power_reduction: float
    strategy: str
    is_defined_risk: bool


@dataclass
class AccountMargin:
    """Current account margin status."""
    net_liquidation: float     # Total account value
    excess_liquidity: float    # Available margin
    buying_power: float        # Total buying power
    initial_margin_used: float
    maintenance_margin_used: float
    margin_cushion: float      # excess_liquidity / net_liquidation
    
    @property
    def margin_usage_pct(self) -> float:
        """Percentage of margin used."""
        if self.net_liquidation == 0:
            return 1.0
        return self.initial_margin_used / self.net_liquidation
    
    @property
    def is_healthy(self) -> bool:
        """Is margin in healthy state?"""
        return self.margin_cushion > 0.20  # >20% cushion


class MarginManager:
    """
    Manages margin calculations and limits for IBKR.
    
    Supports both Reg-T and Portfolio Margin.
    """
    
    # Reg-T margin percentages
    REG_T_NAKED_PUT_PCT = 0.20   # 20% of underlying
    REG_T_NAKED_CALL_PCT = 0.20
    REG_T_MIN_PUT_PCT = 0.10    # Min 10% of strike
    REG_T_MIN_CALL_PCT = 0.10   # Min 10% of underlying
    
    # Portfolio margin is typically 50-70% lower
    PORTFOLIO_MARGIN_REDUCTION = 0.50
    
    # Safety limits
    MAX_MARGIN_USAGE = 0.50      # Never use more than 50% margin
    MIN_EXCESS_LIQUIDITY = 5000  # Keep $5k buffer
    MAX_SINGLE_POSITION_MARGIN = 0.10  # Max 10% of NLV per position
    
    def __init__(self, margin_type: MarginType = MarginType.REG_T):
        """
        Initialize margin manager.
        
        Args:
            margin_type: REG_T or PORTFOLIO margin
        """
        self.margin_type = margin_type
    
    def calculate_margin(self, 
                        strategy: str,
                        underlying_price: float,
                        legs: List[Dict],
                        credit: float = 0) -> MarginRequirement:
        """
        Calculate margin requirement for a strategy.
        
        Args:
            strategy: Strategy name (IRON_CONDOR, SHORT_STRANGLE, etc.)
            underlying_price: Current underlying price
            legs: List of leg dicts with 'strike', 'right' (CALL/PUT), 'action' (BUY/SELL)
            credit: Net credit received
            
        Returns:
            MarginRequirement
        """
        
        # Defined risk strategies - margin = max loss
        defined_risk_strategies = {
            'IRON_CONDOR', 'IRON_BUTTERFLY',
            'PUT_CREDIT_SPREAD', 'CALL_CREDIT_SPREAD',
            'BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD',
            'BULL_PUT_SPREAD', 'BEAR_CALL_SPREAD'
        }
        
        if strategy in defined_risk_strategies:
            return self._calculate_spread_margin(strategy, legs, credit)
        
        # Undefined risk strategies
        if strategy == 'SHORT_STRANGLE':
            return self._calculate_strangle_margin(underlying_price, legs, credit)
        
        if strategy == 'SHORT_PUT' or strategy == 'NAKED_PUT':
            return self._calculate_naked_put_margin(underlying_price, legs, credit)
        
        if strategy == 'SHORT_CALL' or strategy == 'NAKED_CALL':
            return self._calculate_naked_call_margin(underlying_price, legs, credit)
        
        # Default: use underlying value
        logger.warning(f"Unknown strategy {strategy}, using default margin")
        return MarginRequirement(
            initial_margin=underlying_price * 100,
            maintenance_margin=underlying_price * 100 * 0.90,
            buying_power_reduction=underlying_price * 100,
            strategy=strategy,
            is_defined_risk=False
        )
    
    def calculate_from_trade_spec(self, trade_spec) -> MarginRequirement:
        """
        Calculate margin directly from TradeSpec object.
        
        Args:
            trade_spec: TradeSpec from strategy_configurator
            
        Returns:
            MarginRequirement
        """
        # Convert TradeSpec legs to dict format
        legs = []
        for leg in trade_spec.legs:
            legs.append({
                'strike': leg.strike,
                'right': leg.right.value if hasattr(leg.right, 'value') else str(leg.right),
                'action': leg.action.value if hasattr(leg.action, 'value') else str(leg.action)
            })
        
        # Get credit from TradeSpec (credit_received is already in dollars)
        credit_per_share = trade_spec.credit_received / 100 if trade_spec.credit_received > 0 else 0
        
        return self.calculate_margin(
            strategy=trade_spec.strategy.value if hasattr(trade_spec.strategy, 'value') else str(trade_spec.strategy),
            underlying_price=trade_spec.underlying_price,
            legs=legs,
            credit=credit_per_share
        )
    
    def _calculate_spread_margin(self, strategy: str, legs: List[Dict], 
                                  credit: float) -> MarginRequirement:
        """
        Calculate margin for defined-risk spreads.
        Margin = width - credit (always positive)
        """
        # Find width between strikes
        strikes = sorted([leg['strike'] for leg in legs])
        
        if len(strikes) >= 2:
            # For iron condor: use max of put and call spread widths
            if len(strikes) == 4:
                put_width = strikes[1] - strikes[0]
                call_width = strikes[3] - strikes[2]
                width = max(put_width, call_width)
            else:
                width = strikes[-1] - strikes[0]
        else:
            width = 5  # Default $5 wide
        
        max_loss = (width * 100) - (credit if credit > 0 else 0)
        max_loss = max(max_loss, 0)
        
        margin = max_loss
        
        # Portfolio margin is same for defined risk
        return MarginRequirement(
            initial_margin=margin,
            maintenance_margin=margin,
            buying_power_reduction=margin,
            strategy=strategy,
            is_defined_risk=True
        )
    
    def _calculate_naked_put_margin(self, underlying_price: float,
                                     legs: List[Dict], credit: float) -> MarginRequirement:
        """
        Reg-T naked put margin:
        Initial = max(20% * underlying - OTM amount, 10% * strike)
        Maintenance = 90% of initial (stricter than 75%)
        BPR = initial - credit (credit REDUCES BPR)
        """
        put_leg = next((l for l in legs if l.get('right') == 'PUT' 
                        and l.get('action') == 'SELL'), None)
        
        if not put_leg:
            return MarginRequirement(0, 0, 0, 'SHORT_PUT', False)
        
        strike = put_leg['strike']
        credit_dollars = credit * 100 if credit > 0 else 0
        
        otm_amount = max(underlying_price - strike, 0)
        
        margin_1 = (self.REG_T_NAKED_PUT_PCT * underlying_price - otm_amount) * 100
        margin_2 = (self.REG_T_MIN_PUT_PCT * strike) * 100
        
        # Initial margin does NOT include premium
        initial_margin = max(margin_1, margin_2)
        
        # Portfolio margin is lower
        if self.margin_type == MarginType.PORTFOLIO:
            initial_margin *= (1 - self.PORTFOLIO_MARGIN_REDUCTION)
        
        # Maintenance is ~90% of initial (IBKR is stricter)
        maintenance_margin = initial_margin * 0.90
        
        # BPR is reduced by credit received
        buying_power_reduction = max(0, initial_margin - credit_dollars)
        
        return MarginRequirement(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            buying_power_reduction=buying_power_reduction,
            strategy='SHORT_PUT',
            is_defined_risk=False
        )
    
    def _calculate_naked_call_margin(self, underlying_price: float,
                                      legs: List[Dict], credit: float) -> MarginRequirement:
        """
        Reg-T naked call margin:
        Initial = max(20% * underlying - OTM amount, 10% * underlying)
        Maintenance = 90% of initial
        BPR = initial - credit
        """
        call_leg = next((l for l in legs if l.get('right') == 'CALL' 
                         and l.get('action') == 'SELL'), None)
        
        if not call_leg:
            return MarginRequirement(0, 0, 0, 'SHORT_CALL', False)
        
        strike = call_leg['strike']
        credit_dollars = credit * 100 if credit > 0 else 0
        
        otm_amount = max(strike - underlying_price, 0)
        
        margin_1 = (self.REG_T_NAKED_CALL_PCT * underlying_price - otm_amount) * 100
        margin_2 = (self.REG_T_MIN_CALL_PCT * underlying_price) * 100
        
        # Initial margin does NOT include premium
        initial_margin = max(margin_1, margin_2)
        
        if self.margin_type == MarginType.PORTFOLIO:
            initial_margin *= (1 - self.PORTFOLIO_MARGIN_REDUCTION)
        
        maintenance_margin = initial_margin * 0.90
        buying_power_reduction = max(0, initial_margin - credit_dollars)
        
        return MarginRequirement(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            buying_power_reduction=buying_power_reduction,
            strategy='SHORT_CALL',
            is_defined_risk=False
        )
    
    def _calculate_strangle_margin(self, underlying_price: float,
                                    legs: List[Dict], credit: float) -> MarginRequirement:
        """
        Short strangle margin = max(put margin, call margin)
        NOT sum of both (IBKR gives credit for offsetting positions)
        Credit reduces BPR for the whole strangle.
        """
        # Calculate each leg without credit first
        put_margin = self._calculate_naked_put_margin(underlying_price, legs, 0)
        call_margin = self._calculate_naked_call_margin(underlying_price, legs, 0)
        
        # Use greater of the two for initial/maintenance
        initial_margin = max(put_margin.initial_margin, call_margin.initial_margin)
        maintenance_margin = initial_margin * 0.90
        
        # Total credit reduces BPR
        credit_dollars = credit * 100 if credit > 0 else 0
        buying_power_reduction = max(0, initial_margin - credit_dollars)
        
        return MarginRequirement(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            buying_power_reduction=buying_power_reduction,
            strategy='SHORT_STRANGLE',
            is_defined_risk=False
        )
    
    def check_margin_available(self, account: AccountMargin,
                               new_margin: float) -> tuple[bool, str]:
        """
        Check if there's enough margin for a new position.
        
        Args:
            account: Current account margin status
            new_margin: Margin required for new position
            
        Returns:
            (approved, reason)
        """
        # Check margin usage limit
        new_usage = (account.initial_margin_used + new_margin) / account.net_liquidation
        if new_usage > self.MAX_MARGIN_USAGE:
            return False, f"Would exceed {self.MAX_MARGIN_USAGE:.0%} margin limit ({new_usage:.0%})"
        
        # Check excess liquidity
        remaining_excess = account.excess_liquidity - new_margin
        if remaining_excess < self.MIN_EXCESS_LIQUIDITY:
            return False, f"Would leave only ${remaining_excess:.0f} excess (min ${self.MIN_EXCESS_LIQUIDITY})"
        
        # Check single position limit
        if new_margin > account.net_liquidation * self.MAX_SINGLE_POSITION_MARGIN:
            max_allowed = account.net_liquidation * self.MAX_SINGLE_POSITION_MARGIN
            return False, f"Position too large (${new_margin:.0f} > ${max_allowed:.0f})"
        
        return True, "Approved"
    
    def estimate_account_margin(self, net_liquidation: float,
                               positions: List[Dict] = None) -> AccountMargin:
        """
        Estimate account margin status.
        
        For live trading, get real values from IBKR.
        """
        if positions is None:
            positions = []
        
        # Calculate total margin used
        total_initial = 0
        total_maintenance = 0
        
        for pos in positions:
            margin = self.calculate_margin(
                strategy=pos.get('strategy', 'UNKNOWN'),
                underlying_price=pos.get('underlying_price', 100),
                legs=pos.get('legs', []),
                credit=pos.get('credit', 0)
            )
            total_initial += margin.initial_margin
            total_maintenance += margin.maintenance_margin
        
        excess = net_liquidation - total_initial
        buying_power = excess * 2  # Approximate 2:1 for Reg-T
        
        if self.margin_type == MarginType.PORTFOLIO:
            buying_power = excess * 4  # Higher for portfolio margin
        
        return AccountMargin(
            net_liquidation=net_liquidation,
            excess_liquidity=max(0, excess),
            buying_power=max(0, buying_power),
            initial_margin_used=total_initial,
            maintenance_margin_used=total_maintenance,
            margin_cushion=excess / net_liquidation if net_liquidation > 0 else 0
        )


# Singleton
_manager: Optional[MarginManager] = None

def get_margin_manager(margin_type: MarginType = MarginType.REG_T) -> MarginManager:
    """Get singleton MarginManager."""
    global _manager
    if _manager is None:
        _manager = MarginManager(margin_type)
    return _manager


# Quick test
if __name__ == "__main__":
    from core.logger import setup_logger
    setup_logger(level="INFO")
    
    manager = MarginManager(MarginType.REG_T)
    
    print("=== MARGIN CALCULATIONS ===\n")
    
    # Iron Condor
    ic_legs = [
        {'strike': 580, 'right': 'PUT', 'action': 'BUY'},
        {'strike': 585, 'right': 'PUT', 'action': 'SELL'},
        {'strike': 605, 'right': 'CALL', 'action': 'SELL'},
        {'strike': 610, 'right': 'CALL', 'action': 'BUY'},
    ]
    margin = manager.calculate_margin('IRON_CONDOR', 595, ic_legs, credit=1.50)
    print(f"Iron Condor ($5 wide, $1.50 credit):")
    print(f"  Margin: ${margin.initial_margin:.0f}")
    print(f"  Defined Risk: {margin.is_defined_risk}")
    
    # Short Strangle
    strangle_legs = [
        {'strike': 580, 'right': 'PUT', 'action': 'SELL'},
        {'strike': 610, 'right': 'CALL', 'action': 'SELL'},
    ]
    margin = manager.calculate_margin('SHORT_STRANGLE', 595, strangle_legs, credit=3.00)
    print(f"\nShort Strangle (580p/610c, $3.00 credit):")
    print(f"  Margin: ${margin.initial_margin:.0f}")
    print(f"  Defined Risk: {margin.is_defined_risk}")
    
    # Account check
    account = AccountMargin(
        net_liquidation=100000,
        excess_liquidity=60000,
        buying_power=120000,
        initial_margin_used=40000,
        maintenance_margin_used=30000,
        margin_cushion=0.60
    )
    
    approved, reason = manager.check_margin_available(account, 5000)
    print(f"\nMargin check (add $5000): {approved} - {reason}")
    
    approved, reason = manager.check_margin_available(account, 30000)
    print(f"Margin check (add $30000): {approved} - {reason}")
