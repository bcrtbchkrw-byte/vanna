"""
Roll Manager - Position Rolling Logic

Part of Multi-Model Trading Architecture.
Decides when and how to roll existing positions.

Roll Types:
- ROLL_OUT: Extend DTE (same strike, later expiration)
- ROLL_UP: Higher strike (calls) when ITM
- ROLL_DOWN: Lower strike (puts) when ITM
- CLOSE: Don't roll, just close
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime, timedelta

from core.logger import get_logger

logger = get_logger()


class RollAction(Enum):
    """Roll decision types."""
    HOLD = "HOLD"           # Keep position as-is
    ROLL_OUT = "ROLL_OUT"   # Extend expiration
    ROLL_UP = "ROLL_UP"     # Move strike up (calls)
    ROLL_DOWN = "ROLL_DOWN" # Move strike down (puts)
    CLOSE = "CLOSE"         # Close, don't roll


@dataclass
class RollDecision:
    """Roll manager output."""
    action: RollAction
    reason: str
    new_dte: int = 0           # Target DTE after roll
    strike_delta: float = 0     # Strike change ($)
    urgency: float = 0          # How urgent (0-1)
    
    def should_roll(self) -> bool:
        return self.action not in [RollAction.HOLD, RollAction.CLOSE]


@dataclass
class Position:
    """Position state for roll analysis."""
    symbol: str
    option_type: str  # CALL or PUT
    side: str         # BUY or SELL
    strike: float
    dte: int          # Days to expiration
    entry_price: float
    current_price: float
    delta: float = 0.5
    underlying_price: float = 0
    
    @property
    def pnl_pct(self) -> float:
        """P&L as percentage."""
        if self.entry_price == 0:
            return 0
        if self.side == "BUY":
            return (self.current_price - self.entry_price) / self.entry_price
        else:  # SELL
            return (self.entry_price - self.current_price) / self.entry_price
    
    @property
    def is_itm(self) -> bool:
        """Is option in the money?"""
        if self.option_type == "CALL":
            return self.underlying_price > self.strike
        else:
            return self.underlying_price < self.strike


class RollManager:
    """
    Manages position rolling decisions.
    
    Rules:
    1. Take profits at +50%+ (close, don't roll)
    2. Roll out when DTE <= 7 and profitable
    3. Roll up/down when deep ITM (delta > 0.7)
    4. Cut losses at -30% (close or defensive roll)
    """
    
    # Thresholds for 80% win rate
    PROFIT_TAKE = 0.50      # +50% profit = close
    LOSS_CUT = -2.0         # -200% of credit = stop
    MIN_DTE_ROLL = 21       # Roll at 21 DTE
    ITM_DELTA_THRESHOLD = 0.70  # Delta > 0.7 = consider roll
    
    def __init__(self):
        """Initialize roll manager."""
        pass
    
    def analyze(self, position: Position, vix: float = 18) -> RollDecision:
        """
        Analyze position and decide roll action.
        
        Args:
            position: Current position state
            vix: Current VIX level
            
        Returns:
            RollDecision with recommendation
        """
        pnl = position.pnl_pct
        dte = position.dte
        delta = abs(position.delta)
        
        # Rule 1: Profit taking
        if pnl >= self.PROFIT_TAKE:
            return RollDecision(
                action=RollAction.CLOSE,
                reason=f"Take profit at +{pnl:.0%}",
                urgency=0.8
            )
        
        # Rule 2: Stop loss
        if pnl <= self.LOSS_CUT:
            # In high VIX, consider defensive roll instead of loss
            if vix > 25 and position.side == "SELL":
                return RollDecision(
                    action=RollAction.ROLL_OUT,
                    reason=f"Defensive roll at {pnl:.0%} loss (high VIX)",
                    new_dte=21,  # Roll to 3 weeks
                    urgency=0.9
                )
            return RollDecision(
                action=RollAction.CLOSE,
                reason=f"Stop loss at {pnl:.0%}",
                urgency=0.9
            )
        
        # Rule 3: DTE approaching - roll or close
        if dte <= self.MIN_DTE_ROLL:
            if pnl > 0:
                # Profitable near expiry - roll to capture more
                return RollDecision(
                    action=RollAction.ROLL_OUT,
                    reason=f"DTE={dte}, roll to extend profitable position",
                    new_dte=14 if vix < 20 else 21,
                    urgency=0.7
                )
            elif pnl > -0.15:
                # Small loss, roll for recovery
                return RollDecision(
                    action=RollAction.ROLL_OUT,
                    reason=f"DTE={dte}, roll to recover small loss ({pnl:.0%})",
                    new_dte=21,
                    urgency=0.6
                )
            else:
                # Larger loss near expiry - close
                return RollDecision(
                    action=RollAction.CLOSE,
                    reason=f"DTE={dte} with {pnl:.0%} loss - close",
                    urgency=0.8
                )
        
        # Rule 4: Deep ITM - roll strike
        if delta >= self.ITM_DELTA_THRESHOLD and position.is_itm:
            if position.option_type == "CALL":
                return RollDecision(
                    action=RollAction.ROLL_UP,
                    reason=f"Deep ITM call (Δ={delta:.2f}), roll up",
                    strike_delta=5.0,  # Roll up $5
                    urgency=0.5
                )
            else:
                return RollDecision(
                    action=RollAction.ROLL_DOWN,
                    reason=f"Deep ITM put (Δ={delta:.2f}), roll down",
                    strike_delta=-5.0,  # Roll down $5
                    urgency=0.5
                )
        
        # Default: Hold
        return RollDecision(
            action=RollAction.HOLD,
            reason=f"Hold: DTE={dte}, P/L={pnl:.0%}, Δ={delta:.2f}",
            urgency=0
        )
    
    def analyze_portfolio(self, positions: list, vix: float = 18) -> Dict[str, RollDecision]:
        """
        Analyze all positions in portfolio.
        
        Args:
            positions: List of Position objects
            vix: Current VIX
            
        Returns:
            Dict mapping position_id to RollDecision
        """
        decisions = {}
        
        for i, pos in enumerate(positions):
            pos_id = f"{pos.symbol}_{pos.option_type}_{pos.strike}_{pos.dte}"
            decisions[pos_id] = self.analyze(pos, vix)
        
        # Log summary
        actions = [d.action.value for d in decisions.values()]
        logger.info(f"Roll analysis: {len(positions)} positions - "
                   f"HOLD: {actions.count('HOLD')}, ROLL: {sum(1 for a in actions if 'ROLL' in a)}, CLOSE: {actions.count('CLOSE')}")
        
        return decisions


# Singleton
_manager: Optional[RollManager] = None

def get_roll_manager() -> RollManager:
    """Get singleton RollManager."""
    global _manager
    if _manager is None:
        _manager = RollManager()
    return _manager


# Quick test
if __name__ == "__main__":
    from core.logger import setup_logger
    setup_logger(level="INFO")
    
    manager = RollManager()
    
    # Test cases
    test_positions = [
        Position("SPY", "CALL", "BUY", 450, 14, 5.0, 8.0, delta=0.55, underlying_price=460),  # Profit
        Position("SPY", "PUT", "SELL", 440, 5, 3.0, 2.0, delta=-0.25, underlying_price=455),   # Near expiry, profit
        Position("QQQ", "CALL", "BUY", 380, 21, 10.0, 6.5, delta=0.45, underlying_price=375),  # Loss
        Position("AAPL", "CALL", "BUY", 170, 10, 5.0, 12.0, delta=0.82, underlying_price=185), # Deep ITM
    ]
    
    print("\n=== Roll Analysis ===\n")
    for pos in test_positions:
        decision = manager.analyze(pos, vix=18)
        print(f"{pos.symbol} {pos.option_type} ${pos.strike} DTE={pos.dte}")
        print(f"  P/L: {pos.pnl_pct:.0%} | Δ: {pos.delta:.2f}")
        print(f"  → {decision.action.value}: {decision.reason}")
        print()
