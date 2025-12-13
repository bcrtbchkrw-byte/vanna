"""
Exit Management Module

Monitors active positions and manages profit taking / stop losses.
"""
from typing import Any, Optional

from ib_insync import MarketOrder
from loguru import logger

from execution.order_manager import get_order_manager


class ExitManager:
    """
    Manages exit logic for open positions.
    """
    
    def __init__(self):
        self.order_manager = get_order_manager()
        self.take_profit_pct = 0.50 # 50% max profit collected
        self.stop_loss_mult = 2.0   # 2x credit received (Net Loss = 1x credit)
        
    async def check_and_manage_position(
        self,
        position: Any, # ib_insync Position object or dict wrapper
        entry_price: float, # Credit received per contract (positive)
        current_price: float, # Cost to close (debit, negative usually)
                              # BUT IBKR reports Market Value.
                              # Short position has Negative Market Value.
                              # Cost Basis is Total Credit (Positive)? No.
                              # Position.avgCost is usually positive.
                              # Position.position is negative for short.
    ) -> bool:
        """
        Check if position should be closed.
        
        Args:
            position: IBKR Position object
            entry_price: Unit price at entry (Credit > 0)
            current_price: Current market price to close (Debit > 0) 
            
        Returns:
            True if exit order placed
        """
        try:
            # Logic for Short Premium Strategy (Credit Spread)
            # We Sold at Entry Price (Credit). We want to Buy Back at lower price.
            # Profit = Entry - Current.
            # Max Profit = Entry.
            # 50% Profit = Current <= Entry * 0.50.
            
            # Stop Loss:
            # Loss = Current - Entry.
            # risk = 2x Credit.
            # Stop if Current >= Entry * 2.0 -> Loss = 2.0*Credit - Credit = 1.0*Credit? 
            # Usually stop is at 200% of collected premium price.
            
            if entry_price <= 0:
                logger.warning("Entry price should be positive for Credit strategies")
                return False
                
            profit_pct = (entry_price - current_price) / entry_price
            
            symbol = position.contract.localSymbol
            
            logger.debug(
                f"Exit Check {symbol}: Entry={entry_price}, Curr={current_price}, "
                f"Profit={profit_pct:.1%}"
            )
            
            if profit_pct >= self.take_profit_pct:
                logger.info(f"💰 Take Profit Triggered for {symbol} (+{profit_pct:.1%})")
                await self._close_position(position, "Take Profit")
                return True
                
            if current_price >= (entry_price * self.stop_loss_mult):
                msg = f"🛑 SL for {symbol} (Price {current_price} > Limit {entry_price * self.stop_loss_mult})"
                logger.info(msg)
                await self._close_position(position, "Stop Loss")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error in exit check: {e}")
            return False

    async def _close_position(self, position: Any, reason: str):
        """Close the position with error handling."""
        try:
            contract = position.contract
            qty = abs(position.position)
            
            # If short (negative pos), we Buy to Close
            action = 'BUY' if position.position < 0 else 'SELL'
            
            logger.info(f"Closing {contract.localSymbol} ({reason}): {action} {qty}")
            
            # Market order for immediate exit? Or limit?
            # Safety: Market for Stop Loss, Limit for TP?
            # For simplicity, Market order now.
            order = MarketOrder(action, qty)
            
            await self.order_manager.place_order(contract, order)
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to close position ({reason}): {e}")
            # TODO: Notify user via Telegram for critical failures
            raise  # Re-raise to ensure caller knows about failure


# Singleton
_exit_manager: Optional[ExitManager] = None

def get_exit_manager() -> ExitManager:
    global _exit_manager
    if _exit_manager is None:
        _exit_manager = ExitManager()
    return _exit_manager
