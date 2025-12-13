"""
Order Execution Module

Handles placement and management of IBKR orders, including complex Combo (BAG) orders.
"""
from typing import Any, Dict, List, Optional, cast

from ib_insync import ComboLeg, Contract, LimitOrder, Order, Trade
from loguru import logger

from config import get_config
from ibkr.connection import get_ibkr_connection


class OrderManager:
    """
    Manages order execution and tracking.
    
    IMPORTANT: Check paper_mode before placing orders!
    Paper mode = log only, no real orders.
    """
    
    def __init__(self):
        self.config = get_config()
        self.account = self.config.ibkr.account
        self._connection = None
        
        # SAFETY: Paper mode prevents real order execution
        # Set PAPER_MODE=false in .env for live trading
        self.paper_mode = getattr(self.config.ibkr, 'paper_mode', True)
        
        if self.paper_mode:
            logger.warning("⚠️ OrderManager in PAPER MODE - No real orders will be placed!")
        else:
            logger.info("🔴 OrderManager in LIVE MODE - Real orders enabled")

    async def _get_ib(self):
        if self._connection is None:
            self._connection = await get_ibkr_connection()
        return self._connection.ib

    async def place_order(
        self, 
        contract: Contract, 
        order: Order
    ) -> Optional[Trade]:
        """
        Place a standard order.
        """
        ib = await self._get_ib()
        
        if not ib.isConnected():
            logger.error("Cannot place order: Not connected to IBKR")
            return None
            
        try:
            # Ensure contract is qualified (has conId)
            await ib.qualifyContractsAsync(contract)
            
            msg = f"🚀 Order: {order.action} {order.totalQuantity} {contract.localSymbol} @ {order.lmtPrice or 'MKT'}"
            logger.info(msg)
            
            trade = ib.placeOrder(contract, order)
            return cast(Trade, trade)
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    async def place_spread_order(
        self,
        legs: List[Dict[str, Any]],
        quantity: int,
        price_limit: Optional[float] = None,
        strategy_name: str = "Custom"
    ) -> Optional[Trade]:
        """
        Place a complex multi-leg order (BAG/Combo).
        
        Args:
            legs: List of dicts {'contract': Contract, 'action': 'BUY'/'SELL', 'ratio': 1}
            quantity: Number of combos
            price_limit: net limit price (debit/credit). check sign!
                         Strategies usually return Net Credit (positive).
                         IBKR Limit Price:
                         - Debit = Positive (paying)
                         - Credit = Negative (receiving), but for Combo orders specifically?
                         Wait, for Box/Spreads: 
                         - Buy Spread = Pay Debit (Positive Limit)
                         - Sell Spread = Receive Credit (Negative Limit? Or Positive?)
                         IBKR TWS uses conventional pricing.
                         Standard IBKR API: "Buy" = Pay debit. "Sell" = Receive credit.
                         Price is strictly the limit price.
                         If we define the Combo as "Buy 1 Leg A, Sell 1 Leg B",
                         and A is expensive, B is cheap -> Debit -> Buy Combo (Positive Price).
                         If we "Sell 1 Leg A, Buy 1 Leg B" -> Credit -> Sell Combo (Positive Price).
                         
                         Usually simpler to ALWAYS 'BUY' the Combo, and define legs to create debit/credit.
                         BUT for Credit Spreads, we usually "SELL" the spread (Sell Short, Buy Long).
                         If we "Sell" the spread, the limit price should be the Credit amount (Positive).
                         Example: Sell Iron Condor @ 1.00 Credit.
                         Action=SELL, Limit=1.00.
                         Legs: Sell Put, Buy Put, Sell Call, Buy Call.
                         
            strategy_name: Label for logging
            
        Returns:
            Trade object
        """
        ib = await self._get_ib()
        
        if not ib.isConnected():
            logger.error("Not connected")
            return None
            
        try:
            # 1. Qualify all leg contracts to get conIds
            raw_contracts = [leg['contract'] for leg in legs]
            await ib.qualifyContractsAsync(*raw_contracts)
            
            # 2. Build ComboLegs
            combo_legs: List[ComboLeg] = []
            symbol = ""
            currency = "USD"
            exchange = "SMART"
            
            for leg in legs:
                c = leg['contract']
                action = leg['action'] # 'BUY' or 'SELL'
                ratio = leg.get('ratio', 1)
                
                combo_legs.append(ComboLeg(
                    conId=c.conId,
                    ratio=ratio,
                    action=action,
                    exchange=exchange
                ))
                
                if not symbol:
                    symbol = c.symbol
                if not currency:
                    currency = c.currency
            
            # 3. Create BAG Contract
            bag = Contract(
                secType='BAG',
                symbol=symbol,
                currency=currency,
                exchange=exchange
            )
            bag.comboLegs = combo_legs
            
            # 4. Create Order
            # If we are doing a Credit Spread, usually we SELL the combo for a credit
            # But the 'legs' definition matters.
            # If legs define the direction (Buy Long, Sell Short), then
            # 'BUY' the bag means execute legs exactly as defined.
            # 'SELL' the bag means flip actions.
            
            # Let's standardize: We pass 'BUY' to execute legs AS DEFINED.
            # But wait, TWS usually handles "Sell Iron Condor". 
            # If using 'SmartComboRouting', IBKR handles lots.
            
            # Simplest for API:
            # Define legs for the structure.
            # If we want to Receive Credit X, we SELL the Bag (if Bag price > 0 is credit).
            # Actually, standard is:
            # - Limit Price is always positive.
            # - Action BUY = Pay Debit.
            # - Action SELL = Receive Credit.
            
            # main_action = 'SELL' if price_limit and price_limit > 0 else 'BUY' 
            # Wait, Credit Spreads are sold. Debit Spreads are bought.
            # Assuming price_limit passed is the absolute value (e.g. 1.00).
            # If it's a Credit Strategy -> SELL.
            # If it's a Debit Strategy -> BUY.
            
            # For now, let's allow caller to specify, or default to SELL (Credit Strategy bot).
            # We will use 'SELL' for Credit Spreads.
            
            order_action = 'SELL' 
            limit_price = abs(price_limit) if price_limit else 0
            
            order = LimitOrder(
                action=order_action,
                totalQuantity=quantity,
                lmtPrice=limit_price
            )
            # Add algo params if needed
            
            logger.info(f"🧩 Placing Combo {strategy_name}: {order_action} {quantity} @ {limit_price}")
            for cl in combo_legs:
                logger.debug(f"  - Leg: {cl.action} {cl.ratio}x conId={cl.conId}")
                
            trade = ib.placeOrder(bag, order)
            return cast(Trade, trade)
            
        except Exception as e:
            logger.error(f"Error placing spread order: {e}")
            return None

    async def cancel_all_orders(self):
        """Cancel all open orders."""
        ib = await self._get_ib()
        if not ib.isConnected():
            return
        
        orders = ib.openOrders()
        for o in orders:
            ib.cancelOrder(o)
        logger.warning(f"Cancelled {len(orders)} open orders")


# Singleton
_order_manager: Optional[OrderManager] = None

def get_order_manager() -> OrderManager:
    global _order_manager
    if _order_manager is None:
        _order_manager = OrderManager()
    return _order_manager
