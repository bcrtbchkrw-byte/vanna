"""
Order Execution Module

Handles placement and management of IBKR orders, including complex Combo (BAG) orders.
"""
from typing import Any, Dict, List, Optional, cast

from ib_insync import ComboLeg, Contract, LimitOrder, Order, Trade
from loguru import logger

from config import get_config
from ibkr.connection import get_ibkr_connection
from core.exceptions import (
    OrderValidationError, 
    OrderPlacementError,
    IBKRDisconnectedError
)
from core.audit_logger import get_audit_logger
from core.circuit_breaker import get_ibkr_circuit_breaker


# Validation constants
MAX_QUANTITY = 100  # Max contracts per order
MAX_PRICE = 10000.0  # Max price sanity check
MIN_PRICE = 0.01  # Min price (avoid 0)
MAX_LEGS = 4  # Max legs in spread


class OrderManager:
    """
    Manages order execution and tracking.
    
    IMPORTANT: Check paper_mode before placing orders!
    Paper mode = log only, no real orders.
    
    Features:
    - Input validation before order placement
    - Audit logging of all order attempts
    - Circuit breaker protection
    """
    
    def __init__(self):
        self.config = get_config()
        self.account = self.config.ibkr.account
        self._connection = None
        self._audit = get_audit_logger()
        self._circuit_breaker = get_ibkr_circuit_breaker()
        
        # SAFETY: Paper mode prevents real order execution
        # Set PAPER_TRADING=false in .env for live trading
        # Note: paper_trading is in TradingConfig, not IBKRConfig
        self.paper_mode = self.config.trading.paper_trading
        
        if self.paper_mode:
            logger.warning("⚠️ OrderManager in PAPER MODE - No real orders will be placed!")
        else:
            logger.info("🔴 OrderManager in LIVE MODE - Real orders enabled")

    async def _get_ib(self):
        if self._connection is None:
            self._connection = await get_ibkr_connection()
        return self._connection.ib
    
    def _validate_spread_order(
        self,
        legs: List[Dict[str, Any]],
        quantity: int,
        price_limit: Optional[float]
    ) -> None:
        """
        Validate spread order parameters.
        
        Raises:
            OrderValidationError: If any parameter is invalid
        """
        # Validate legs
        if not legs:
            raise OrderValidationError("Legs list cannot be empty")
        
        if len(legs) > MAX_LEGS:
            raise OrderValidationError(f"Too many legs: {len(legs)} > {MAX_LEGS}")
        
        for i, leg in enumerate(legs):
            if 'contract' not in leg:
                raise OrderValidationError(f"Leg {i}: missing 'contract'")
            if 'action' not in leg:
                raise OrderValidationError(f"Leg {i}: missing 'action'")
            if leg['action'] not in ('BUY', 'SELL'):
                raise OrderValidationError(f"Leg {i}: action must be 'BUY' or 'SELL', got '{leg['action']}'")
        
        # Validate quantity
        if quantity <= 0:
            raise OrderValidationError(f"Quantity must be positive, got {quantity}")
        if quantity > MAX_QUANTITY:
            raise OrderValidationError(f"Quantity {quantity} exceeds max {MAX_QUANTITY}")
        
        # Validate price
        if price_limit is not None:
            if price_limit < MIN_PRICE:
                raise OrderValidationError(f"Price {price_limit} below minimum {MIN_PRICE}")
            if price_limit > MAX_PRICE:
                raise OrderValidationError(f"Price {price_limit} exceeds max {MAX_PRICE}")

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
            price_limit: Net limit price (positive for credit)
            strategy_name: Label for logging
            
        Returns:
            Trade object or None if failed
            
        Raises:
            OrderValidationError: If parameters are invalid
        """
        # Extract symbol for logging
        symbol = legs[0]['contract'].symbol if legs else "UNKNOWN"
        
        # === STEP 1: Validate inputs BEFORE any external calls ===
        try:
            self._validate_spread_order(legs, quantity, price_limit)
        except OrderValidationError as e:
            self._audit.log_validation_error(symbol, "SPREAD_ORDER", str(e))
            raise  # Re-raise to caller
        
        # === STEP 2: Check circuit breaker ===
        self._circuit_breaker.check()  # Raises CircuitBreakerOpenError if open
        
        # === STEP 3: Audit log BEFORE placement ===
        correlation_id = self._audit.log_order_attempt(
            symbol=symbol,
            action="SELL",  # Credit spreads
            quantity=quantity,
            price=price_limit,
            strategy=strategy_name,
            order_type="SPREAD",
            extra={"legs": len(legs)}
        )
        
        # === STEP 4: Execute order ===
        ib = await self._get_ib()
        
        if not ib.isConnected():
            self._circuit_breaker.record_failure()
            self._audit.log_order_result(correlation_id, success=False, error="Not connected to IBKR")
            raise IBKRDisconnectedError("Not connected to IBKR")
            
        try:
            # 1. Qualify all leg contracts to get conIds
            raw_contracts = [leg['contract'] for leg in legs]
            await ib.qualifyContractsAsync(*raw_contracts)
            
            # 2. Build ComboLegs
            combo_legs: List[ComboLeg] = []
            currency = "USD"
            exchange = "SMART"
            
            for leg in legs:
                c = leg['contract']
                action = leg['action']
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
            
            # 4. Create Order (SELL for credit spreads)
            order_action = 'SELL' 
            limit_price = abs(price_limit) if price_limit else 0
            
            order = LimitOrder(
                action=order_action,
                totalQuantity=quantity,
                lmtPrice=limit_price
            )
            
            logger.info(f"🧩 Placing Combo {strategy_name}: {order_action} {quantity} @ {limit_price}")
            for cl in combo_legs:
                logger.debug(f"  - Leg: {cl.action} {cl.ratio}x conId={cl.conId}")
                
            trade = ib.placeOrder(bag, order)
            
            # === STEP 5: Record success ===
            self._circuit_breaker.record_success()
            self._audit.log_order_result(
                correlation_id, 
                success=True, 
                order_id=str(trade.order.orderId) if trade else None
            )
            
            return cast(Trade, trade)
            
        except Exception as e:
            # Record failure for circuit breaker
            self._circuit_breaker.record_failure()
            self._audit.log_order_result(correlation_id, success=False, error=str(e))
            logger.error(f"Error placing spread order: {e}")
            raise OrderPlacementError(f"Failed to place spread order: {e}") from e

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
