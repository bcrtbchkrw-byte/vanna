"""
IBKR Execution Client

Handles order placement and management via ib_insync.
"""
import asyncio
from typing import Optional, List, Dict
from decimal import Decimal

from ib_insync import Contract, Order, LimitOrder, MarketOrder, Trade, OrderStatus, TagValue

from core.logger import get_logger
from ibkr.connection import get_ibkr_connection

logger = get_logger()

class IBKRExecutionClient:
    """
    Handles trading operations including order placement and modification.
    """
    
    def __init__(self):
        self._connection = None
        
    async def _get_ib(self):
        """Get IBKR connection."""
        if self._connection is None:
            self._connection = await get_ibkr_connection()
        return self._connection.ib
        
    async def place_limit_order(
        self, 
        contract: Contract, 
        action: str, 
        quantity: int, 
        limit_price: float,
        adaptive: bool = True,
        priority: Optional[str] = None
    ) -> Trade:
        """
        Place a LIMIT order.
        
        Args:
            contract: IBKR Contract object
            action: 'BUY' or 'SELL'
            quantity: Number of contracts
            limit_price: Limit price
            adaptive: Use IBKR Adaptive Algo (default True)
            priority: Adaptive priority ('Urgent', 'Normal', 'Patient')
            
        Returns:
            Trade object (live update wrapper)
        """
        ib = await self._get_ib()
        
        # Qualify contract to ensure we have ConId
        await ib.qualifyContractsAsync(contract)
        
        order = LimitOrder(action, quantity, limit_price)
        order.tif = 'DAY' # Safer for options
        
        if adaptive:
            # Use config default if priority not specified
            if not priority:
                from config import get_config
                priority = get_config().trading.adaptive_priority
                
            order.algoStrategy = 'Adaptive'
            order.algoParams = [TagValue('adaptivePriority', priority)]
            order.transmit = True # Ensure transmit is set
        
        trade = ib.placeOrder(contract, order)
        algo_info = f" (Adaptive/{priority})" if adaptive else ""
        logger.info(f"📨 Order Placed: {action} {quantity} {contract.localSymbol} @ {limit_price:.2f}{algo_info}")
        
        return trade

    async def place_market_order(
        self, 
        contract: Contract, 
        action: str, 
        quantity: int
    ) -> Trade:
        """Place MARKET order (Use with caution!)."""
        ib = await self._get_ib()
        await ib.qualifyContractsAsync(contract)
        order = MarketOrder(action, quantity)
        trade = ib.placeOrder(contract, order)
        logger.warning(f"📨 MARKET Order Placed: {action} {quantity} {contract.localSymbol}")
        return trade
        
    async def get_current_positions(self) -> List[Dict]:
        """Get list of current open positions."""
        ib = await self._get_ib()
        positions = ib.positions()
        
        results = []
        for p in positions:
            contract = p.contract
            # Ensure details
            if not contract.localSymbol:
                await ib.qualifyContractsAsync(contract)
                
            results.append({
                'symbol': contract.symbol,
                'localSymbol': contract.localSymbol,
                'conId': contract.conId,
                'secType': contract.secType,
                'position': p.position,
                'avgCost': p.avgCost
            })
        return results

    async def wait_for_fill(self, trade: Trade, timeout: float = 10.0) -> bool:
        """Wait for trade to be filled."""
        ib = await self._get_ib()
        start = asyncio.get_event_loop().time()
        
        while not trade.isDone():
            await asyncio.sleep(0.5)
            if asyncio.get_event_loop().time() - start > timeout:
                logger.warning(f"Order timeout waiting for fill: {trade.order}")
                return False
                
        if trade.orderStatus.status == 'Filled':
            logger.info(f"✅ Order Filled: {trade.contract.localSymbol} @ {trade.orderStatus.avgFillPrice}")
            return True
            
        return False
