"""Phase 7 Tests - Order Management and Exit Manager."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from ib_insync import Contract, Order, Trade

from execution.exit_manager import get_exit_manager
from execution.order_manager import get_order_manager


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class TestOrderManager:
    """Tests for order management."""
    
    @pytest.mark.asyncio
    async def test_order_manager_instantiation(self):
        """Test order manager can be created."""
        manager = get_order_manager()
        assert manager is not None
        assert hasattr(manager, 'paper_mode')
    
    @pytest.mark.asyncio
    async def test_spread_order(self):
        """Test spread order construction."""
        print("\n--- Testing Order Manager ---")
        
        # Mock IBKR Connection
        connection_mock = AsyncMock()
        ib_mock = MagicMock()
        connection_mock.ib = ib_mock
        ib_mock.isConnected.return_value = True
        ib_mock.qualifyContractsAsync = AsyncMock()
        
        # Mock placeOrder return
        mock_trade = Trade(Contract(), Order(), MagicMock())
        ib_mock.placeOrder.return_value = mock_trade
        
        # Inject Mock
        manager = get_order_manager()
        manager._connection = connection_mock
        
        # Test Spread Order Construction
        legs = [
            {'contract': Contract(conId=1, symbol='AAPL', secType='OPT'), 'action': 'BUY', 'ratio': 1},
            {'contract': Contract(conId=2, symbol='AAPL', secType='OPT'), 'action': 'SELL', 'ratio': 1}
        ]
        
        print("Placing simulated spread order...")
        trade = await manager.place_spread_order(
            legs=legs,
            quantity=1,
            price_limit=1.50,  # Credit
            strategy_name="Bull Put Info"
        )
        
        assert trade is not None, "Order placement failed"
            
        # Verify call args
        args = ib_mock.placeOrder.call_args
        contract_arg = args[0][0]
        order_arg = args[0][1]
        
        # Check Contract
        assert contract_arg.secType == 'BAG', f"Contract type {contract_arg.secType} != BAG"
        assert len(contract_arg.comboLegs) == 2, f"Combo Legs count {len(contract_arg.comboLegs)} != 2"
            
        print("✅ Contract construction OK (BAG with 2 legs)")
        
        # Check Order
        assert order_arg.action == 'SELL', f"Order Action {order_arg.action} != SELL (for positive credit)"
        assert order_arg.lmtPrice == 1.50, f"Limit Price {order_arg.lmtPrice} != 1.50"
            
        print("✅ Order execution logic OK")


class TestExitManager:
    """Tests for exit management."""
    
    def test_exit_manager_instantiation(self):
        """Test exit manager can be created."""
        exit_manager = get_exit_manager()
        assert exit_manager is not None
    
    @pytest.mark.asyncio
    async def test_exit_manager_logic(self):
        """Test exit manager decision logic."""
        print("\n--- Testing Exit Manager ---")
        
        exit_manager = get_exit_manager()
        
        # Mock Order Manager inside Exit Manager
        mock_om = AsyncMock()
        exit_manager.order_manager = mock_om
        
        # Mock Position
        pos = MagicMock()
        pos.contract.localSymbol = "AAPL_OPT"
        pos.position = -1  # Short 1 (Credit Spread)
        
        # 1. Check NO Exit (Profit 10%)
        res = await exit_manager.check_and_manage_position(pos, entry_price=1.00, current_price=0.90)
        assert not res, "Triggered early exit (should be False)"
        print("✅ No exit at 10% profit")
        
        # 2. Check TP Exit (Profit 60%)
        res = await exit_manager.check_and_manage_position(pos, entry_price=1.00, current_price=0.40)
        assert res, "Failed to trigger TP"
        print("✅ TP Triggered at 60% profit")
        
        # Verify order placed
        assert mock_om.place_order.call_count == 1, "place_order not called"
            
        args = mock_om.place_order.call_args
        order_arg = args[0][1]  # 2nd arg
        assert order_arg.action == 'BUY', f"ID Exit Order Action {order_arg.action} != BUY (Closing short)"

        # 3. Check SL Exit (Loss 200%)
        mock_om.reset_mock()
        res = await exit_manager.check_and_manage_position(pos, entry_price=1.00, current_price=2.50)
        assert res, "Failed to trigger SL"
        print("✅ SL Triggered at 2.5x entry")
