"""
Phase 7 Test - Execution

Tests:
- OrderManager spread construction
- ExitManager TP/SL logic

Run: python tests/test_phase7.py
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ib_insync import Contract, Order, Trade

from core.logger import setup_logger
from execution.exit_manager import get_exit_manager
from execution.order_manager import get_order_manager


async def test_order_manager():
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
    
    # 1. Test Spread Order Construction
    legs = [
        {'contract': Contract(conId=1, symbol='AAPL', secType='OPT'), 'action': 'BUY', 'ratio': 1},
        {'contract': Contract(conId=2, symbol='AAPL', secType='OPT'), 'action': 'SELL', 'ratio': 1}
    ]
    
    print("Placing simulated spread order...")
    trade = await manager.place_spread_order(
        legs=legs,
        quantity=1,
        price_limit=1.50, # Credit
        strategy_name="Bull Put Info"
    )
    
    if trade is None:
        print("❌ Order placement failed")
        return False
        
    # Verify call args
    args = ib_mock.placeOrder.call_args
    contract_arg = args[0][0]
    order_arg = args[0][1]
    
    # Check Contract
    if contract_arg.secType != 'BAG':
        print(f"❌ Contract type {contract_arg.secType} != BAG")
        return False
    
    if len(contract_arg.comboLegs) != 2:
        print(f"❌ Combo Legs count {len(contract_arg.comboLegs)} != 2")
        return False
        
    print("✅ Contract construction OK (BAG with 2 legs)")
    
    # Check Order
    # Credit Spread -> SELL action expected with Positive limit? 
    # Or BUY with Negative limit?
    # Our implementation: 'SELL' if price_limit > 0.
    if order_arg.action != 'SELL':
        print(f"❌ Order Action {order_arg.action} != SELL (for positive credit)")
        return False
        
    if order_arg.lmtPrice != 1.50:
        print(f"❌ Limit Price {order_arg.lmtPrice} != 1.50")
        return False
        
    print("✅ Order execution logic OK")
    return True

async def test_exit_manager():
    print("\n--- Testing Exit Manager ---")
    
    exit_manager = get_exit_manager()
    
    # Mock Order Manager inside Exit Manager
    mock_om = AsyncMock()
    exit_manager.order_manager = mock_om
    
    # Mock Position
    pos = MagicMock()
    pos.contract.localSymbol = "AAPL_OPT"
    pos.position = -1 # Short 1 (Credit Spread)
    
    # 1. Check NO Exit (Profit 10%)
    # Entry 1.00. Current 0.90. Profit 0.10 / 1.00 = 10%.
    res = await exit_manager.check_and_manage_position(pos, entry_price=1.00, current_price=0.90)
    if res:
        print("❌ Triggered early exit (should be False)")
        return False
    print("✅ No exit at 10% profit")
    
    # 2. Check TP Exit (Profit 60%)
    # Entry 1.00. Current 0.40. Profit 0.60 / 1.00 = 60%.
    res = await exit_manager.check_and_manage_position(pos, entry_price=1.00, current_price=0.40)
    if not res:
        print("❌ Failed to trigger TP")
        return False
    print("✅ TP Triggered at 60% profit")
    
    # Verify order placed
    # Closing Short -> BUY
    # Call count should be 1
    if mock_om.place_order.call_count != 1:
        print("❌ place_order not called")
        return False
        
    args = mock_om.place_order.call_args
    order_arg = args[0][1] # 2nd arg
    if order_arg.action != 'BUY':
        print(f"❌ ID Exit Order Action {order_arg.action} != BUY (Closing short)")
        return False

    # 3. Check SL Exit (Loss 200%)
    # Entry 1.00. Current 2.50. > 2.0 limit.
    mock_om.reset_mock()
    res = await exit_manager.check_and_manage_position(pos, entry_price=1.00, current_price=2.50)
    if not res:
        print("❌ Failed to trigger SL")
        return False
    print("✅ SL Triggered at 2.5x entry")

    return True

async def run_tests():
    setup_logger()
    
    results = []
    results.append(await test_order_manager())
    results.append(await test_exit_manager())
    
    if all(results):
        print("\n✅ ALL PHASE 7 TESTS PASSED")
        return 0
    else:
        print("\n❌ TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(run_tests()))
