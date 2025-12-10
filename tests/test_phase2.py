"""
Phase 2 Test - IBKR Connection

Tests:
- IBKR connection
- Account info
- VIX fetch
- Stock quote
- Disconnect

Run: python tests/test_phase2.py

IMPORTANT: Requires running IB Gateway!
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_connection():
    """Test IBKR connection."""
    print("Testing IBKR connection...", end=" ")
    
    from ibkr.connection import get_ibkr_connection
    
    conn = await get_ibkr_connection()
    success = await conn.connect()
    
    if not success:
        print("❌ FAILED - Could not connect")
        return False
    
    print("✅ PASSED")
    return True


async def test_account_info(conn):
    """Test account info retrieval."""
    print("Testing account info...", end=" ")
    
    try:
        summary = conn.get_account_summary()
        
        assert 'NetLiquidation' in summary, "Missing NetLiquidation"
        assert summary['NetLiquidation'] > 0, "Invalid NetLiquidation"
        
        print(f"✅ PASSED (Net Liq: ${summary['NetLiquidation']:,.2f})")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def test_vix(conn):
    """Test VIX data fetch."""
    print("Testing VIX fetch...", end=" ")
    
    from ibkr.data_fetcher import get_data_fetcher
    
    fetcher = get_data_fetcher()
    vix = await fetcher.get_vix()
    
    if vix and vix > 0:
        print(f"✅ PASSED (VIX: {vix:.2f})")
        return True
    else:
        print("❌ FAILED - Could not fetch VIX")
        return False


async def test_stock_quote(conn):
    """Test stock quote fetch."""
    print("Testing stock quote (SPY)...", end=" ")
    
    from ibkr.data_fetcher import get_data_fetcher
    
    fetcher = get_data_fetcher()
    quote = await fetcher.get_stock_quote("SPY")
    
    if quote and quote.get('last'):
        print(f"✅ PASSED (SPY: ${quote['last']:.2f})")
        return True
    else:
        print("❌ FAILED - Could not fetch quote")
        return False


async def test_disconnect(conn):
    """Test clean disconnect."""
    print("Testing disconnect...", end=" ")
    
    await conn.disconnect()
    
    if not conn.is_connected:
        print("✅ PASSED")
        return True
    else:
        print("❌ FAILED - Still connected")
        return False


async def run_tests():
    """Run all Phase 2 tests."""
    print("=" * 50)
    print("PHASE 2 TESTS - IBKR Connection")
    print("=" * 50)
    print()
    print("⚠️  Requires running IB Gateway!")
    print()
    
    # Initialize logger
    from core.logger import setup_logger
    setup_logger(level="INFO")
    
    results = []
    conn = None
    
    # Test 1: Connection
    try:
        from ibkr.connection import get_ibkr_connection
        conn = await get_ibkr_connection()
        success = await conn.connect()
        if success:
            print("Testing IBKR connection... ✅ PASSED")
            results.append(True)
        else:
            print("Testing IBKR connection... ❌ FAILED")
            results.append(False)
            # Can't continue without connection
            print("=" * 50)
            print("❌ Connection failed - cannot run remaining tests")
            print("   Make sure IB Gateway is running and healthy")
            return 1
    except Exception as e:
        print(f"Testing IBKR connection... ❌ FAILED: {e}")
        results.append(False)
        return 1
    
    # Test 2: Account Info
    try:
        results.append(await test_account_info(conn))
    except Exception as e:
        print(f"Testing account info... ❌ FAILED: {e}")
        results.append(False)
    
    # Test 3: VIX
    try:
        results.append(await test_vix(conn))
    except Exception as e:
        print(f"Testing VIX fetch... ❌ FAILED: {e}")
        results.append(False)
    
    # Test 4: Stock Quote
    try:
        results.append(await test_stock_quote(conn))
    except Exception as e:
        print(f"Testing stock quote... ❌ FAILED: {e}")
        results.append(False)
    
    # Test 5: Disconnect
    try:
        results.append(await test_disconnect(conn))
    except Exception as e:
        print(f"Testing disconnect... ❌ FAILED: {e}")
        results.append(False)
    
    # Summary
    print()
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("=" * 50)
        print("Phase 2 COMPLETE - Ready for Phase 3 (Market Analysis)")
        return 0
    else:
        print(f"❌ TESTS FAILED ({passed}/{total})")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)
