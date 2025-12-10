"""
Phase 1 Test - Foundation

Tests:
- Config loading
- Logger initialization
- Database connection

Run: python tests/test_phase1.py
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config():
    """Test configuration loading."""
    print("Testing config...", end=" ")
    
    from config import get_config
    
    config = get_config()
    
    # Check basic attributes exist
    assert hasattr(config, 'ibkr'), "Missing ibkr config"
    assert hasattr(config, 'ai'), "Missing ai config"
    assert hasattr(config, 'trading'), "Missing trading config"
    assert hasattr(config, 'log'), "Missing log config"
    
    # Check defaults
    assert config.ibkr.port == 4002, f"Expected port 4002, got {config.ibkr.port}"
    assert config.trading.max_risk_per_trade == 120, "Wrong max risk default"
    
    print("✅ PASSED")
    return True


def test_logger():
    """Test logger initialization."""
    print("Testing logger...", end=" ")
    
    from core.logger import setup_logger, get_logger
    
    # Setup logger
    setup_logger(level="DEBUG", log_dir="logs")
    logger = get_logger()
    
    # Test logging
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    
    print("✅ PASSED")
    return True


async def test_database():
    """Test database operations."""
    print("Testing database...", end=" ")
    
    from core.database import Database
    
    # Use test database
    db = Database(db_path="data/test.db")
    await db.connect()
    
    # Test insert
    trade_id = await db.insert_trade(
        symbol="TEST",
        strategy="test_strategy",
        entry_price=100.0,
        quantity=1,
        notes="Test trade"
    )
    assert trade_id > 0, "Failed to insert trade"
    
    # Test fetch
    trades = await db.get_open_trades()
    assert len(trades) > 0, "No trades found"
    assert trades[0]['symbol'] == "TEST", "Wrong symbol"
    
    # Test close
    await db.close_trade(trade_id, exit_price=110.0, pnl=10.0)
    
    # Cleanup
    await db.disconnect()
    
    # Remove test database
    import os
    if os.path.exists("data/test.db"):
        os.remove("data/test.db")
    
    print("✅ PASSED")
    return True


async def run_tests():
    """Run all Phase 1 tests."""
    print("=" * 50)
    print("PHASE 1 TESTS - Foundation")
    print("=" * 50)
    
    results = []
    
    # Test 1: Config
    try:
        results.append(test_config())
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results.append(False)
    
    # Test 2: Logger
    try:
        results.append(test_logger())
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results.append(False)
    
    # Test 3: Database
    try:
        results.append(await test_database())
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results.append(False)
    
    # Summary
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("=" * 50)
        print("Phase 1 COMPLETE - Ready for Phase 2 (IBKR)")
        return 0
    else:
        print(f"❌ TESTS FAILED ({passed}/{total})")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)
