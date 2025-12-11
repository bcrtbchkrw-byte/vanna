import os

import pytest


def test_config():
    """Test configuration loading."""
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

def test_logger():
    """Test logger initialization."""
    from core.logger import get_logger, setup_logger
    
    # Setup logger
    setup_logger(level="DEBUG", log_dir="logs")
    logger = get_logger()
    
    # Test logging
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")

@pytest.mark.asyncio
async def test_database():
    """Test database operations."""

    from core.database import Database
    
    # Use test database
    db_path = "data/test_phase1.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    db = Database(db_path=db_path)
    await db.connect()
    
    try:
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
        
    finally:
        # Cleanup
        await db.disconnect()
        if os.path.exists(db_path):
            os.remove(db_path)
