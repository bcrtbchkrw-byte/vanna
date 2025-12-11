import asyncio

import nest_asyncio
import pytest

nest_asyncio.apply()



from core.logger import get_logger, setup_logger  # noqa: E402
from ibkr.connection import get_ibkr_connection  # noqa: E402
from ibkr.data_fetcher import get_data_fetcher  # noqa: E402


@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
async def conn():
    """Shared IBKR connection for Phase 2 tests."""
    setup_logger(level="INFO")
    connection = await get_ibkr_connection()
    connected = await connection.connect()
    assert connected, "Failed to connect to IBKR - Is Gateway running?"
    
    yield connection
    
    await connection.disconnect()

@pytest.mark.asyncio
async def test_account_info(conn):
    """Test account info retrieval."""
    logger = get_logger()
    summary = await conn.get_account_summary()
    logger.info(f"Account Summary: {summary}")
    assert 'NetLiquidation' in summary, "Missing NetLiquidation"
    assert summary['NetLiquidation'] > 0, "Invalid NetLiquidation"

@pytest.mark.asyncio
async def test_vix(conn):
    """Test VIX data fetch."""
    fetcher = get_data_fetcher()
    vix = await fetcher.get_vix()
    assert vix is not None
    assert vix > 0, "VIX should be positive"

@pytest.mark.asyncio
async def test_stock_quote(conn):
    """Test stock quote fetch."""
    fetcher = get_data_fetcher()
    quote = await fetcher.get_stock_quote("SPY")
    assert quote is not None, "Failed to get quote"
    assert quote.get('last') is not None, "Quote missing 'last' price"

@pytest.mark.asyncio
async def test_disconnect_verification(conn):
    """Verify connection is active before teardown."""
    assert conn.is_connected
