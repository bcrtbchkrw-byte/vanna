import asyncio

import nest_asyncio
import pytest

nest_asyncio.apply()


from core.logger import get_logger, setup_logger  # noqa: E402
from tests.conftest import MockDataFetcher, MockIBKRConnection, is_ci_environment  # noqa: E402

# Only import real IBKR modules if not in CI
if not is_ci_environment():
    from ibkr.connection import get_ibkr_connection
    from ibkr.data_fetcher import get_data_fetcher


@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def conn():
    """Shared IBKR connection for Phase 2 tests (uses mocks in CI)."""
    setup_logger(level="INFO")
    
    if is_ci_environment():
        # Use mock in CI
        yield MockIBKRConnection()
    else:
        # Use real connection locally
        connection = await get_ibkr_connection()
        connected = await connection.connect()
        assert connected, "Failed to connect to IBKR - Is Gateway running?"
        yield connection
        await connection.disconnect()


@pytest.fixture(scope="module")
def fetcher():
    """Data fetcher fixture (uses mocks in CI)."""
    if is_ci_environment():
        return MockDataFetcher()
    else:
        return get_data_fetcher()


@pytest.mark.asyncio
async def test_account_info(conn):
    """Test account info retrieval."""
    logger = get_logger()
    
    if is_ci_environment():
        # Mock test
        value = conn.get_account_value("NetLiquidation")
        logger.info(f"[MOCK] NetLiquidation: {value}")
        assert value > 0
    else:
        summary = await conn.get_account_summary()
        logger.info(f"Account Summary: {summary}")
        assert 'NetLiquidation' in summary
        assert summary['NetLiquidation'] > 0


@pytest.mark.asyncio
async def test_vix(conn, fetcher):
    """Test VIX data fetch."""
    vix = await fetcher.get_vix()
    assert vix is not None
    assert vix > 0, "VIX should be positive"


@pytest.mark.asyncio
async def test_stock_quote(conn, fetcher):
    """Test stock quote fetch."""
    quote = await fetcher.get_stock_quote("SPY")
    assert quote is not None, "Failed to get quote"
    assert quote.get('last') is not None, "Quote missing 'last' price"


@pytest.mark.asyncio
async def test_disconnect_verification(conn):
    """Verify connection is active before teardown."""
    assert conn.is_connected

