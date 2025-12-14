"""Phase 2 Tests - IBKR Connection and Data Fetching."""
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


@pytest.fixture
def mock_conn():
    """Mock IBKR connection for CI tests."""
    return MockIBKRConnection()


@pytest.fixture
def mock_fetcher():
    """Mock data fetcher for CI tests."""
    return MockDataFetcher()


class TestIBKRConnection:
    """Tests for IBKR connection (uses mocks in CI)."""
    
    def test_account_info_mock(self, mock_conn):
        """Test account info retrieval with mock."""
        logger = get_logger()
        
        value = mock_conn.get_account_value("NetLiquidation")
        logger.info(f"[MOCK] NetLiquidation: {value}")
        assert value > 0
    
    def test_account_value_available_funds(self, mock_conn):
        """Test AvailableFunds retrieval."""
        value = mock_conn.get_account_value("AvailableFunds")
        assert value == 50000.0
    
    def test_connection_status(self, mock_conn):
        """Test connection status check."""
        assert mock_conn.is_connected is True


class TestDataFetcher:
    """Tests for data fetching (uses mocks in CI)."""
    
    @pytest.mark.asyncio
    async def test_vix(self, mock_fetcher):
        """Test VIX data fetch."""
        vix = await mock_fetcher.get_vix()
        assert vix is not None
        assert vix > 0, "VIX should be positive"
    
    @pytest.mark.asyncio
    async def test_stock_quote(self, mock_fetcher):
        """Test stock quote fetch."""
        quote = await mock_fetcher.get_stock_quote("SPY")
        assert quote is not None, "Failed to get quote"
        assert quote.get('last') is not None, "Quote missing 'last' price"
    
    @pytest.mark.asyncio
    async def test_option_chain(self, mock_fetcher):
        """Test option chain fetch."""
        chain = await mock_fetcher.get_option_chain("SPY")
        assert chain is not None
        assert len(chain) > 0
