"""Phase 3 Tests - Market Analysis (VIX, Earnings, Liquidity, Screener)."""
import asyncio

import nest_asyncio
import pytest

nest_asyncio.apply()


from core.logger import setup_logger  # noqa: E402
from tests.conftest import MockDataFetcher, MockIBKRConnection, is_ci_environment  # noqa: E402


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_conn():
    """Mock IBKR connection for CI tests."""
    setup_logger()
    return MockIBKRConnection()


@pytest.fixture
def mock_fetcher():
    """Mock data fetcher for CI tests."""
    return MockDataFetcher()


class TestVIXMonitor:
    """Tests for VIX monitoring."""
    
    @pytest.mark.asyncio
    async def test_vix_fetch(self, mock_fetcher):
        """Test VIX data fetch."""
        vix = await mock_fetcher.get_vix()
        print(f"[MOCK] Current VIX: {vix}")
        assert vix is not None
        assert vix > 0
    
    def test_vix_monitor_existence(self):
        """Test VIX monitor can be instantiated."""
        from analysis.vix_monitor import get_vix_monitor
        monitor = get_vix_monitor()
        assert monitor is not None


class TestEarningsChecker:
    """Tests for earnings blackout checking."""
    
    @pytest.mark.asyncio
    async def test_earnings_checker_instantiation(self):
        """Test earnings checker can be created."""
        from analysis.earnings_checker import get_earnings_checker
        
        checker = get_earnings_checker()
        assert checker is not None
    
    @pytest.mark.asyncio
    async def test_spy_earnings_check(self):
        """Test SPY (ETF) earnings check - should be safe."""
        from analysis.earnings_checker import get_earnings_checker
        
        checker = get_earnings_checker()
        result = await checker.check_blackout('SPY')
        
        print(f"SPY earnings check result: {result}")
        # SPY is an ETF - either safe or no data
        assert result['reason'] in ['SAFE', 'NO_DATA']


class TestLiquidityChecker:
    """Tests for liquidity checking."""
    
    def test_liquidity_checker_instantiation(self):
        """Test liquidity checker can be created."""
        from analysis.liquidity import get_liquidity_checker
        
        checker = get_liquidity_checker()
        assert checker is not None


class TestScreener:
    """Tests for daily screener."""
    
    def test_screener_instantiation(self):
        """Test screener can be created."""
        # Screener may not exist in all versions
        try:
            from analysis.screener import DailyOptionsScreener
            screener = DailyOptionsScreener()
            assert screener is not None
        except ImportError:
            # Try alternative import
            from analysis.screener import get_daily_screener
            screener = get_daily_screener()
            assert screener is not None
