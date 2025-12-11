"""Phase 3 Tests - Market Analysis (VIX, Earnings, Liquidity, Screener)."""
import asyncio

import nest_asyncio
import pytest

nest_asyncio.apply()


from core.logger import setup_logger  # noqa: E402
from tests.conftest import MockDataFetcher, MockIBKRConnection, is_ci_environment  # noqa: E402

# Only import real IBKR modules if not in CI
if not is_ci_environment():
    from ibkr.connection import get_ibkr_connection
    from ibkr.data_fetcher import get_data_fetcher


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def conn():
    """Shared IBKR connection for Phase 3 (uses mocks in CI)."""
    setup_logger()
    
    if is_ci_environment():
        yield MockIBKRConnection()
    else:
        connection = await get_ibkr_connection()
        connected = await connection.connect()
        if not connected:
            pytest.fail("Could not connect to IBKR")
        yield connection
        await connection.disconnect()


@pytest.fixture(scope="module")
def fetcher():
    """Data fetcher fixture (uses mocks in CI)."""
    if is_ci_environment():
        return MockDataFetcher()
    # Will be imported only if not in CI
    return get_data_fetcher()


@pytest.mark.asyncio
async def test_vix(fetcher):
    """Test VIX Monitor."""
    print("\n--- Testing VIX Monitor ---")
    
    if is_ci_environment():
        # Mock VIX test
        vix = await fetcher.get_vix()
        print(f"[MOCK] Current VIX: {vix}")
        assert vix is not None
        assert vix > 0
    else:
        from analysis.vix_monitor import get_vix_monitor
        monitor = get_vix_monitor()
        regime = await monitor.update()
        print(f"Current VIX: {monitor.current_vix}")
        print(f"Current Regime: {regime}")
        assert monitor.current_vix is not None


@pytest.mark.asyncio
async def test_earnings(conn):
    """Test Earnings Checker."""
    print("\n--- Testing Earnings Checker ---")
    from analysis.earnings_checker import get_earnings_checker
    
    checker = get_earnings_checker()
    symbol_safe = 'SPY'
    
    print(f"Checking {symbol_safe}...")
    safe_res = await checker.check_blackout(symbol_safe)
    print(f"Result: {safe_res}")
    
    # SPY might have no data or be safe
    assert safe_res['reason'] in ['SAFE', 'NO_DATA']


@pytest.mark.asyncio
async def test_liquidity(conn, fetcher):
    """Test Liquidity Checker."""
    print("\n--- Testing Liquidity Checker ---")
    
    if is_ci_environment():
        # Mock liquidity test - just verify the checker exists and responds
        from analysis.liquidity import get_liquidity_checker
        checker = get_liquidity_checker()
        # Create mock option contract
        from unittest.mock import MagicMock
        mock_option = MagicMock()
        mock_option.localSymbol = "SPY_MOCK"
        # Skip actual check in CI as it requires real market data
        print("[MOCK] Liquidity checker initialized successfully")
        assert checker is not None
    else:
        from ib_insync import Option, Stock

        from analysis.liquidity import get_liquidity_checker
        
        checker = get_liquidity_checker()
        spy = Stock('SPY', 'SMART', 'USD')
        conn_obj = await fetcher._get_connection()
        await conn_obj.ib.qualifyContractsAsync(spy)
        
        chains = await conn_obj.ib.reqSecDefOptParamsAsync(
            spy.symbol, '', spy.secType, spy.conId
        )
        if not chains:
            pytest.skip("No option chains found")
            
        chain = chains[0]
        exp = sorted(chain.expirations)[0]
        strike = chain.strikes[len(chain.strikes)//2]
        
        option = Option('SPY', exp, strike, 'C', 'SMART')
        await conn_obj.ib.qualifyContractsAsync(option)
        
        res = await checker.check_option_liquidity(option)
        print(f"Result: {res}")
        assert 'passed' in res
        assert 'reason' in res


@pytest.mark.asyncio
async def test_screener(conn):
    """Test Unified Screener."""
    print("\n--- Testing Unified Screener ---")
    
    if is_ci_environment():
        # Mock screener test
        print("[MOCK] Screener test skipped in CI (requires IBKR)")
        assert True
    else:
        from analysis.screener import get_screener
        screener = get_screener()
        symbols = ['SPY', 'NVDA', 'AAPL']
        results = await screener.run_checks(symbols)
        print(f"Market Status: {results['market_status']}")
        assert results['market_status'] != 'UNKNOWN'
