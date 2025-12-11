import asyncio

import nest_asyncio
import pytest

nest_asyncio.apply()



from core.logger import setup_logger  # noqa: E402
from ibkr.connection import get_ibkr_connection  # noqa: E402


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
async def conn():
    """Shared IBKR connection for Phase 3."""
    setup_logger()
    connection = await get_ibkr_connection()
    connected = await connection.connect()
    if not connected:
        pytest.fail("Could not connect to IBKR")
    yield connection
    await connection.disconnect()

@pytest.mark.asyncio
async def test_vix():
    print("\n--- Testing VIX Monitor ---")
    from analysis.vix_monitor import get_vix_monitor
    
    monitor = get_vix_monitor()
    regime = await monitor.update()
    
    print(f"Current VIX: {monitor.current_vix}")
    print(f"Current Regime: {regime}")
    print(f"Trading Allowed: {monitor.is_trading_allowed()}")
    
    assert monitor.current_vix is not None

@pytest.mark.asyncio
async def test_earnings(conn):
    # Note: conn fixture ensures connection is active for checkers that might need it
    print("\n--- Testing Earnings Checker ---")
    from analysis.earnings_checker import get_earnings_checker
    
    checker = get_earnings_checker()
    symbol_safe = 'SPY'
    
    # Test potentially safe symbol (SPY usually safe/ETF)
    print(f"Checking {symbol_safe}...")
    safe_res = await checker.check_blackout(symbol_safe)
    print(f"Result: {safe_res}")
    
    # SPY might have no data or be safe
    assert safe_res['reason'] in ['SAFE', 'NO_DATA']

@pytest.mark.asyncio
async def test_liquidity(conn):
    print("\n--- Testing Liquidity Checker ---")
    from ib_insync import Option, Stock

    from analysis.liquidity import get_liquidity_checker
    from ibkr.data_fetcher import get_data_fetcher
    
    fetcher = get_data_fetcher()
    checker = get_liquidity_checker()
    
    # Get a real option to test
    spy = Stock('SPY', 'SMART', 'USD')
    # Use internal connection 
    conn_obj = await fetcher._get_connection()
    await conn_obj.ib.qualifyContractsAsync(spy)
    
    # Get chains
    chains = await conn_obj.ib.reqSecDefOptParamsAsync(spy.symbol, '', spy.secType, spy.conId)
    if not chains:
        pytest.skip("No option chains found")
        
    # Pick first chain, first expiry, first strike
    chain = chains[0]
    exp = sorted(chain.expirations)[0]
    strike = chain.strikes[len(chain.strikes)//2] # Middle strike
    
    option = Option('SPY', exp, strike, 'C', 'SMART')
    await conn_obj.ib.qualifyContractsAsync(option)
    
    print(f"Checking liquidity for {option.localSymbol}...")
    res = await checker.check_option_liquidity(option)
    print(f"Result: {res}")
    
    # We accept failure if data is valid but just illiquid (market hours)
    # But assert result structure is valid
    assert 'passed' in res
    assert 'reason' in res

@pytest.mark.asyncio
async def test_screener(conn):
    print("\n--- Testing Unified Screener ---")
    from analysis.screener import get_screener
    
    screener = get_screener()
    symbols = ['SPY', 'NVDA', 'AAPL']
    
    results = await screener.run_checks(symbols)
    print(f"Market Status: {results['market_status']}")
    
    assert results['market_status'] != 'UNKNOWN'

