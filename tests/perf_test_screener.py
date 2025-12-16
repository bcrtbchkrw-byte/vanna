import asyncio
import time
from core.logger import setup_logger

async def test_screener_performance():
    setup_logger(level="INFO")
    
    print("=" * 60)
    print("🚀 Running Screener Performance Test")
    print("=" * 60)
    
    # Import locally to avoid issues
    from analysis.screener import get_daily_screener, SCREENER_UNIVERSE
    
    # Mock SCREENER_UNIVERSE to use only 20 stocks for testing
    # We choose liquid ones and some illiquid ones
    TEST_UNIVERSE = [
        'SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 
        'TSLA', 'AMD', 'META', 'GOOGL', 'AMZN',
        'IWM', 'GLD', 'TLT', 'SLV', 'EEM',
        'COIN', 'PLTR', 'SOFI', 'MARA', 'GME'
    ]
    
    # Monkey patch the universe for the test
    import analysis.screener
    analysis.screener.SCREENER_UNIVERSE = TEST_UNIVERSE
    
    print(f"Test Universe: {len(TEST_UNIVERSE)} stocks")
    
    start_time = time.time()
    
    screener = get_daily_screener()
    # Force run to bypass cache
    watchlist = await screener.run_morning_screen(force=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print(f"✅ Test Complete")
    print(f"⏱️ Duration: {duration:.2f} seconds")
    print(f"📦 Stocks Processed: {len(TEST_UNIVERSE)}")
    print(f"⚡ Rate: {duration / len(TEST_UNIVERSE):.2f} sec/stock")
    print("=" * 60)
    
    if duration < 60:
        print("✅ PASS: Performance is acceptable (< 60s)")
    else:
        print("❌ FAIL: Too slow (> 60s)")
        
    print(f"Top results: {watchlist[:5]}")

async def test_full_pipeline_performance():
    """Test full pipeline including Step 2 (ML Filter)."""
    print("\n" + "=" * 60)
    print("🚀 Running Full Pipeline Performance Test (Screener + ML)")
    print("=" * 60)
    
    from core.stock_screener import get_stock_screener
    
    # We need to mock the internal screener's universe again or rely on the cached result
    # Best way: patch SCREENER_UNIVERSE in analysis.screener before getting screener
    import analysis.screener
    # Reuse same small universe for speed
    TEST_UNIVERSE = [
        'SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 
        'TSLA', 'AMD', 'META', 'GOOGL', 'AMZN',
        'IWM', 'GLD', 'TLT', 'SLV', 'EEM',
        'COIN', 'PLTR', 'SOFI', 'MARA', 'GME'
    ]
    analysis.screener.SCREENER_UNIVERSE = TEST_UNIVERSE
    
    start_time = time.time()
    
    stock_screener = get_stock_screener()
    # Mocking _ml_filter_top_10 to run on all passed stocks is done by the code logic (up to 50)
    # Since we have < 50 stocks, it will process all 20.
    
    result = await stock_screener.run_morning_screening()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print(f"✅ Full Pipeline Complete")
    print(f"⏱️ Duration: {duration:.2f} seconds")
    print(f"📦 Top 10 Selected: {result.top_10}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_screener_performance())
    asyncio.run(test_full_pipeline_performance())
