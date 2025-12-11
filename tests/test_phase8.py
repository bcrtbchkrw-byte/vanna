"""
Phase 8 Test - Automation

Tests:
- Scheduler job firing
- Watchdog connectivity monitor

Run: python tests/test_phase8.py
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automation.scheduler import BotScheduler
from automation.watchdog import SystemWatchdog
from core.logger import setup_logger


async def test_scheduler():
    print("\n--- Testing Scheduler ---")
    sched = BotScheduler()
    sched.start()
    
    # Event flag
    event = asyncio.Event()
    
    async def job():
        print("✅ Job Fired!")
        event.set()
        
    # Schedule job for 1 second later
    sched.add_job(job, 'interval', seconds=1)
    
    print("Waiting for job (max 3s)...")
    try:
        await asyncio.wait_for(event.wait(), timeout=3.0)
        print("✅ Scheduler works")
    except asyncio.TimeoutError:
        print("❌ Scheduler job timed out")
        sched.shutdown()
        return False
        
    sched.shutdown()
    return True

async def test_watchdog():
    print("\n--- Testing Watchdog ---")
    
    # Mock Connectivity
    ib_conn_mock = MagicMock()
    ib_conn_mock.is_connected = False # Sim disconnected
    ib_conn_mock.connect = AsyncMock()
    
    # Patch get_ibkr_connection in watchdog
    import automation.watchdog
    
    # Mocking the module level function requires patching where it's imported OR
    # simpler: just instantiate Watchdog and rely on injecting mock if possible,
    # but Watchdog calls `get_ibkr_connection` inside `check_health`.
    # We will MonkeyPatch the function in the module namespace.
    
    original_getter = automation.watchdog.get_ibkr_connection
    
    async def mock_getter():
        return ib_conn_mock
        
    automation.watchdog.get_ibkr_connection = mock_getter
    
    try:
        wd = SystemWatchdog()
        
        # Run health check manually
        print("Simulating disconnected state...")
        await wd.check_health()
        
        # Verify connect called
        if ib_conn_mock.connect.called:
            print("✅ Watchdog attempted reconnect")
            return True
        else:
            print("❌ Watchdog did NOT attempt reconnect")
            return False
            
    finally:
        # Restore
        automation.watchdog.get_ibkr_connection = original_getter


async def run_tests():
    setup_logger()
    
    results = []
    results.append(await test_scheduler())
    results.append(await test_watchdog())
    
    if all(results):
        print("\n✅ ALL PHASE 8 TESTS PASSED")
        return 0
    else:
        print("\n❌ TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(run_tests()))
