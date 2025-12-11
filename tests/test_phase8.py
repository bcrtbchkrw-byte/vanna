import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from automation.scheduler import BotScheduler
from automation.watchdog import SystemWatchdog


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.mark.asyncio
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
        sched.shutdown()
        pytest.fail("Scheduler job timed out")
        
    sched.shutdown()

@pytest.mark.asyncio
async def test_watchdog(monkeypatch):
    print("\n--- Testing Watchdog ---")
    
    # Mock Connectivity
    ib_conn_mock = MagicMock()
    ib_conn_mock.is_connected = False # Sim disconnected
    ib_conn_mock.connect = AsyncMock()
    
    # Mock return value
    async def mock_getter():
        return ib_conn_mock
        
    # Patch get_ibkr_connection in watchdog using pytest monkeypatch
    monkeypatch.setattr("automation.watchdog.get_ibkr_connection", mock_getter)
    
    wd = SystemWatchdog()
    
    # Run health check manually
    print("Simulating disconnected state...")
    await wd.check_health()
    
    # Verify connect called
    assert ib_conn_mock.connect.called, "Watchdog did NOT attempt reconnect"
    print("✅ Watchdog attempted reconnect")
