"""
System Watchdog

Monitors application health and connectivity.
"""
from typing import Optional

from loguru import logger

from automation.scheduler import get_scheduler
from ibkr.connection import get_ibkr_connection


class SystemWatchdog:
    """
    Monitors system health.
    """
    
    def __init__(self):
        self.scheduler = get_scheduler()
        self.check_interval_seconds = 60
        
    def start(self):
        """Start the watchdog checks."""
        self.scheduler.add_job(
            self.check_health, 
            'interval', 
            seconds=self.check_interval_seconds
        )
        logger.info("🐕 Watchdog started")
        
    async def check_health(self):
        """
        Run health checks.
        """
        logger.debug("🐕 Watchdog checking health...")
        
        # 1. IBKR Connection
        conn = await get_ibkr_connection()
        if not conn.is_connected:
            logger.error("🐕 Watchdog Alert: IBKR Not Connected! Attempting reconnect...")
            try:
                await conn.connect()
            except Exception as e:
                logger.error(f"🐕 Watchdog Reconnect Failed: {e}")
        else:
            # Ping?
            pass
            
        # 2. Database (TODO: Check if DB is locked?)
        
        # 3. Disk Space (Optional)


# Singleton
_watchdog: Optional[SystemWatchdog] = None

def get_watchdog() -> SystemWatchdog:
    global _watchdog
    if _watchdog is None:
        _watchdog = SystemWatchdog()
    return _watchdog
