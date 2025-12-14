"""
Scheduler Module

Handles background maintenance tasks:
1. Monthly Data Maintenance (1st of month)
2. Weekly ML/RL Training (Saturdays)
3. Other periodic jobs
"""

import asyncio
from datetime import datetime
from loguru import logger

from automation.data_maintenance import get_maintenance_manager

class Scheduler:
    """
    Background task scheduler.
    Independent of market hours.
    """
    
    def __init__(self):
        self._shutdown_requested = False
        self._last_maintenance_date = None
        self._last_training_date = None
        self.check_interval = 60 # Check every minute
        
        logger.info("Scheduler initialized")

    async def start(self):
        """Start the scheduler loop."""
        logger.info("⏳ Scheduler started (background)")
        
        while not self._shutdown_requested:
            try:
                await self._check_schedule()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
    
    def stop(self):
        """Stop scheduler."""
        self._shutdown_requested = True
        logger.info("Scheduler stopped")

    async def _check_schedule(self):
        """Check if any jobs need running."""
        now = datetime.now()
        today = now.date()
        
        # 1. Monthly Maintenance (1st of month, 00:00-01:00)
        if now.day == 1 and now.hour == 0:
            if self._last_maintenance_date != today:
                logger.info("🔧 Running monthly data maintenance...")
                try:
                    manager = get_maintenance_manager()
                    await manager.run_maintenance()
                    self._last_maintenance_date = today
                    logger.info("✅ Monthly maintenance complete")
                except Exception as e:
                    logger.error(f"Maintenance failed: {e}")
        
        # 2. Saturday Training (Saturday, 06:00)
        # 5 = Saturday
        if now.weekday() == 5 and now.hour == 6:
            if self._last_training_date != today:
                logger.info("🗓️ SATURDAY: Starting weekly ML/RL training...")
                try:
                    from automation.saturday_training import run_saturday_training
                    
                    # Run training
                    results = await run_saturday_training(skip_rl=False, rl_timesteps=50_000)
                    logger.info(f"✅ Saturday training complete: {results}")
                    
                    self._last_training_date = today
                except Exception as e:
                    logger.error(f"Saturday training failed: {e}")

# Singleton
_scheduler = None

def get_scheduler() -> Scheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = Scheduler()
    return _scheduler
