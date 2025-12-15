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
        
        # 2. Daily Live Data Merge (After market close, 16:30)
        # Merge live bars from SQLite to parquet daily (Mon-Fri)
        if now.weekday() < 5 and now.hour == 16 and now.minute >= 30 and now.minute < 31:
            merge_key = f"merge_{today}"
            if not hasattr(self, '_last_merge_date') or self._last_merge_date != today:
                logger.info("📊 Daily merge: Saving live data to parquet...")
                try:
                    manager = get_maintenance_manager()
                    results = await manager.merge_live_to_parquet()
                    self._last_merge_date = today
                    total = sum(v for v in results.values() if v > 0)
                    logger.info(f"✅ Daily merge complete: {total} new bars saved")
                except Exception as e:
                    logger.error(f"Daily merge failed: {e}")
        
        # 3. Saturday Data + Training (Saturday, 06:00)
        # 5 = Saturday
        if now.weekday() == 5 and now.hour == 6:
            if self._last_training_date != today:
                logger.info("🗓️ SATURDAY: Starting weekly data maintenance + training...")
                try:
                    manager = get_maintenance_manager()
                    
                    # Step 1: Merge live data to parquet (final cleanup)
                    logger.info("📊 Step 1/3: Merging live data...")
                    await manager.merge_live_to_parquet()
                    
                    # Step 2: Check and patch any gaps
                    logger.info("🔧 Step 2/3: Checking for data gaps...")
                    await manager.run_maintenance()
                    
                    # Step 3: Run ML/RL training
                    logger.info("🧠 Step 3/3: Running ML/RL training...")
                    from automation.saturday_training import run_saturday_training
                    results = await run_saturday_training(skip_rl=False, rl_timesteps=50_000)
                    
                    logger.info(f"✅ Saturday routine complete: {results}")
                    self._last_training_date = today
                except Exception as e:
                    logger.error(f"Saturday routine failed: {e}")

# Singleton
_scheduler = None

def get_scheduler() -> Scheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = Scheduler()
    return _scheduler
