"""
Bot Scheduler Strategy

Manages the timing of bot operations using AsyncIOScheduler.
"""
import asyncio
from typing import Callable, Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from datetime import datetime
import pytz

class BotScheduler:
    """
    Schedules trading tasks.
    """
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone=pytz.timezone('US/Eastern'))
        self._running = False
        
    def start(self):
        """Start the scheduler."""
        if not self._running:
            self.scheduler.start()
            self._running = True
            logger.info("📅 Scheduler started (Timezone: US/Eastern)")
            
    def shutdown(self):
        """Stop the scheduler."""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            logger.info("📅 Scheduler stopped")
            
    def add_job(self, func: Callable, trigger_type: str, **kwargs):
        """
        Add a job to the scheduler.
        
        Args:
            func: Async function to call
            trigger_type: 'cron' or 'interval'
            kwargs: Trigger args (e.g. minute='*/15' or minutes=15)
        """
        if trigger_type == 'cron':
            trigger = CronTrigger(**kwargs)
        elif trigger_type == 'interval':
            trigger = IntervalTrigger(**kwargs)
        else:
            logger.error(f"Unknown trigger type: {trigger_type}")
            return
            
        job = self.scheduler.add_job(func, trigger)
        logger.info(f"📅 Added job: {func.__name__} ({trigger})")
        return job

    def run_now(self, func: Callable):
        """Run a job immediately (fire and forget task)."""
        self.scheduler.add_job(func)


# Singleton
_scheduler: Optional[BotScheduler] = None

def get_scheduler() -> BotScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = BotScheduler()
    return _scheduler
