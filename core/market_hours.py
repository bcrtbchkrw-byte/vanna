"""
Market Hours Utility

Provides market hours checking for US stock exchanges.
Split from trading_pipeline.py for reuse across modules.
"""
from datetime import datetime, time
from zoneinfo import ZoneInfo
from typing import Tuple

from core.logger import get_logger

logger = get_logger()

# US Eastern timezone
ET = ZoneInfo("America/New_York")

# Market hours (NYSE/NASDAQ)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Pre/Post market
PRE_MARKET_OPEN = time(4, 0)
POST_MARKET_CLOSE = time(20, 0)


def is_market_open() -> bool:
    """
    Check if US stock market is currently open (NYSE/NASDAQ hours).
    
    Returns:
        True if market is open (9:30 AM - 4:00 PM ET, Mon-Fri)
    """
    now = datetime.now(ET)
    
    # Check weekday (0=Mon, 6=Sun)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    
    current_time = now.time()
    return MARKET_OPEN <= current_time < MARKET_CLOSE


def is_pre_market() -> bool:
    """Check if pre-market hours (4:00 AM - 9:30 AM ET)."""
    now = datetime.now(ET)
    
    if now.weekday() >= 5:
        return False
    
    current_time = now.time()
    return PRE_MARKET_OPEN <= current_time < MARKET_OPEN


def is_post_market() -> bool:
    """Check if post-market hours (4:00 PM - 8:00 PM ET)."""
    now = datetime.now(ET)
    
    if now.weekday() >= 5:
        return False
    
    current_time = now.time()
    return MARKET_CLOSE <= current_time < POST_MARKET_CLOSE


def is_extended_hours() -> bool:
    """Check if pre-market or post-market."""
    return is_pre_market() or is_post_market()


def get_market_status() -> str:
    """
    Get current market status as string.
    
    Returns:
        One of: 'OPEN', 'PRE_MARKET', 'POST_MARKET', 'CLOSED'
    """
    if is_market_open():
        return 'OPEN'
    elif is_pre_market():
        return 'PRE_MARKET'
    elif is_post_market():
        return 'POST_MARKET'
    else:
        return 'CLOSED'


def time_until_open() -> Tuple[int, int]:
    """
    Get time until market opens.
    
    Returns:
        Tuple of (hours, minutes) until open, or (0, 0) if open
    """
    if is_market_open():
        return (0, 0)
    
    now = datetime.now(ET)
    
    # Calculate next open
    next_open = now.replace(
        hour=MARKET_OPEN.hour,
        minute=MARKET_OPEN.minute,
        second=0,
        microsecond=0
    )
    
    # If past today's open, use tomorrow
    if now.time() >= MARKET_OPEN:
        from datetime import timedelta
        next_open += timedelta(days=1)
    
    # Skip weekends
    while next_open.weekday() >= 5:
        from datetime import timedelta
        next_open += timedelta(days=1)
    
    diff = next_open - now
    hours = int(diff.total_seconds() // 3600)
    minutes = int((diff.total_seconds() % 3600) // 60)
    
    return (hours, minutes)


def time_until_close() -> Tuple[int, int]:
    """
    Get time until market closes.
    
    Returns:
        Tuple of (hours, minutes) until close, or (0, 0) if closed
    """
    if not is_market_open():
        return (0, 0)
    
    now = datetime.now(ET)
    close_time = now.replace(
        hour=MARKET_CLOSE.hour,
        minute=MARKET_CLOSE.minute,
        second=0,
        microsecond=0
    )
    
    diff = close_time - now
    hours = int(diff.total_seconds() // 3600)
    minutes = int((diff.total_seconds() % 3600) // 60)
    
    return (hours, minutes)
