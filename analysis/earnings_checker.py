"""
Earnings Checker
Prevent trading near earnings announcements to avoid binary events.
"""
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger
from config import get_config
from ibkr.data_fetcher import get_data_fetcher

class EarningsChecker:
    """Check earnings dates using IBKR fundamental data."""
    
    def __init__(self):
        self.config = get_config()
        self.data_fetcher = get_data_fetcher()
        self.blackout_hours = self.config.trading.earnings_blackout_hours
        self.cache = {} # Simple cache to avoid spamming IBKR
    
    async def check_blackout(self, symbol: str) -> Dict[str, Any]:
        """
        Check if symbol is in earnings blackout.
        
        Returns:
            Dict with status, reason, etc.
        """
        # Check cache (valid for 24h)
        if symbol in self.cache:
            timestamp, result = self.cache[symbol]
            if (datetime.now() - timestamp).total_seconds() < 86400:
                return result
        
        earnings_date = await self.data_fetcher.get_earnings_date(symbol)
        
        result = {
            'symbol': symbol,
            'in_blackout': False,
            'reason': 'SAFE',
            'earnings_date': None,
            'hours_until': None
        }
        
        if earnings_date:
            result['earnings_date'] = earnings_date.strftime('%Y-%m-%d')
            
            # Ensure timezone naive for comparison
            if earnings_date.tzinfo:
                earnings_date = earnings_date.replace(tzinfo=None)
                
            now = datetime.now()
            diff = earnings_date - now
            hours_until = diff.total_seconds() / 3600
            result['hours_until'] = round(hours_until, 1)
            
            # Check blackout window (e.g. within 48h before)
            # Also typically avoid immediately after? For now just before.
            if 0 < hours_until <= self.blackout_hours:
                result['in_blackout'] = True
                result['reason'] = 'EARNINGS_TOO_CLOSE'
                logger.warning(f"⚠️ {symbol} in earnings blackout ({hours_until:.1f}h until {result['earnings_date']})")
            
            # Check if earnings just passed (within 24h) - optional safety
            elif -24 < hours_until <= 0:
                 result['in_blackout'] = True
                 result['reason'] = 'EARNINGS_JUST_PASSED'
                 logger.warning(f"⚠️ {symbol} earnings just passed ({abs(hours_until):.1f}h ago)")

        
        # Update cache
        self.cache[symbol] = (datetime.now(), result)
        return result

# Singleton
_earnings_checker: Optional[EarningsChecker] = None

def get_earnings_checker() -> EarningsChecker:
    global _earnings_checker
    if _earnings_checker is None:
        _earnings_checker = EarningsChecker()
    return _earnings_checker
