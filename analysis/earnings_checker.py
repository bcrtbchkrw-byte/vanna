"""
Earnings Checker

Dual mode:
1. BLACKOUT: Block trading near earnings (default - safety)
2. OPPORTUNITY: Identify earnings plays for straddles/strangles (high risk/reward)
"""
from datetime import datetime
from typing import Any, Dict, Optional, cast

from loguru import logger

from config import get_config
from ibkr.data_fetcher import get_data_fetcher


class EarningsChecker:
    """
    Check earnings dates - both for avoidance AND opportunity.
    
    Modes:
    - Safety mode: Block trading within blackout window
    - Opportunity mode: Flag stocks with earnings for earnings plays
    """
    
    def __init__(self):
        self.config = get_config()
        self.data_fetcher = get_data_fetcher()
        self.blackout_hours = self.config.trading.earnings_blackout_hours
        self.cache = {}  # Simple cache to avoid spamming IBKR
        
        # Opportunity settings
        self.opportunity_window_hours = 72  # 3 days before earnings
        self.min_iv_for_play = 0.5  # Min IV for earnings play
    
    async def check_blackout(self, symbol: str) -> Dict[str, Any]:
        """
        Check if symbol is in earnings blackout (SAFE mode).
        
        Returns:
            Dict with status, reason, etc.
        """
        # Check cache (valid for 24h)
        if symbol in self.cache:
            timestamp, result = self.cache[symbol]
            if (datetime.now() - timestamp).total_seconds() < 86400:
                return cast(Dict[str, Any], result)
        
        earnings_date = await self.data_fetcher.get_earnings_date(symbol)
        
        result = {
            'symbol': symbol,
            'in_blackout': False,
            'is_opportunity': False,
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
    
    async def check_opportunity(self, symbol: str) -> Dict[str, Any]:
        """
        Check if symbol has earnings opportunity (AGGRESSIVE mode).
        
        Good for:
        - Straddles before earnings (bet on movement)
        - Strangles (cheaper, needs bigger move)
        - Iron Condors AFTER earnings (sell the vol crush)
        
        Returns:
            Dict with opportunity details
        """
        blackout_result = await self.check_blackout(symbol)
        hours_until = blackout_result.get('hours_until')
        
        result = {
            'symbol': symbol,
            'is_opportunity': False,
            'play_type': None,
            'earnings_date': blackout_result.get('earnings_date'),
            'hours_until': hours_until,
            'reason': 'NO_EARNINGS'
        }
        
        if hours_until is None:
            return result
        
        # Pre-earnings opportunity (straddle/strangle)
        if 0 < hours_until <= self.opportunity_window_hours:
            result['is_opportunity'] = True
            result['play_type'] = 'STRADDLE_PRE_EARNINGS'
            result['reason'] = f'Earnings in {hours_until:.0f}h - high IV expected'
            logger.info(f"🎯 {symbol} EARNINGS OPPORTUNITY: {hours_until:.0f}h until earnings")
        
        # Post-earnings opportunity (sell vol crush)
        elif -48 < hours_until <= -2:
            result['is_opportunity'] = True
            result['play_type'] = 'IRON_CONDOR_POST_EARNINGS'
            result['reason'] = f'Earnings passed {abs(hours_until):.0f}h ago - vol crush opportunity'
            logger.info(f"🎯 {symbol} VOL CRUSH OPPORTUNITY: Earnings passed, sell premium")
        
        return result
    
    async def get_earnings_plays(self, symbols: list) -> list:
        """
        Scan symbols for earnings opportunities.
        
        Returns list of symbols with upcoming earnings plays.
        """
        opportunities = []
        
        for symbol in symbols:
            try:
                opp = await self.check_opportunity(symbol)
                if opp['is_opportunity']:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug(f"Error checking {symbol}: {e}")
        
        return opportunities


# Singleton
_earnings_checker: Optional[EarningsChecker] = None

def get_earnings_checker() -> EarningsChecker:
    global _earnings_checker
    if _earnings_checker is None:
        _earnings_checker = EarningsChecker()
    return _earnings_checker

