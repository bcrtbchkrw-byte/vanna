"""
Screener
Unified market analysis interface.
Aggregates VIX, Earnings, and Liquidity checks to filter trading candidates.
"""
from typing import Any, Dict, List, Optional

from loguru import logger

from analysis.earnings_checker import get_earnings_checker
from analysis.liquidity import get_liquidity_checker
from analysis.vix_monitor import get_vix_monitor


class Screener:
    """
    Market Screener.
    Filters symbols based on global market conditions and asset-specific safety checks.
    """
    
    def __init__(self):
        self.vix_monitor = get_vix_monitor()
        self.earnings_checker = get_earnings_checker()
        self.liquidity_checker = get_liquidity_checker()
    
    async def run_checks(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Run all checks for a list of symbols.
        
        Returns:
            Dict containing 'market_status' and 'results' (per symbol)
        """
        passed_symbols: List[str] = []
        failed_symbols: Dict[str, str] = {}
        details: Dict[str, Any] = {}

        results: Dict[str, Any] = {
            'market_status': 'UNKNOWN',
            'passed_symbols': passed_symbols,
            'failed_symbols': failed_symbols,
            'details': details
        }
        
        # 1. Global Market Check (VIX)
        regime = await self.vix_monitor.update()
        results['market_status'] = regime
        
        if not self.vix_monitor.is_trading_allowed():
            logger.warning(f"🛑 Market Screener HALTED: VIX Regime {regime}")
            return results
            
        logger.info(f"✅ Market Safe (VIX: {regime}). Screening {len(symbols)} symbols...")
        
        # 2. Asset-Specific Checks (Earnings)
        # We don't check liquidity here yet because that requires specific option contracts.
        # Screener focuses on symbol safety first.
        
        for symbol in symbols:
            # Check Earnings
            earnings_result = await self.earnings_checker.check_blackout(symbol)
            
            if earnings_result['in_blackout']:
                failed_symbols[symbol] = f"Earnings Blackout ({earnings_result['reason']})"
                logger.info(f"❌ {symbol} failed: Earnings Blackout")
                continue
                
            # If passed earnings, we consider it a candidate
            # (Liquidity is checked later on specific contracts found for this symbol)
            passed_symbols.append(symbol)
            details[symbol] = {'earnings': earnings_result}
            
        logger.info(f"🛡️ Screening Complete: {len(passed_symbols)}/{len(symbols)} passed")
        return results

# Singleton
_screener: Optional[Screener] = None

def get_screener() -> Screener:
    global _screener
    if _screener is None:
        _screener = Screener()
    return _screener
