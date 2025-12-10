"""
Liquidity Checker
Validates option liquidity (Spread, Volume/OI) to avoid slippage.
"""
from typing import Dict, Any, Optional
from ib_insync import Contract
from loguru import logger
from config import get_config
from ibkr.data_fetcher import get_data_fetcher

class LiquidityChecker:
    """Check option liquidity before trading."""
    
    def __init__(self):
        self.config = get_config()
        self.data_fetcher = get_data_fetcher()
        self.max_spread = self.config.liquidity.max_bid_ask_spread
        self.min_vol_oi = self.config.liquidity.min_volume_oi_ratio
        
    async def check_option_liquidity(self, contract: Contract) -> Dict[str, Any]:
        """
        Check if contract meets liquidity requirements.
        """
        greeks = await self.data_fetcher.get_option_greeks(contract)
        
        if not greeks:
            return {'passed': False, 'reason': 'NO_DATA'}
            
        bid = greeks.get('bid', 0)
        ask = greeks.get('ask', 0)
        volume = greeks.get('volume', 0)
        oi = greeks.get('open_interest', 0)
        
        # 1. Spread Check
        spread = ask - bid
        mid = (ask + bid) / 2
        spread_pct = (spread / mid * 100) if mid > 0 else 100
        
        # Allow wider spreads if percentage is low (e.g. expensive options)
        spread_ok = (spread <= self.max_spread) or (spread_pct <= 1.0)
        
        if not spread_ok:
            return {
                'passed': False, 
                'reason': f'SPREAD_TOO_WIDE: ${spread:.2f} ({spread_pct:.1f}%)',
                'spread': spread,
                'volume': volume,
                'oi': oi
            }
            
        # 2. Volume/OI Check
        # If OI is 0, we require significant volume? Or just fail?
        # Let's say if OI > 0, check ratio. If OI=0, fail unless purely volume based? 
        # Follow legacy: if OI > 0 check ratio. 
        
        vol_oi_ratio = 0
        if oi > 0:
            vol_oi_ratio = (volume / oi) * 100
            
        # If OI is 0, fail (illiquid)
        if oi == 0:
             return {'passed': False, 'reason': 'ZERO_OPEN_INTEREST', 'spread': spread, 'volume': volume, 'oi': 0}

        if vol_oi_ratio < self.min_vol_oi:
             # Strict check could block valid trades early in day. 
             # Warning only? No, let's enforce provided config.
             return {
                 'passed': False, 
                 'reason': f'LOW_VOL_OI_RATIO: {vol_oi_ratio:.1f}% < {self.min_vol_oi}%',
                 'spread': spread,
                 'volume': volume,
                 'oi': oi
             }
             
        return {
            'passed': True,
            'reason': 'LIQUID',
            'spread': spread,
            'volume': volume,
            'oi': oi,
            'vol_oi_ratio': vol_oi_ratio
        }

# Singleton
_liquidity_checker: Optional[LiquidityChecker] = None

def get_liquidity_checker() -> LiquidityChecker:
    global _liquidity_checker
    if _liquidity_checker is None:
        _liquidity_checker = LiquidityChecker()
    return _liquidity_checker
