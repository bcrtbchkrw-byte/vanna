"""
VIX Monitor - Market Regime Detection
Monitors VIX and classifies market into regimes (Low/Normal/High/Extreme).
"""
from typing import Optional

from loguru import logger

from config import get_config
from ibkr.data_fetcher import get_data_fetcher


class VIXMonitor:
    """
    Monitor VIX and classify market regime.
    """
    
    def __init__(self):
        self.config = get_config()
        self.data_fetcher = get_data_fetcher()
        self.current_vix: Optional[float] = None
        self._current_regime: str = "UNKNOWN"
        
        # Panic threshold from config
        self.panic_threshold = self.config.trading.vix_panic_threshold
    
    async def update(self) -> str:
        """
        Fetch latest VIX and update regime.
        
        Returns:
            Current regime string
        """
        vix = await self.data_fetcher.get_vix()
        
        if vix is None:
            logger.warning("Could not fetch VIX, keeping previous regime")
            return self._current_regime
            
        self.current_vix = vix
        self._current_regime = self._determine_regime(vix)
        
        return self._current_regime
    
    def _determine_regime(self, vix: float) -> str:
        """Determine regime based on VIX level."""
        if vix >= 40:
            return "EXTREME"
        elif vix >= self.panic_threshold: # Default 30
            return "HIGH_VOL"
        elif vix >= 20:
            return "ELEVATED"
        elif vix >= 15:
            return "NORMAL"
        else:
            return "LOW_VOL"
            
    def is_trading_allowed(self) -> bool:
        """
        Check if trading is allowed based on VIX.
        EXTREME and HIGH_VOL (> panic_threshold) block new trades.
        """
        if self._current_regime in ["EXTREME", "HIGH_VOL"]:
            logger.warning(f"🛑 TRADING BLOCKED: VIX in {self._current_regime} mode ({self.current_vix:.2f})")
            return False
        return True

    def get_current_regime(self) -> str:
        return self._current_regime


# Singleton
_vix_monitor: Optional[VIXMonitor] = None

def get_vix_monitor() -> VIXMonitor:
    global _vix_monitor
    if _vix_monitor is None:
        _vix_monitor = VIXMonitor()
    return _vix_monitor
