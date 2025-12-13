"""
DTE Optimizer - Optimal Days to Expiration Selector

Selects optimal DTE based on VIX term structure.
Uses rule-based logic with ML enhancement.
"""
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np

from core.logger import get_logger

logger = get_logger()


@dataclass
class DTERecommendation:
    """DTE optimization result."""
    dte: int
    confidence: float
    reason: str
    vix: float
    vix_ratio: float


class DTEOptimizer:
    """
    Optimal DTE selection based on VIX term structure.
    
    Logic:
    - Contango (VIX < VIX3M): Long DTE (45-60 days)
      - IV tends to stay low, theta decay is gentle
    - Backwardation (VIX > VIX3M): Short DTE (21-30 days)
      - IV spike, faster decay, less gamma risk
    - Neutral: Medium DTE (30-45 days)
    
    For short puts/credit spreads:
    - Higher VIX → Shorter DTE to capture decay faster
    - Lower VIX → Longer DTE for higher premium
    """
    
    # DTE ranges by market condition
    DTE_RANGES = {
        'contango_low_vix': (45, 60),    # VIX < 15, contango
        'contango_normal': (35, 50),      # VIX 15-20, contango
        'contango_elevated': (30, 45),    # VIX 20-25, contango
        'neutral': (30, 45),              # VIX/VIX3M ≈ 1.0
        'backwardation_mild': (25, 35),   # VIX > VIX3M slightly
        'backwardation_strong': (14, 25), # VIX >> VIX3M (fear)
        'crisis': (7, 14),                # VIX > 35, extreme
    }
    
    def __init__(self, default_dte: int = 30):
        self.default_dte = default_dte
        logger.info("DTEOptimizer initialized")
    
    def get_optimal_dte(
        self,
        vix: float,
        vix3m: float = None,
        strategy: str = 'short_put'
    ) -> DTERecommendation:
        """
        Get optimal DTE based on current market conditions.
        
        Args:
            vix: Current VIX level
            vix3m: VIX3M for term structure (optional)
            strategy: Trading strategy type
            
        Returns:
            DTERecommendation with optimal DTE
        """
        # Calculate VIX ratio (term structure)
        if vix3m and vix3m > 0:
            vix_ratio = vix / vix3m
        else:
            # Estimate VIX3M if not provided
            vix3m = vix * 1.05  # Typical slight contango
            vix_ratio = vix / vix3m
        
        # Determine market condition
        condition, dte_range = self._classify_condition(vix, vix_ratio)
        
        # Select DTE within range
        dte_min, dte_max = dte_range
        
        # Bias within range based on VIX level
        if vix < 15:
            # Low VIX: go longer for more premium
            dte = dte_max
        elif vix > 30:
            # High VIX: go shorter to reduce risk
            dte = dte_min
        else:
            # Middle: use midpoint
            dte = (dte_min + dte_max) // 2
        
        # Adjust for strategy
        if strategy in ['short_put', 'credit_spread']:
            pass  # Default logic applies
        elif strategy in ['iron_condor', 'strangle']:
            dte = max(dte, 30)  # Need more time for adjustments
        elif strategy == 'calendar':
            dte = 45  # Calendars need specific timing
        
        # Calculate confidence
        confidence = self._calculate_confidence(vix, vix_ratio, condition)
        
        return DTERecommendation(
            dte=dte,
            confidence=confidence,
            reason=f"{condition}: VIX={vix:.1f}, ratio={vix_ratio:.2f}",
            vix=vix,
            vix_ratio=vix_ratio
        )
    
    def _classify_condition(
        self,
        vix: float,
        vix_ratio: float
    ) -> Tuple[str, Tuple[int, int]]:
        """Classify market condition."""
        
        # Crisis mode
        if vix > 35:
            return 'crisis', self.DTE_RANGES['crisis']
        
        # Backwardation (fear/spike)
        if vix_ratio > 1.1:
            return 'backwardation_strong', self.DTE_RANGES['backwardation_strong']
        if vix_ratio > 1.02:
            return 'backwardation_mild', self.DTE_RANGES['backwardation_mild']
        
        # Neutral zone
        if 0.98 <= vix_ratio <= 1.02:
            return 'neutral', self.DTE_RANGES['neutral']
        
        # Contango (normal/calm)
        if vix < 15:
            return 'contango_low_vix', self.DTE_RANGES['contango_low_vix']
        if vix < 20:
            return 'contango_normal', self.DTE_RANGES['contango_normal']
        
        return 'contango_elevated', self.DTE_RANGES['contango_elevated']
    
    def _calculate_confidence(
        self,
        vix: float,
        vix_ratio: float,
        condition: str
    ) -> float:
        """Calculate confidence in recommendation."""
        
        # Higher confidence in:
        # - Clear contango/backwardation (ratio far from 1.0)
        # - Typical VIX levels (12-25)
        
        ratio_distance = abs(vix_ratio - 1.0)
        ratio_conf = min(1.0, ratio_distance * 5)  # 0.2 distance = 100%
        
        if 12 <= vix <= 25:
            vix_conf = 0.9
        elif vix < 12 or vix > 35:
            vix_conf = 0.5
        else:
            vix_conf = 0.7
        
        return (ratio_conf + vix_conf) / 2
    
    def get_dte_for_vix_percentile(
        self,
        vix_percentile: float
    ) -> int:
        """
        Quick DTE lookup by VIX percentile.
        
        Args:
            vix_percentile: 0-1 (0=low VIX, 1=high VIX)
            
        Returns:
            Recommended DTE
        """
        # Linear interpolation: low percentile = long DTE
        min_dte = 14
        max_dte = 60
        
        # Inverse: high percentile = short DTE
        dte = int(max_dte - (max_dte - min_dte) * vix_percentile)
        
        return max(min_dte, min(max_dte, dte))


# Singleton
_optimizer: Optional[DTEOptimizer] = None


def get_dte_optimizer() -> DTEOptimizer:
    """Get or create DTE optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = DTEOptimizer()
    return _optimizer
