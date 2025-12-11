"""
Implied Volatility Surface for Vanna Trading Bot.

Constructs and analyzes the IV surface across strikes and expirations.
Provides insights for identifying mispriced options and optimal strike
selection based on volatility term structure.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import griddata

from core.logger import get_logger


@dataclass
class IVSurfacePoint:
    """Single point on the IV surface."""
    strike: float
    days_to_expiry: int
    implied_volatility: float
    moneyness: float  # strike / underlying


@dataclass
class IVSurfaceAnalysis:
    """Analysis results from IV surface."""
    term_structure: str  # "CONTANGO", "BACKWARDATION", "FLAT"
    skew_type: str  # "PUT_SKEW", "CALL_SKEW", "SMILE"
    atm_iv: float
    surface_quality: float  # Data quality score 0-1


class IVSurface:
    """
    Implied Volatility Surface analyzer.
    
    Constructs a 3D surface of IV across strikes (moneyness) and
    time to expiration. Used for:
    - Identifying cheap/expensive options
    - Understanding term structure
    - Detecting skew patterns
    """
    
    def __init__(self) -> None:
        self.logger = get_logger()
        self._surface_data: list[IVSurfacePoint] = []
        self._underlying_price: float = 0.0
    
    def build_surface(
        self,
        option_chain: list[dict[str, Any]],
        underlying_price: float
    ) -> bool:
        """
        Build IV surface from option chain data.
        
        Args:
            option_chain: List of options with iv, strike, dte
            underlying_price: Current underlying price
            
        Returns:
            True if surface was built successfully
        """
        self._underlying_price = underlying_price
        self._surface_data = []
        
        for opt in option_chain:
            iv = opt.get("iv", opt.get("impliedVolatility", 0))
            strike = opt.get("strike", 0)
            dte = opt.get("dte", opt.get("daysToExpiry", 0))
            
            if iv > 0 and strike > 0 and dte > 0:
                moneyness = strike / underlying_price
                
                self._surface_data.append(IVSurfacePoint(
                    strike=strike,
                    days_to_expiry=dte,
                    implied_volatility=iv,
                    moneyness=moneyness
                ))
        
        self.logger.info(f"Built IV surface with {len(self._surface_data)} points")
        return len(self._surface_data) >= 10
    
    def get_iv_at(
        self,
        strike: float,
        days_to_expiry: int
    ) -> float | None:
        """
        Get IV at a specific strike and DTE.
        
        Uses interpolation if exact match not found.
        """
        if not self._surface_data or self._underlying_price == 0:
            return None
        
        # Try exact match first
        for point in self._surface_data:
            if abs(point.strike - strike) < 0.5 and point.days_to_expiry == days_to_expiry:
                return point.implied_volatility
        
        # Interpolate
        try:
            moneyness_target = strike / self._underlying_price
            
            points = np.array([
                [p.moneyness, p.days_to_expiry] for p in self._surface_data
            ])
            ivs = np.array([p.implied_volatility for p in self._surface_data])
            
            # Grid interpolation
            result = griddata(
                points, ivs, 
                [(moneyness_target, days_to_expiry)],
                method='linear'
            )
            
            if not np.isnan(result[0]):
                return float(result[0])
        except Exception as e:
            self.logger.debug(f"IV interpolation failed: {e}")
        
        return None
    
    def analyze_term_structure(self) -> str:
        """
        Analyze the term structure of volatility.
        
        Returns:
            "CONTANGO" if near-term IV < long-term IV
            "BACKWARDATION" if near-term IV > long-term IV
            "FLAT" if relatively flat
        """
        if not self._surface_data:
            return "UNKNOWN"
        
        # Get ATM options at different DTEs
        atm_options = [
            p for p in self._surface_data
            if 0.95 <= p.moneyness <= 1.05
        ]
        
        if len(atm_options) < 2:
            return "UNKNOWN"
        
        # Sort by DTE
        atm_options.sort(key=lambda x: x.days_to_expiry)
        
        # Compare front-month to back-month
        front_month = [p for p in atm_options if p.days_to_expiry <= 30]
        back_month = [p for p in atm_options if p.days_to_expiry > 45]
        
        if not front_month or not back_month:
            return "FLAT"
        
        front_iv = sum(p.implied_volatility for p in front_month) / len(front_month)
        back_iv = sum(p.implied_volatility for p in back_month) / len(back_month)
        
        ratio = front_iv / back_iv if back_iv > 0 else 1.0
        
        if ratio > 1.1:
            return "BACKWARDATION"
        elif ratio < 0.9:
            return "CONTANGO"
        else:
            return "FLAT"
    
    def get_analysis(self) -> IVSurfaceAnalysis:
        """Get comprehensive surface analysis."""
        term_structure = self.analyze_term_structure()
        
        # Get ATM IV
        atm_options = [
            p for p in self._surface_data
            if 0.98 <= p.moneyness <= 1.02
        ]
        atm_iv = sum(p.implied_volatility for p in atm_options) / len(atm_options) if atm_options else 0.0
        
        # Determine skew type (simplified)
        put_side = [p for p in self._surface_data if p.moneyness < 0.95]
        call_side = [p for p in self._surface_data if p.moneyness > 1.05]
        
        put_avg_iv = sum(p.implied_volatility for p in put_side) / len(put_side) if put_side else 0
        call_avg_iv = sum(p.implied_volatility for p in call_side) / len(call_side) if call_side else 0
        
        if put_avg_iv > atm_iv > call_avg_iv:
            skew_type = "PUT_SKEW"
        elif call_avg_iv > atm_iv > put_avg_iv:
            skew_type = "CALL_SKEW"
        elif put_avg_iv > atm_iv and call_avg_iv > atm_iv:
            skew_type = "SMILE"
        else:
            skew_type = "NORMAL"
        
        # Surface quality
        quality = min(len(self._surface_data) / 50, 1.0)
        
        return IVSurfaceAnalysis(
            term_structure=term_structure,
            skew_type=skew_type,
            atm_iv=atm_iv,
            surface_quality=quality
        )


# Singleton
_iv_surface: IVSurface | None = None


def get_iv_surface() -> IVSurface:
    """Get global IV surface instance."""
    global _iv_surface
    if _iv_surface is None:
        _iv_surface = IVSurface()
    return _iv_surface
