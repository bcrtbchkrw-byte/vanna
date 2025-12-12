"""
Vanna Calculator - Black-Scholes Greeks Calculation

Calculates Vanna (∂²V/∂S∂σ), Charm (∂Δ/∂t), Volga (∂²V/∂σ²) and other Greeks.
Uses analytical Black-Scholes formula with dynamic risk-free rate.
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

from core.logger import get_logger

logger = get_logger()


@dataclass
class GreeksResult:
    """Container for all Greeks calculations."""
    delta: float
    gamma: float
    theta: float
    vega: float
    vanna: float  # ∂Δ/∂σ = ∂Vega/∂S
    charm: float  # ∂Δ/∂t (Delta decay)
    volga: float  # ∂²V/∂σ² (Vega convexity)
    rho: float
    
    # Inputs used
    S: float  # Underlying price
    K: float  # Strike
    T: float  # Time to expiry (years)
    sigma: float  # IV
    r: float  # Risk-free rate
    option_type: str  # 'call' or 'put'


class VannaCalculator:
    """
    Calculate Vanna and other second-order Greeks using Black-Scholes.
    
    Vanna measures sensitivity of Delta to changes in volatility.
    Critical for stress testing options strategies.
    
    Features:
    - Analytical Black-Scholes formula (precise)
    - Numerical method (for validation)
    - Dynamic risk-free rate from environment/IBKR
    - Full Greeks surface calculation
    """
    
    DEFAULT_RISK_FREE_RATE = 0.045  # 4.5% fallback
    
    def __init__(self, risk_free_rate: Optional[float] = None):
        """
        Initialize Vanna calculator.
        
        Args:
            risk_free_rate: Risk-free rate (e.g., 0.045 = 4.5%)
                           If None, uses environment or default.
        """
        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate
        else:
            import os
            self.risk_free_rate = float(
                os.getenv('RISK_FREE_RATE', str(self.DEFAULT_RISK_FREE_RATE))
            )
        
        logger.debug(f"VannaCalculator initialized with r={self.risk_free_rate:.4f}")
    
    def _d1_d2(self, S: float, K: float, T: float, sigma: float, r: float) -> tuple:
        """Calculate d1 and d2 from Black-Scholes."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def calculate_vanna(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        r: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate Vanna using Black-Scholes analytical formula.
        
        Vanna = ∂²V/∂S∂σ = -φ(d1) × d2 / (S × σ × √T)
        
        Vanna is the same for calls and puts.
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility (decimal, e.g. 0.25 for 25%)
            r: Risk-free rate (optional, uses instance default)
            
        Returns:
            Vanna value or None if invalid inputs
        """
        try:
            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
                return None
            
            r = r if r is not None else self.risk_free_rate
            d1, d2 = self._d1_d2(S, K, T, sigma, r)
            
            # Black-Scholes Vanna formula
            phi_d1 = norm.pdf(d1)
            vanna = -(phi_d1 * d2) / (S * sigma * np.sqrt(T))
            
            logger.debug(
                f"Vanna: S={S:.2f}, K={K:.2f}, T={T:.3f}y, σ={sigma:.2%}, "
                f"d1={d1:.3f}, d2={d2:.3f}, Vanna={vanna:.6f}"
            )
            
            return vanna
            
        except Exception as e:
            logger.error(f"Error calculating Vanna: {e}")
            return None
    
    def calculate_charm(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str = 'call',
        r: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate Charm (Delta decay) - ∂Δ/∂t.
        
        Measures how Delta changes as time passes.
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility
            option_type: 'call' or 'put'
            r: Risk-free rate
            
        Returns:
            Charm value
        """
        try:
            if T <= 0 or sigma <= 0 or S <= 0:
                return None
            
            r = r if r is not None else self.risk_free_rate
            d1, d2 = self._d1_d2(S, K, T, sigma, r)
            
            phi_d1 = norm.pdf(d1)
            sqrt_T = np.sqrt(T)
            
            # Charm formula
            term1 = phi_d1 * (r / (sigma * sqrt_T) - d2 / (2 * T))
            
            if option_type.lower() == 'call':
                charm = -term1
            else:
                charm = -term1  # Same for put when considering the sign convention
            
            return charm
            
        except Exception as e:
            logger.error(f"Error calculating Charm: {e}")
            return None
    
    def calculate_volga(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        r: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate Volga (Vomma) - ∂²V/∂σ² = Vega × d1 × d2 / σ.
        
        Measures the convexity of Vega (how Vega changes with IV).
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility
            r: Risk-free rate
            
        Returns:
            Volga value
        """
        try:
            if T <= 0 or sigma <= 0 or S <= 0:
                return None
            
            r = r if r is not None else self.risk_free_rate
            d1, d2 = self._d1_d2(S, K, T, sigma, r)
            
            # Vega calculation first
            vega = S * norm.pdf(d1) * np.sqrt(T)
            
            # Volga = Vega × d1 × d2 / σ
            volga = vega * d1 * d2 / sigma
            
            return volga
            
        except Exception as e:
            logger.error(f"Error calculating Volga: {e}")
            return None
    
    def calculate_all_greeks(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str = 'call',
        r: Optional[float] = None
    ) -> Optional[GreeksResult]:
        """
        Calculate all Greeks including second-order.
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility
            option_type: 'call' or 'put'
            r: Risk-free rate
            
        Returns:
            GreeksResult with all Greeks
        """
        try:
            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
                return None
            
            r = r if r is not None else self.risk_free_rate
            d1, d2 = self._d1_d2(S, K, T, sigma, r)
            
            phi_d1 = norm.pdf(d1)
            sqrt_T = np.sqrt(T)
            
            # First-order Greeks
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
                theta = (
                    -S * phi_d1 * sigma / (2 * sqrt_T)
                    - r * K * np.exp(-r * T) * norm.cdf(d2)
                ) / 365  # Daily theta
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                delta = norm.cdf(d1) - 1
                theta = (
                    -S * phi_d1 * sigma / (2 * sqrt_T)
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)
                ) / 365
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
            gamma = phi_d1 / (S * sigma * sqrt_T)
            vega = S * phi_d1 * sqrt_T / 100  # Per 1% move
            
            # Second-order Greeks
            vanna = -(phi_d1 * d2) / (S * sigma * sqrt_T)
            charm = -phi_d1 * (r / (sigma * sqrt_T) - d2 / (2 * T))
            volga = vega * d1 * d2 / sigma
            
            return GreeksResult(
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                vanna=vanna,
                charm=charm,
                volga=volga,
                rho=rho,
                S=S, K=K, T=T, sigma=sigma, r=r,
                option_type=option_type
            )
            
        except Exception as e:
            logger.error(f"Error calculating all Greeks: {e}")
            return None
    
    def calculate_vanna_surface(
        self,
        S: float,
        strikes: List[float],
        expiries: List[float],  # Years to expiration
        ivs: Dict[tuple, float],  # (strike, expiry) -> IV
        option_type: str = 'call'
    ) -> List[Dict[str, Any]]:
        """
        Calculate Vanna for a surface of strikes/expiries.
        
        Args:
            S: Underlying price
            strikes: List of strike prices
            expiries: List of expiration times (years)
            ivs: Dict mapping (strike, expiry) to IV
            option_type: 'call' or 'put'
            
        Returns:
            List of dicts with strike, expiry, vanna, and other Greeks
        """
        results = []
        
        for K in strikes:
            for T in expiries:
                key = (K, T)
                sigma = ivs.get(key, 0.25)  # Default 25% IV
                
                greeks = self.calculate_all_greeks(S, K, T, sigma, option_type)
                
                if greeks:
                    results.append({
                        'strike': K,
                        'expiry_years': T,
                        'iv': sigma,
                        'vanna': greeks.vanna,
                        'delta': greeks.delta,
                        'gamma': greeks.gamma,
                        'vega': greeks.vega,
                        'charm': greeks.charm,
                        'volga': greeks.volga
                    })
        
        logger.info(f"Calculated Vanna surface: {len(results)} points")
        return results
    
    def calculate_vanna_numerical(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str = 'call',
        d_sigma: float = 0.01
    ) -> Optional[float]:
        """
        Calculate Vanna using numerical finite difference.
        
        Useful for validation against analytical formula.
        Vanna ≈ [Delta(σ + Δσ) - Delta(σ)] / Δσ
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration
            sigma: Implied volatility
            option_type: 'call' or 'put'
            d_sigma: IV increment (default 1%)
            
        Returns:
            Numerical Vanna approximation
        """
        try:
            if T <= 0 or sigma <= 0 or S <= 0:
                return None
            
            r = self.risk_free_rate
            
            # Delta at current IV
            d1_1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            delta_1 = norm.cdf(d1_1) if option_type.lower() == 'call' else norm.cdf(d1_1) - 1
            
            # Delta at IV + d_sigma
            sigma_2 = sigma + d_sigma
            d1_2 = (np.log(S / K) + (r + 0.5 * sigma_2**2) * T) / (sigma_2 * np.sqrt(T))
            delta_2 = norm.cdf(d1_2) if option_type.lower() == 'call' else norm.cdf(d1_2) - 1
            
            vanna_numerical = (delta_2 - delta_1) / d_sigma
            
            return vanna_numerical
            
        except Exception as e:
            logger.error(f"Error calculating numerical Vanna: {e}")
            return None


# Singleton
_vanna_calculator: Optional[VannaCalculator] = None


def get_vanna_calculator(risk_free_rate: Optional[float] = None) -> VannaCalculator:
    """Get or create singleton Vanna calculator."""
    global _vanna_calculator
    if _vanna_calculator is None:
        _vanna_calculator = VannaCalculator(risk_free_rate)
    return _vanna_calculator
