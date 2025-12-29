#!/usr/bin/env python3
"""
greeks_engine.py

GREEKS ENGINE PRO ML/PPO TRADING - OPTIMIZED VERSION

Optimalizováno pro:
- Konzistentní jednotky napříč všemi Greeks
- RYCHLÝ VECTORIZED IV VÝPOČET (Newton-Raphson)
- Normalizované hodnoty vhodné pro neuronové sítě
- Rychlé vectorized výpočty pro DataFrames

JEDNOTKY (všechny Greeks):
- Delta: [-1, 1] - bezrozměrné
- Gamma: per $1 změny underlying
- Theta: per 1 KALENDÁŘNÍ den (záporné = ztráta hodnoty)
- Vega: per 1% změny IV (0.01 sigma)
- Rho: per 1% změny rate (0.01 r)
- Vanna: per 1% změny IV
- Charm: per 1 kalendářní den
- Vomma: per 1% změny IV
- Veta: per 1 kalendářní den
- Speed: per $1 změny underlying
- Zomma: per 1% změny IV
- Color: per 1 kalendářní den
- Ultima: per 1% změny IV

POUŽITÍ:
    from greeks_engine import GreeksEngine, create_engine
    
    engine = create_engine()  # S RatesProvider
    # nebo
    engine = GreeksEngine()   # Bez RatesProvider
    
    # Všechny Greeks pro jednu opci
    greeks = engine.calculate_greeks(
        S=595.50, K=600.0, option_price=11.50, dte=30,
        is_call=True, symbol='SPY'
    )
    
    # RYCHLÝ vectorized IV pro arrays/DataFrames
    ivs = engine.calculate_iv_vectorized(prices, S, K, T, r, q, is_call)
    
    # Zpracování celého DataFrame
    df_with_greeks = engine.process_dataframe(df, symbol='SPY')
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, date
from typing import Optional, Union, Dict, List
import warnings

try:
    from rates_provider import RatesProvider
except ImportError:
    RatesProvider = None


class GreeksEngine:
    """
    Greeks engine optimalizovaný pro ML/PPO.
    
    Všechny Greeks mají konzistentní jednotky vhodné pro neuronové sítě.
    """
    
    # Škálovací faktor pro theta (ThetaData kompatibilita)
    THETA_SCALE = 0.9332  # ≈ 252/270
    
    def __init__(self, rates_provider: Optional['RatesProvider'] = None):
        """
        Args:
            rates_provider: RatesProvider pro dynamické r a q
        """
        self.rates = rates_provider
    
    # =========================================================================
    # BLACK-SCHOLES CORE
    # =========================================================================
    
    def _d1(self, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """d1 parametr Black-Scholes."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    def _d2(self, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """d2 parametr Black-Scholes."""
        return self._d1(S, K, T, r, q, sigma) - sigma*np.sqrt(T)
    
    def bs_price(self, S: float, K: float, T: float, r: float, q: float,
                 sigma: float, is_call: bool = True) -> float:
        """Black-Scholes cena opce."""
        if T <= 0:
            return max(0, S - K) if is_call else max(0, K - S)
        if sigma <= 0:
            df = np.exp(-r*T)
            fwd = S * np.exp((r-q)*T)
            return df * max(0, fwd - K) if is_call else df * max(0, K - fwd)
        
        d1 = self._d1(S, K, T, r, q, sigma)
        d2 = d1 - sigma*np.sqrt(T)
        
        if is_call:
            return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
    
    def bs_price_vectorized(self, S: np.ndarray, K: np.ndarray, T: np.ndarray,
                            r: float, q: float, sigma: np.ndarray, 
                            is_call: np.ndarray) -> np.ndarray:
        """Vectorized Black-Scholes price calculation."""
        n = len(S)
        prices = np.zeros(n)
        
        # Valid mask
        valid = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
        
        if valid.any():
            S_v, K_v, T_v, sigma_v = S[valid], K[valid], T[valid], sigma[valid]
            is_call_v = is_call[valid]
            
            sqrt_T = np.sqrt(T_v)
            d1 = (np.log(S_v/K_v) + (r - q + 0.5*sigma_v**2)*T_v) / (sigma_v*sqrt_T)
            d2 = d1 - sigma_v*sqrt_T
            
            exp_qT = np.exp(-q*T_v)
            exp_rT = np.exp(-r*T_v)
            
            call_price = S_v*exp_qT*norm.cdf(d1) - K_v*exp_rT*norm.cdf(d2)
            put_price = K_v*exp_rT*norm.cdf(-d2) - S_v*exp_qT*norm.cdf(-d1)
            
            prices[valid] = np.where(is_call_v, call_price, put_price)
        
        # Handle expired options
        expired = T <= 0
        if expired.any():
            intrinsic_call = np.maximum(0, S[expired] - K[expired])
            intrinsic_put = np.maximum(0, K[expired] - S[expired])
            prices[expired] = np.where(is_call[expired], intrinsic_call, intrinsic_put)
        
        return prices
    
    def calculate_iv(self, price: float, S: float, K: float, T: float,
                     r: float, q: float, is_call: bool = True,
                     tol: float = 1e-6, max_iter: int = 100) -> float:
        """
        Výpočet implied volatility z ceny opce (Brent's method).
        
        Returns:
            IV jako desetinné číslo (např. 0.20 = 20%)
            np.nan pokud nelze vypočítat
        """
        if T <= 0 or price <= 0:
            return np.nan
        
        # Intrinsic value check
        intrinsic = max(0, S*np.exp(-q*T) - K*np.exp(-r*T)) if is_call else max(0, K*np.exp(-r*T) - S*np.exp(-q*T))
        if price < intrinsic - 0.01:
            return np.nan
        
        def objective(sigma):
            return self.bs_price(S, K, T, r, q, sigma, is_call) - price
        
        try:
            iv = brentq(objective, 0.001, 5.0, xtol=tol, maxiter=max_iter)
            return iv
        except (ValueError, RuntimeError):
            return np.nan
    
    def calculate_iv_vectorized(self, prices: np.ndarray, S: np.ndarray, 
                                 K: np.ndarray, T: np.ndarray,
                                 r: float, q: float, is_call: np.ndarray,
                                 max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
        """
        FAST vectorized IV calculation using Newton-Raphson method.
        
        ~100x faster than row-by-row Brent's method for large arrays.
        
        Args:
            prices: Option prices array
            S: Underlying prices array
            K: Strike prices array
            T: Time to expiry array (in years)
            r: Risk-free rate
            q: Dividend yield
            is_call: Boolean array (True for calls)
            max_iter: Maximum Newton-Raphson iterations
            tol: Convergence tolerance
        
        Returns:
            Array of implied volatilities (np.nan where calculation fails)
        """
        n = len(prices)
        iv = np.full(n, np.nan)
        
        # Valid mask - need positive prices, T, S, K
        valid = (prices > 0) & (T > 0) & (S > 0) & (K > 0)
        
        if not valid.any():
            return iv
        
        # Extract valid data
        p_v = prices[valid]
        S_v = S[valid]
        K_v = K[valid]
        T_v = T[valid]
        is_call_v = is_call[valid]
        
        # Check for arbitrage violations (price below intrinsic)
        exp_qT = np.exp(-q * T_v)
        exp_rT = np.exp(-r * T_v)
        intrinsic = np.where(
            is_call_v,
            np.maximum(0, S_v * exp_qT - K_v * exp_rT),
            np.maximum(0, K_v * exp_rT - S_v * exp_qT)
        )
        price_valid = p_v >= intrinsic - 0.01
        
        if not price_valid.any():
            return iv
        
        # Work only with price-valid options
        idx_valid = np.where(valid)[0][price_valid]
        p_w = p_v[price_valid]
        S_w = S_v[price_valid]
        K_w = K_v[price_valid]
        T_w = T_v[price_valid]
        is_call_w = is_call_v[price_valid]
        exp_qT_w = exp_qT[price_valid]
        exp_rT_w = exp_rT[price_valid]
        
        # Initial guess using Brenner-Subrahmanyam approximation
        # sigma_0 ≈ sqrt(2*pi/T) * price / S
        sigma = np.sqrt(2 * np.pi / T_w) * p_w / S_w
        sigma = np.clip(sigma, 0.01, 3.0)  # Reasonable bounds
        
        sqrt_T = np.sqrt(T_w)
        
        # Newton-Raphson iteration
        for _ in range(max_iter):
            # Calculate d1, d2
            d1 = (np.log(S_w / K_w) + (r - q + 0.5 * sigma**2) * T_w) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            
            # BS price
            cdf_d1 = norm.cdf(d1)
            cdf_d2 = norm.cdf(d2)
            cdf_neg_d1 = norm.cdf(-d1)
            cdf_neg_d2 = norm.cdf(-d2)
            
            bs_call = S_w * exp_qT_w * cdf_d1 - K_w * exp_rT_w * cdf_d2
            bs_put = K_w * exp_rT_w * cdf_neg_d2 - S_w * exp_qT_w * cdf_neg_d1
            bs_price = np.where(is_call_w, bs_call, bs_put)
            
            # Vega (derivative w.r.t. sigma)
            pdf_d1 = norm.pdf(d1)
            vega = S_w * exp_qT_w * pdf_d1 * sqrt_T
            
            # Price difference
            diff = bs_price - p_w
            
            # Check convergence
            converged = np.abs(diff) < tol
            if converged.all():
                break
            
            # Newton-Raphson update (avoid division by zero)
            vega_safe = np.maximum(vega, 1e-10)
            sigma_new = sigma - diff / vega_safe
            
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma_new, 0.001, 5.0)
        
        # Store results
        # Mark as NaN if didn't converge well
        final_valid = np.abs(diff) < 0.01  # Allow small error
        iv[idx_valid[final_valid]] = sigma[final_valid]
        
        # Log convergence stats
        n_converged = final_valid.sum()
        n_attempted = len(sigma)
        n_failed = n_attempted - n_converged
        if n_failed > 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"IV convergence: {n_converged}/{n_attempted} ({100*n_converged/n_attempted:.1f}%), {n_failed} failed")
        
        return iv
    
    # =========================================================================
    # FIRST ORDER GREEKS
    # =========================================================================
    
    def delta(self, S: float, K: float, T: float, r: float, q: float,
              sigma: float, is_call: bool = True) -> float:
        """Delta: ∂V/∂S. Call: [0, 1], Put: [-1, 0]"""
        if T <= 0:
            if is_call:
                return 1.0 if S > K else (0.5 if S == K else 0.0)
            else:
                return -1.0 if S < K else (-0.5 if S == K else 0.0)
        if sigma <= 0:
            if is_call:
                return np.exp(-q*T) if S > K else 0.0
            else:
                return -np.exp(-q*T) if S < K else 0.0
        
        d1 = self._d1(S, K, T, r, q, sigma)
        if is_call:
            return np.exp(-q*T) * norm.cdf(d1)
        else:
            return np.exp(-q*T) * (norm.cdf(d1) - 1)
    
    def gamma(self, S: float, K: float, T: float, r: float, q: float,
              sigma: float) -> float:
        """Gamma: ∂²V/∂S². Vždy kladné."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, q, sigma)
        return np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def theta(self, S: float, K: float, T: float, r: float, q: float,
              sigma: float, is_call: bool = True) -> float:
        """Theta: ∂V/∂t (per kalendářní den). Typicky záporné."""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, q, sigma)
        d2 = d1 - sigma*np.sqrt(T)
        sqrt_T = np.sqrt(T)
        exp_qT = np.exp(-q*T)
        exp_rT = np.exp(-r*T)
        
        term1 = -S * exp_qT * norm.pdf(d1) * sigma / (2*sqrt_T)
        
        if is_call:
            term2 = q*S*exp_qT*norm.cdf(d1) - r*K*exp_rT*norm.cdf(d2)
        else:
            term2 = -q*S*exp_qT*norm.cdf(-d1) + r*K*exp_rT*norm.cdf(-d2)
        
        return (term1 + term2) / 365.0 * self.THETA_SCALE
    
    def vega(self, S: float, K: float, T: float, r: float, q: float,
             sigma: float) -> float:
        """Vega: ∂V/∂σ (per 1% změny IV). Vždy kladné."""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, q, sigma)
        return S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T) * 0.01
    
    def rho(self, S: float, K: float, T: float, r: float, q: float,
            sigma: float, is_call: bool = True) -> float:
        """Rho: ∂V/∂r (per 1% změny rate)."""
        if T <= 0:
            return 0.0
        if sigma <= 0:
            return K*T*np.exp(-r*T)*0.01 if is_call else -K*T*np.exp(-r*T)*0.01
        
        d2 = self._d2(S, K, T, r, q, sigma)
        if is_call:
            return K * T * np.exp(-r*T) * norm.cdf(d2) * 0.01
        else:
            return -K * T * np.exp(-r*T) * norm.cdf(-d2) * 0.01
    
    # =========================================================================
    # SECOND ORDER GREEKS
    # =========================================================================
    
    def vanna(self, S: float, K: float, T: float, r: float, q: float,
              sigma: float) -> float:
        """Vanna: ∂Delta/∂σ = ∂Vega/∂S (per 1% změny IV)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, q, sigma)
        d2 = d1 - sigma*np.sqrt(T)
        return -np.exp(-q*T) * norm.pdf(d1) * d2 / sigma * 0.01
    
    def charm(self, S: float, K: float, T: float, r: float, q: float,
              sigma: float, is_call: bool = True) -> float:
        """Charm: ∂Delta/∂t (per kalendářní den)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, q, sigma)
        d2 = d1 - sigma*np.sqrt(T)
        sqrt_T = np.sqrt(T)
        exp_qT = np.exp(-q*T)
        
        term1 = q*exp_qT*norm.cdf(d1) if is_call else -q*exp_qT*norm.cdf(-d1)
        term2 = exp_qT * norm.pdf(d1) * (2*(r-q)*T - d2*sigma*sqrt_T) / (2*T*sigma*sqrt_T)
        
        if is_call:
            return (term1 - term2) / 365.0
        else:
            return (term1 + term2) / 365.0
    
    def vomma(self, S: float, K: float, T: float, r: float, q: float,
              sigma: float) -> float:
        """Vomma: ∂Vega/∂σ (per 1% změny IV)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, q, sigma)
        d2 = d1 - sigma*np.sqrt(T)
        vega_raw = S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)
        return vega_raw * d1 * d2 / sigma * 0.01
    
    def veta(self, S: float, K: float, T: float, r: float, q: float,
             sigma: float) -> float:
        """Veta: ∂Vega/∂t (per kalendářní den)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, q, sigma)
        d2 = d1 - sigma*np.sqrt(T)
        sqrt_T = np.sqrt(T)
        vega_raw = S * np.exp(-q*T) * norm.pdf(d1) * sqrt_T
        
        term1 = q
        term2 = (r - q) * d1 / (sigma * sqrt_T)
        term3 = (1 + d1 * d2) / (2 * T)
        
        return vega_raw * (term1 + term2 - term3) / 365.0 * 0.01
    
    # =========================================================================
    # THIRD ORDER GREEKS
    # =========================================================================
    
    def speed(self, S: float, K: float, T: float, r: float, q: float,
              sigma: float) -> float:
        """Speed: ∂Gamma/∂S."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, q, sigma)
        gamma_val = self.gamma(S, K, T, r, q, sigma)
        return -gamma_val / S * (d1 / (sigma * np.sqrt(T)) + 1)
    
    def zomma(self, S: float, K: float, T: float, r: float, q: float,
              sigma: float) -> float:
        """Zomma: ∂Gamma/∂σ (per 1% změny IV)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, q, sigma)
        d2 = d1 - sigma*np.sqrt(T)
        gamma_val = self.gamma(S, K, T, r, q, sigma)
        return gamma_val * (d1 * d2 - 1) / sigma * 0.01
    
    def color(self, S: float, K: float, T: float, r: float, q: float,
              sigma: float) -> float:
        """Color: ∂Gamma/∂t (per kalendářní den)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, q, sigma)
        d2 = d1 - sigma*np.sqrt(T)
        sqrt_T = np.sqrt(T)
        exp_qT = np.exp(-q*T)
        
        term1 = 2*q*T + 1
        term2 = d1 * (2*(r-q)*T - d2*sigma*sqrt_T) / (sigma*sqrt_T)
        dgamma_dT = -exp_qT * norm.pdf(d1) / (2*S*T*sigma*sqrt_T) * (term1 + term2)
        
        return -dgamma_dT / 365.0
    
    def ultima(self, S: float, K: float, T: float, r: float, q: float,
               sigma: float) -> float:
        """Ultima: ∂Vomma/∂σ (per 1% změny IV)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, q, sigma)
        d2 = d1 - sigma*np.sqrt(T)
        vega_raw = S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)
        
        return -vega_raw / (sigma**2) * (d1*d2*(1 - d1*d2) + d1**2 + d2**2) * 0.0001
    
    # =========================================================================
    # COMBINED CALCULATIONS
    # =========================================================================
    
    def calculate_greeks_from_iv(self, S: float, K: float, T: float, r: float,
                                  q: float, iv: float, is_call: bool = True) -> Dict[str, float]:
        """Calculate all Greeks from given IV."""
        if np.isnan(iv) or iv <= 0:
            return {name: np.nan for name in [
                'delta', 'gamma', 'theta', 'vega', 'rho',
                'vanna', 'charm', 'vomma', 'veta',
                'speed', 'zomma', 'color', 'ultima'
            ]}
        
        return {
            'delta': self.delta(S, K, T, r, q, iv, is_call),
            'gamma': self.gamma(S, K, T, r, q, iv),
            'theta': self.theta(S, K, T, r, q, iv, is_call),
            'vega': self.vega(S, K, T, r, q, iv),
            'rho': self.rho(S, K, T, r, q, iv, is_call),
            'vanna': self.vanna(S, K, T, r, q, iv),
            'charm': self.charm(S, K, T, r, q, iv, is_call),
            'vomma': self.vomma(S, K, T, r, q, iv),
            'veta': self.veta(S, K, T, r, q, iv),
            'speed': self.speed(S, K, T, r, q, iv),
            'zomma': self.zomma(S, K, T, r, q, iv),
            'color': self.color(S, K, T, r, q, iv),
            'ultima': self.ultima(S, K, T, r, q, iv),
        }
    
    def calculate_greeks(self, S: float, K: float, option_price: float,
                         dte: int, is_call: bool, symbol: str,
                         trade_date: Optional[Union[str, date]] = None,
                         r: Optional[float] = None,
                         q: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate all Greeks for a single option.
        
        Args:
            S: Underlying price
            K: Strike price
            option_price: Option mid price
            dte: Days to expiration
            is_call: True for call
            symbol: Stock symbol (for rate lookup)
            trade_date: Trade date
            r: Override SOFR (None to fetch)
            q: Override dividend yield (None to fetch)
        
        Returns:
            Dict with iv, delta, gamma, theta, vega, rho, and higher order Greeks
        """
        T = dte / 365.0
        
        # Get rates
        if r is None or q is None:
            if self.rates is not None:
                rates = self.rates.get_rates(symbol, trade_date, S)
                r = r if r is not None else rates['r']
                q = q if q is not None else rates['q']
            else:
                r = r if r is not None else 0.05
                q = q if q is not None else 0.01
        
        # Calculate IV
        iv = self.calculate_iv(option_price, S, K, T, r, q, is_call)
        
        result = {'iv': iv, 'r': r, 'q': q}
        result.update(self.calculate_greeks_from_iv(S, K, T, r, q, iv, is_call))
        
        return result
    
    def calculate_greeks_vectorized(self, S: np.ndarray, K: np.ndarray,
                                     T: np.ndarray, r: float, q: float,
                                     sigma: np.ndarray, is_call: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Vectorized calculation of all Greeks.
        
        Args:
            S, K, T, sigma: Arrays of same length
            r, q: Scalars
            is_call: Boolean array
        
        Returns:
            Dict of arrays for each Greek
        """
        n = len(S)
        
        # Initialize output arrays
        delta = np.full(n, np.nan)
        gamma = np.full(n, np.nan)
        theta = np.full(n, np.nan)
        vega = np.full(n, np.nan)
        rho = np.full(n, np.nan)
        vanna = np.full(n, np.nan)
        charm = np.full(n, np.nan)
        vomma = np.full(n, np.nan)
        veta = np.full(n, np.nan)
        speed = np.full(n, np.nan)
        zomma = np.full(n, np.nan)
        color = np.full(n, np.nan)
        ultima = np.full(n, np.nan)
        
        # Valid mask
        valid = (T > 0) & (sigma > 0) & (S > 0) & (K > 0) & ~np.isnan(sigma)
        
        if valid.any():
            S_v, K_v, T_v, sigma_v = S[valid], K[valid], T[valid], sigma[valid]
            is_call_v = is_call[valid]
            
            sqrt_T = np.sqrt(T_v)
            d1 = (np.log(S_v/K_v) + (r - q + 0.5*sigma_v**2)*T_v) / (sigma_v*sqrt_T)
            d2 = d1 - sigma_v*sqrt_T
            
            exp_qT = np.exp(-q*T_v)
            exp_rT = np.exp(-r*T_v)
            pdf_d1 = norm.pdf(d1)
            cdf_d1, cdf_d2 = norm.cdf(d1), norm.cdf(d2)
            
            # === FIRST ORDER ===
            delta[valid] = np.where(is_call_v, exp_qT*cdf_d1, exp_qT*(cdf_d1-1))
            gamma[valid] = exp_qT*pdf_d1 / (S_v*sigma_v*sqrt_T)
            
            term1 = -S_v*exp_qT*pdf_d1*sigma_v / (2*sqrt_T)
            term2_c = q*S_v*exp_qT*cdf_d1 - r*K_v*exp_rT*cdf_d2
            term2_p = -q*S_v*exp_qT*norm.cdf(-d1) + r*K_v*exp_rT*norm.cdf(-d2)
            theta[valid] = (term1 + np.where(is_call_v, term2_c, term2_p)) / 365.0 * self.THETA_SCALE
            
            vega[valid] = S_v*exp_qT*pdf_d1*sqrt_T*0.01
            
            rho_c = K_v*T_v*exp_rT*cdf_d2*0.01
            rho_p = -K_v*T_v*exp_rT*norm.cdf(-d2)*0.01
            rho[valid] = np.where(is_call_v, rho_c, rho_p)
            
            # === SECOND ORDER ===
            vanna[valid] = -exp_qT * pdf_d1 * d2 / sigma_v * 0.01
            
            term1_ch = np.where(is_call_v, q*exp_qT*cdf_d1, -q*exp_qT*norm.cdf(-d1))
            term2_ch = exp_qT * pdf_d1 * (2*(r-q)*T_v - d2*sigma_v*sqrt_T) / (2*T_v*sigma_v*sqrt_T)
            charm[valid] = (np.where(is_call_v, term1_ch - term2_ch, term1_ch + term2_ch)) / 365.0
            
            vega_raw = S_v * exp_qT * pdf_d1 * sqrt_T
            vomma[valid] = vega_raw * d1 * d2 / sigma_v * 0.01
            
            term1_ve = q
            term2_ve = (r - q) * d1 / (sigma_v * sqrt_T)
            term3_ve = (1 + d1 * d2) / (2 * T_v)
            veta[valid] = vega_raw * (term1_ve + term2_ve - term3_ve) / 365.0 * 0.01
            
            # === THIRD ORDER ===
            gamma_v = gamma[valid]
            speed[valid] = -gamma_v / S_v * (d1 / (sigma_v * sqrt_T) + 1)
            zomma[valid] = gamma_v * (d1 * d2 - 1) / sigma_v * 0.01
            
            term1_co = 2*q*T_v + 1
            term2_co = d1 * (2*(r-q)*T_v - d2*sigma_v*sqrt_T) / (sigma_v*sqrt_T)
            dgamma_dT = -exp_qT * pdf_d1 / (2*S_v*T_v*sigma_v*sqrt_T) * (term1_co + term2_co)
            color[valid] = -dgamma_dT / 365.0
            
            ultima[valid] = -vega_raw / (sigma_v**2) * (d1*d2*(1 - d1*d2) + d1**2 + d2**2) * 0.0001
        
        return {
            'delta': delta, 'gamma': gamma, 'theta': theta,
            'vega': vega, 'rho': rho,
            'vanna': vanna, 'charm': charm, 'vomma': vomma, 'veta': veta,
            'speed': speed, 'zomma': zomma, 'color': color, 'ultima': ultima
        }
    
    def process_dataframe(self, df: pd.DataFrame, symbol: str,
                          trade_date: Optional[str] = None,
                          r: Optional[float] = None,
                          q: Optional[float] = None) -> pd.DataFrame:
        """
        Přidá všechny Greeks do DataFrame.
        
        Uses FAST vectorized IV calculation.
        
        Očekává sloupce:
            - underlying_price nebo S
            - strike nebo K  
            - dte nebo DTE
            - is_call nebo right/option_type
            - mid nebo option_price (nebo bid+ask)
        """
        df = df.copy()
        
        # Standardize column names
        col_map = {
            'underlying_price': 'S', 'stock_price': 'S',
            'strike': 'K', 'strike_price': 'K',
            'DTE': 'dte', 'days_to_expiry': 'dte',
            'mid': 'option_price', 'price': 'option_price'
        }
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
        
        # Calculate mid if needed
        if 'option_price' not in df.columns and 'bid' in df.columns and 'ask' in df.columns:
            df['option_price'] = (df['bid'] + df['ask']) / 2
        
        # Determine is_call
        if 'is_call' not in df.columns:
            if 'right' in df.columns:
                df['is_call'] = df['right'].str.upper().isin(['C', 'CALL'])
            elif 'option_type' in df.columns:
                df['is_call'] = df['option_type'].str.upper().isin(['C', 'CALL'])
        
        # Get rates
        if r is None or q is None:
            if self.rates is not None:
                S_sample = df['S'].iloc[0]
                rates = self.rates.get_rates(symbol, trade_date, S_sample)
                r = r if r is not None else rates['r']
                q = q if q is not None else rates['q']
            else:
                r = r if r is not None else 0.05
                q = q if q is not None else 0.01
        
        df['r'] = r
        df['q'] = q
        
        # FAST vectorized IV calculation
        T_arr = df['dte'].values / 365.0
        df['iv'] = self.calculate_iv_vectorized(
            prices=df['option_price'].values,
            S=df['S'].values,
            K=df['K'].values,
            T=T_arr,
            r=r, q=q,
            is_call=df['is_call'].values
        )
        
        # Vectorized Greeks calculation
        greeks = self.calculate_greeks_vectorized(
            S=df['S'].values,
            K=df['K'].values,
            T=T_arr,
            r=r, q=q,
            sigma=df['iv'].values,
            is_call=df['is_call'].values
        )
        
        # Add to DataFrame
        for name, values in greeks.items():
            df[name] = values
        
        return df
    
    # =========================================================================
    # ML FEATURE HELPERS
    # =========================================================================
    
    def get_normalized_greeks(self, S: float, K: float, T: float, r: float, 
                               q: float, iv: float, is_call: bool) -> Dict[str, float]:
        """
        Vrací normalizované Greeks vhodné přímo pro ML input.
        """
        greeks = self.calculate_greeks_from_iv(S, K, T, r, q, iv, is_call)
        
        return {
            'delta': greeks['delta'],
            'gamma_normalized': greeks['gamma'] * S,
            'theta_pct': greeks['theta'] / S * 100,
            'vega_pct': greeks['vega'] / S * 100,
            'vanna_normalized': greeks['vanna'] * S,
            'charm_pct': greeks['charm'] / S * 100,
            'vomma_normalized': greeks['vomma'],
            'moneyness': np.log(S / K),
            'time_to_expiry': T,
            'iv': iv,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_engine(cache_dir: str = "rates_cache") -> GreeksEngine:
    """Vytvoří engine s RatesProvider."""
    try:
        provider = RatesProvider(cache_dir=cache_dir)
        return GreeksEngine(provider)
    except:
        return GreeksEngine(None)


if __name__ == "__main__":
    import time
    
    engine = GreeksEngine()
    
    print("=== GREEKS ENGINE TEST ===\n")
    
    # Single option test
    S, K, T = 595.50, 600.0, 30/365
    r, q, sigma = 0.0433, 0.012, 0.19
    
    print(f"Single option: S={S}, K={K}, T={T:.4f}, σ={sigma}")
    greeks = engine.calculate_greeks_from_iv(S, K, T, r, q, sigma, is_call=True)
    print(f"  Delta: {greeks['delta']:.6f}")
    print(f"  Gamma: {greeks['gamma']:.6f}")
    print(f"  Theta: {greeks['theta']:.6f}")
    
    # Vectorized IV speed test
    print("\n=== VECTORIZED IV SPEED TEST ===")
    n = 100000
    
    np.random.seed(42)
    S_arr = np.random.uniform(400, 600, n)
    K_arr = np.random.uniform(380, 620, n)
    T_arr = np.random.uniform(0.01, 1.0, n)
    sigma_true = np.random.uniform(0.1, 0.5, n)
    is_call_arr = np.random.choice([True, False], n)
    
    # Generate prices from known IV
    prices = engine.bs_price_vectorized(S_arr, K_arr, T_arr, r, q, sigma_true, is_call_arr)
    
    # Time vectorized IV
    start = time.time()
    iv_calc = engine.calculate_iv_vectorized(prices, S_arr, K_arr, T_arr, r, q, is_call_arr)
    elapsed = time.time() - start
    
    # Check accuracy
    valid = ~np.isnan(iv_calc) & ~np.isnan(sigma_true)
    error = np.abs(iv_calc[valid] - sigma_true[valid])
    
    print(f"  {n:,} options in {elapsed:.2f}s ({n/elapsed:,.0f} opts/sec)")
    print(f"  IV recovery - Mean error: {error.mean():.6f}, Max: {error.max():.6f}")
    print(f"  Valid IVs: {valid.sum():,}/{n:,} ({100*valid.sum()/n:.1f}%)")