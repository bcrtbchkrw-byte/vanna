"""
Portfolio Risk Manager

Manages aggregate portfolio risk, focusing on Beta-Weighted Delta.
Ensures the portfolio is not overexposed to systematic market risk.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Optional, Any

from datetime import datetime

from config import get_config
from ibkr.data_fetcher import get_data_fetcher
from greeks.greeks_engine import GreeksEngine

class PortfolioManager:
    """
    Manages portfolio-level risk.
    
    Key Metrics:
    - Beta-Weighted Delta: SPY-equivalent exposure
    - Portfolio Leverage: Gross Notional / Net Liquidity
    """
    
    def __init__(self):
        self.config = get_config()
        self.data_fetcher = get_data_fetcher()
        self.greeks_engine = GreeksEngine()
        self.spy_history: Optional[pd.DataFrame] = None
        self.beta_cache: Dict[str, float] = {}
        
        # Risk Limits
        self.MAX_BETA_DELTA = self.config.portfolio.max_beta_weighted_delta
        self.MAX_LEVERAGE = self.config.portfolio.max_portfolio_leverage
        
        logger.info(f"🛡️ PortfolioManager initialized. Max Delta: +/-{self.MAX_BETA_DELTA}, Max Lev: {self.MAX_LEVERAGE}x")

    async def get_beta(self, symbol: str, lookback_days: int = 90) -> float:
        """
        Calculate Beta of a symbol vs SPY.
        Returns cached value if available and recent.
        """
        # TODO: Implement proper caching expiration
        if symbol in self.beta_cache:
            return self.beta_cache[symbol]
            
        try:
            # Check cache with simplistic expiration (e.g. check if calculated today?)
            # For now, just simplistic check
            
            # Fetch Daily History for Symbol and SPY
            # We use 6 months of daily bars for Beta
            logger.debug(f"Calculate Beta for {symbol}...")
            
            df_sym = await self.data_fetcher.get_historical_data(symbol, duration='6 M', bar_size='1 day')
            
            if self.spy_history is None:
                 self.spy_history = await self.data_fetcher.get_historical_data('SPY', duration='6 M', bar_size='1 day')
                 
            if df_sym is None or self.spy_history is None:
                logger.warning(f"Could not fetch history for Beta calc ({symbol}). Using default.")
                # Fallback logic
                beta = 1.0
                if symbol in ['NVDA', 'AMD', 'TSLA', 'COIN', 'NDX', 'QQQ']:
                    beta = 1.5
                elif symbol in ['GLD', 'TLT', 'XLP']:
                    beta = 0.5
                self.beta_cache[symbol] = beta
                return beta
            
            # Calculate Daily Returns
            # We align indices
            # df_sym['close'] vs self.spy_history['close']
            
            # Ensure indices are dates
            # (Already handled in get_historical_data)
            
            # Inner join on index
            df = pd.DataFrame({
                'sym': df_sym['close'],
                'spy': self.spy_history['close']
            }).dropna()
            
            if len(df) < 20: # Need at least a month or so
                 logger.warning(f"Insufficient overlap for Beta calc {symbol}")
                 return 1.0
                 
            # Percentage Changes
            returns = df.pct_change().dropna()
            
            # Covariance Matrix
            # [[Var(sym), Cov(sym,spy)], [Cov(spy,sym), Var(spy)]]
            cov_matrix = np.cov(returns['sym'], returns['spy'])
            
            cov_sym_spy = cov_matrix[0][1]
            var_spy = cov_matrix[1][1]
            
            beta = cov_sym_spy / var_spy
            
            logger.info(f"Computed Beta for {symbol}: {beta:.2f} (based on {len(returns)} days)")
            
            self.beta_cache[symbol] = beta
            return float(beta)
            
        except Exception as e:
            logger.error(f"Error calculating beta for {symbol}: {e}")
            return 1.0

    def _get_years_to_expiry(self, expiry_str: str) -> float:
        """Parse IBKR expiry 'YYYYMMDD' to years."""
        try:
            if not expiry_str or len(expiry_str) < 8:
                return 0.0
            
            # Handle YYYYMMDD
            exp_date = datetime.strptime(expiry_str[:8], "%Y%m%d")
            now = datetime.now()
            
            delta = exp_date - now
            return max(delta.days / 365.0, 0.001) # Min 1 day
        except Exception:
            return 0.0

    def _calculate_manual_delta(self, pos: Any, under_price: float) -> float:
        """Calculate Delta manually using VannaCalculator if IBKR data missing."""
        try:
            contract = pos.contract
            
            # Extract inputs
            K = contract.strike
            T = self._get_years_to_expiry(contract.lastTradeDateOrContractMonth)
            right = contract.right # 'C' or 'P'
            option_type = 'call' if right == 'C' else 'put'
            
            # IV Estimation
            # If we don't have IV, we are guessing.
            # But we can assume a conservative Vol for risk (e.g. 50% or higher).
            # Or better, we define S, K, T. Delta isn't linear with Vol, but reasonable.
            # Ideally we use an IV from a similar option or VIX index mapping.
            # For now, using 0.4 (40%) as a generic baseline for risk estimation 
            # if we absolutely have no data.
            sigma = 0.4 
            
            # Calculate delta using GreeksEngine
            is_call = option_type == 'call'
            delta = self.greeks_engine.delta(
                S=under_price,
                K=K,
                T=T,
                r=0.05,  # Default risk-free rate
                q=0.0,   # Default dividend yield
                sigma=sigma,
                is_call=is_call
            )
            
            return delta
                
            return 1.0 # Fail safe (worst case for stock/ITM) or 0.0?
            
        except Exception as e:
            logger.error(f"Manual delta calc failed for {pos.contract.symbol}: {e}")
            return 1.0

    async def calculate_portfolio_risk(self, positions: List[Any], current_prices: Dict[str, float], net_liq: float) -> Dict[str, float]:
        """
        Calculate aggregate portfolio risk metrics.
        
        Args:
            positions: List of IBKR Position objects
            current_prices: Dict of {symbol: price}
            net_liq: Account Net Liquidation Value
            
        Returns:
            Dict with 'beta_weighted_delta', 'leverage', etc.
        """
        total_beta_weighted_delta = 0.0
        total_gross_notional = 0.0
        
        spy_price = current_prices.get('SPY', 400.0) # Fallback
        
        for pos in positions:
            symbol = pos.contract.symbol
            qty = pos.position
            price = current_prices.get(symbol, pos.avgCost)
            
            # 1. Get Delta (Unit Delta)
            # For stocks, delta = 1. 
            # For options, we need the delta from the contract (pos.modelGreeks.delta) if avlbl
            # IBKR provides modelGreeks if requested.
            
            unit_delta = 1.0
            if pos.contract.secType == 'OPT':
                # Use provided greeks or approximate
                # If modelGreeks is None, we are blind. 
                # Ideally we ask IBKR or compute it.
                # For MVP, we assume delta based on moneyness if missing, but better to skip or warn.
                if hasattr(pos, 'modelGreeks') and pos.modelGreeks and pos.modelGreeks.delta is not None:
                     unit_delta = pos.modelGreeks.delta
                else:
                     logger.warning(f"⚠️ No greeks for {symbol} option. Calculating manually...")
                     unit_delta = self._calculate_manual_delta(pos, price)
            
            # 2. Get Beta
            beta = await self.get_beta(symbol)
            
            # 3. Calculate Beta-Weighted Delta
            # BWD = (UnitDelta * Qty * Price * Beta) / SPY_Price
            position_delta_dollars = unit_delta * qty * price
            position_bwd = (position_delta_dollars * beta) / spy_price
            
            total_beta_weighted_delta += position_bwd
            total_gross_notional += abs(qty * price)
            
        leverage = total_gross_notional / net_liq if net_liq > 0 else 0
        
        return {
            'beta_weighted_delta': total_beta_weighted_delta,
            'leverage': leverage,
            'spy_price': spy_price
        }

    async def check_trade(self, signal: Any, current_positions: List[Any], net_liq: float) -> Dict[str, Any]:
        """
        Validate if a new trade adheres to portfolio risk limits.
        
        Args:
            signal: TradeSignal (symbol, action, etc.)
            current_positions: List of current IBKR positions
            net_liq: Account Net Liq
            
        Returns:
            {'allowed': bool, 'reason': str}
        """
        try:
            # 1. Get Market Data (needed for calc)
            # We need prices for all positions + new symbol
            # This is expensive if we do it sequentially.
            # In a real system we pass the market_data snapshot.
            
            # For now, simplistic approximation:
            # We primarily need SPY price and Symbol price.
            # Other positions prices we can use avgCost or assume close to last known?
            # Better to fetch them or assume caller provides snapshot.
            # To keep it simple, I will rely on data_fetcher for the specific symbol
            # and use avgCost for others if live price not cheap.
            
            quote = await self.data_fetcher.get_stock_quote(signal.symbol)
            price = quote.get('last', 0)
            spy_quote = await self.data_fetcher.get_stock_quote('SPY')
            spy_price = spy_quote.get('last', 400.0)
            
            prices = {signal.symbol: price, 'SPY': spy_price}
            
            # 2. Calculate Current Risk
            current_risk = await self.calculate_portfolio_risk(current_positions, prices, net_liq)
            current_bwd = current_risk['beta_weighted_delta']
            
            # 3. Calculate Trade Impact (Dynamic Greeks)
            # Instead of hardcoding 0.20, we estimate the delta of the intended strategy.
            # Assumption: Standard entry is Short Put / Bull Put Spread approx 5% OTM, 30 DTE.
            # This makes risk check sensitive to Volatility (Vanna).
            
            trade_qty = 1 # contract
            
            # Determine params for theoretical contract
            t_days = 30
            T = t_days / 365.0
            
            # Estimate IV from VIX if available, else 0.4
            vix = await self.data_fetcher.get_vix()
            sigma = (vix / 100.0) if vix else 0.4
            
            # Determine Strike (5% OTM for Puts)
            if "PUT" in str(signal.action) or "BULL" in str(signal.action) or signal.action == "OPEN":
                 # Bullish: Short Put at 0.95 * Price
                 K = price * 0.95
                 option_type = 'put'
                 direction = 1.0 # Long delta (Short Put is bullish/positive delta)
            else:
                 # Bearish: Long Put or Short Call? 
                 # Assuming Long Put for simplicity or Short Call OTM
                 K = price * 1.05
                 option_type = 'call' 
                 direction = -1.0 # Short delta
            
            # Calculate REAL delta for this theoretical contract
            is_call = option_type == 'call'
            raw_delta = self.greeks_engine.delta(
                S=price, K=K, T=T, r=0.05, q=0.0, sigma=sigma, is_call=is_call
            )
            
            if raw_delta is not None:
                # If we are selling the option (Credits), we flip the sign of the option delta.
                # Put Delta is negative. Short Put Delta is Positive.
                # Call Delta is positive. Short Call Delta is Negative.
                
                # If Bullish (Short Put)
                if direction > 0:
                     # Put delta is neg. We want pos.
                     unit_delta = abs(raw_delta) 
                else:
                     # Bearish (Long Put)
                     unit_delta = -abs(raw_delta)
                     
                logger.debug(f"Estimated Trade Delta (S={price:.2f}, K={K:.2f}, Vol={sigma:.2f}): {unit_delta:.3f}")
            else:
                unit_delta = 0.20 * direction
            
            beta = await self.get_beta(signal.symbol)
            
            trade_bwd = (unit_delta * 100 * trade_qty * price * beta) / spy_price
            
            new_bwd = current_bwd + trade_bwd
            
            logger.info(f"📊 Portfolio Risk Check: Current Delta={current_bwd:.1f}, Trade Impact={trade_bwd:.1f}, New={new_bwd:.1f}")
            
            # 4. Validate
            if abs(new_bwd) > self.MAX_BETA_DELTA:
                 return {
                     'allowed': False,
                     'reason': f"Portfolio Delta Limit: {new_bwd:.1f} exceeds +/-{self.MAX_BETA_DELTA}"
                 }
            
            return {'allowed': True, 'reason': "OK"}
            
        except Exception as e:
            logger.error(f"Portfolio check failed: {e}")
            return {'allowed': False, 'reason': f"Error: {e}"}

# Singleton
_portfolio_manager = None

def get_portfolio_manager() -> PortfolioManager:
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = PortfolioManager()
    return _portfolio_manager
