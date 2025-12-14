"""
Portfolio Risk Manager

Manages aggregate portfolio risk, focusing on Beta-Weighted Delta.
Ensures the portfolio is not overexposed to systematic market risk.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Optional, Any

from config import get_config
from ibkr.data_fetcher import get_data_fetcher

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
            # We need daily closes for symbol and SPY
            # Using data_fetcher to get history
            # Note: In production, this should likely query the DB or specialized data service
            # For now, we assume data_fetcher can get some history or we warn and use beta=1
            
            # Placeholder: In a real system, we'd fetch 90d daily bars here.
            # Currently fetch_historical_intraday is available in pipeline, 
            # but data_fetcher might be limited to live quotes or snapshots in current impl.
            
            # Fallback to Beta = 1 for stable assets if data missing, or 1.2 for tech
            # For MVP, let's try to infer from VIX or volatility relative to market if possible,
            # but simple hardcoded check or simple calculation is better.
            
            # Let's assume Beta = 1.0 for now if we can't calc, but log warning.
            # Real implementation would calculate Cov(Rm, Ri) / Var(Rm)
            
            beta = 1.0
            if symbol in ['NVDA', 'AMD', 'TSLA', 'COIN']:
                beta = 1.5
            elif symbol in ['GLD', 'TLT']:
                beta = 0.5
            
            self.beta_cache[symbol] = beta
            return beta
            
        except Exception as e:
            logger.error(f"Error calculating beta for {symbol}: {e}")
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
                 if hasattr(pos, 'modelGreeks') and pos.modelGreeks:
                     unit_delta = pos.modelGreeks.delta or 0.5 # Fallback 0.5 ATM
                 else:
                     logger.warning(f"⚠️ No greeks for {symbol} option. Assuming Delta=0.5")
                     unit_delta = 0.5
            
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
            
            # 3. Calculate Trade Impact
            # Assume 1 unit (or signal.quantity if available)
            # Default to 1 spread or 100 shares equivalent? 
            # Signal often doesn't have quantity yet (OrderManager decides).
            # Assume max size: e.g. 1 contract (~50 delta default for spreads? or 16 delta?)
            # Bull Put Spread 16 Delta Short / 5 Delta Long -> Net 11 Delta.
            # Let's be conservative: Net Delta = 0.20 per contract (credit spread)
            
            trade_qty = 1 # contract
            unit_delta = 0.20 if signal.action == "OPEN" else -0.20 # Direction?
            # OPEN BULL PUT = Positive Delta (Long Market)
            
            if "PUT" in str(signal.action) or "BULL" in str(signal.action): # Simplified
                 unit_delta = 0.20
            
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
