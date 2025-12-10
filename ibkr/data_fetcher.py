"""
Vanna IBKR Data Fetcher Module

Fetches market data, option chains, and Greeks from IBKR.
"""

import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from ib_insync import Stock, Option, Index, Contract, Ticker

from core.logger import get_logger
from ibkr.connection import get_ibkr_connection

logger = get_logger()


class IBKRDataFetcher:
    """
    IBKR Data Fetcher.
    
    Provides methods to fetch:
    - Stock quotes
    - Option chains
    - Greeks (Delta, Theta, Vega, Gamma)
    - VIX data
    - Historical data
    """
    
    def __init__(self):
        self._connection = None
    
    async def _get_connection(self):
        """Get IBKR connection (lazy initialization)."""
        if self._connection is None:
            self._connection = await get_ibkr_connection()
        return self._connection
    
    # =========================================================================
    # Stock Data
    # =========================================================================
    
    async def get_stock_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current stock quote.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
        
        Returns:
            Dict with bid, ask, last, volume, etc.
        """
        conn = await self._get_connection()
        
        if not conn.is_connected:
            logger.error("Not connected to IBKR")
            return None
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            conn.ib.qualifyContracts(contract)
            
            ticker = conn.ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(2)  # Wait for data
            
            quote = {
                "symbol": symbol,
                "bid": ticker.bid if ticker.bid > 0 else None,
                "ask": ticker.ask if ticker.ask > 0 else None,
                "last": ticker.last if ticker.last > 0 else None,
                "volume": ticker.volume if ticker.volume else 0,
                "high": ticker.high if ticker.high > 0 else None,
                "low": ticker.low if ticker.low > 0 else None,
                "close": ticker.close if ticker.close > 0 else None,
                "timestamp": datetime.now()
            }
            
            conn.ib.cancelMktData(contract)
            logger.info(f"📊 {symbol}: Last=${quote['last']}, Bid=${quote['bid']}, Ask=${quote['ask']}")
            
            return quote
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    # =========================================================================
    # VIX Data
    # =========================================================================
    
    async def get_vix(self) -> Optional[float]:
        """
        Get current VIX value.
        
        Returns:
            VIX value (e.g., 18.5)
        """
        conn = await self._get_connection()
        
        if not conn.is_connected:
            logger.error("Not connected to IBKR")
            return None
        
        try:
            vix = Index('VIX', 'CBOE')
            conn.ib.qualifyContracts(vix)
            
            ticker = conn.ib.reqMktData(vix, '', False, False)
            await asyncio.sleep(2)
            
            vix_value = ticker.last if ticker.last and ticker.last > 0 else ticker.close
            
            conn.ib.cancelMktData(vix)
            
            if vix_value and vix_value > 0:
                logger.info(f"📈 VIX: {vix_value:.2f}")
                return vix_value
            else:
                logger.warning("Could not fetch VIX value")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return None
    
    # =========================================================================
    # Option Chain
    # =========================================================================
    
    async def get_option_chain(
        self,
        symbol: str,
        expiry: str = None,
        strikes: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Get option chain for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            expiry: Expiration date YYYYMMDD (default: nearest monthly)
            strikes: Number of strikes above/below ATM
        
        Returns:
            Dict with calls and puts option data
        """
        conn = await self._get_connection()
        
        if not conn.is_connected:
            logger.error("Not connected to IBKR")
            return None
        
        try:
            # Get underlying stock
            stock = Stock(symbol, 'SMART', 'USD')
            conn.ib.qualifyContracts(stock)
            
            # Get current price
            ticker = conn.ib.reqMktData(stock, '', False, False)
            await asyncio.sleep(1)
            current_price = ticker.last or ticker.close
            conn.ib.cancelMktData(stock)
            
            if not current_price:
                logger.warning(f"Could not get price for {symbol}")
                return None
            
            # Get option chains
            chains = conn.ib.reqSecDefOptParams(
                stock.symbol, '', stock.secType, stock.conId
            )
            
            if not chains:
                logger.warning(f"No option chains for {symbol}")
                return None
            
            # Get nearest expiry if not specified
            chain = chains[0]  # SMART exchange
            
            if not expiry:
                available_expiries = sorted(chain.expirations)
                expiry = available_expiries[0] if available_expiries else None
            
            if not expiry:
                logger.warning(f"No expiries available for {symbol}")
                return None
            
            # Get strikes around ATM
            available_strikes = sorted([s for s in chain.strikes 
                                        if abs(s - current_price) < current_price * 0.2])
            
            result = {
                "symbol": symbol,
                "underlying_price": current_price,
                "expiry": expiry,
                "calls": [],
                "puts": []
            }
            
            # Limit strikes
            atm_idx = min(range(len(available_strikes)), 
                         key=lambda i: abs(available_strikes[i] - current_price))
            selected_strikes = available_strikes[max(0, atm_idx - strikes):atm_idx + strikes]
            
            logger.info(f"⚙️ Fetching option chain for {symbol} exp={expiry}, {len(selected_strikes)} strikes")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {e}")
            return None
    
    # =========================================================================
    # Greeks
    # =========================================================================
    
    async def get_option_greeks(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str  # 'C' or 'P'
    ) -> Optional[Dict[str, float]]:
        """
        Get Greeks for a specific option.
        
        Args:
            symbol: Underlying symbol
            expiry: Expiration YYYYMMDD
            strike: Strike price
            right: 'C' for call, 'P' for put
        
        Returns:
            Dict with delta, gamma, theta, vega, impliedVol
        """
        conn = await self._get_connection()
        
        if not conn.is_connected:
            logger.error("Not connected to IBKR")
            return None
        
        try:
            option = Option(symbol, expiry, strike, right, 'SMART')
            conn.ib.qualifyContracts(option)
            
            ticker = conn.ib.reqMktData(option, '', False, False)
            await asyncio.sleep(2)
            
            greeks = ticker.modelGreeks or ticker.lastGreeks
            
            if greeks:
                result = {
                    "delta": greeks.delta,
                    "gamma": greeks.gamma,
                    "theta": greeks.theta,
                    "vega": greeks.vega,
                    "impliedVol": greeks.impliedVol
                }
                logger.info(f"Greeks {symbol} {expiry} {strike}{right}: "
                           f"Δ={result['delta']:.3f}, Θ={result['theta']:.3f}, "
                           f"V={result['vega']:.3f}")
            else:
                result = None
                logger.warning(f"No Greeks available for {symbol} {strike}{right}")
            
            conn.ib.cancelMktData(option)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching Greeks: {e}")
            return None


# Singleton instance
_data_fetcher: Optional[IBKRDataFetcher] = None


def get_data_fetcher() -> IBKRDataFetcher:
    """Get the global data fetcher instance."""
    global _data_fetcher
    
    if _data_fetcher is None:
        _data_fetcher = IBKRDataFetcher()
    
    return _data_fetcher
