"""
Vanna IBKR Data Fetcher Module

Fetches market data, option chains, and Greeks from IBKR.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

from ib_insync import Contract, Index, Stock

from core.logger import get_logger
from core.exceptions import DataError, IBKRDisconnectedError
from ibkr.connection import get_ibkr_connection

import pandas as pd

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
        # Load timeout from config
        from config import get_config
        self._data_timeout = get_config().ibkr.data_timeout
    
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
            await conn.ib.qualifyContractsAsync(contract)
            
            ticker = conn.ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(self._data_timeout)  # Configurable timeout
            
            # Fallback to close if last is missing (common on weekends/ah)
            last_price = ticker.last if (ticker.last and ticker.last > 0) else ticker.close
            
            quote = {
                "symbol": symbol,
                "bid": ticker.bid if (ticker.bid and ticker.bid > 0) else None,
                "ask": ticker.ask if (ticker.ask and ticker.ask > 0) else None,
                "last": last_price if (last_price and last_price > 0) else None,
                "volume": ticker.volume if ticker.volume else 0,
                "high": ticker.high if (ticker.high and ticker.high > 0) else None,
                "low": ticker.low if (ticker.low and ticker.low > 0) else None,
                "close": ticker.close if (ticker.close and ticker.close > 0) else None,
                "timestamp": datetime.now()
            }
            
            conn.ib.cancelMktData(contract)
            
            # Log with context
            price_src = "Last" if ticker.last and ticker.last > 0 else "Close"
            logger.info(f"📊 {symbol}: {price_src}=${quote['last'] or '?'}, Bid=${quote['bid'] or '?'}, Ask=${quote['ask'] or '?'}")
            
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
            await conn.ib.qualifyContractsAsync(vix)
            
            ticker = conn.ib.reqMktData(vix, '', False, False)
            await asyncio.sleep(self._data_timeout)  # Configurable timeout
            
            vix_value = ticker.last if ticker.last and ticker.last > 0 else ticker.close
            
            conn.ib.cancelMktData(vix)
            
            if vix_value and vix_value > 0:
                logger.info(f"📈 VIX: {vix_value:.2f}")
                return float(vix_value) if vix_value else None
            else:
                logger.warning("Could not fetch VIX value")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return None
    
    async def get_vix3m(self) -> Optional[float]:
        """
        Get current VIX3M (3-month VIX) value.
        
        Used for VIX term structure analysis (contango/backwardation).
        
        Returns:
            VIX3M value (e.g., 20.5)
        """
        conn = await self._get_connection()
        
        if not conn.is_connected:
            logger.error("Not connected to IBKR")
            return None
        
        try:
            vix3m = Index('VIX3M', 'CBOE')
            await conn.ib.qualifyContractsAsync(vix3m)
            
            ticker = conn.ib.reqMktData(vix3m, '', False, False)
            await asyncio.sleep(self._data_timeout)
            
            vix3m_value = ticker.last if ticker.last and ticker.last > 0 else ticker.close
            
            conn.ib.cancelMktData(vix3m)
            
            if vix3m_value and vix3m_value > 0:
                logger.info(f"📈 VIX3M: {vix3m_value:.2f}")
                return float(vix3m_value)
            else:
                logger.warning("Could not fetch VIX3M value")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching VIX3M: {e}")
            return None
    
    async def get_options_market_data(self, symbol: str) -> Dict[str, float]:
        """
        Get aggregate options market data for a symbol.
        
        Used for live feature construction (put/call ratio, ATM IV, volume).
        
        Returns:
            Dict with:
            - iv_atm: ATM implied volatility
            - put_call_ratio: Put volume / Call volume
            - volume_norm: Normalized options volume (z-score)
            - total_volume: Raw total options volume
        """
        defaults = {
            'iv_atm': 0.2,
            'put_call_ratio': 0.8,
            'volume_norm': 0.5,
            'total_volume': 0,
        }
        
        try:
            # Get option chain for ATM strike
            chain = await self.get_option_chain(symbol, strikes=3)
            
            if not chain or (not chain.get('calls') and not chain.get('puts')):
                return defaults
            
            # Extract ATM IV (closest to underlying price)
            underlying_price = chain.get('underlying_price', 0)
            
            atm_iv = 0.2
            if chain['calls']:
                # Find ATM call
                atm_call = min(chain['calls'], 
                              key=lambda x: abs(x['strike'] - underlying_price))
                if atm_call.get('iv') and atm_call['iv'] > 0:
                    atm_iv = atm_call['iv']
            
            # Calculate put/call volume ratio
            total_call_volume = sum(c.get('volume', 0) for c in chain['calls'])
            total_put_volume = sum(p.get('volume', 0) for p in chain['puts'])
            total_volume = total_call_volume + total_put_volume
            
            if total_call_volume > 0:
                put_call_ratio = total_put_volume / total_call_volume
            else:
                put_call_ratio = 1.0  # Default if no call volume
            
            # Normalize volume (rough z-score based on typical SPY volume)
            # SPY typically has ~2M options contracts daily
            typical_volume = 50000  # Per expiry cycle
            volume_norm = min(2.0, total_volume / typical_volume) - 1  # Center around 0
            
            result = {
                'iv_atm': atm_iv,
                'put_call_ratio': min(put_call_ratio, 3.0),  # Cap at 3.0
                'volume_norm': volume_norm,
                'total_volume': total_volume,
            }
            
            logger.info(f"📊 {symbol} Options: IV={atm_iv:.1%}, P/C={put_call_ratio:.2f}, Vol={total_volume:,}")
            return result
            
        except Exception as e:
            logger.warning(f"Error fetching options market data for {symbol}: {e}")
            return defaults
    
    # =========================================================================
    # Option Chain
    # =========================================================================
    
    async def get_option_chain(
        self,
        symbol: str,
        expiry: Optional[str] = None,
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
            await conn.ib.qualifyContractsAsync(stock)
            
            # Get current price
            ticker = conn.ib.reqMktData(stock, '', False, False)
            await asyncio.sleep(self._data_timeout)  # Configurable timeout
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
            
            # Fetch option data for each strike
            from ib_insync import Option
            
            for strike in selected_strikes:
                for right in ['C', 'P']:
                    try:
                        opt_contract = Option(
                            symbol=symbol,
                            lastTradeDateOrContractMonth=expiry,
                            strike=strike,
                            right=right,
                            exchange='SMART',
                            currency='USD'
                        )
                        
                        # Qualify and get data
                        await conn.ib.qualifyContractsAsync(opt_contract)
                        ticker = conn.ib.reqMktData(opt_contract, '101,106', False, False)
                        await asyncio.sleep(0.5)  # Brief wait for each option
                        
                        greeks = ticker.modelGreeks or ticker.lastGreeks
                        
                        option_data = {
                            "strike": strike,
                            "right": right,
                            "bid": ticker.bid if ticker.bid and ticker.bid > 0 else 0,
                            "ask": ticker.ask if ticker.ask and ticker.ask > 0 else 0,
                            "last": ticker.last if ticker.last and ticker.last > 0 else 0,
                            "volume": ticker.volume if ticker.volume else 0,
                            "delta": greeks.delta if greeks else 0,
                            "gamma": greeks.gamma if greeks else 0,
                            "theta": greeks.theta if greeks else 0,
                            "vega": greeks.vega if greeks else 0,
                            "iv": greeks.impliedVol if greeks else 0
                        }
                        
                        conn.ib.cancelMktData(opt_contract)
                        
                        if right == 'C':
                            result["calls"].append(option_data)
                        else:
                            result["puts"].append(option_data)
                            
                    except Exception as opt_e:
                        logger.debug(f"Could not fetch {symbol} {strike}{right}: {opt_e}")
                        continue
            
            logger.info(f"✅ Option chain: {len(result['calls'])} calls, {len(result['puts'])} puts")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {e}")
            return None
    
    # =========================================================================
    # Greeks
    # =========================================================================
    
    async def get_option_greeks(
        self,
        contract: Contract
    ) -> Optional[Dict[str, float]]:
        """
        Get Greeks for a specific option.
        
        Args:
            contract: Option contract
        
        Returns:
            Dict with delta, gamma, theta, vega, impliedVol
        """
        conn = await self._get_connection()
        
        if not conn.is_connected:
            logger.error("Not connected to IBKR")
            return None
        
        try:
            await conn.ib.qualifyContractsAsync(contract)
            
            # 106 = Option Implied Vol, 101 = Open Interest
            ticker = conn.ib.reqMktData(contract, '101,106', False, False)
            await asyncio.sleep(self._data_timeout + 1)  # Greeks need slightly more time
            
            greeks = ticker.modelGreeks or ticker.lastGreeks
            
            # Extract Open Interest safely
            # ib_insync populates callOpenInterest/putOpenInterest based on the contract right? 
            # Or simplified: try to get any non-zero, non-None OI value
            raw_oi = 0
            if ticker.callOpenInterest and ticker.callOpenInterest > 0:
                raw_oi = ticker.callOpenInterest
            elif ticker.putOpenInterest and ticker.putOpenInterest > 0:
                raw_oi = ticker.putOpenInterest
            
            # Handle potential nan (e.g. from nan + number)
            import math
            if hasattr(raw_oi, 'real') and math.isnan(raw_oi):
                 raw_oi = 0

            if greeks:
                result = {
                    "delta": greeks.delta,
                    "gamma": greeks.gamma,
                    "theta": greeks.theta,
                    "vega": greeks.vega,
                    "impliedVol": greeks.impliedVol,
                    
                    # Also include price data for liquidity check
                    "bid": ticker.bid,
                    "ask": ticker.ask,
                    "volume": ticker.volume if ticker.volume else 0,
                    "open_interest": int(raw_oi)
                }
                
                logger.info(f"Greeks {contract.localSymbol}: "
                           f"Δ={result['delta']:.3f}, Θ={result['theta']:.3f}, "
                           f"V={result['vega']:.3f}")
            else:
                # Even if no Greeks, return price data if available (for liquidity check)
                result = {
                    "delta": 0, "gamma": 0, "theta": 0, "vega": 0, "impliedVol": 0,
                    "bid": ticker.bid,
                    "ask": ticker.ask,
                    "volume": ticker.volume if ticker.volume else 0,
                    "open_interest": int(raw_oi)
                }
                logger.warning(f"No Greeks available for {contract.localSymbol}, returning price data")
            
            conn.ib.cancelMktData(contract)
            return result

            
        except Exception as e:
            logger.error(f"Error fetching Greeks: {e}")
            return None


    
    # =========================================================================
    # Fundamental Data
    # =========================================================================
    
    async def get_earnings_date(self, symbol: str) -> Optional[datetime]:
        """
        Get next earnings date from IBKR fundamental data
        
        Uses IBKR's CalendarReport with rate limiting to avoid pacing violations.
        IBKR limit: ~60 fundamental data requests per 10 minutes.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Next earnings datetime or None
        """
        try:
            conn = await self._get_connection()
            
            if not conn.is_connected:
                logger.error("Not connected to IBKR")
                return None
            
            # Create stock contract
            stock = Stock(symbol, 'SMART', 'USD')
            await conn.ib.qualifyContractsAsync(stock)
            
            # Request fundamental data with retry logic for pacing violations
            logger.debug(f"Fetching earnings calendar for {symbol} from IBKR...")
            
            max_retries = 3
            retry_delay = 5  # Start with 5 seconds
            
            calendar_xml = None
            
            for attempt in range(max_retries):
                try:
                    # Add small delay to avoid pacing violations (error 162)
                    if attempt > 0:
                        logger.info(f"Retry {attempt}/{max_retries} for {symbol} after {retry_delay}s")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    
                    calendar_xml = await conn.ib.reqFundamentalDataAsync(
                        stock,
                        'CalendarReport'  # Contains earnings dates
                    )
                    
                    # Success - break retry loop
                    break
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # Check for pacing violation (error 162)
                    if '162' in error_msg or 'pacing' in error_msg.lower():
                        logger.warning(
                            f"IBKR pacing violation for {symbol} (attempt {attempt+1}/{max_retries}). "
                            f"Waiting {retry_delay}s..."
                        )
                        if attempt == max_retries - 1:
                            logger.error(f"Failed to get earnings for {symbol} after {max_retries} retries")
                            return None
                        continue
                    else:
                        # Different error - log and return None (don't crash app)
                        logger.warning(f"Error fetching fundamental data: {e}")
                        return None
            
            if not calendar_xml:
                logger.debug(f"No calendar data for {symbol}")
                return None
            
            # Parse XML to get earnings date
            from xml.etree import ElementTree as ET
            
            root = ET.fromstring(calendar_xml)
            
            # Look for earnings announcement date
            # XML structure: <CalendarReport><EarningsDate>...</EarningsDate></CalendarReport>
            earnings_elements = root.findall('.//EarningsDate')
            
            if not earnings_elements:
                # Try alternative path
                earnings_elements = root.findall('.//Event[@Type="Earnings"]')
            
            if earnings_elements:
                # Get the first (next) earnings date
                earnings_date_str = earnings_elements[0].text
                
                if earnings_date_str:
                    # Parse date (format varies, try common formats)
                    for fmt in ['%Y-%m-%d', '%Y%m%d', '%m/%d/%Y']:
                        try:
                            earnings_date = datetime.strptime(earnings_date_str.strip(), fmt)
                            # Logic to ensure we don't return past earnings
                            # But usually CalendarReport returns upcoming events or recent past
                            logger.info(f"📅 {symbol} next earnings: {earnings_date.strftime('%Y-%m-%d')}")
                            return earnings_date
                        except ValueError:
                            continue
            
            logger.debug(f"No earnings date found in calendar for {symbol}")
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching earnings from IBKR for {symbol}: {e}")
            return None



    
    # =========================================================================
    # Historical Data
    # =========================================================================

    async def get_historical_data(
        self, 
        symbol: str, 
        duration: str = '6 M', 
        bar_size: str = '1 day'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a symbol.
        
        Args:
            symbol: Stock symbol
            duration: Duration to fetch (e.g. '6 M', '1 Y')
            bar_size: Bar size (e.g. '1 day', '1 hour')
            
        Returns:
            DataFrame with Index=Date, Columns=[open, high, low, close, volume]
        """
        conn = await self._get_connection()
        if not conn.is_connected:
            logger.error("Not connected to IBKR")
            return None
            
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            await conn.ib.qualifyContractsAsync(contract)
            
            # Request historical data
            bars = await conn.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if not bars:
                logger.warning(f"No historical data for {symbol}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Error fetching history for {symbol}: {e}")
            return None


# Singleton instance
_data_fetcher: Optional[IBKRDataFetcher] = None


def get_data_fetcher() -> IBKRDataFetcher:
    """Get the global data fetcher instance."""
    global _data_fetcher
    
    if _data_fetcher is None:
        _data_fetcher = IBKRDataFetcher()
    return _data_fetcher

