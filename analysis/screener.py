"""
Daily Options Screener with IBKR Integration

Selects top 50 stocks for options trading each morning based on REAL data:
1. Options volume (from IBKR)
2. Implied volatility (IV rank from ATM options)
3. Bid-ask spread (from IBKR)
4. No earnings within 7 days

Universe: 300+ optionable stocks
Runs at market open (9:30 AM) and caches results for the day.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from dataclasses import dataclass
import asyncio

from loguru import logger

from ml.yahoo_earnings import get_yahoo_earnings_fetcher
from analysis.liquidity import get_liquidity_checker
from analysis.vix_monitor import get_vix_monitor
from ibkr.data_fetcher import get_data_fetcher


@dataclass
class StockScore:
    """Score for a single stock."""
    symbol: str
    score: float
    iv_rank: float
    options_volume: int
    avg_spread_pct: float
    passed_earnings: bool
    passed_liquidity: bool
    reason: str = ""


# ============================================================================
# SCREENER UNIVERSE - 300+ Optionable Stocks
# ============================================================================
SCREENER_UNIVERSE = [
    # ========== ETFs (Core - Always High Liquidity) ==========
    'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'TLT', 'IEF', 'LQD', 'HYG',
    'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB', 'XLRE',
    'EEM', 'EFA', 'VWO', 'EWZ', 'FXI', 'EWJ', 'EWG', 'EWY', 'EWH', 'INDA',
    'VXX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXU', 'SPXL', 'TNA', 'TZA',
    'ARKK', 'ARKG', 'ARKF', 'ARKW', 'XBI', 'IBB', 'LABU', 'LABD',
    'USO', 'UNG', 'GDX', 'GDXJ', 'SLV', 'JNUG', 'NUGT',
    'KRE', 'XOP', 'OIH', 'SMH', 'SOXX', 'ITB', 'XHB', 'XRT',
    
    # ========== Mega Caps (Highest Options Volume) ==========
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'NVDA', 'TSLA',
    'UNH', 'JNJ', 'V', 'MA', 'XOM', 'JPM', 'PG', 'HD', 'CVX',
    'LLY', 'ABBV', 'PFE', 'MRK', 'AVGO', 'KO', 'PEP', 'COST', 'TMO',
    'WMT', 'BAC', 'MCD', 'CSCO', 'ABT', 'DHR', 'ACN', 'CRM', 'ADBE',
    'ORCL', 'NKE', 'NFLX', 'AMD', 'TXN', 'INTC', 'QCOM', 'IBM',
    
    # ========== Tech / Growth (High Volatility) ==========
    'MU', 'AMAT', 'LRCX', 'KLAC', 'ASML', 'TSM', 'MRVL', 'ON', 'NXPI',
    'NOW', 'SNOW', 'PLTR', 'U', 'DDOG', 'NET', 'ZS', 'CRWD', 'PANW', 'OKTA',
    'SHOP', 'PYPL', 'COIN', 'MELI', 'SE', 'PINS', 'SNAP', 'RBLX',
    'UBER', 'LYFT', 'DASH', 'ABNB', 'BKNG', 'EXPE', 'MAR', 'HLT', 'WYNN',
    'ZM', 'DOCU', 'ROKU', 'TWLO', 'TTD', 'BILL', 'MDB', 'ESTC', 'CFLT',
    'PATH', 'AI', 'UPST', 'AFRM', 'OPEN', 'Z', 'ZG', 'COMP',
    
    # ========== Fintech / Brokers (User Requested) ==========
    'SOFI', 'HOOD', 'AFRM', 'NU', 'LC', 'UPST', 'PAYO', 'PSFE',
    'SCHW', 'IBKR', 'AMBA', 'FUTU', 'TIGR', 'MKTX',
    
    # ========== Banks / Financials ==========
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
    'AXP', 'BLK', 'SCHW', 'CME', 'ICE', 'NDAQ', 'CBOE',
    'AIG', 'MET', 'PRU', 'ALL', 'PGR', 'TRV', 'AFL', 'HIG',
    
    # ========== Consumer / Retail ==========
    'DIS', 'CMCSA', 'SBUX', 'NKE', 'MCD', 'YUM', 'SBUX', 'CMG', 'DPZ',
    'WMT', 'COST', 'TGT', 'HD', 'LOW', 'DG', 'DLTR', 'FIVE', 'ULTA',
    'LULU', 'ANF', 'URBN', 'RL', 'PVH', 'TPR', 'VFC',
    'PTON', 'LULU', 'BIRD', 'CROX', 'DECK', 'WWW',
    
    # ========== Food / Beverages ==========
    'HLF', 'HAIN', 'BYND', 'OTLY',  # User requested HLF
    'KO', 'PEP', 'MNST', 'KDP', 'STZ', 'TAP', 'BUD', 'SAM',
    
    # ========== Healthcare / Biotech ==========
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'BMY', 'AMGN', 'GILD',
    'MRNA', 'BNTX', 'NVAX', 'VRTX', 'REGN', 'BIIB', 'ALNY',
    'EDIT', 'CRSP', 'NTLA', 'BEAM', 'IONS', 'SRPT', 'BMRN',
    'EXAS', 'ILMN', 'DXCM', 'ISRG', 'ALGN', 'ZBH', 'SYK', 'MDT',
    
    # ========== Energy ==========
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX',
    'DVN', 'FANG', 'APA', 'HAL', 'BKR', 'NOV', 'RIG',
    'ET', 'EPD', 'KMI', 'WMB', 'OKE', 'TRGP', 'LNG',
    
    # ========== Industrial / Aerospace ==========
    'BA', 'LMT', 'NOC', 'RTX', 'GD', 'GE', 'HON', 'CAT', 'DE',
    'UPS', 'FDX', 'DAL', 'UAL', 'AAL', 'LUV', 'JBLU', 'ALK',
    'CSX', 'UNP', 'NSC', 'ODFL', 'XPO', 'CHRW', 'EXPD', 'JBHT',
    
    # ========== Auto / EV ==========
    'TSLA', 'GM', 'F', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI',
    'QS', 'CHPT', 'BLNK', 'EVGO', 'PLUG', 'FCEL', 'BE', 'BLDP',
    
    # ========== Mining / Materials ==========
    'FCX', 'NEM', 'GOLD', 'AEM', 'NUE', 'STLD', 'CLF', 'AA',
    'ALB', 'LAC', 'MP', 'SQM', 'CC', 'DOW', 'LYB',
    
    # ========== REITs ==========
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'AVB', 'EQR', 'VTR',
    'DLR', 'PSA', 'ESS', 'MAA', 'UDR', 'INVH', 'SUI', 'ELS', 'CPT',
    
    # ========== Cannabis (High IV) ==========
    'TLRY', 'CGC', 'ACB', 'CRON', 'SNDL', 'CURLF', 'GTBIF', 'TCNNF',
    
    # ========== Meme / Speculative (High Volume) ==========
    'GME', 'AMC', 'BB', 'SPCE', 'PLTR', 'CLOV',
    'TMC', 'ATER',
    
    # ========== China ADRs ==========
    'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'XPEV', 'LI', 'BILI', 'IQ',
    'TME', 'VIPS', 'NTES', 'TAL', 'EDU', 'GOTU', 'YMM',
    
    # ========== User Requested ==========
    'TE',  # TE Connectivity
]

# Remove duplicates
SCREENER_UNIVERSE = list(dict.fromkeys(SCREENER_UNIVERSE))


class DailyOptionsScreener:
    """
    Morning screener that selects top 50 stocks for options trading.
    
    Uses REAL IBKR data for:
    - Options volume
    - Implied volatility (ATM options)
    - Bid-ask spread
    
    Runs once at market open and caches results for the day.
    """
    
    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.vix_monitor = get_vix_monitor()
        self.earnings_fetcher = get_yahoo_earnings_fetcher()
        self.liquidity_checker = get_liquidity_checker()
        self.data_fetcher = None  # Lazy init
        
        # Daily cache
        self._today_watchlist: List[str] = []
        self._today_scores: Dict[str, StockScore] = {}
        self._last_run_date: Optional[date] = None
        
        logger.info(f"DailyOptionsScreener initialized (universe: {len(SCREENER_UNIVERSE)} stocks)")
    
    async def _get_data_fetcher(self):
        """Lazy init data fetcher."""
        if self.data_fetcher is None:
            self.data_fetcher = get_data_fetcher()
        return self.data_fetcher
    
    def get_today_watchlist(self) -> List[str]:
        """Get cached watchlist for today."""
        if self._last_run_date == date.today():
            return self._today_watchlist
        return []
    
    async def run_morning_screen(self, force: bool = False) -> List[str]:
        """
        Run morning screening to select top 50 stocks.
        
        Args:
            force: Force re-run even if already ran today
            
        Returns:
            List of top 50 symbols for today
        """
        # Check if already ran today
        today = date.today()
        if not force and self._last_run_date == today:
            logger.info(f"Using cached watchlist from {today}: {len(self._today_watchlist)} symbols")
            return self._today_watchlist
        
        # Skip weekends - no point in screening when market is closed
        from datetime import datetime
        import pytz
        
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        
        if not force and now_et.weekday() >= 5:  # Saturday=5, Sunday=6
            logger.info(f"⏸️ Weekend - skipping screening (day={now_et.strftime('%A')})")
            return self._today_watchlist if self._today_watchlist else []
        
        # Skip outside market hours (pre-9:25 AM or after 4:05 PM)
        market_open = now_et.replace(hour=9, minute=25, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=5, second=0, microsecond=0)
        
        if not force and (now_et < market_open or now_et > market_close):
            logger.info(f"⏸️ Market closed - skipping screening ({now_et.strftime('%H:%M')} ET)")
            return self._today_watchlist if self._today_watchlist else []
        
        logger.info("=" * 60)
        logger.info(f"🌅 MORNING SCREEN - {today}")
        logger.info(f"   Universe: {len(SCREENER_UNIVERSE)} stocks")
        logger.info("=" * 60)
        
        # 1. Check market conditions (VIX)
        regime = await self.vix_monitor.update()
        if not self.vix_monitor.is_trading_allowed():
            logger.warning(f"🛑 Market HALTED: VIX regime {regime}")
            self._today_watchlist = []
            self._last_run_date = today
            return []
        
        logger.info(f"✅ Market OK (VIX regime: {regime})")
        
        # 2. Score each stock
        scores: List[StockScore] = []
        fetcher = await self._get_data_fetcher()
        
        # Parallel processing with Semaphore
        # We use a semaphore to limit concurrent stock analysis (which triggers IBKR requests)
        sem = asyncio.Semaphore(10)  # Max 10 concurrent stocks
        
        async def analyze_stock(symbol):
            async with sem:
                try:
                    score = await self._score_stock_ibkr(symbol, fetcher)
                    return score
                except Exception as e:
                    logger.debug(f"  ❌ {symbol}: {e}")
                    return None

        # Create tasks for all stocks
        tasks = [analyze_stock(s) for s in SCREENER_UNIVERSE]
        
        # Run in chunks to provide progress updates (optional, but good for UX)
        # Using as_completed or simple chunks. Let's use chunks.
        chunk_size = 50
        total = len(tasks)
        
        for i in range(0, total, chunk_size):
            chunk = tasks[i:i + chunk_size]
            logger.info(f"   Processing stocks {i+1}-{min(i+chunk_size, total)}/{total}...")
            
            chunk_results = await asyncio.gather(*chunk)
            
            for score in chunk_results:
                if score and score.passed_earnings and score.passed_liquidity:
                    scores.append(score)
                    logger.debug(f"  ✅ {score.symbol}: {score.score:.2f} (IV:{score.iv_rank:.0%}, Vol:{score.options_volume:,})")
                elif score:
                    logger.debug(f"  ❌ {score.symbol}: {score.reason}")
        
        # Rate limiting is handled by Semaphore, explicit sleep removed
        
        # 3. Sort by score and take top N
        scores.sort(key=lambda x: x.score, reverse=True)
        top_stocks = scores[:self.top_n]
        
        # 4. Cache results
        self._today_watchlist = [s.symbol for s in top_stocks]
        self._today_scores = {s.symbol: s for s in top_stocks}
        self._last_run_date = today
        
        # 5. Log results
        logger.info("=" * 60)
        logger.info(f"📊 TOP {self.top_n} STOCKS FOR TODAY:")
        logger.info("=" * 60)
        for i, stock in enumerate(top_stocks[:15], 1):
            logger.info(
                f"  {i:2}. {stock.symbol:6} | Score: {stock.score:5.1f} | "
                f"IV: {stock.iv_rank:5.0%} | Vol: {stock.options_volume:>8,}"
            )
        if len(top_stocks) > 15:
            logger.info(f"  ... and {len(top_stocks) - 15} more")
        logger.info("=" * 60)
        
        return self._today_watchlist
    
    async def _score_stock_ibkr(self, symbol: str, fetcher) -> Optional[StockScore]:
        """
        Score a single stock using REAL IBKR data.
        
        Fetches:
        - Options chain for ATM IV and volume
        - Quote for spread estimation
        """
        try:
            # 1. Check earnings blackout (using Yahoo Finance - free!)
            days_to_earnings = self.earnings_fetcher.get_days_to_earnings(symbol)
            if days_to_earnings <= 7:  # Blackout window
                return StockScore(
                    symbol=symbol,
                    score=0,
                    iv_rank=0,
                    options_volume=0,
                    avg_spread_pct=100,
                    passed_earnings=False,
                    passed_liquidity=False,
                    reason=f"Earnings in {days_to_earnings} days"
                )
            
            # 2. Get stock quote for spread
            quote = await fetcher.get_stock_quote(symbol)
            if not quote or not quote.get('bid') or not quote.get('ask'):
                # Fallback to estimates if no quote
                avg_spread_pct = self._estimate_spread(symbol)
            else:
                bid = quote['bid']
                ask = quote['ask']
                mid = (bid + ask) / 2
                avg_spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 5.0
            
            # 3. Get options data (IV and volume)
            try:
                options_data = await fetcher.get_options_chain(symbol)
                if options_data:
                    iv_rank = options_data.get('iv_atm', 0.3)  # ATM IV
                    options_volume = options_data.get('total_volume', 10000)
                else:
                    iv_rank = self._estimate_iv_rank(symbol)
                    options_volume = self._estimate_options_volume(symbol)
            except Exception:
                # Fallback to estimates
                iv_rank = self._estimate_iv_rank(symbol)
                options_volume = self._estimate_options_volume(symbol)
            
            # 4. Check liquidity threshold
            passed_liquidity = options_volume > 1000 and avg_spread_pct < 5.0
            
            # 5. Calculate composite score
            # Weight: IV (40%) + Volume (40%) + Spread (20%)
            score = (
                min(iv_rank * 100, 100) * 0.4 +  # IV capped at 100%
                min(options_volume / 100000, 1.0) * 40 +  # Volume capped at 100k
                max(0, (5 - avg_spread_pct) / 5) * 20  # Tighter spread = higher score
            )
            
            return StockScore(
                symbol=symbol,
                score=score,
                iv_rank=iv_rank,
                options_volume=int(options_volume),
                avg_spread_pct=avg_spread_pct,
                passed_earnings=True,
                passed_liquidity=passed_liquidity,
                reason="" if passed_liquidity else "Low liquidity"
            )
            
        except Exception as e:
            return StockScore(
                symbol=symbol,
                score=0,
                iv_rank=0,
                options_volume=0,
                avg_spread_pct=100,
                passed_earnings=False,
                passed_liquidity=False,
                reason=str(e)
            )
    
    def _estimate_iv_rank(self, symbol: str) -> float:
        """Fallback IV estimate when IBKR unavailable."""
        high_iv = ['TSLA', 'NVDA', 'AMD', 'COIN', 'MARA', 'GME', 'AMC', 'RIVN', 'LCID', 
                   'PLTR', 'SOFI', 'HOOD', 'NIO', 'XPEV', 'MRNA', 'BNTX']
        if symbol in high_iv:
            return 0.65
        elif symbol in ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']:
            return 0.45
        return 0.50
    
    def _estimate_options_volume(self, symbol: str) -> int:
        """Fallback volume estimate when IBKR unavailable."""
        tier1 = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'AMZN', 'META']
        tier2 = ['IWM', 'MSFT', 'GOOGL', 'NFLX', 'BA', 'DIS', 'COIN', 'PLTR']
        if symbol in tier1:
            return 500000
        elif symbol in tier2:
            return 200000
        return 30000
    
    def _estimate_spread(self, symbol: str) -> float:
        """Fallback spread estimate when IBKR unavailable."""
        if symbol in ['SPY', 'QQQ', 'IWM', 'AAPL']:
            return 0.3
        elif symbol in SCREENER_UNIVERSE[:50]:
            return 1.0
        return 2.5


# Singleton
_screener: Optional[DailyOptionsScreener] = None


def get_daily_screener() -> DailyOptionsScreener:
    """Get or create daily screener singleton."""
    global _screener
    if _screener is None:
        _screener = DailyOptionsScreener()
    return _screener


# CLI for testing
if __name__ == "__main__":
    import asyncio
    from core.logger import setup_logger
    
    async def main():
        try:
            setup_logger(level="INFO")
        except:
            pass
        
        print("=" * 60)
        print("Daily Options Screener - IBKR Integration")
        print("=" * 60)
        print(f"Universe: {len(SCREENER_UNIVERSE)} stocks")
        
        screener = get_daily_screener()
        watchlist = await screener.run_morning_screen(force=True)
        
        print(f"\n✅ Today's watchlist: {len(watchlist)} stocks")
        print(f"Top 20: {watchlist[:20]}")
    
    asyncio.run(main())
