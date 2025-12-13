"""
Daily Options Screener

Selects top 50 stocks for options trading each morning based on:
1. Options liquidity (bid-ask spread, open interest)
2. Implied volatility (IV rank, IV percentile)
3. Volume (average daily volume)
4. No earnings within 7 days

Runs at market open (9:30 AM) and caches results for the day.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from dataclasses import dataclass
import asyncio

from loguru import logger

from analysis.earnings_checker import get_earnings_checker
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


# Universe of optionable stocks (high liquidity ETFs + popular stocks)
# This is the base universe to screen from
SCREENER_UNIVERSE = [
    # ETFs (always included - highest options liquidity)
    'SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI',
    'EEM', 'HYG', 'LQD', 'SLV', 'USO', 'EWZ', 'FXI', 'VXX', 'ARKK', 'XBI',
    
    # Mega caps (high options volume)
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'COIN',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'PYPL', 'SQ',
    
    # Tech
    'CRM', 'ADBE', 'INTC', 'MU', 'QCOM', 'AVGO', 'TXN', 'AMAT', 'LRCX', 'KLAC',
    'NOW', 'SNOW', 'PLTR', 'U', 'DDOG', 'NET', 'ZS', 'CRWD', 'PANW', 'OKTA',
    
    # Consumer
    'DIS', 'SBUX', 'NKE', 'MCD', 'WMT', 'COST', 'TGT', 'HD', 'LOW', 'LULU',
    
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'BMY', 'AMGN', 'GILD', 'MRNA',
    
    # Energy
    'XOM', 'CVX', 'OXY', 'SLB', 'HAL', 'DVN', 'EOG', 'PXD', 'MPC', 'VLO',
    
    # Other popular
    'BA', 'CAT', 'DE', 'UPS', 'FDX', 'GM', 'F', 'RIVN', 'LCID', 'NIO',
]


class DailyOptionsScreener:
    """
    Morning screener that selects top 50 stocks for options trading.
    
    Runs once at market open and caches results for the day.
    """
    
    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.vix_monitor = get_vix_monitor()
        self.earnings_checker = get_earnings_checker()
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
            self.data_fetcher = await get_data_fetcher()
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
        
        for symbol in SCREENER_UNIVERSE:
            try:
                score = await self._score_stock(symbol)
                if score and score.passed_earnings and score.passed_liquidity:
                    scores.append(score)
                    logger.debug(f"  ✅ {symbol}: {score.score:.2f}")
                else:
                    logger.debug(f"  ❌ {symbol}: {score.reason if score else 'Error'}")
            except Exception as e:
                logger.debug(f"  ❌ {symbol}: Error - {e}")
            
            # Small delay to avoid IBKR pacing
            await asyncio.sleep(0.1)
        
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
        for i, stock in enumerate(top_stocks[:10], 1):
            logger.info(f"  {i:2}. {stock.symbol:6} | Score: {stock.score:.1f} | IV: {stock.iv_rank:.0%}")
        if len(top_stocks) > 10:
            logger.info(f"  ... and {len(top_stocks) - 10} more")
        logger.info("=" * 60)
        
        return self._today_watchlist
    
    async def _score_stock(self, symbol: str) -> Optional[StockScore]:
        """
        Score a single stock for options trading.
        
        Scoring criteria:
        - IV rank (0-100): Higher = better for selling premium
        - Options volume: Higher = better liquidity
        - Spread: Lower = better execution
        - No earnings: Must not be in earnings blackout
        """
        try:
            # Check earnings blackout
            earnings = await self.earnings_checker.check_blackout(symbol)
            if earnings['in_blackout']:
                return StockScore(
                    symbol=symbol,
                    score=0,
                    iv_rank=0,
                    options_volume=0,
                    avg_spread_pct=100,
                    passed_earnings=False,
                    passed_liquidity=False,
                    reason=f"Earnings blackout: {earnings['reason']}"
                )
            
            # Get market data (simplified - would need real IBKR calls)
            # For now, use heuristics based on symbol type
            iv_rank = self._estimate_iv_rank(symbol)
            options_volume = self._estimate_options_volume(symbol)
            avg_spread_pct = self._estimate_spread(symbol)
            
            # Check liquidity threshold
            passed_liquidity = options_volume > 1000 and avg_spread_pct < 5.0
            
            # Calculate composite score
            # Weight: IV (40%) + Volume (40%) + Spread (20%)
            score = (
                iv_rank * 0.4 +
                min(options_volume / 100000, 1.0) * 40 +  # Cap at 100k volume
                max(0, (5 - avg_spread_pct) / 5) * 20  # Lower spread = higher score
            )
            
            return StockScore(
                symbol=symbol,
                score=score,
                iv_rank=iv_rank,
                options_volume=options_volume,
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
        """Estimate IV rank (0-100). Would use real data in production."""
        # ETFs typically have lower IV
        if symbol in ['SPY', 'QQQ', 'IWM']:
            return 45
        elif symbol in ['GLD', 'TLT', 'XLF']:
            return 35
        # Tech/growth has higher IV
        elif symbol in ['TSLA', 'NVDA', 'AMD', 'COIN']:
            return 65
        # Default
        return 50
    
    def _estimate_options_volume(self, symbol: str) -> int:
        """Estimate options volume. Would use real data in production."""
        # Top tier (millions)
        if symbol in ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD']:
            return 500000
        # Second tier
        elif symbol in ['IWM', 'META', 'AMZN', 'MSFT', 'GOOGL']:
            return 200000
        # Third tier
        elif symbol in SCREENER_UNIVERSE[:30]:
            return 50000
        # Default
        return 10000
    
    def _estimate_spread(self, symbol: str) -> float:
        """Estimate bid-ask spread %. Would use real data in production."""
        # ETFs have tightest spreads
        if symbol in ['SPY', 'QQQ', 'IWM']:
            return 0.5
        # Large caps
        elif symbol in SCREENER_UNIVERSE[:30]:
            return 1.5
        # Default
        return 3.0


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
        print("Daily Options Screener Test")
        print("=" * 60)
        
        screener = get_daily_screener()
        watchlist = await screener.run_morning_screen()
        
        print(f"\nToday's watchlist: {len(watchlist)} stocks")
        print(watchlist[:20])
    
    asyncio.run(main())
