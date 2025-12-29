#!/usr/bin/env python3
"""
ML-Enhanced Daily Options Screener

REPLACES: screener.py

Combines traditional screening with ML predictions for 80%+ win rate targeting.

Flow:
1. Universe Filter (300+ stocks)
2. Liquidity Check (volume, spread)
3. Earnings Blackout (7 days)
4. ML Scoring:
   - XGBoost win probability
   - Regime classification
   - Strategy recommendation
   - DTE bucket assignment
5. Final Ranking → Top 50 with strategies

Output per symbol:
- score: Combined ML + traditional score
- strategy: Recommended strategy (IRON_CONDOR, PUT_CREDIT_SPREAD, etc.)
- dte_bucket: Recommended DTE (0=0DTE, 1=WEEKLY, 2=MONTHLY, 3=LEAPS)
- win_probability: XGBoost prediction
- regime: Market regime (CALM, NORMAL, ELEVATED, HIGH_VOL, CRISIS)

Usage:
    from ml_enhanced_screener import get_screener
    
    screener = get_screener()
    watchlist = await screener.run_morning_screen()
    
    for stock in watchlist:
        print(f"{stock.symbol}: {stock.strategy} ({stock.win_probability:.0%})")
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Try to import project dependencies, provide fallbacks
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    import pytz
    ET = pytz.timezone('US/Eastern')
except ImportError:
    ET = None


# =============================================================================
# ENUMS
# =============================================================================

class MarketRegime(Enum):
    CALM = 0        # VIX < 15
    NORMAL = 1      # VIX 15-20
    ELEVATED = 2    # VIX 20-25
    HIGH_VOL = 3    # VIX 25-35
    CRISIS = 4      # VIX > 35


class Strategy(Enum):
    CASH = "CASH"
    IRON_CONDOR = "IRON_CONDOR"
    PUT_CREDIT_SPREAD = "PUT_CREDIT_SPREAD"
    CALL_CREDIT_SPREAD = "CALL_CREDIT_SPREAD"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"
    SHORT_STRANGLE = "SHORT_STRANGLE"
    LONG_STRADDLE = "LONG_STRADDLE"
    LONG_STRANGLE = "LONG_STRANGLE"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StockScore:
    """Complete scoring for a single stock."""
    symbol: str
    
    # Traditional metrics
    iv_rank: float = 0.0
    options_volume: int = 0
    avg_spread_pct: float = 100.0
    
    # Filter results
    passed_earnings: bool = False
    passed_liquidity: bool = False
    
    # ML predictions
    win_probability: float = 0.5
    regime: MarketRegime = MarketRegime.NORMAL
    trend: str = "NEUTRAL"
    model_agreement: float = 0.5
    
    # Recommendations
    strategy: Strategy = Strategy.CASH
    dte_bucket: int = 2  # Default MONTHLY
    
    # Final score
    score: float = 0.0
    
    # Metadata
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_tradeable(self) -> bool:
        """Can we trade this stock?"""
        return (
            self.passed_earnings and 
            self.passed_liquidity and 
            self.strategy != Strategy.CASH and
            self.win_probability >= 0.5
        )
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'score': self.score,
            'strategy': self.strategy.value,
            'dte_bucket': self.dte_bucket,
            'win_probability': self.win_probability,
            'iv_rank': self.iv_rank,
            'options_volume': self.options_volume,
            'avg_spread_pct': self.avg_spread_pct,
            'regime': self.regime.name,
            'trend': self.trend,
            'model_agreement': self.model_agreement,
            'is_tradeable': self.is_tradeable,
        }


@dataclass
class ScreenerResult:
    """Full screener output."""
    watchlist: List[StockScore]
    regime: MarketRegime
    vix: float
    timestamp: datetime
    total_scanned: int
    total_passed: int
    
    @property
    def symbols(self) -> List[str]:
        return [s.symbol for s in self.watchlist]
    
    @property
    def tradeable(self) -> List[StockScore]:
        return [s for s in self.watchlist if s.is_tradeable]


# =============================================================================
# SYMBOL TIERS BY PRICE (for small accounts)
# =============================================================================

# Tier by approximate stock price and typical spread margin
SYMBOL_PRICE_TIERS = {
    # TIER 1: Ultra cheap ($5-20) - Min account ~$200
    'ultra_cheap': [
        'F', 'FORD', 'SNDL', 'AMC', 'SOFI', 'PLTR', 'NIO', 'LCID', 
        'RIVN', 'HOOD', 'SNAP', 'PINS', 'BB', 'NOK', 'PLUG', 'FCEL',
        'TLRY', 'CGC', 'ACB', 'CLOV', 'WISH', 'RIG', 'ET',
    ],
    
    # TIER 2: Cheap ($20-50) - Min account ~$500
    'cheap': [
        'BAC', 'WFC', 'C', 'USB', 'INTC', 'T', 'VZ', 'KO', 'PFE',
        'CSCO', 'MU', 'GM', 'AAL', 'DAL', 'UAL', 'CCL', 'NCLH',
        'XLF', 'XLE', 'GDX', 'SLV', 'GOLD', 'FCX', 'CLF', 'AA',
    ],
    
    # TIER 3: Medium ($50-150) - Min account ~$1,500
    'medium': [
        'AMD', 'PYPL', 'DIS', 'NFLX', 'BA', 'JPM', 'GS', 'MS',
        'SBUX', 'NKE', 'TGT', 'LOW', 'CVS', 'WBA', 'MRK', 'ABBV',
        'QQQ', 'IWM', 'XLK', 'XLV', 'SMH', 'ARKK', 'COIN',
    ],
    
    # TIER 4: Expensive ($150-300) - Min account ~$3,000
    'expensive': [
        'AAPL', 'MSFT', 'GOOGL', 'META', 'V', 'MA', 'HD', 'UNH',
        'JNJ', 'PG', 'COST', 'ADBE', 'CRM', 'NOW', 'PANW',
        'SPY', 'DIA',
    ],
    
    # TIER 5: Ultra expensive ($300+) - Min account ~$5,000
    'ultra_expensive': [
        'NVDA', 'TSLA', 'AMZN', 'AVGO', 'LLY', 'NFLX', 'ISRG',
        'CMG', 'BKNG', 'SHOP', 'MELI',
    ],
}

def get_affordable_symbols(account_size: float, 
                           max_margin_per_trade_pct: float = 0.10) -> List[str]:
    """
    Vrátí symboly které si účet může dovolit.
    
    Args:
        account_size: Velikost účtu v $
        max_margin_per_trade_pct: Max % účtu na jeden trade (default 10%)
    
    Returns:
        List symbolů které jsou affordable
    """
    max_margin = account_size * max_margin_per_trade_pct
    affordable = []
    
    # Ultra cheap: margin ~$50-100 → need $500+ account
    if max_margin >= 50:
        affordable.extend(SYMBOL_PRICE_TIERS['ultra_cheap'])
    
    # Cheap: margin ~$100-200 → need $1,000+ account
    if max_margin >= 100:
        affordable.extend(SYMBOL_PRICE_TIERS['cheap'])
    
    # Medium: margin ~$200-400 → need $2,000+ account
    if max_margin >= 200:
        affordable.extend(SYMBOL_PRICE_TIERS['medium'])
    
    # Expensive: margin ~$400-600 → need $4,000+ account
    if max_margin >= 400:
        affordable.extend(SYMBOL_PRICE_TIERS['expensive'])
    
    # Ultra expensive: margin ~$600-1000 → need $6,000+ account
    if max_margin >= 600:
        affordable.extend(SYMBOL_PRICE_TIERS['ultra_expensive'])
    
    return list(set(affordable))  # Remove duplicates


def estimate_spread_margin(underlying_price: float, wing_width_pct: float = 0.01) -> float:
    """
    Odhadne margin pro credit spread.
    
    Margin ≈ wing_width * 100 - credit
    Pro konzervativní odhad: margin ≈ wing_width * 100 * 0.8
    
    Args:
        underlying_price: Cena underlying
        wing_width_pct: Šířka wing jako % underlying (default 1%)
    
    Returns:
        Estimated margin per contract in $
    """
    wing_width = underlying_price * wing_width_pct
    # Round to standard strikes
    if underlying_price < 50:
        wing_width = max(1, round(wing_width))  # $1 strikes
    elif underlying_price < 200:
        wing_width = max(2.5, round(wing_width / 2.5) * 2.5)  # $2.5 strikes
    else:
        wing_width = max(5, round(wing_width / 5) * 5)  # $5 strikes
    
    # Margin = width * 100, assume 20% credit received
    margin = wing_width * 100 * 0.80
    return margin

SCREENER_UNIVERSE = [
    # ========== ETFs (Core - Always High Liquidity) ==========
    'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'TLT', 'IEF', 'LQD', 'HYG',
    'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB', 'XLRE',
    'EEM', 'EFA', 'VWO', 'EWZ', 'FXI', 'EWJ', 'EWG', 'EWY', 'EWH', 'INDA',
    'VXX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXU', 'SPXL', 'TNA', 'TZA',
    'ARKK', 'ARKG', 'ARKF', 'ARKW', 'XBI', 'IBB', 'LABU', 'LABD',
    'USO', 'UNG', 'GDX', 'GDXJ', 'JNUG', 'NUGT',
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
    
    # ========== Fintech / Brokers ==========
    'SOFI', 'HOOD', 'NU', 'LC', 'PAYO', 'PSFE',
    'SCHW', 'IBKR', 'FUTU', 'TIGR', 'MKTX',
    
    # ========== Banks / Financials ==========
    'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
    'AXP', 'BLK', 'CME', 'ICE', 'NDAQ', 'CBOE',
    'AIG', 'MET', 'PRU', 'ALL', 'PGR', 'TRV', 'AFL', 'HIG',
    
    # ========== Consumer / Retail ==========
    'DIS', 'CMCSA', 'SBUX', 'YUM', 'CMG', 'DPZ',
    'TGT', 'LOW', 'DG', 'DLTR', 'FIVE', 'ULTA',
    'LULU', 'ANF', 'URBN', 'RL', 'PVH', 'TPR', 'VFC',
    'PTON', 'CROX', 'DECK',
    
    # ========== Healthcare / Biotech ==========
    'BMY', 'AMGN', 'GILD', 'MRNA', 'BNTX', 'NVAX', 'VRTX', 'REGN', 'BIIB',
    'ALNY', 'EDIT', 'CRSP', 'NTLA', 'BEAM', 'IONS', 'SRPT', 'BMRN',
    'EXAS', 'ILMN', 'DXCM', 'ISRG', 'ALGN', 'ZBH', 'SYK', 'MDT',
    
    # ========== Energy ==========
    'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX',
    'DVN', 'FANG', 'APA', 'HAL', 'BKR', 'NOV', 'RIG',
    'ET', 'EPD', 'KMI', 'WMB', 'OKE', 'TRGP', 'LNG',
    
    # ========== Industrial / Aerospace ==========
    'BA', 'LMT', 'NOC', 'RTX', 'GD', 'GE', 'HON', 'CAT', 'DE',
    'UPS', 'FDX', 'DAL', 'UAL', 'AAL', 'LUV', 'JBLU', 'ALK',
    'CSX', 'UNP', 'NSC', 'ODFL', 'XPO', 'CHRW', 'EXPD', 'JBHT',
    
    # ========== Auto / EV ==========
    'GM', 'F', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI',
    'QS', 'CHPT', 'BLNK', 'EVGO', 'PLUG', 'FCEL', 'BE', 'BLDP',
    
    # ========== Mining / Materials ==========
    'FCX', 'NEM', 'GOLD', 'AEM', 'NUE', 'STLD', 'CLF', 'AA',
    'ALB', 'LAC', 'MP', 'SQM', 'CC', 'DOW', 'LYB',
    
    # ========== REITs ==========
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'AVB', 'EQR', 'VTR',
    'DLR', 'PSA', 'ESS', 'MAA', 'UDR', 'INVH', 'SUI', 'ELS',
    
    # ========== Cannabis (High IV) ==========
    'TLRY', 'CGC', 'ACB', 'CRON', 'SNDL',
    
    # ========== Meme / Speculative (High Volume) ==========
    'GME', 'AMC', 'BB', 'SPCE', 'CLOV',
    
    # ========== China ADRs ==========
    'BABA', 'JD', 'PDD', 'BIDU', 'BILI', 'IQ',
    'TME', 'VIPS', 'NTES',
]

# Remove duplicates
SCREENER_UNIVERSE = list(dict.fromkeys(SCREENER_UNIVERSE))


# =============================================================================
# STRATEGY SELECTION MATRIX
# =============================================================================

# (regime, trend) -> (strategy, base_win_prob)
STRATEGY_MATRIX = {
    # CALM regime - theta harvesting
    (MarketRegime.CALM, "BULLISH"): (Strategy.PUT_CREDIT_SPREAD, 0.80),
    (MarketRegime.CALM, "BEARISH"): (Strategy.CALL_CREDIT_SPREAD, 0.75),
    (MarketRegime.CALM, "NEUTRAL"): (Strategy.IRON_CONDOR, 0.82),
    
    # NORMAL regime
    (MarketRegime.NORMAL, "BULLISH"): (Strategy.PUT_CREDIT_SPREAD, 0.78),
    (MarketRegime.NORMAL, "BEARISH"): (Strategy.CALL_CREDIT_SPREAD, 0.73),
    (MarketRegime.NORMAL, "NEUTRAL"): (Strategy.IRON_CONDOR, 0.80),
    
    # ELEVATED - more caution
    (MarketRegime.ELEVATED, "BULLISH"): (Strategy.BULL_CALL_SPREAD, 0.65),
    (MarketRegime.ELEVATED, "BEARISH"): (Strategy.BEAR_PUT_SPREAD, 0.65),
    (MarketRegime.ELEVATED, "NEUTRAL"): (Strategy.IRON_CONDOR, 0.70),
    
    # HIGH_VOL - defined risk only
    (MarketRegime.HIGH_VOL, "BULLISH"): (Strategy.BULL_CALL_SPREAD, 0.55),
    (MarketRegime.HIGH_VOL, "BEARISH"): (Strategy.BEAR_PUT_SPREAD, 0.60),
    (MarketRegime.HIGH_VOL, "NEUTRAL"): (Strategy.CASH, 0.50),
    
    # CRISIS - mostly cash
    (MarketRegime.CRISIS, "BULLISH"): (Strategy.CASH, 0.40),
    (MarketRegime.CRISIS, "BEARISH"): (Strategy.BEAR_PUT_SPREAD, 0.55),
    (MarketRegime.CRISIS, "NEUTRAL"): (Strategy.CASH, 0.40),
}

# DTE bucket based on regime
DTE_BUCKET_MATRIX = {
    MarketRegime.CALM: 2,      # MONTHLY - best theta
    MarketRegime.NORMAL: 2,    # MONTHLY
    MarketRegime.ELEVATED: 2,  # MONTHLY - more time
    MarketRegime.HIGH_VOL: 3,  # LEAPS - avoid gamma
    MarketRegime.CRISIS: 3,    # LEAPS - safety
}


# =============================================================================
# ML-ENHANCED SCREENER
# =============================================================================

class MLEnhancedScreener:
    """
    Daily Options Screener with ML Integration.
    
    Combines:
    1. Traditional metrics (IV, volume, spread)
    2. ML predictions (win probability, regime, strategy)
    3. Risk filters (earnings, liquidity)
    4. AFFORDABILITY filter (account size based)
    """
    
    def __init__(self, 
                 account_size: float = 100000,
                 top_n: int = 50,
                 min_win_prob: float = 0.50,
                 xgb_predictor=None,
                 lstm_model=None):
        """
        Args:
            account_size: Velikost účtu v $ (pro affordability filter)
            top_n: Number of stocks to return
            min_win_prob: Minimum win probability threshold
            xgb_predictor: Optional XGBoost model for win prediction
            lstm_model: Optional LSTM for direction forecast
        """
        self.account_size = account_size
        self.top_n = top_n
        self.min_win_prob = min_win_prob
        self.xgb = xgb_predictor
        self.lstm = lstm_model
        
        # Get affordable symbols for this account
        self.affordable_symbols = get_affordable_symbols(account_size)
        
        # Filter universe to only affordable symbols
        self.universe = [s for s in SCREENER_UNIVERSE if s in self.affordable_symbols]
        
        # If very small account, use only ultra_cheap + cheap
        if account_size < 1000:
            cheap_symbols = (SYMBOL_PRICE_TIERS['ultra_cheap'] + 
                           SYMBOL_PRICE_TIERS['cheap'])
            self.universe = [s for s in self.universe if s in cheap_symbols]
        
        # External dependencies (lazy loaded)
        self._vix_monitor = None
        self._earnings_fetcher = None
        self._data_fetcher = None
        
        # Cache
        self._last_result: Optional[ScreenerResult] = None
        self._last_run_date: Optional[date] = None
        
        logger.info(f"MLEnhancedScreener initialized:")
        logger.info(f"  Account size: ${account_size:,.0f}")
        logger.info(f"  Universe: {len(self.universe)} affordable stocks "
                   f"(from {len(SCREENER_UNIVERSE)} total)")
        
        if account_size < 1000:
            logger.warning(f"  ⚠️ SMALL ACCOUNT MODE - limited to cheap stocks")
    
    def update_account_size(self, new_size: float):
        """Update account size and recalculate affordable symbols."""
        self.account_size = new_size
        self.affordable_symbols = get_affordable_symbols(new_size)
        self.universe = [s for s in SCREENER_UNIVERSE if s in self.affordable_symbols]
        
        if new_size < 1000:
            cheap_symbols = (SYMBOL_PRICE_TIERS['ultra_cheap'] + 
                           SYMBOL_PRICE_TIERS['cheap'])
            self.universe = [s for s in self.universe if s in cheap_symbols]
        
        logger.info(f"Account size updated to ${new_size:,.0f}")
        logger.info(f"New universe: {len(self.universe)} affordable stocks")
    
    # =========================================================================
    # MAIN API
    # =========================================================================
    
    async def run_morning_screen(self, 
                                  vix: float = None,
                                  force: bool = False) -> ScreenerResult:
        """
        Run morning screening with ML scoring.
        
        Args:
            vix: Current VIX (if None, will fetch)
            force: Force re-run even if cached
            
        Returns:
            ScreenerResult with top 50 stocks + strategies
        """
        today = date.today()
        
        # Return cached if available
        if not force and self._last_run_date == today and self._last_result:
            logger.info(f"Using cached result from {today}")
            return self._last_result
        
        # Check market hours
        if not force and not self._is_market_hours():
            logger.info("Market closed - returning cached or empty result")
            return self._last_result or ScreenerResult(
                watchlist=[], regime=MarketRegime.NORMAL, vix=18,
                timestamp=datetime.now(), total_scanned=0, total_passed=0
            )
        
        logger.info("=" * 60)
        logger.info("🚀 STARTING ML-ENHANCED SCREENING")
        logger.info("=" * 60)
        
        # 1. Get VIX and classify regime
        vix = vix or await self._get_vix()
        regime = self._classify_regime(vix)
        logger.info(f"VIX: {vix:.1f} → Regime: {regime.name}")
        
        # 2. Screen all stocks
        scores = await self._screen_universe(vix, regime)
        
        # 3. Sort by score and take top N
        scores.sort(key=lambda x: x.score, reverse=True)
        top_stocks = scores[:self.top_n]
        
        # 4. Create result
        result = ScreenerResult(
            watchlist=top_stocks,
            regime=regime,
            vix=vix,
            timestamp=datetime.now(),
            total_scanned=len(SCREENER_UNIVERSE),
            total_passed=len([s for s in scores if s.is_tradeable])
        )
        
        # 5. Cache
        self._last_result = result
        self._last_run_date = today
        
        # 6. Log results
        self._log_results(result)
        
        return result
    
    def get_cached_result(self) -> Optional[ScreenerResult]:
        """Get cached screening result."""
        return self._last_result
    
    def get_stock_score(self, symbol: str) -> Optional[StockScore]:
        """Get score for specific symbol from cache."""
        if not self._last_result:
            return None
        for stock in self._last_result.watchlist:
            if stock.symbol == symbol:
                return stock
        return None
    
    # =========================================================================
    # SCREENING LOGIC
    # =========================================================================
    
    async def _screen_universe(self, vix: float, regime: MarketRegime) -> List[StockScore]:
        """Screen all affordable stocks in universe."""
        
        scores = []
        semaphore = asyncio.Semaphore(20)  # Limit concurrency
        
        async def analyze_stock(symbol: str) -> Optional[StockScore]:
            async with semaphore:
                try:
                    return await self._score_stock(symbol, vix, regime)
                except Exception as e:
                    logger.debug(f"Error scoring {symbol}: {e}")
                    return None
        
        # Use self.universe (filtered by affordability) instead of SCREENER_UNIVERSE
        universe = self.universe
        
        # Process in chunks for progress logging
        chunk_size = 50
        total = len(universe)
        
        for i in range(0, total, chunk_size):
            chunk = universe[i:i + chunk_size]
            logger.info(f"Processing {i+1}-{min(i+chunk_size, total)}/{total}...")
            
            tasks = [analyze_stock(s) for s in chunk]
            results = await asyncio.gather(*tasks)
            
            for score in results:
                if score:
                    scores.append(score)
        
        logger.info(f"Screened {total} affordable stocks, {len(scores)} scored")
        return scores
    
    async def _score_stock(self, symbol: str, vix: float, 
                           regime: MarketRegime) -> StockScore:
        """
        Score a single stock with ML enhancement.
        """
        score = StockScore(symbol=symbol, regime=regime)
        
        # 1. EARNINGS CHECK
        days_to_earnings = await self._get_days_to_earnings(symbol)
        if days_to_earnings is not None and days_to_earnings <= 7:
            score.reason = f"Earnings in {days_to_earnings} days"
            score.passed_earnings = False
            return score
        score.passed_earnings = True
        
        # 2. GET MARKET DATA
        iv_rank, options_volume, spread_pct = await self._get_market_data(symbol)
        score.iv_rank = iv_rank
        score.options_volume = options_volume
        score.avg_spread_pct = spread_pct
        
        # 3. LIQUIDITY CHECK
        if options_volume < 1000 or spread_pct > 5.0:
            score.reason = f"Low liquidity (vol={options_volume}, spread={spread_pct:.1f}%)"
            score.passed_liquidity = False
            return score
        score.passed_liquidity = True
        
        # 4. DETECT TREND
        trend, trend_strength = await self._detect_trend(symbol)
        score.trend = trend
        
        # 5. GET STRATEGY & WIN PROBABILITY
        strategy, base_win_prob = self._get_strategy(regime, trend)
        score.strategy = strategy
        
        # 6. ML WIN PROBABILITY (if model available)
        if self.xgb:
            features = self._build_features(symbol, vix, iv_rank, trend_strength)
            ml_win_prob = self._predict_win_prob(features)
            # Blend base + ML
            score.win_probability = base_win_prob * 0.4 + ml_win_prob * 0.6
        else:
            # Adjust base probability by IV rank
            iv_adjustment = 0.05 if iv_rank > 0.5 else -0.05  # High IV = better for selling
            score.win_probability = base_win_prob + iv_adjustment
        
        # 7. DTE BUCKET
        score.dte_bucket = self._get_dte_bucket(regime, vix)
        
        # 8. MODEL AGREEMENT (simplified without full LSTM)
        score.model_agreement = self._calculate_agreement(
            strategy, trend, score.win_probability
        )
        
        # 9. FINAL SCORE
        score.score = self._calculate_final_score(score)
        
        return score
    
    def _get_strategy(self, regime: MarketRegime, trend: str) -> Tuple[Strategy, float]:
        """Get recommended strategy based on regime and trend."""
        key = (regime, trend)
        if key in STRATEGY_MATRIX:
            return STRATEGY_MATRIX[key]
        # Default to neutral
        return STRATEGY_MATRIX.get((regime, "NEUTRAL"), (Strategy.CASH, 0.5))
    
    def _get_dte_bucket(self, regime: MarketRegime, vix: float) -> int:
        """Get recommended DTE bucket."""
        base_bucket = DTE_BUCKET_MATRIX.get(regime, 2)
        
        # 0DTE only in calm market with low VIX
        if regime == MarketRegime.CALM and vix < 14:
            return 0  # 0DTE allowed
        
        return base_bucket
    
    def _calculate_agreement(self, strategy: Strategy, trend: str, 
                             win_prob: float) -> float:
        """Calculate model agreement score."""
        agreement = 0.5  # Base
        
        # Strategy-trend alignment
        bullish_strategies = {Strategy.PUT_CREDIT_SPREAD, Strategy.BULL_CALL_SPREAD}
        bearish_strategies = {Strategy.CALL_CREDIT_SPREAD, Strategy.BEAR_PUT_SPREAD}
        
        if strategy in bullish_strategies and trend == "BULLISH":
            agreement += 0.2
        elif strategy in bearish_strategies and trend == "BEARISH":
            agreement += 0.2
        elif strategy == Strategy.IRON_CONDOR and trend == "NEUTRAL":
            agreement += 0.2
        
        # Win probability confidence
        if win_prob > 0.7:
            agreement += 0.15
        elif win_prob > 0.6:
            agreement += 0.1
        
        return min(agreement, 1.0)
    
    def _calculate_final_score(self, stock: StockScore) -> float:
        """Calculate final composite score."""
        if not stock.passed_earnings or not stock.passed_liquidity:
            return 0.0
        
        if stock.strategy == Strategy.CASH:
            return 0.0
        
        # Components:
        # 40% - Win probability
        # 25% - IV rank (higher = more premium)
        # 20% - Model agreement
        # 15% - Liquidity score
        
        win_score = stock.win_probability * 40
        iv_score = min(stock.iv_rank, 1.0) * 25
        agreement_score = stock.model_agreement * 20
        
        # Liquidity: volume + tight spread
        vol_score = min(stock.options_volume / 100000, 1.0) * 10
        spread_score = max(0, (5 - stock.avg_spread_pct) / 5) * 5
        liquidity_score = vol_score + spread_score
        
        return win_score + iv_score + agreement_score + liquidity_score
    
    # =========================================================================
    # DATA FETCHING (with fallbacks)
    # =========================================================================
    
    async def _get_vix(self) -> float:
        """Get current VIX level."""
        if self._vix_monitor:
            try:
                return self._vix_monitor.current_vix
            except:
                pass
        return 18.0  # Default
    
    async def _get_days_to_earnings(self, symbol: str) -> Optional[int]:
        """Get days until next earnings."""
        if self._earnings_fetcher:
            try:
                return self._earnings_fetcher.get_days_to_earnings(symbol)
            except:
                pass
        return None  # Unknown = allow
    
    async def _get_market_data(self, symbol: str) -> Tuple[float, int, float]:
        """
        Get IV rank, options volume, and spread.
        
        Returns:
            (iv_rank, options_volume, spread_pct)
        """
        if self._data_fetcher:
            try:
                # Real IBKR data
                quote = await self._data_fetcher.get_stock_quote(symbol)
                options = await self._data_fetcher.get_options_chain(symbol)
                
                iv_rank = options.get('iv_atm', 0.5) if options else 0.5
                volume = options.get('total_volume', 10000) if options else 10000
                
                if quote and quote.get('bid') and quote.get('ask'):
                    mid = (quote['bid'] + quote['ask']) / 2
                    spread = (quote['ask'] - quote['bid']) / mid * 100 if mid > 0 else 2.0
                else:
                    spread = self._estimate_spread(symbol)
                
                return iv_rank, volume, spread
            except:
                pass
        
        # Fallback estimates
        return (
            self._estimate_iv_rank(symbol),
            self._estimate_volume(symbol),
            self._estimate_spread(symbol)
        )
    
    async def _detect_trend(self, symbol: str) -> Tuple[str, float]:
        """
        Detect trend for symbol.
        
        Returns:
            (trend, strength) where trend is BULLISH/BEARISH/NEUTRAL
        """
        # TODO: Integrate with real price data / LSTM
        # For now, use simple heuristics
        
        # High-beta tech = follow SPY trend
        high_beta = ['TSLA', 'NVDA', 'AMD', 'COIN', 'PLTR', 'SOFI', 'RIVN', 'NIO']
        if symbol in high_beta:
            # Would check SPY trend here
            return "NEUTRAL", 0.3
        
        # ETFs = neutral bias
        etfs = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
        if symbol in etfs:
            return "NEUTRAL", 0.5
        
        return "NEUTRAL", 0.3
    
    def _build_features(self, symbol: str, vix: float, iv_rank: float,
                        trend_strength: float) -> Dict[str, float]:
        """Build feature dict for XGBoost."""
        return {
            'vix': vix,
            'iv_rank': iv_rank,
            'trend_strength': trend_strength,
            # Add more features as needed
        }
    
    def _predict_win_prob(self, features: Dict) -> float:
        """Get XGBoost win probability prediction."""
        if self.xgb:
            try:
                return self.xgb.predict(features)
            except:
                pass
        return 0.5
    
    # =========================================================================
    # FALLBACK ESTIMATES
    # =========================================================================
    
    def _estimate_iv_rank(self, symbol: str) -> float:
        """Estimate IV rank when real data unavailable."""
        high_iv = {'TSLA', 'NVDA', 'AMD', 'COIN', 'GME', 'AMC', 'RIVN', 'LCID', 
                   'PLTR', 'SOFI', 'HOOD', 'NIO', 'MRNA', 'BNTX', 'MARA', 'RIOT'}
        low_iv = {'SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'JNJ', 'PG', 'KO', 'PEP'}
        
        if symbol in high_iv:
            return 0.65
        elif symbol in low_iv:
            return 0.35
        return 0.50
    
    def _estimate_volume(self, symbol: str) -> int:
        """Estimate options volume when real data unavailable."""
        tier1 = {'SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'AMZN', 'META'}
        tier2 = {'IWM', 'MSFT', 'GOOGL', 'NFLX', 'BA', 'DIS', 'COIN', 'PLTR', 'GME'}
        tier3 = {'SOFI', 'HOOD', 'RIVN', 'NIO', 'BABA', 'JPM', 'BAC'}
        
        if symbol in tier1:
            return 500000
        elif symbol in tier2:
            return 200000
        elif symbol in tier3:
            return 100000
        return 30000
    
    def _estimate_spread(self, symbol: str) -> float:
        """Estimate bid-ask spread when real data unavailable."""
        tight = {'SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'AMZN', 'GOOGL'}
        medium = {'TSLA', 'NVDA', 'AMD', 'META', 'NFLX', 'BA', 'JPM'}
        
        if symbol in tight:
            return 0.3
        elif symbol in medium:
            return 1.0
        return 2.5
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _classify_regime(self, vix: float) -> MarketRegime:
        """Classify market regime from VIX."""
        if vix < 15:
            return MarketRegime.CALM
        elif vix < 20:
            return MarketRegime.NORMAL
        elif vix < 25:
            return MarketRegime.ELEVATED
        elif vix < 35:
            return MarketRegime.HIGH_VOL
        else:
            return MarketRegime.CRISIS
    
    def _is_market_hours(self) -> bool:
        """Check if within market hours."""
        if ET is None:
            return True  # Can't check, assume yes
        
        now = datetime.now(ET)
        
        # Weekend
        if now.weekday() >= 5:
            return False
        
        # Before 9:25 or after 16:05
        market_open = now.replace(hour=9, minute=25, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=5, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _log_results(self, result: ScreenerResult):
        """Log screening results."""
        logger.info("=" * 60)
        logger.info(f"📊 ML-ENHANCED SCREENING COMPLETE")
        logger.info(f"   Regime: {result.regime.name} | VIX: {result.vix:.1f}")
        logger.info(f"   Scanned: {result.total_scanned} | Passed: {result.total_passed}")
        logger.info("=" * 60)
        logger.info(f"TOP {len(result.watchlist)} STOCKS:")
        logger.info("-" * 60)
        
        for i, stock in enumerate(result.watchlist[:20], 1):
            tradeable = "✅" if stock.is_tradeable else "❌"
            logger.info(
                f"{i:2}. {stock.symbol:6} | {tradeable} | "
                f"Score: {stock.score:5.1f} | "
                f"Strategy: {stock.strategy.value:20} | "
                f"WinProb: {stock.win_probability:.0%} | "
                f"DTE: {stock.dte_bucket}"
            )
        
        if len(result.watchlist) > 20:
            logger.info(f"    ... and {len(result.watchlist) - 20} more")
        
        logger.info("=" * 60)
        
        # Strategy distribution
        strategy_counts = {}
        for stock in result.watchlist:
            s = stock.strategy.value
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        
        logger.info("STRATEGY DISTRIBUTION:")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            logger.info(f"   {strategy:20}: {count}")
        
        logger.info("=" * 60)


# =============================================================================
# SINGLETON & FACTORY
# =============================================================================

_screener: Optional[MLEnhancedScreener] = None


def get_screener(account_size: float = 100000, top_n: int = 50, **kwargs) -> MLEnhancedScreener:
    """Get or create screener singleton."""
    global _screener
    if _screener is None:
        _screener = MLEnhancedScreener(account_size=account_size, top_n=top_n, **kwargs)
    elif _screener.account_size != account_size:
        # Account size changed, update
        _screener.update_account_size(account_size)
    return _screener


# =============================================================================
# CLI
# =============================================================================

async def main():
    """Test the screener."""
    print("=" * 60)
    print("ML-Enhanced Options Screener")
    print("=" * 60)
    
    screener = get_screener(top_n=50)
    result = await screener.run_morning_screen(vix=17.5, force=True)
    
    print(f"\n✅ Screening complete!")
    print(f"   Regime: {result.regime.name}")
    print(f"   Tradeable: {len(result.tradeable)} stocks")
    print(f"\nTop 10 symbols: {result.symbols[:10]}")
    
    # Show strategy breakdown
    print("\nStrategy recommendations:")
    for stock in result.watchlist[:10]:
        print(f"  {stock.symbol:6} → {stock.strategy.value:20} "
              f"(WP: {stock.win_probability:.0%}, DTE: {stock.dte_bucket})")


if __name__ == "__main__":
    asyncio.run(main())
