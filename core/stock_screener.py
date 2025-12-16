"""
Stock Screener Module

Handles the morning screening routine to select top stocks.
Split from trading_pipeline.py for better maintainability.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio

from core.logger import get_logger
from analysis.screener import get_daily_screener
from ml.trade_success_predictor import get_trade_success_predictor

logger = get_logger()


@dataclass
class ScreenerResult:
    """Result of stock screening."""
    top_50: List[str]
    top_10: List[str]
    ml_scores: Dict[str, float]
    timestamp: datetime


class StockScreener:
    """
    Stock screening and ML filtering.
    
    Morning routine:
    1. Screen 402 stocks → Top 50
    2. ML predict → Top 10
    """
    
    def __init__(self):
        self._screener = get_daily_screener()
        self._predictor = get_trade_success_predictor()
        self._top_50: List[str] = []
        self._top_10: List[str] = []
        self._ml_scores: Dict[str, float] = {}
    
    async def run_morning_screening(self) -> ScreenerResult:
        """
        Run full morning screening routine.
        
        Returns:
            ScreenerResult with top stocks and scores
        """
        logger.info("📊 Starting morning screening routine...")
        
        # Step 1: Run screener
        screener_result = await self._screener.run_morning_screen()
        
        # Original code expects a dict, but run_morning_screen returns a list of symbols (watchlist). 
        # Wait, let's check analysis/screener.py return type.
        # analysis/screener.py: returns List[str] (watchlist)
        
        # Logic update needed:
        if not screener_result:
            logger.warning("Screener returned no results, using defaults")
            self._top_50 = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']
        else:
            self._top_50 = screener_result[:50]
        
        logger.info(f"✅ Top 50 selected: {len(self._top_50)} stocks")
        
        # Step 2: ML filter to Top 10
        self._top_10 = await self._ml_filter_top_10(self._top_50)
        
        logger.info(f"✅ Top 10 ML filtered: {self._top_10}")
        
        return ScreenerResult(
            top_50=self._top_50.copy(),
            top_10=self._top_10.copy(),
            ml_scores=self._ml_scores.copy(),
            timestamp=datetime.now()
        )
    
    async def _ml_filter_top_10(self, stocks: List[str]) -> List[str]:
        """
        Use ML to filter Top 50 → Top 10.
        
        Returns stocks with highest predicted probability.
        """
        if self._predictor is None:
            logger.warning("ML Predictor not available, returning first 10")
            return stocks[:10]
        
        scores: List[tuple[str, float]] = []
        
        # Parallel processing with Semaphore
        sem = asyncio.Semaphore(10)  # Limit concurrent feature fetching/scoring
        
        async def score_stock(symbol):
            async with sem:
                try:
                    features = await self._get_stock_features(symbol)
                    if features:
                        prob = self._predictor.predict(features)
                        return (symbol, prob)
                except Exception as e:
                    logger.debug(f"Could not score {symbol}: {e}")
                return None

        # Create tasks
        tasks = [score_stock(s) for s in stocks[:50]]
        
        # Run concurrently
        results = await asyncio.gather(*tasks)
        
        # Collect results
        for res in results:
            if res:
                symbol, prob = res
                scores.append((symbol, prob))
                self._ml_scores[symbol] = prob
        
        # Sort by probability descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        top_10 = [s[0] for s in scores[:10]]
        
        logger.info(
            f"ML Top 10: {[(s[0], f'{s[1]:.1%}') for s in scores[:10]]}"
        )
        
        return top_10
    
    async def _get_stock_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get features for ML prediction."""
        # Placeholder - in production, fetch real features
        return {
            'symbol': symbol,
            'vix': 18.0,
            'return_1d': 0.01,
            'return_5d': 0.02,
            'volume_ratio': 1.2,
        }
    
    @property
    def top_50(self) -> List[str]:
        """Get current Top 50 stocks."""
        return self._top_50.copy()
    
    @property
    def top_10(self) -> List[str]:
        """Get current Top 10 stocks."""
        return self._top_10.copy()
    
    def get_ml_score(self, symbol: str) -> Optional[float]:
        """Get ML score for a symbol."""
        return self._ml_scores.get(symbol)


# Singleton
_screener: Optional[StockScreener] = None


def get_stock_screener() -> StockScreener:
    """Get or create stock screener singleton."""
    global _screener
    if _screener is None:
        _screener = StockScreener()
    return _screener
