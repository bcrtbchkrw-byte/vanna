"""
Stock Screener Module

Handles the morning screening routine to select top stocks.
Split from trading_pipeline.py for better maintainability.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.logger import get_logger
from analysis.screener import get_daily_screener
from ml.trade_success_predictor import get_trade_success_predictor
from ibkr.data_fetcher import get_data_fetcher
from ml.live_feature_builder import get_live_feature_builder
from ml.daily_feature_calculator import get_daily_feature_calculator
from ml.vanna_data_pipeline import get_vanna_pipeline

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
        self._data_fetcher = get_data_fetcher()
        self._feature_builder = get_live_feature_builder()
        self._vanna_pipeline = get_vanna_pipeline()
        self._daily_calc = get_daily_feature_calculator()
        
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
        
        if not screener_result:
            logger.warning("Screener returned no results, using defaults")
            self._top_50 = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']
        else:
            self._top_50 = screener_result[:50]
        
        logger.info(f"✅ Top 50 selected: {len(self._top_50)} stocks")
        
        # Determine VIX once for all stocks
        vix = await self._data_fetcher.get_vix()
        if vix is None:
            logger.warning("Could not fetch VIX, defaulting to 20.0")
            vix = 20.0
        
        logger.info(f"   Using VIX={vix:.2f} for ML filtering")

        # Step 2: ML filter to Top 10
        self._top_10 = await self._ml_filter_top_10(self._top_50, vix)
        
        logger.info(f"✅ Top 10 ML filtered: {self._top_10}")
        
        return ScreenerResult(
            top_50=self._top_50.copy(),
            top_10=self._top_10.copy(),
            ml_scores=self._ml_scores.copy(),
            timestamp=datetime.now()
        )
    
    async def _ml_filter_top_10(self, stocks: List[str], vix: float) -> List[str]:
        """
        Use ML to filter Top 50 → Top 10.
        
        Returns stocks with highest predicted probability.
        """
        if self._predictor is None:
            logger.warning("ML Predictor not available, returning first 10")
            return stocks[:10]
        
        self._ml_scores.clear()
        scores: List[tuple[str, float]] = []
        
        # Parallel processing with Semaphore
        # We need to be careful not to overload IBKR with 50 option request chains
        # IBKRDataFetcher has its own semaphore (20) for option chains.
        # But here we also fetch quotes.
        sem = asyncio.Semaphore(10)  # Limit concurrent full stock analyses
        
        async def score_stock(symbol):
            async with sem:
                try:
                    features = await self._get_stock_features(symbol, vix)
                    if features:
                        prob = self._predictor.predict(features)
                        # Log probability for visibility
                        if prob > 0.5:
                            logger.info(f"   🎯 {symbol}: {prob:.1%}")
                        else:
                            logger.debug(f"   ❌ {symbol}: {prob:.1%}")
                        return (symbol, prob)
                    else:
                        logger.debug(f"   ⚠️ {symbol}: No features")
                except Exception as e:
                    logger.debug(f"Could not score {symbol}: {e}")
                return None

        # Create tasks
        tasks = [score_stock(s) for s in stocks]
        
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
    
    async def _get_stock_features(self, symbol: str, vix: float) -> Optional[Dict[str, Any]]:
        """
        Get features for ML prediction using LiveFeatureBuilder.
        """
        try:
            # 1. Fetch Quote (Price, Volume)
            quote = await self._data_fetcher.get_stock_quote(symbol)
            if not quote or not quote.get('last'):
                return None
            
            price = quote['last']
            
            # 2. Fetch Options Data (IV, P/C Ratio) - This is the "expensive" part
            # We can skip it if we want speed and use VIX estimates, but for accuracy we need it.
            # TradeSuccessPredictor might rely on 'options_iv_atm'.
            options_data = await self._data_fetcher.get_options_market_data(symbol)
            
            # 3. Get Daily Features (Technicals)
            # Try to load from parquet first (fast), else calculating is hard without history.
            # We assume data maintenance runs daily so parquet should depend on yesterday.
            # But for live calculation we might need to load parquet and add today's candle?
            # For simplicity/speed in screener, we might rely on cached daily data or simple approximation.
            # 3. Get Daily Features (Technicals)
            daily_features = None
            try:
                # We need history.
                df = self._vanna_pipeline.get_training_data([symbol], timeframe='1day')
                
                # Check if data is missing/insufficient
                if df is None or len(df) < 50:
                    logger.info(f"   ⚠️ {symbol}: Missing history, attempting on-demand download...")
                    from automation.data_maintenance import get_maintenance_manager
                    manager = get_maintenance_manager()
                    
                    # Download data (this might take 30-60s)
                    success = await manager.ensure_data_for_symbol(symbol)
                    
                    if success:
                        # Retry loading
                        df = self._vanna_pipeline.get_training_data([symbol], timeframe='1day')
                    else:
                        logger.warning(f"   ❌ {symbol}: Download failed, skipping ML features")

                # Process if we have data now
                if df is not None and len(df) > 50:
                    df = self._daily_calc.add_technical_features(df)
                    row = df.iloc[-1]
                    daily_features = row.to_dict()
                    # Prefix with 'day_' to match LiveFeatureBuilder expectation
                    daily_features = {f"day_{k}": v for k, v in daily_features.items()}
            except Exception as e:
                logger.debug(f"   Daily features error for {symbol}: {e}")

            # 4. Build All Features
            features = self._feature_builder.build_all_features(
                symbol=symbol,
                price=price,
                vix=vix,
                quote=quote,
                options_data=options_data,
                daily_features=daily_features
            )
            
            return features
            
        except Exception as e:
            logger.debug(f"Error building features for {symbol}: {e}")
            return None
    
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

