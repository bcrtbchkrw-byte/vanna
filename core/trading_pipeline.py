"""
Trading Pipeline Orchestrator

Coordinates the complete trading workflow:
1. Morning: Screener (402) → Top 50
2. Morning: ML Filter → Top 10
3. All Day: RL Agent monitors 10 stocks
4. On Signal: Gemini sentiment check
5. If Approved: Order Manager → IBKR

This is the main trading engine.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, date, time
from dataclasses import dataclass
import asyncio
from zoneinfo import ZoneInfo

from loguru import logger

# Components
from analysis.screener import get_daily_screener
from ml.trade_success_predictor import get_trade_success_predictor
from ml.regime_classifier import get_regime_classifier
from rl.ppo_agent import get_trading_agent
from ai.gemini_client import get_gemini_client
from execution.order_manager import get_order_manager
from ibkr.data_fetcher import get_data_fetcher
from ibkr.connection import get_ibkr_connection


@dataclass
class TradeSignal:
    """Signal from RL agent."""
    symbol: str
    action: str  # OPEN, CLOSE, HOLD
    confidence: float
    features: Dict[str, float]
    timestamp: datetime


@dataclass
class TradeDecision:
    """Final trade decision after Gemini check."""
    signal: TradeSignal
    gemini_approved: bool
    gemini_reason: str
    executed: bool = False


class TradingPipeline:
    """
    Main trading pipeline orchestrator.
    
    Flow:
    - 9:30 AM: Screener selects Top 50 from 402 universe
    - 9:35 AM: ML filters to Top 10 with highest probability
    - 9:40+ AM: RL agent continuously monitors Top 10
    - On RL signal: Gemini checks sentiment/risk
    - If approved: Order executed via IBKR
    """
    
    def __init__(self):
        # Components (lazy init)
        self._screener = None
        self._ml_predictor = None
        self._regime_classifier = None
        self._rl_agent = None
        self._gemini = None
        self._order_manager = None
        self._data_fetcher = None
        self._ibkr = None
        
        # Daily state
        self._top_50: List[str] = []
        self._top_10: List[str] = []
        self._last_screen_date: Optional[date] = None
        
        # Trading state
        self._active_positions: Dict[str, Any] = {}
        self._daily_trades: int = 0
        self._max_daily_trades: int = 10
        self._max_positions: int = 5
        
        # RL loop control
        self._rl_interval_seconds: int = 60  # Check every minute
        self._min_rl_confidence: float = 0.7  # Min confidence for signal
        
        # Dynamic Top 10 refresh
        self._refresh_interval_minutes: int = 30  # Re-evaluate Top 10 every 30 min
        self._last_refresh_time: Optional[datetime] = None
        self._min_stock_probability: float = 0.4  # Replace if prob drops below this
        
        logger.info("TradingPipeline initialized")
    
    async def _init_components(self):
        """Initialize all components."""
        self._screener = get_daily_screener()
        self._ml_predictor = get_trade_success_predictor()
        self._regime_classifier = get_regime_classifier()
        self._gemini = get_gemini_client()
        self._order_manager = get_order_manager()
        
        try:
            self._rl_agent = get_trading_agent()
            self._rl_agent.load()  # Load trained model
        except Exception as e:
            logger.warning(f"Could not load RL agent: {e}")
            self._rl_agent = None
        
        try:
            self._ibkr = await get_ibkr_connection()
            self._data_fetcher = get_data_fetcher()
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
    
    # =========================================================================
    # MORNING ROUTINE
    # =========================================================================
    
    async def run_morning_routine(self) -> List[str]:
        """
        Run morning screening and ML filtering.
        
        Called once at market open (9:30 AM).
        
        Returns:
            Top 10 stocks for RL to monitor
        """
        today = date.today()
        
        # Check if already ran today
        if self._last_screen_date == today and self._top_10:
            logger.info(f"Using cached Top 10 from {today}")
            return self._top_10
        
        logger.info("=" * 70)
        logger.info("🌅 MORNING ROUTINE - Starting")
        logger.info("=" * 70)
        
        await self._init_components()
        
        # Step 1: Screener → Top 50
        logger.info("\n📊 Step 1: Screener (402 → 50)")
        self._top_50 = await self._screener.run_morning_screen()
        logger.info(f"   Screener selected: {len(self._top_50)} stocks")
        
        if len(self._top_50) < 10:
            logger.error("Not enough stocks passed screening!")
            return []
        
        # Step 2: ML Filter → Top 10
        logger.info("\n🤖 Step 2: ML Filter (50 → 10)")
        self._top_10 = await self._ml_filter_top_10(self._top_50)
        logger.info(f"   ML selected: {self._top_10}")
        
        self._last_screen_date = today
        self._daily_trades = 0
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✅ MORNING COMPLETE - RL will monitor: {self._top_10}")
        logger.info("=" * 70)
        
        return self._top_10
    
    async def _ml_filter_top_10(self, stocks: List[str]) -> List[str]:
        """
        Use ML to filter Top 50 → Top 10.
        
        Uses TradeSuccessPredictor to score each stock.
        Returns stocks with highest predicted probability.
        """
        scored_stocks: List[tuple] = []
        
        for symbol in stocks:
            try:
                # Get features for prediction
                features = await self._get_stock_features(symbol)
                if features is None:
                    continue
                
                # ML prediction
                prob = self._ml_predictor.predict_proba(features)
                scored_stocks.append((symbol, prob))
                
            except Exception as e:
                logger.debug(f"ML scoring failed for {symbol}: {e}")
                continue
        
        # Sort by probability and take top 10
        scored_stocks.sort(key=lambda x: x[1], reverse=True)
        top_10 = [s[0] for s in scored_stocks[:10]]
        
        # Log scores
        for symbol, prob in scored_stocks[:10]:
            logger.info(f"   {symbol}: {prob:.1%}")
        
        return top_10
    
    async def _get_stock_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get features for ML prediction."""
        try:
            if self._data_fetcher is None:
                return None
            
            quote = await self._data_fetcher.get_stock_quote(symbol)
            vix = await self._data_fetcher.get_vix()
            
            # Build feature dict (simplified)
            return {
                'price': quote.get('last', 0),
                'volume': quote.get('volume', 0),
                'vix': vix or 18.0,
                'symbol': symbol
            }
        except Exception:
            return None
    
    # =========================================================================
    # RL CONTINUOUS LOOP
    # =========================================================================
    
    async def run_rl_loop(self):
        """
        Run RL agent continuously during market hours.
        
        Checks Top 10 stocks every minute for trade opportunities.
        Refreshes Top 10 every 30 minutes to replace underperforming stocks.
        """
        logger.info("🔄 Starting RL continuous loop...")
        
        while self._is_market_open():
            try:
                # Dynamic refresh: Check if any stock should be replaced
                await self._maybe_refresh_top_10()
                
                for symbol in self._top_10:
                    signal = await self._rl_evaluate(symbol)
                    
                    if signal and signal.action in ['OPEN', 'CLOSE']:
                        # RL found opportunity!
                        logger.info(f"🎯 RL Signal: {signal.action} {signal.symbol} (conf: {signal.confidence:.1%})")
                        
                        # Pass to Gemini for final check
                        decision = await self._gemini_check(signal)
                        
                        if decision.gemini_approved:
                            await self._execute_trade(decision)
                        else:
                            logger.info(f"❌ Gemini rejected: {decision.gemini_reason}")
                
                await asyncio.sleep(self._rl_interval_seconds)
                
            except Exception as e:
                logger.error(f"RL loop error: {e}")
                await asyncio.sleep(5)
        
        logger.info("RL loop ended - market closed")
    
    async def _maybe_refresh_top_10(self):
        """
        Periodically re-evaluate Top 10 and replace underperforming stocks.
        
        Runs every 30 minutes. Stocks with probability below threshold
        are replaced by next best from Top 50.
        """
        now = datetime.now()
        
        # Check if enough time passed since last refresh
        if self._last_refresh_time:
            minutes_since = (now - self._last_refresh_time).total_seconds() / 60
            if minutes_since < self._refresh_interval_minutes:
                return
        
        logger.info("🔄 Refreshing Top 10 - checking for underperformers...")
        self._last_refresh_time = now
        
        # Re-score current Top 10
        to_replace: List[str] = []
        current_scores: Dict[str, float] = {}
        
        for symbol in self._top_10:
            # Skip if we have active position - don't remove mid-trade
            if symbol in self._active_positions:
                logger.debug(f"  {symbol}: Keeping (active position)")
                current_scores[symbol] = 1.0  # Keep at top
                continue
                
            try:
                features = await self._get_stock_features(symbol)
                if features:
                    prob = self._ml_predictor.predict_proba(features)
                    current_scores[symbol] = prob
                    
                    if prob < self._min_stock_probability:
                        to_replace.append(symbol)
                        logger.info(f"  ❌ {symbol}: {prob:.1%} - BELOW THRESHOLD, will replace")
                    else:
                        logger.debug(f"  ✅ {symbol}: {prob:.1%}")
                else:
                    to_replace.append(symbol)
            except Exception as e:
                logger.debug(f"  {symbol}: Error - {e}")
        
        if not to_replace:
            logger.info("  All Top 10 stocks still performing well ✅")
            return
        
        # Find replacements from Top 50 (excluding current Top 10)
        candidates = [s for s in self._top_50 if s not in self._top_10]
        replacements: List[tuple] = []
        
        for symbol in candidates[:20]:  # Check first 20 candidates
            try:
                features = await self._get_stock_features(symbol)
                if features:
                    prob = self._ml_predictor.predict_proba(features)
                    if prob >= self._min_stock_probability:
                        replacements.append((symbol, prob))
            except:
                pass
        
        # Sort by probability
        replacements.sort(key=lambda x: x[1], reverse=True)
        
        # Perform replacement
        for old_symbol in to_replace:
            if replacements:
                new_symbol, new_prob = replacements.pop(0)
                idx = self._top_10.index(old_symbol)
                self._top_10[idx] = new_symbol
                logger.info(f"  🔄 Replaced {old_symbol} → {new_symbol} ({new_prob:.1%})")
            else:
                logger.warning(f"  ⚠️ No replacement found for {old_symbol}")
        
        logger.info(f"  Updated Top 10: {self._top_10}")
    
    async def _rl_evaluate(self, symbol: str) -> Optional[TradeSignal]:
        """
        Evaluate a single stock with RL agent.
        
        Returns signal if action != HOLD and confidence > threshold.
        """
        if self._rl_agent is None or self._rl_agent.model is None:
            return None
        
        try:
            # Get live features
            features = await self._get_live_features(symbol)
            if features is None:
                return None
            
            # RL prediction
            import numpy as np
            obs = np.array(list(features.values()), dtype=np.float32)
            action = self._rl_agent.predict(obs, deterministic=True)
            
            # Map action to signal
            action_map = {0: 'HOLD', 1: 'OPEN', 2: 'CLOSE', 3: 'INCREASE', 4: 'DECREASE'}
            action_name = action_map.get(action, 'HOLD')
            
            if action_name == 'HOLD':
                return None
            
            # Check position state
            has_position = symbol in self._active_positions
            
            # Validate action
            if action_name == 'OPEN' and has_position:
                return None  # Already have position
            if action_name == 'CLOSE' and not has_position:
                return None  # No position to close
            
            return TradeSignal(
                symbol=symbol,
                action=action_name,
                confidence=0.8,  # TODO: Get actual confidence from RL
                features=features,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.debug(f"RL eval failed for {symbol}: {e}")
            return None
    
    async def _get_live_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get live features for RL prediction (49 features)."""
        # TODO: Implement full feature extraction matching TradingEnv
        # For now, return mock features
        return None
    
    # =========================================================================
    # GEMINI SENTIMENT CHECK
    # =========================================================================
    
    async def _gemini_check(self, signal: TradeSignal) -> TradeDecision:
        """
        Ask Gemini to verify trade with sentiment/risk check.
        """
        try:
            # Get current market context
            vix = await self._data_fetcher.get_vix() if self._data_fetcher else 18
            quote = await self._data_fetcher.get_stock_quote(signal.symbol) if self._data_fetcher else {}
            price = quote.get('last', 0)
            
            prompt = f"""
Trade Verification Request:
Symbol: {signal.symbol}
Action: {signal.action}
Current Price: ${price:.2f}
VIX: {vix:.1f}
Time: {signal.timestamp.strftime('%H:%M')}

Quick checks:
1. Any breaking news affecting {signal.symbol}?
2. Is current market sentiment acceptable?
3. Any unusual risk factors now?

Respond with exactly one word: APPROVED or REJECTED
Then a brief reason on the next line.
"""
            
            response = await self._gemini.generate_text(prompt)
            
            # Parse response
            lines = response.strip().split('\n')
            first_word = lines[0].strip().upper()
            reason = lines[1] if len(lines) > 1 else ""
            
            approved = 'APPROVED' in first_word
            
            return TradeDecision(
                signal=signal,
                gemini_approved=approved,
                gemini_reason=reason
            )
            
        except Exception as e:
            logger.error(f"Gemini check failed: {e}")
            # Default to rejected on error
            return TradeDecision(
                signal=signal,
                gemini_approved=False,
                gemini_reason=f"Gemini error: {e}"
            )
    
    # =========================================================================
    # ORDER EXECUTION
    # =========================================================================
    
    async def _execute_trade(self, decision: TradeDecision):
        """Execute approved trade via Order Manager."""
        signal = decision.signal
        
        # Check daily limits
        if self._daily_trades >= self._max_daily_trades:
            logger.warning(f"Daily trade limit ({self._max_daily_trades}) reached")
            return
        
        if signal.action == 'OPEN' and len(self._active_positions) >= self._max_positions:
            logger.warning(f"Max positions ({self._max_positions}) reached")
            return
        
        # Check paper mode
        if self._order_manager.paper_mode:
            logger.info(f"📝 PAPER TRADE: {signal.action} {signal.symbol}")
            self._daily_trades += 1
            if signal.action == 'OPEN':
                self._active_positions[signal.symbol] = {
                    'entry_time': datetime.now(),
                    'signal': signal
                }
            elif signal.action == 'CLOSE':
                self._active_positions.pop(signal.symbol, None)
            return
        
        # Real trading
        try:
            logger.info(f"🚀 EXECUTING: {signal.action} {signal.symbol}")
            
            # TODO: Implement actual order placement
            # await self._order_manager.place_order(...)
            
            self._daily_trades += 1
            decision.executed = True
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _is_market_open(self) -> bool:
        """Check if US stock market is currently open (NYSE/NASDAQ hours)."""
        # CRITICAL: Use Eastern Time, not local time!
        ET = ZoneInfo("America/New_York")
        now_et = datetime.now(ET)
        
        # Weekend check
        if now_et.weekday() >= 5:
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        return market_open <= now_et.time() <= market_close
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'top_50': len(self._top_50),
            'top_10': self._top_10,
            'active_positions': list(self._active_positions.keys()),
            'daily_trades': self._daily_trades,
            'market_open': self._is_market_open(),
            'rl_agent_loaded': self._rl_agent is not None and self._rl_agent.model is not None
        }


# Singleton
_pipeline: Optional[TradingPipeline] = None


def get_trading_pipeline() -> TradingPipeline:
    """Get or create trading pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = TradingPipeline()
    return _pipeline


# CLI for testing
if __name__ == "__main__":
    async def main():
        from core.logger import setup_logger
        setup_logger(level="INFO")
        
        print("=" * 60)
        print("Trading Pipeline Test")
        print("=" * 60)
        
        pipeline = get_trading_pipeline()
        
        # Run morning routine
        top_10 = await pipeline.run_morning_routine()
        print(f"\nTop 10 for RL: {top_10}")
        
        # Show status
        status = pipeline.get_status()
        print(f"\nStatus: {status}")
    
    asyncio.run(main())
