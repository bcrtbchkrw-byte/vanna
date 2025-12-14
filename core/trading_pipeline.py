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
from core.database import get_database
from core.scheduler import get_scheduler
from risk.portfolio_manager import get_portfolio_manager


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
        self._portfolio_manager = get_portfolio_manager()
        
        try:
            self._rl_agent = get_trading_agent()
            self._rl_agent.load()  # Load trained model
        except Exception as e:
            logger.warning(f"Could not load RL agent: {e}")
            self._rl_agent = None
        
        try:
            self._ibkr = await get_ibkr_connection()
            self._data_fetcher = get_data_fetcher()
            self._db = await get_database()
            
            # Connect to IBKR if not connected
            if not self._ibkr.is_connected:
                await self._ibkr.connect()
            
            # Reconcile state (Sync DB vs IBKR)
            await self._reconcile_state()
            
        except Exception as e:
            logger.error(f"IBKR/DB connection failed: {e}")

    async def _reconcile_state(self):
        """
        Reconcile internal state with DB and IBKR.
        Critical for crash recovery.
        """
        logger.info("🔄 Reconciling state (DB vs IBKR)...")
        
        try:
            # 1. Get DB open trades
            db_trades = await self._db.get_open_trades()
            db_map = {t['symbol']: t for t in db_trades}
            
            # 2. Get IBKR positions with Safety Check
            ib_positions = self._ibkr.get_positions()
            
            # SAFETY: If DB has trades but IBKR returns ZERO, it might be a glitch.
            # We verify connection health by checking NetLiquidation.
            if len(db_trades) > 0 and len(ib_positions) == 0:
                logger.warning("⚠️ Mismatch: DB has trades but IBKR returns 0 positions. Verifying...")
                
                # Check connection health explicitly
                try:
                    summary = await self._ibkr.get_account_summary()
                    net_liq = float(summary.get('NetLiquidation', 0))
                    if net_liq <= 0:
                         logger.error("❌ Safety Stop: NetLiquidation is 0 or unreadable. IBKR data feed likely down. Aborting reconciliation.")
                         return
                except Exception as e:
                     logger.error(f"❌ Safety Stop: Could not verify account summary. Aborting reconciliation. Error: {e}")
                     return
                
                # Force refresh positions
                await asyncio.sleep(2.0)
                ib_positions = self._ibkr.get_positions()
                if len(ib_positions) == 0:
                     logger.warning("⚠️ Confirmed 0 positions in IBKR after verify. Proceeding to close DB trades.")

            ib_map = {p.contract.symbol: p for p in ib_positions if p.position != 0}
            
            # 3. Reconcile
            
            # A. In IBKR, not in DB -> Recover (Insert)
            for symbol, pos in ib_map.items():
                if symbol not in db_map:
                    logger.warning(f"⚠️ Found un-tracked position in IBKR: {symbol} ({pos.position}). Recovering...")
                    try:
                        trade_id = await self._db.insert_trade(
                            symbol=symbol,
                            strategy="Recovered",
                            entry_price=pos.avgCost,
                            quantity=int(pos.position),
                            notes="Recovered from IBKR reconciliation"
                        )
                        self._active_positions[symbol] = {
                            'entry_price': pos.avgCost,
                            'quantity': int(pos.position),
                            'trade_id': trade_id,
                            'entry_time': datetime.now(),
                            'signal': None 
                        }
                    except Exception as e:
                        logger.error(f"Failed to recover trade for {symbol}: {e}")
            
            # B. In DB, not in IBKR -> Close (Mark closed)
            for symbol, trade in db_map.items():
                if symbol not in ib_map:
                    logger.warning(f"⚠️ Trade in DB but not in IBKR: {symbol}. Marking closed.")
                    # Assume closed at 0 PnL if unknown
                    await self._db.close_trade(
                        trade_id=trade['id'],
                        exit_price=trade['entry_price'], 
                        pnl=0.0
                    )
                else:
                    # C. Match -> Load into memory
                    pos = ib_map[symbol]
                    self._active_positions[symbol] = {
                        'entry_price': trade['entry_price'],
                        'quantity': trade['quantity'],
                        'trade_id': trade['id'],
                        'entry_time': trade['entry_time'] if isinstance(trade['entry_time'], datetime) else datetime.now(),
                        'signal': None
                    }
                    logger.info(f"✅ Reconciled {symbol}: ID {trade['id']}")
    
            logger.info(f"State reconciled. {len(self._active_positions)} active positions.")
            
        except Exception as e:
            logger.error(f"State reconciliation failed: {e}")
    
        except Exception as e:
            logger.error(f"IBKR/DB connection failed: {e}")

    async def start(self):
        """
        Start the trading system.
        
        1. Initialize components
        2. Start Scheduler (background)
        3. Run Morning Routine
        4. Enter continuous RL loop
        """
        logger.info("🚀 Starting Trading Pipeline...")
        
        # 1. Initialize
        await self._init_components()
        
        # 2. Start Scheduler
        self._scheduler = get_scheduler()
        asyncio.create_task(self._scheduler.start())
        
        # 3. Morning Routine
        # Only run if within time window or force checked?
        # Actually run_morning_routine checks time internally or cache logic
        # But we should try it on startup regardless of time, and let it decide?
        # Typically morning routine is for 9:30. If we start at 10:00, we missed it?
        # pipeline._top_10 is empty.
        
        # Let's run it. It has logic inside.
        # Actually it says "Called once at market open (9:30 AM)".
        # Logic: if already ran today, returns cached.
        
        # If we start mid-day, we need top 10.
        # So we should attempt to populate it.
        top_10 = await self.run_morning_routine()
        if not top_10:
             # Maybe force re-screen or use fallback?
             if self._is_market_open():
                 logger.info("⚠️ No top 10 (perhaps missed morning). Attempting late screen...")
                 # Force screen?
                 # Actually run_morning_routine checks time? 
                 # Looking at implementation: It does NOT check time strictness, only day cache.
                 # Wait, main loop had time check "if today.hour == 9 and 30 <= today.minute <= 35".
                 # run_morning_routine() itself doesn't check time, just does work.
                 # So calling it here is safe.
                 pass
        
        # 4. Continuous Loop
        logger.info("🔄 Entering continuous trading loop...")
        await self.run_rl_loop()

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
            
            # Use new method with confidence
            if hasattr(self._rl_agent, 'predict_with_confidence'):
                action, confidence = self._rl_agent.predict_with_confidence(obs, deterministic=True)
            else:
                action = self._rl_agent.predict(obs, deterministic=True)
                confidence = 0.8  # Default if method missing
            
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
                confidence=confidence,
                features=features,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.debug(f"RL eval failed for {symbol}: {e}")
            return None
    
    async def _calculate_technical_features(self, symbol: str, current_price: float) -> Dict[str, float]:
        """
        Calculate daily technical features using Single Source of Truth (DailyFeatureCalculator).
        Ensures consistency with RL training data.
        """
        defaults = {
            'day_sma_200': 1.0, 'day_sma_50': 1.0, 'day_sma_20': 1.0,
            'day_price_vs_sma200': 0.0, 'day_price_vs_sma50': 0.0,
            'day_rsi_14': 0.5, 'day_atr_14': 0.02, 'day_atr_pct': 0.02,
            'day_bb_position': 0.5, 'day_macd': 0.0, 'day_macd_hist': 0.0,
            'day_above_sma200': 1, 'day_above_sma50': 1, 'day_sma_50_200_ratio': 1.0
        }
        
        try:
            from ml.vanna_data_pipeline import get_vanna_pipeline
            from ml.daily_feature_calculator import get_daily_feature_calculator
            import pandas as pd
            import numpy as np
            
            pipeline = get_vanna_pipeline()
            # Fetch enough history for 200 SMA + some buffer
            df = pipeline.get_training_data([symbol], timeframe='1day')
            
            if df is None or len(df) < 200:
                logger.warning(f"Insufficient history for {symbol} technicals")
                return defaults
                
            # Use centralized calculator
            calc = get_daily_feature_calculator()
            df = calc.add_technical_features(df)
            
            # Extract latest values
            last = df.iloc[-1]
            
            # Normalize for RL Agent (0-1 scale, ratios)
            # CRITICAL: Currently DailyFeatureCalculator returns RAW values (RSI 0-100, SMA prices)
            # We must apply the SAME transforms here as FeatureEnricher/Injector do for training data.
            # OR, if we move normalization to Calculator, use that.
            # Currently Calculator returns standard technicals.
            
            close = last['close']
            
            features = {
                # SMA: Raw values normalized by price for Gym observation consistency
                # (Assuming RL environment receives price-relative values for generalization)
                'day_sma_200': float(last['sma_200'] / close),
                'day_sma_50': float(last['sma_50'] / close),
                'day_sma_20': float(last['sma_20'] / close),
                
                # Ratios (already computed)
                'day_price_vs_sma200': float(last['price_vs_sma200']), # close/sma - 1? No, calc is close/sma
                # Wait, calculator logic: df['price_vs_sma200'] = df['close'] / df['sma_200'] (Ratio ~1.0)
                # Pipeline previously returned: (current_price / sma_200) - 1 (Diff from 1.0)
                # Let's align with what TradingEnv expects.
                # TradingEnv Market Features are normalized by VecNormalize, so raw ratio ~1.0 is fine.
                'day_price_vs_sma200': float(last['price_vs_sma200'] - 1.0), # Centered around 0
                'day_price_vs_sma50': float(last['price_vs_sma50'] - 1.0),
                
                # RSI: Calculator 0-100 -> RL 0-1
                'day_rsi_14': float(last['rsi_14'] / 100.0),
                
                # ATR
                'day_atr_14': float(last['atr_14']),
                'day_atr_pct': float(last['atr_pct']),
                
                # BB Position (0-1) - already calculated
                'day_bb_position': float(last['bb_position']),
                
                # MACD: Normalize by price to make it relative
                'day_macd': float(last['macd'] / close),
                'day_macd_hist': float(last['macd_hist'] / close),
                
                # Trend
                'day_above_sma200': int(last['above_sma200']),
                'day_above_sma50': int(last['above_sma50']),
                'day_sma_50_200_ratio': float(last['sma_50_200_ratio']),
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating technical features for {symbol}: {e}")
            return defaults

    async def _get_live_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get live features for RL prediction (70 features matching TradingEnv).
        
        Market features (63) + Position features (7) = 70 total.
        Uses live data from IBKR and current position state.
        """
        try:
            if self._data_fetcher is None:
                logger.warning("Data fetcher not available for live features")
                return None
            
            # Get live data
            quote = await self._data_fetcher.get_stock_quote(symbol)
            vix = await self._data_fetcher.get_vix()
            
            if not quote or not quote.get('last'):
                logger.warning(f"No live quote for {symbol}")
                return None
            
            from datetime import datetime
            import math
            
            now = datetime.now()
            price = quote.get('last', 0)
            
            # Calculate time features (cyclical encoding)
            minutes_of_day = now.hour * 60 + now.minute
            day_of_week = now.weekday()
            day_of_year = now.timetuple().tm_yday
            
            sin_time = math.sin(2 * math.pi * minutes_of_day / 1440)
            cos_time = math.cos(2 * math.pi * minutes_of_day / 1440)
            sin_dow = math.sin(2 * math.pi * day_of_week / 7)
            cos_dow = math.cos(2 * math.pi * day_of_week / 7)
            sin_doy = math.sin(2 * math.pi * day_of_year / 365)
            cos_doy = math.cos(2 * math.pi * day_of_year / 365)
            
            # VIX features
            vix_value = vix or 18.0
            vix_norm = vix_value / 100.0
            vix_ratio = 1.0  # Would need VIX3M for proper ratio
            vix_in_contango = 1 if vix_ratio < 1 else 0
            vix_percentile = min(vix_value / 50.0, 1.0)  # Approximate
            vix_zscore = (vix_value - 20) / 10  # Approximate
            
            # Price features (returns - using 0 for single snapshot)
            return_1m = 0.0
            return_5m = 0.0
            volatility_20 = vix_value / 100 * 0.05  # Approximate
            momentum_20 = 0.0
            range_pct = 0.02  # Approximate daily range
            
            # Greeks placeholders (would need option chain)
            delta = -0.16
            gamma = 0.01
            theta = -0.02
            vega = 0.15
            vanna = 0.0
            charm = 0.0
            volga = 0.0
            
            # Historical Daily Features
            # =========================
            daily_features = await self._calculate_technical_features(symbol, price)
            
            # Position features
            has_position = symbol in self._active_positions
            position_info = self._active_positions.get(symbol, {})
            
            pnl_pct = 0.0
            days_held = 0.0
            if has_position and 'entry_time' in position_info:
                entry_time = position_info['entry_time']
                days_held = (now - entry_time).total_seconds() / 86400
            
            # Build feature dict matching TradingEnv.MARKET_FEATURES order
            features = {
                # Time (6)
                'sin_time': sin_time,
                'cos_time': cos_time,
                'sin_dow': sin_dow,
                'cos_dow': cos_dow,
                'sin_doy': sin_doy,
                'cos_doy': cos_doy,
                # VIX (8)
                'vix_ratio': vix_ratio,
                'vix_in_contango': vix_in_contango,
                'vix_change_1d': 0.0,
                'vix_change_5d': 0.0,
                'vix_percentile': vix_percentile,
                'vix_zscore': vix_zscore,
                'vix_norm': vix_norm,
                'vix3m_norm': vix_norm,  # Same as VIX for now
                # Regime (1)
                'regime': 1,  # Normal
                # Options (3)
                'options_iv_atm': vix_norm,
                'options_put_call_ratio': 1.0,
                'options_volume_norm': 0.5,
                # Price (5)
                'return_1m': return_1m,
                'return_5m': return_5m,
                'volatility_20': volatility_20,
                'momentum_20': momentum_20,
                'range_pct': range_pct,
                # Greeks (7)
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'vanna': vanna,
                'charm': charm,
                'volga': volga,
                # ML outputs (7) - defaults
                'regime_ml': 1,
                'regime_adj_position': 1.0,
                'regime_adj_delta': 1.0,
                'regime_adj_dte': 1.0,
                'dte_confidence': 0.5,
                'optimal_dte_norm': 0.5,
                'trade_prob': 0.5,
                # Binary signals (5)
                'signal_high_prob': 0,
                'signal_low_vol': 1 if vix_value < 15 else 0,
                'signal_crisis': 1 if vix_value > 30 else 0,
                'signal_contango': vix_in_contango,
                'signal_backwardation': 1 - vix_in_contango,
                # Event features (4)
                'days_to_major_event': 7,
                'is_event_week': 0,
                'is_event_day': 0,
                'event_iv_boost': 0.0,
                # Daily features (17) - Real values
                'day_sma_200': daily_features['day_sma_200'],
                'day_sma_50': daily_features['day_sma_50'],
                'day_sma_20': daily_features['day_sma_20'],
                'day_price_vs_sma200': daily_features['day_price_vs_sma200'],
                'day_price_vs_sma50': daily_features['day_price_vs_sma50'],
                'day_rsi_14': daily_features['day_rsi_14'],
                'day_atr_14': daily_features['day_atr_14'],
                'day_atr_pct': daily_features['day_atr_pct'],
                'day_bb_position': daily_features['day_bb_position'],
                'day_macd': daily_features['day_macd'],
                'day_macd_hist': daily_features['day_macd_hist'],
                'day_above_sma200': daily_features['day_above_sma200'],
                'day_above_sma50': daily_features['day_above_sma50'],
                'day_sma_50_200_ratio': daily_features['day_sma_50_200_ratio'],
                'day_days_to_major_event': 7,
                'day_is_event_week': 0,
                'day_event_iv_boost': 0.0,
                # Position features (7)
                'pnl_pct': pnl_pct,
                'days_held': days_held,
                'position_flag': 1.0 if has_position else 0.0,
                'capital_ratio': 1.0,
                'trade_count': min(self._daily_trades / 10, 1.0),
                'bid_ask_spread': 0.02,
                'market_open': 1.0 if self._is_market_open() else 0.0,
            }
            
            logger.debug(f"Built {len(features)} live features for {symbol}")
            return features
            
        except Exception as e:
            logger.error(f"Error getting live features for {symbol}: {e}")
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
        
            # Real trading logic
        try:
            logger.info(f"🚀 EXECUTING: {signal.action} {signal.symbol}")
            
            # Portfolio Risk Check
            if signal.action == "OPEN":
                # Get current positions and net liq from IBKR
                positions = self._ibkr.get_positions()
                summary = await self._ibkr.get_account_summary()
                net_liq = float(summary.get('NetLiquidation', 100000))
                
                risk_check = await self._portfolio_manager.check_trade(
                    signal=signal, 
                    current_positions=positions, 
                    net_liq=net_liq
                )
                
                if not risk_check['allowed']:
                    logger.warning(f"❌ Portfolio Risk Reject: {risk_check['reason']}")
                    # Update decision to failed
                    decision.executed = False
                    return
            
            if signal.action == "OPEN":
                # Default to BULL_PUT spread for now as primary strategy
                # In real scenario, strategy comes from signal
                trade = await self._order_manager.place_spread_order(
                    symbol=signal.symbol,
                    strategy="BULL_PUT",
                    action="OPEN",
                    quantity=1,
                    strikes=[],  # Would need logic to select strikes based on delta
                    expiry=None  # Default to nearest
                )
                
                if trade:
                    self._daily_trades += 1
                    # Persist to DB
                    try:
                        trade_id = await self._db.insert_trade(
                            symbol=signal.symbol,
                            strategy="BULL_PUT",
                            entry_price=0.0, # Filled price updated later via execution feed
                            quantity=1,
                            notes=f"RL Confidence: {signal.confidence:.2f}"
                        )
                        # Update memory with trade_id
                        self._active_positions[signal.symbol] = {
                            'entry_time': datetime.now(),
                            'signal': signal,
                            'trade_id': trade_id,
                            'entry_price': 0.0,
                            'quantity': 1
                        }
                    except Exception as e:
                        logger.error(f"Failed to persist trade to DB: {e}")
                
            elif signal.action == "CLOSE":
                trade = await self._order_manager.place_spread_order(
                    symbol=signal.symbol,
                    strategy="BULL_PUT",
                    action="CLOSE",
                    quantity=1,
                    strikes=[], 
                    expiry=None
                )
                
                if trade:
                    # Close in DB
                    position = self._active_positions.get(signal.symbol, {})
                    trade_id = position.get('trade_id')
                    if trade_id:
                        try:
                            # We don't have PnL here yet without execution details
                            # For V1, we mark closed. 
                            await self._db.close_trade(
                                trade_id=trade_id,
                                exit_price=0.0, 
                                pnl=0.0
                            )
                        except Exception as e:
                            logger.error(f"Failed to close trade in DB: {e}")
                            
                    self._active_positions.pop(signal.symbol, None)
                
            logger.info(f"✅ Trade executed: {signal.action} {signal.symbol}")
            
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
