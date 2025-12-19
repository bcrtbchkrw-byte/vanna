"""
Trading Pipeline Orchestrator

Coordinates the complete trading workflow:
1. Initializes Data, Analysis, and Execution layers.
2. Runs the main analysis loop (Watchlist -> Features -> TradeExecutor -> StrategyEngine).
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, date, time
import asyncio
from zoneinfo import ZoneInfo

from loguru import logger

# New Core Components
from core.trade_executor import TradeExecutor, Action, TradeDecision
from core.strategy_engine import StrategyExecutionEngine
from ibkr.data_fetcher import IBKRDataFetcher
from ml.live_feature_builder import get_live_feature_builder
from core.scheduler import get_scheduler

class TradingPipeline:
    """
    Main trading pipeline orchestrator (The Central Nervous System).
    
    Flow:
    - Continuous Loop:
        - Fetch Market Data (Quote, Options, VIX)
        - Build Live Features (Market + Position)
        - TradeExecutor Decide (PPO + XGBoost + LSTM)
        - StrategyEngine Execute (Open/Close Position)
    """
    
    def __init__(self, agent=None):
        # 1. Initialize Components
        self.fetcher = IBKRDataFetcher()
        self.builder = get_live_feature_builder()
        self.executor = TradeExecutor(agent=agent)
        self.engine = StrategyExecutionEngine()
        self.scheduler = get_scheduler()
        
        # 2. State
        self.symbols = ['SPY', 'QQQ', 'IWM'] # Default Watchlist
        self._loop_delay = 60 # Check every 60s
        self._is_running = False
        
        logger.info("🧠 TradingPipeline initialized with TradeExecutor & StrategyEngine")
        
    async def start(self):
        """Start the main trading loop."""
        logger.info("🚀 Starting Trading Pipeline...")
        self._is_running = True
        
        # Ensure connection
        conn = await self.fetcher._get_connection()
        if not await conn.connect():
            logger.error("❌ Initial connection to IBKR failed! Will retry...")
            # We don't exit, just loop and hope connection.py auto-reconnects or we retry
        else:
            logger.info("✅ Pipeline connected to IBKR")
        
        while self._is_running:
            try:
                # 0. Check Connection & Reconnect if needed
                if not conn.is_connected:
                    logger.warning("⚠️ IBKR not connected. Attempting reconnection...")
                    # We use connect() because reconnect() forces disconnect first, 
                    # but if we are already disconnected, connect() is fine. 
                    # Actually connect() creates new IB instance, so maybe cleaner to ensure clean slate.
                    # But connect() has built-in retries.
                    if not await conn.connect():
                        logger.error("❌ Reconnection failed. Sleeping before retry...")
                        await asyncio.sleep(self._loop_delay)
                        continue
                    else:
                        logger.info("✅ Reconnected to IBKR successfully")

                # 1. Process Watchlist
                for symbol in self.symbols:
                    await self._process_symbol(symbol)
                    
                logger.info(f"💤 Sleeping {self._loop_delay}s...")
                await asyncio.sleep(self._loop_delay)
                
            except asyncio.CancelledError:
                logger.info("Pipeline stopped (Cancelled)")
                break
            except Exception as e:
                logger.error(f"Pipeline Loop Error: {e}")
                await asyncio.sleep(10) # Backoff
                
    async def _process_symbol(self, symbol: str):
        """Process a single symbol through the pipeline."""
        try:
            logger.debug(f"--- Analyzing {symbol} ---")
            
            # A. Fetch Data (Live Features)
            quote = await self.fetcher.get_stock_quote(symbol)
            if not quote:
                # logger.warning(f"No quote for {symbol}")
                return
                
            # Safe extraction of price
            bid = quote.get('bid') or 0
            ask = quote.get('ask') or 0
            price = (bid + ask) / 2
            if price <= 0: 
                price = quote.get('last') or 0
            
            # VIX (Global Market Risk) - Use dedicated method
            vix = await self.fetcher.get_vix()
            if not vix or vix <= 0: 
                vix = 15.0 # Fallback
            
            # Options Stats (OI, Volume)
            options_data = await self.fetcher.get_options_market_data(symbol)
            
            # Position Data
            positions = await self.engine.execution_client.get_current_positions()
            has_position = any(p['symbol'] == symbol for p in positions)
            
            # B. Build Features
            features = self.builder.build_all_features(
                symbol=symbol,
                price=price,
                vix=vix,
                quote=quote,
                options_data=options_data,
                has_position=has_position,
                # daily_features=... (Defaulting to 0s for now/MVP)
            )
            
            # C. Decide (The Council)
            decision = await self.executor.get_trade_decision(symbol, features)
            
            if decision.action != Action.HOLD:
                logger.info(f"🧠 {symbol} Decision: {decision.action.name} ({decision.reason})")
                
                # D. Execute (The Body)
                success = await self.engine.execute_decision(decision)
                if success:
                    logger.info(f"✅ Trade Executed for {symbol}")
            else:
                # Log HOLD reasoning occasionally
                # logger.debug(f"{symbol}: HOLD ({decision.reason})")
                pass
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    # Helper function for main.py compatibility
    async def stop(self):
        self._is_running = False

def get_trading_pipeline() -> TradingPipeline:
    """Factory method."""
    return TradingPipeline()
