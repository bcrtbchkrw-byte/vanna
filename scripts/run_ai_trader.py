"""
RUN AI TRADER (Demo Script)

Integrates:
- IBKR Data Fetcher
- Live Feature Builder
- Trade Executor (PPO+XGBoost+LSTM)
- Strategy Engine (Execution)

Runs a continuous loop of: Fetch -> Analyze -> Decide -> Execute.
"""
import asyncio
import logging
from core.logger import setup_logger, get_logger
from core.trade_executor import TradeExecutor, Action
from core.strategy_engine import StrategyExecutionEngine
from ibkr.data_fetcher import IBKRDataFetcher
from ml.live_feature_builder import get_live_feature_builder
from config import get_config

async def run_trader():
    setup_logger(level="INFO")
    logger = get_logger()
    
    logger.info("🚀 Starting AI Trader...")
    
    # 1. Initialize Components
    fetcher = IBKRDataFetcher()
    builder = get_live_feature_builder()
    executor = TradeExecutor()
    engine = StrategyExecutionEngine()
    
    # Symbols to trade
    symbols = ['SPY', 'QQQ', 'IWM']
    
    logger.info(f"👀 Watching: {symbols}")
    
    while True:
        for symbol in symbols:
            try:
                logger.info(f"--- Analyzing {symbol} ---")
                
                # A. Fetch Data (Live Features)
                # We need raw data to build features
                # Fetcher needs a method to get ALL raw data needed for builder
                # For MVP, let's use fetcher to get Quote + Options
                
                quote = await fetcher.get_stock_quote(symbol)
                price = (quote['bid'] + quote['ask']) / 2
                
                vix_quote = await fetcher.get_stock_quote('VIX') # Approximate
                vix = (vix_quote['bid'] + vix_quote['ask']) / 2
                
                options_data = await fetcher.get_options_market_data(symbol)
                
                # B. Build Features
                features = builder.build_all_features(
                    symbol=symbol,
                    price=price,
                    vix=vix,
                    quote=quote,
                    options_data=options_data,
                    # daily_features need to be fetched/cached too. 
                    # Assuming builder handles missing daily features via fallback for now.
                )
                
                # C. Decide (The Council)
                decision = await executor.get_trade_decision(symbol, features)
                
                logger.info(f"🧠 Decision for {symbol}: {decision.action.name} ({decision.reason})")
                
                # D. Execute (The Body)
                if decision.action != Action.HOLD:
                    success = await engine.execute_decision(decision)
                    if success:
                        logger.info(f"✅ Trade Executed for {symbol}")
                    else:
                        logger.warning(f"❌ Execution failed for {symbol}")
            
            except Exception as e:
                logger.error(f"Loop error for {symbol}: {e}")
                
        logger.info("💤 Sleeping 60s...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(run_trader())
    except KeyboardInterrupt:
        pass
