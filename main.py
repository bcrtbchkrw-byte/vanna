"""
Vanna - AI Options Trading Bot

Entry point for the trading system.
Integrates IBKR, data pipeline, ML models, and trading strategies.
"""

import asyncio
import signal
import sys
from datetime import datetime

from config import get_config
from core.database import get_database
from core.logger import get_logger, setup_logger

# IBKR
from ibkr.connection import get_ibkr_connection
from ibkr.data_fetcher import get_data_fetcher

# ML & Data
from ml.vanna_data_pipeline import get_vanna_pipeline
from ml.vanna_calculator import get_vanna_calculator
from ml.regime_classifier import get_regime_classifier

# Strategies & Risk
from strategies.strategy_selector import get_strategy_selector
from risk.greeks_validator import get_greeks_validator
from risk.position_sizer import get_position_sizer

# Data Maintenance
from automation.data_maintenance import get_maintenance_manager

# Daily Screener
from analysis.screener import get_daily_screener


# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    logger = get_logger()
    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    _shutdown_requested = True


async def check_ibkr_connection(ibkr_conn, logger) -> bool:
    """Check and reconnect to IBKR if needed."""
    if not ibkr_conn.is_connected:
        logger.warning("⚠️ IBKR disconnected, attempting reconnect...")
        connected = await ibkr_conn.connect()
        if not connected:
            logger.error("❌ Failed to reconnect to IBKR")
            return False
    return True


async def _calculate_smas(symbol: str, current_price: float) -> tuple[float, float, float]:
    """
    Calculate SMA 20, 50, 200 from historical daily data.
    Falls back to approximations if data is missing.
    """
    try:
        from ml.vanna_data_pipeline import get_vanna_pipeline
        pipeline = get_vanna_pipeline()
        
        # Get daily data
        df = pipeline.get_training_data([symbol], timeframe='1day')
        
        if df is None or len(df) < 200:
            # logger not in scope, use get_logger()
            get_logger().warning(f"Insufficient history for {symbol} SMAs, using approximations")
            return current_price * 0.99, current_price * 0.98, current_price * 0.95
            
        # Sort by date
        df = df.sort_values('timestamp')
        closes = df['close'].values
        
        # Calculate SMAs
        import pandas as pd
        series = pd.Series(closes)
        
        # Check if enough data points exist for each window
        sma_20 = series.rolling(window=20).mean().iloc[-1] if len(series) >= 20 else current_price
        sma_50 = series.rolling(window=50).mean().iloc[-1] if len(series) >= 50 else current_price * 0.98
        sma_200 = series.rolling(window=200).mean().iloc[-1] if len(series) >= 200 else current_price * 0.95
        
        # Handle recent IPOs or short history (failsafe)
        if pd.isna(sma_200): sma_200 = sma_50 * 0.95
        if pd.isna(sma_50): sma_50 = sma_20 * 0.98
        if pd.isna(sma_20): sma_20 = current_price
        
        return float(sma_20), float(sma_50), float(sma_200)
        
    except Exception as e:
        get_logger().error(f"Error calculating SMAs: {e}")
        # Fallback
        return current_price * 0.99, current_price * 0.98, current_price * 0.95


async def collect_market_data(data_fetcher, data_pipeline, logger) -> dict:
    """Collect current market data for all symbols."""
    market_data = {}
    
    try:
        # Get VIX
        vix = await data_fetcher.get_vix()
        market_data['vix'] = vix if vix else 18.0
        logger.debug(f"VIX: {market_data['vix']:.2f}")
        
        # Get quotes for main symbols
        for symbol in data_pipeline.SYMBOLS:
            try:
                quote = await data_fetcher.get_stock_quote(symbol)
                if quote:
                    market_data[symbol] = {
                        'price': quote.get('last', quote.get('close', 0)),
                        'bid': quote.get('bid'),
                        'ask': quote.get('ask'),
                        'volume': quote.get('volume', 0)
                    }
                    logger.debug(f"{symbol}: ${market_data[symbol]['price']:.2f}")
            except Exception as e:
                logger.debug(f"Could not fetch {symbol}: {e}")
        
        # Record live bar
        await data_pipeline.record_live_bar()
        
    except Exception as e:
        logger.error(f"Error collecting market data: {e}")
    
    return market_data


async def analyze_market(market_data, regime_classifier, strategy_selector, logger) -> dict:
    """Analyze market conditions and select strategies."""
    analysis = {
        'regime': 'unknown',
        'recommended_strategies': [],
        'vix_level': 'normal',
        'using_simulated_data': False  # Flag for downstream safety checks
    }
    
    try:
        vix = market_data.get('vix', 18.0)
        if vix is None:
            vix = 18.0  # Fallback for market closed
            analysis['using_simulated_data'] = True
            logger.warning("⚠️ VIX data unavailable - using fallback value 18.0")
        
        spy_data = market_data.get('SPY', {})
        current_price = spy_data.get('price')
        
        # Handle None price (market closed / no data)
        if current_price is None or current_price == 0:
            analysis['using_simulated_data'] = True
            logger.warning("⚠️ SPY price unavailable - skipping strategy analysis")
            # Don't use hardcoded price - return early with unknown analysis
            return analysis
        
        # Classify VIX level
        if vix < 15:
            analysis['vix_level'] = 'low'
        elif vix > 25:
            analysis['vix_level'] = 'high'
        else:
            analysis['vix_level'] = 'normal'
        
        # Calculate real SMAs from daily historical data
        sma_20, sma_50, sma_200 = await _calculate_smas('SPY', current_price)
        
        strategies = strategy_selector.select_strategies(
            vix=vix,
            current_price=current_price,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            available_capital=10000  # Will get from account
        )
        
        analysis['recommended_strategies'] = strategies
        if strategies:
            analysis['primary_strategy'] = strategies[0].get('strategy')
            analysis['regime'] = strategies[0].get('regime', 'unknown')
        
        logger.info(f"📊 Market: VIX={vix:.1f} ({analysis['vix_level']}), Regime={analysis['regime']}")
        if analysis.get('primary_strategy'):
            logger.info(f"📈 Primary Strategy: {analysis['primary_strategy']}")
        
    except Exception as e:
        logger.error(f"Error analyzing market: {e}")
    
    return analysis


async def check_positions(ibkr_conn, logger) -> list:
    """Check current positions."""
    positions = []
    
    try:
        if ibkr_conn.is_connected:
            positions = ibkr_conn.get_positions()
            if positions:
                logger.info(f"📋 Current positions: {len(positions)}")
                for pos in positions:
                    logger.debug(f"   {pos.contract.symbol}: {pos.position} @ ${pos.avgCost:.2f}")
    except Exception as e:
        logger.debug(f"Could not fetch positions: {e}")
    
    return positions


async def main_loop(logger, config, db, ibkr_conn, data_fetcher, data_pipeline,
                   regime_classifier, strategy_selector, greeks_validator, position_sizer,
                   maintenance_manager):
    """
    Main trading loop - runs continuously.
    
    Executes every minute:
    1. Check IBKR connection
    2. Collect market data
    3. Analyze market & select strategies
    4. Check existing positions
    5. Calculate Vanna for options
    6. Monthly maintenance (1st of month)
    """
    global _shutdown_requested
    
    loop_interval = 60  # seconds
    iteration = 0
    last_maintenance_day = None
    
    logger.info("🔄 Starting main trading loop...")
    
    while not _shutdown_requested:
        try:
            iteration += 1
            loop_start = datetime.now()
            
            # 0. Morning Screening (9:30-9:35 AM) - Select top 50 options stocks
            today = datetime.now()
            if today.hour == 9 and 30 <= today.minute <= 35:
                screener = get_daily_screener()
                watchlist = screener.get_today_watchlist()
                if not watchlist:
                    logger.info("🌅 Running morning options screening...")
                    try:
                        watchlist = await screener.run_morning_screen()
                        logger.info(f"✅ Today's watchlist: {len(watchlist)} stocks")
                    except Exception as e:
                        logger.error(f"Screening error: {e}")
            
            # 1. Check IBKR connection
            connected = await check_ibkr_connection(ibkr_conn, logger)
            
            if connected:
                # 2. Collect market data
                market_data = await collect_market_data(data_fetcher, data_pipeline, logger)
                
                # 3. Analyze market
                analysis = await analyze_market(
                    market_data, regime_classifier, strategy_selector, logger
                )
                
                # 4. Check positions
                positions = await check_positions(ibkr_conn, logger)
                
                # 5. Calculate Vanna for key options (placeholder for now)
                # TODO: Integrate option chain fetching and Vanna surface
                
            else:
                logger.warning("⏸️ Skipping iteration - IBKR not connected")
            
            # Log heartbeat every 10 iterations (10 minutes)
            if iteration % 10 == 0:
                elapsed = (datetime.now() - loop_start).total_seconds()
                logger.info(f"💓 Heartbeat: iteration {iteration}, loop time {elapsed:.1f}s")
            
            # 6. Monthly maintenance check (1st of month, 00:00-01:00)
            today = datetime.now()
            if today.day == 1 and today.hour == 0 and last_maintenance_day != today.date():
                logger.info("🔧 Running monthly data maintenance...")
                try:
                    await maintenance_manager.run_maintenance()
                    last_maintenance_day = today.date()
                    logger.info("✅ Monthly maintenance complete")
                except Exception as e:
                    logger.error(f"Maintenance error: {e}")
            
            # 7. Saturday training (every Saturday at 6 AM)
            if today.weekday() == 5 and today.hour == 6:
                from automation.saturday_training import should_run_saturday_training, run_saturday_training
                
                # Only run once per Saturday (use maintenance_day tracking)
                if last_maintenance_day != today.date():
                    logger.info("🗓️ SATURDAY: Starting weekly ML/RL training...")
                    try:
                        # Skip RL if not enough data
                        results = await run_saturday_training(skip_rl=False, rl_timesteps=50_000)
                        logger.info(f"✅ Saturday training complete: {results}")
                        last_maintenance_day = today.date()  # Reuse flag to prevent re-run
                    except Exception as e:
                        logger.error(f"Saturday training error: {e}")
            
            # Wait for next iteration
            await asyncio.sleep(loop_interval)
            
        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            await asyncio.sleep(5)
    
    logger.info("Main loop ended")


async def main():
    """Main entry point."""
    global _shutdown_requested
    
    # Initialize configuration
    config = get_config()
    
    # Initialize logger
    setup_logger(level=config.log.level)
    logger = get_logger()
    
    logger.info("=" * 60)
    logger.info("🚀 VANNA - AI Options Trading Bot")
    logger.info("=" * 60)
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        logger.warning("Running in limited mode due to config errors")
    
    # Initialize database
    db = await get_database()
    logger.info("✅ Database initialized")
    
    # Initialize IBKR connection
    logger.info(f"🔌 Connecting to IBKR: {config.ibkr.host}:{config.ibkr.port}")
    ibkr_conn = await get_ibkr_connection()
    connected = await ibkr_conn.connect()
    
    if connected:
        logger.info("✅ IBKR connected")
        try:
            account_summary = await ibkr_conn.get_account_summary()
            logger.info(f"💰 Account: NetLiq=${account_summary.get('NetLiquidation', 0):,.2f}")
        except Exception as e:
            logger.debug(f"Could not get account summary: {e}")
    else:
        logger.warning("⚠️ IBKR not connected - will retry in main loop")
    
    # Initialize components
    data_fetcher = get_data_fetcher()
    data_pipeline = get_vanna_pipeline()
    vanna_calc = get_vanna_calculator()
    regime_classifier = get_regime_classifier()
    strategy_selector = get_strategy_selector()
    greeks_validator = get_greeks_validator()
    position_sizer = get_position_sizer()
    
    logger.info("✅ All components initialized")
    
    # Check/download historical data on startup
    logger.info("🔍 Checking historical data availability...")
    maintenance_manager = get_maintenance_manager()
    await maintenance_manager.ensure_historical_data()
    
    # Log configuration summary
    logger.info(f"📊 Trading Mode: {config.ibkr.trading_mode}")
    logger.info(f"📄 Paper Trading: {config.trading.paper_trading}")
    logger.info(f"💵 Max Risk Per Trade: ${config.trading.max_risk_per_trade}")
    logger.info(f"🎯 Symbols: {', '.join(data_pipeline.SYMBOLS)}")
    
    logger.info("=" * 60)
    logger.info("✅ Initialization Complete - Starting Main Loop")
    logger.info("Press Ctrl+C to stop gracefully")
    logger.info("=" * 60)
    
    try:
        # NEW: Use TradingPipeline instead of old main_loop
        from core.trading_pipeline import get_trading_pipeline
        
        pipeline = get_trading_pipeline()
        
        # Morning routine: Screener → ML → Top 10
        logger.info("🌅 Running morning screening and ML filtering...")
        top_10 = await pipeline.run_morning_routine()
        
        if top_10:
            logger.info(f"🎯 Today's Top 10 for RL: {top_10}")
            
            # Run RL loop continuously during market hours
            await pipeline.run_rl_loop()
        else:
            logger.warning("No stocks selected - running minimal loop")
            
            # Fallback: Run old main_loop for maintenance tasks only
            await main_loop(
                logger=logger,
                config=config,
                db=db,
                ibkr_conn=ibkr_conn,
                data_fetcher=data_fetcher,
                data_pipeline=data_pipeline,
                regime_classifier=regime_classifier,
                strategy_selector=strategy_selector,
                greeks_validator=greeks_validator,
                position_sizer=position_sizer,
                maintenance_manager=maintenance_manager
            )
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up...")
        await ibkr_conn.disconnect()
        await db.disconnect()
        logger.info("👋 Vanna shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
