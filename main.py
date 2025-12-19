"""
Vanna - AI Options Trading Bot

Entry point for the trading system.
Integrates IBKR, data pipeline, ML models, and trading strategies.
"""

import asyncio
import signal
import sys
import os
# CRITICAL STABILITY: Force single-threaded execution for Linear Algebra libs on RPi
# This MUST occur before numpy/torch are imported.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import nest_asyncio
nest_asyncio.apply()
from loguru import logger
import faulthandler
faulthandler.enable()

from config import get_config
from core.database import get_database
from core.logger import setup_logger
from core.trading_pipeline import TradingPipeline
from rl.ppo_agent import TradingAgent
from pathlib import Path

# CRITICAL DEBUG: Try loading PPO here
try:
    logger.info("MAIN: Attempting Early PPO Load...")
    test_agent = TradingAgent(model_path=Path("data/models/ppo_trading_agent"))
    if test_agent.load():
        logger.info("MAIN: Early PPO Load SUCCESS!")
    else:
        logger.error("MAIN: Early PPO Load FAILED (not found?)")
except Exception as e:
    logger.critical(f"MAIN: Early PPO Load CRASHED: {e}")

# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    # We just log it here. The actual shutdown is handled by the asyncio loop
    # or the Pipeline monitoring this flag if we passed it down,
    # but Pipeline has its own internal loop. 
    # Ideal: pipeline.stop()
    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    # sys.exit(0) # Brutal exit for now to ensure container restart or we can try to cancel tasks
    # Proper way: set event
    raise KeyboardInterrupt


async def main():
    """Main entry point."""
    
    # 1. Initialize Configuration
    config = get_config()
    setup_logger(level=config.log.level)
    
    logger.info("=" * 60)
    logger.info("🚀 VANNA - AI Options Trading Bot (Refactored)")
    logger.info("=" * 60)
    
    # 2. Setup Signal Handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 3. Validate Config
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        logger.warning("Running in limited mode due to config errors")
        
    # 4. Initialize Core Services (DB)
    # Pipeline initializes DB internally, but we can do a preliminary check or init here
    # to fail fast.
    try:
        db = await get_database()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}")
        return

    # Initialize Pipeline (Pass pre-loaded agent to avoid crash)
    # pipeline = get_trading_pipeline()
    pipeline = TradingPipeline(agent=test_agent)
    
    # Register Signals
    loop = asyncio.get_running_loop()
    
    try:
        logger.info(f"📊 Trading Mode: {config.ibkr.trading_mode}")
        logger.info(f"📄 Paper Trading: {config.trading.paper_trading}")
        
        # This will run forever until cancelled
        await pipeline.start()
        
    except asyncio.CancelledError:
        logger.info("Main task cancelled")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.critical(f"Fatal error in pipeline: {e}")
        import traceback
        if isinstance(e, SystemExit):
            logger.critical(f"SystemExit caught: {e.code}")
        logger.debug(traceback.format_exc())
    except BaseException as e:
        logger.critical(f"CRITICAL: Uncaught BaseException: {type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    finally:
        logger.info("🧹 Shutting down...")
        await db.disconnect()
        # await pipeline.stop() # If implemented
        logger.info("👋 Vanna shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

