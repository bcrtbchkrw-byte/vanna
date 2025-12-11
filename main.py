"""
Vanna - AI Options Trading Bot

Entry point for the trading system.
"""

import asyncio

from config import get_config
from core.database import get_database
from core.logger import get_logger, setup_logger


async def main():
    """Main entry point."""
    # Initialize configuration
    config = get_config()
    
    # Initialize logger
    setup_logger(level=config.log.level)
    logger = get_logger()
    
    logger.info("=" * 60)
    logger.info("🚀 VANNA - AI Options Trading Bot")
    logger.info("=" * 60)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        logger.warning("Running in limited mode due to config errors")
    
    # Initialize database
    db = await get_database()
    logger.info("Database initialized")
    
    # Log configuration summary
    logger.info(f"IBKR Host: {config.ibkr.host}:{config.ibkr.port}")
    logger.info(f"Trading Mode: {config.ibkr.trading_mode}")
    logger.info(f"Paper Trading: {config.trading.paper_trading}")
    logger.info(f"Max Risk Per Trade: ${config.trading.max_risk_per_trade}")
    
    # TODO: Phase 2 - IBKR Connection
    # TODO: Phase 3 - Market Analysis
    # TODO: Phase 4 - AI Clients
    # TODO: Phase 5 - Risk Management
    # TODO: Phase 6 - Strategies
    # TODO: Phase 7 - Order Execution
    # TODO: Phase 8 - Automation
    
    logger.info("=" * 60)
    logger.info("Phase 1 Complete - Foundation Ready")
    logger.info("=" * 60)
    
    # Cleanup
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
