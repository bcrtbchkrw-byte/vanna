
"""
Vanna Weekend Update Script 🛠️

Run this script on weekends (e.g., Saturday) to:
1. Update historical data from IBKR (fetch latest bars)
2. Retrain the RL model on the new data

Usage:
    python3 run_weekend_update.py
"""
import asyncio
import sys
from loguru import logger
from core.logger import setup_logger

# Components
from ml.vanna_data_pipeline import get_vanna_pipeline
from rl.ppo_agent import TradingAgent, get_available_symbols

async def main():
    setup_logger(level="INFO")
    
    logger.info("=" * 60)
    logger.info("🛠️ VANNA WEEKEND UPDATE UTILITY")
    logger.info("=" * 60)
    
    # =========================================================================
    # STEP 1: UPDATE DATA
    # =========================================================================
    logger.info("\n⬇️ STEP 1: UPDATING HISTORICAL DATA")
    logger.info("   Connecting to IBKR...")
    
    try:
        pipeline = get_vanna_pipeline()
        
        # Ensure connection
        conn = await pipeline._get_connection()
        if not conn.is_connected:
            await conn.connect()
            await asyncio.sleep(2) # Wait for handshake
            
        if not conn.is_connected:
            logger.error("❌ Could not connect to IBKR. Please ensure TWS/Gateway is running.")
            sys.exit(1)
            
        logger.info("   Fetch started (this may take 10-20 minutes)...")
        # Update both Intraday (550 days) and Daily (10 years)
        await pipeline.fetch_all_historical(days=550, years=10)
        
        logger.info("✅ Data update complete!")
        
    except Exception as e:
        logger.critical(f"❌ Data update failed: {e}")
        sys.exit(1)

    # =========================================================================
    # STEP 2: RETRAIN MODEL
    # =========================================================================
    logger.info("\n🧠 STEP 2: RETRAINING RL MODEL")
    
    # Verify data exists
    symbols = get_available_symbols()
    logger.info(f"   Found training data for: {symbols}")
    
    if not symbols:
        logger.error("❌ No data found after update! Aborting training.")
        sys.exit(1)
        
    try:
        agent = TradingAgent(
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64, 
            n_epochs=10
        )
        
        logger.info("   Creating environment...")
        agent.create_env(symbols=symbols)
        
        logger.info("   Starting training (100,000 steps)...")
        agent.train(
            total_timesteps=100_000,
            eval_freq=10_000,
            checkpoint_freq=50_000
        )
        
        logger.info("✅ Model retraining complete!")
        logger.info(f"   Saved to: {agent.model_path}")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user.")
        agent.save()
    except Exception as e:
        logger.critical(f"❌ Training failed: {e}")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("🎉 WEEKEND ROUTINE COMPLETE!")
    logger.info("   You can now start the bot normally: python3 main.py")
    logger.info("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
