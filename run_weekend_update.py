
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
    # STEP 1: UPDATE DATA (Smart - only missing symbols)
    # =========================================================================
    logger.info("\n⬇️ STEP 1: UPDATING HISTORICAL DATA")
    logger.info("   Using smart download (only missing symbols)...")
    
    try:
        from automation.data_maintenance import get_maintenance_manager
        manager = get_maintenance_manager()
        
        # Step 1a: Ensure all symbols have historical data
        logger.info("   Checking for missing symbols...")
        await manager.ensure_historical_data()
        
        # Step 1b: Merge any accumulated live data
        logger.info("   Merging live data to parquet...")
        await manager.merge_live_to_parquet()
        
        # Step 1c: Check for gaps and patch them
        logger.info("   Checking for data gaps...")
        await manager.run_maintenance()
        
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
        
        logger.info("   Starting training (2,000,000 steps)...")
        agent.train(
            total_timesteps=2_000_000,
            eval_freq=50_000,
            checkpoint_freq=500_000
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
