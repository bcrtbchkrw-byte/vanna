"""
Saturday Training Pipeline

Orchestrates the complete weekly retraining workflow:
1. Merge live data from SQLite to parquet
2. Recalculate Greeks (vectorized)
3. Train ML models (TradeSuccessPredictor, RegimeClassifier)
4. Enrich data with ML outputs
5. Train RL agent (PPO)

Run every Saturday when markets are closed.
"""
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import time

from core.logger import get_logger, setup_logger

logger = get_logger()


class SaturdayTrainingPipeline:
    """
    Complete retraining pipeline for Saturday execution.
    
    Steps:
    1. merge_live_data()    - SQLite live_bars → *_1min.parquet
    2. calculate_greeks()   - *_1min.parquet → *_1min_vanna.parquet
    3. train_ml_models()    - Train TradeSuccessPredictor, RegimeClassifier
    4. enrich_features()    - *_vanna.parquet → *_rl.parquet
    5. train_rl_agent()     - PPO training on *_rl.parquet
    """
    
    def __init__(
        self,
        data_dir: str = "data/vanna_ml",
        models_dir: str = "data/models",
        rl_timesteps: int = 100_000,
        skip_rl: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.rl_timesteps = rl_timesteps
        self.skip_rl = skip_rl
        
        self.results: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
    
    async def run(self) -> Dict[str, Any]:
        """Execute complete training pipeline."""
        self.start_time = time.time()
        
        logger.info("=" * 70)
        logger.info("🗓️ SATURDAY TRAINING PIPELINE")
        logger.info(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        try:
            # Step 0: Run Data Maintenance (Patch Gaps)
            await self._step_run_data_maintenance()

            # Step 1: Merge live data
            await self._step_merge_live_data()
            
            # Step 2: Calculate Greeks
            await self._step_calculate_greeks()
            
            # Step 2b: Calculate daily features (NEW)
            await self._step_calculate_daily_features()
            
            # Step 2c: Add earnings data (NEW)
            await self._step_add_earnings_data()
            
            # Step 2d: Inject daily features into 1min (NEW)
            await self._step_inject_daily_features()
            
            # Step 3: Train ML models
            await self._step_train_ml_models()
            
            # Step 4: Enrich features
            await self._step_enrich_features()
            
            # Step 5: Train RL agent
            if not self.skip_rl:
                await self._step_train_rl_agent()
            else:
                logger.info("⏭️ Skipping RL training (skip_rl=True)")
            
            self._log_summary()
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            self.results['error'] = str(e)
            raise
        
        return self.results
    
    async def _step_run_data_maintenance(self):
        """Step 0: Run data maintenance (patch gaps)."""
        logger.info("\n" + "=" * 50)
        logger.info("🔧 STEP 0: Run Data Maintenance (Patch Gaps)")
        logger.info("=" * 50)
        
        from automation.data_maintenance import get_maintenance_manager
        
        manager = get_maintenance_manager()
        results = await manager.run_maintenance()
        
        self.results['maintenance'] = results
        logger.info(f"✅ Maintenance complete: {results.get('gaps_patched', 0)} gaps patched")

    async def _step_merge_live_data(self):
        """Step 1: Merge live SQLite data to parquet."""
        logger.info("\n" + "=" * 50)
        logger.info("📥 STEP 1: Merge Live Data")
        logger.info("=" * 50)
        
        from automation.data_maintenance import get_maintenance_manager
        
        manager = get_maintenance_manager()
        merge_results = await manager.merge_live_to_parquet()
        
        self.results['merge'] = merge_results
        total_new = sum(v for v in merge_results.values() if v > 0)
        logger.info(f"✅ Merged {total_new} new bars")
    
    async def _step_calculate_greeks(self):
        """Step 2: Calculate Greeks using vectorized calculator."""
        logger.info("\n" + "=" * 50)
        logger.info("📐 STEP 2: Calculate Greeks")
        logger.info("=" * 50)
        
        from ml.vectorized_greeks import VectorizedGreeksCalculator
        from ml.yahoo_earnings import get_yahoo_earnings_fetcher
        
        calculator = VectorizedGreeksCalculator()
        fetcher = get_yahoo_earnings_fetcher()
        symbols_processed = []
        
        # Process all *_1min.parquet files
        for parquet_file in self.data_dir.glob("*_1min.parquet"):
            if "_vanna" in parquet_file.name or "_rl" in parquet_file.name:
                continue
            
            output_path = self.data_dir / f"{parquet_file.stem}_vanna.parquet"
            symbol = parquet_file.stem.split('_')[0]
            
            try:
                # Fetch dividend yield
                div_yield = fetcher.get_dividend_yield(symbol)
                
                calculator.process_parquet_file(
                    str(parquet_file), 
                    str(output_path),
                    dividend_yield=div_yield
                )
                
                symbols_processed.append(symbol)
                logger.info(f"   ✅ {symbol} (div: {div_yield:.2%})")
            except Exception as e:
                logger.error(f"   ❌ {parquet_file.name}: {e}")
        
        self.results['greeks'] = symbols_processed
        logger.info(f"✅ Greeks calculated for {len(symbols_processed)} symbols")
    
    async def _step_calculate_daily_features(self):
        """Step 2b: Calculate daily technical indicators (SMA200, RSI, etc.)."""
        logger.info("\n" + "=" * 50)
        logger.info("📊 STEP 2b: Calculate Daily Features")
        logger.info("=" * 50)
        
        try:
            from ml.daily_feature_calculator import get_daily_feature_calculator
            
            calculator = get_daily_feature_calculator()
            results = calculator.calculate_all()
            
            self.results['daily_features'] = results
            success = sum(results.values())
            logger.info(f"✅ Daily features calculated for {success}/{len(results)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Daily features failed: {e}")
            self.results['daily_features'] = {'error': str(e)}
    
    async def _step_add_earnings_data(self):
        """Step 2c: Add major event data (FOMC, CPI, mega-cap earnings)."""
        logger.info("\n" + "=" * 50)
        logger.info("📅 STEP 2c: Add Major Events Data")
        logger.info("   SPY/QQQ: AAPL,MSFT,NVDA,AMZN,GOOGL earnings")
        logger.info("   TLT/GLD: FOMC meetings, CPI releases")
        logger.info("=" * 50)
        
        try:
            from ml.earnings_data_fetcher import get_major_events_calculator
            
            calculator = get_major_events_calculator()
            results = calculator.calculate_all()
            
            self.results['major_events'] = results
            success = sum(results.values())
            logger.info(f"✅ Major events added for {success}/{len(results)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Major events failed: {e}")
            self.results['major_events'] = {'error': str(e)}
    
    async def _step_inject_daily_features(self):
        """Step 2d: Inject daily features into 1min data (NO LOOK-AHEAD!)."""
        logger.info("\n" + "=" * 50)
        logger.info("💉 STEP 2d: Inject Daily Features into 1-min Data")
        logger.info("   ⚠️ Using yesterday's data to prevent look-ahead bias")
        logger.info("=" * 50)
        
        try:
            from ml.daily_feature_injector import get_daily_feature_injector
            
            injector = get_daily_feature_injector()
            results = injector.inject_all(target_suffix="1min_vanna")
            
            self.results['daily_injection'] = results
            success = sum(results.values())
            logger.info(f"✅ Daily features injected for {success}/{len(results)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Daily injection failed: {e}")
            self.results['daily_injection'] = {'error': str(e)}
    
    async def _step_train_ml_models(self):
        """Step 3: Train ML models."""
        logger.info("\n" + "=" * 50)
        logger.info("🧠 STEP 3: Train ML Models")
        logger.info("=" * 50)
        
        ml_results = {}
        
        # 3a. Train TradeSuccessPredictor
        try:
            from ml.trade_success_predictor import TradeSuccessPredictor
            import pandas as pd
            import numpy as np
            
            logger.info("   Training TradeSuccessPredictor...")
            
            # Load all vanna data
            dfs = []
            # Load all vanna data
            dfs = []
            for f in self.data_dir.glob("*_1min_vanna.parquet"):
                try:
                    df = pd.read_parquet(f)
                    
                    # Calculate FUTURE return (look ahead 5 minutes) for correct labeling
                    # Must be done per-file (per-symbol) to avoid cross-symbol shifting
                    if 'close' in df.columns:
                        # Future return = (Close[t+5] / Close[t]) - 1
                        df['future_close'] = df['close'].shift(-5)
                        df['future_return_5m'] = (df['future_close'] / df['close']) - 1
                        
                        # Target: Did price go up in next 5 mins?
                        df['is_successful'] = (df['future_return_5m'] > 0).astype(int)
                        
                        # Drop last 5 rows where future return is unknown
                        df = df.dropna(subset=['future_return_5m'])
                        
                        dfs.append(df)
                except Exception as e:
                    logger.warning(f"Skipping {f.name}: {e}")
            
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                
                # Old logic removed (was using return_5m)
                
                predictor = TradeSuccessPredictor()
                metrics = predictor.train(combined, target_col='is_successful')
                ml_results['trade_predictor'] = metrics
                logger.info(f"   ✅ TradeSuccessPredictor: {metrics.get('accuracy', 0):.2%}")
            
        except Exception as e:
            logger.error(f"   ❌ TradeSuccessPredictor failed: {e}")
            ml_results['trade_predictor'] = {'error': str(e)}
        
        # 3b. Train RegimeClassifier
        try:
            from ml.regime_classifier import RegimeClassifier
            import pandas as pd
            
            logger.info("   Training RegimeClassifier...")
            
            # Load vanna data
            dfs = []
            for f in self.data_dir.glob("*_1min_vanna.parquet"):
                df = pd.read_parquet(f)
                dfs.append(df)
            
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                
                # Synthetic labels from VIX levels
                def vix_to_regime(vix):
                    if vix < 15: return 0
                    if vix < 20: return 1
                    if vix < 25: return 2
                    if vix < 35: return 3
                    return 4
                
                combined['regime_target'] = combined['vix'].apply(vix_to_regime)
                
                classifier = RegimeClassifier()
                metrics = classifier.train(combined, target_col='regime_target')
                ml_results['regime_classifier'] = metrics
                logger.info(f"   ✅ RegimeClassifier: {metrics.get('accuracy', 0):.2%}")
            
        except Exception as e:
            logger.error(f"   ❌ RegimeClassifier failed: {e}")
            ml_results['regime_classifier'] = {'error': str(e)}
        
        self.results['ml_training'] = ml_results
    
    async def _step_enrich_features(self):
        """Step 4: Enrich data with ML outputs."""
        logger.info("\n" + "=" * 50)
        logger.info("🔧 STEP 4: Enrich Features")
        logger.info("=" * 50)
        
        from ml.feature_enricher import FeatureEnricher
        
        enricher = FeatureEnricher()
        enrich_results = enricher.process_all()
        
        self.results['enrich'] = enrich_results
        logger.info(f"✅ Enriched {len(enrich_results)} symbol files")
    
    async def _step_train_rl_agent(self):
        """Step 5: Train RL agent."""
        logger.info("\n" + "=" * 50)
        logger.info("🤖 STEP 5: Train RL Agent")
        logger.info("=" * 50)
        
        try:
            from rl.ppo_agent import TradingAgent
            from rl.vec_env import get_available_symbols
            
            symbols = get_available_symbols()
            
            if not symbols:
                logger.warning("No RL data available")
                self.results['rl_training'] = {'error': 'No data'}
                return
            
            logger.info(f"   Symbols: {symbols}")
            logger.info(f"   Timesteps: {self.rl_timesteps:,}")
            
            agent = TradingAgent()
            agent.create_env(symbols=symbols)
            agent.create_model()
            
            train_result = agent.train(
                total_timesteps=self.rl_timesteps,
                eval_freq=10_000,
                checkpoint_freq=25_000
            )
            
            self.results['rl_training'] = train_result
            logger.info(f"✅ RL training complete: {train_result}")
            
        except Exception as e:
            logger.error(f"❌ RL training failed: {e}")
            self.results['rl_training'] = {'error': str(e)}
    
    def _log_summary(self):
        """Log final summary."""
        elapsed = time.time() - self.start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"   Duration: {elapsed/60:.1f} minutes")
        logger.info(f"   Merge: {self.results.get('merge', {})}")
        logger.info(f"   Greeks: {self.results.get('greeks', [])}")
        logger.info(f"   ML: {self.results.get('ml_training', {})}")
        logger.info(f"   Enrich: {self.results.get('enrich', {})}")
        logger.info(f"   RL: {self.results.get('rl_training', {})}")
        logger.info("=" * 70)


def should_run_saturday_training() -> bool:
    """Check if today is Saturday."""
    return datetime.now().weekday() == 5


async def run_saturday_training(
    skip_rl: bool = False,
    rl_timesteps: int = 100_000
) -> Dict[str, Any]:
    """
    Run complete Saturday training pipeline.
    
    Args:
        skip_rl: Skip RL training (useful for quick tests)
        rl_timesteps: Number of RL training steps
        
    Returns:
        Results dict
    """
    pipeline = SaturdayTrainingPipeline(
        skip_rl=skip_rl,
        rl_timesteps=rl_timesteps
    )
    return await pipeline.run()


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Saturday Training Pipeline")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL training")
    parser.add_argument("--rl-steps", type=int, default=100_000, help="RL timesteps")
    parser.add_argument("--force", action="store_true", help="Run even if not Saturday")
    args = parser.parse_args()
    
    try:
        setup_logger(level="INFO")
    except:
        pass
    
    logger = get_logger()
    
    # Check if Saturday
    if not args.force and not should_run_saturday_training():
        logger.warning(f"Today is not Saturday (weekday={datetime.now().weekday()})")
        logger.info("Use --force to run anyway")
        exit(0)
    
    # Run pipeline
    asyncio.run(run_saturday_training(
        skip_rl=args.skip_rl,
        rl_timesteps=args.rl_steps
    ))
