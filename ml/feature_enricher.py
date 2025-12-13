"""
Feature Enrichment for RL Training Data

Adds ML model outputs to parquet files:
- RegimeClassifier: regime, regime_adjustments
- DTEOptimizer: optimal_dte, dte_confidence
- TradeSuccessPredictor: trade_prob

Creates enriched *_1min_rl.parquet files ready for RL training.
"""
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np

from ml.regime_classifier import get_regime_classifier
from ml.dte_optimizer import get_dte_optimizer
from ml.trade_success_predictor import get_trade_success_predictor
from core.logger import get_logger

logger = get_logger()


class FeatureEnricher:
    """
    Enriches parquet data with ML model outputs for RL.
    
    Adds columns:
    - regime_ml: ML-classified regime (0-4)
    - regime_adj_position: Position size multiplier
    - regime_adj_delta: Delta target adjustment
    - optimal_dte: DTEOptimizer recommendation
    - dte_confidence: Confidence in DTE recommendation
    - trade_prob: TradeSuccessPredictor probability
    """
    
    def __init__(self, data_dir: str = "data/vanna_ml"):
        self.data_dir = Path(data_dir)
        
        # Initialize ML models
        self.regime_clf = get_regime_classifier()
        self.dte_opt = get_dte_optimizer()
        
        try:
            self.trade_predictor = get_trade_success_predictor()
        except Exception:
            self.trade_predictor = None
            logger.warning("TradeSuccessPredictor not available")
        
        logger.info("FeatureEnricher initialized")
    
    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ML features to DataFrame.
        
        Args:
            df: DataFrame with VIX, Greeks, etc.
            
        Returns:
            Enriched DataFrame
        """
        df = df.copy()
        n = len(df)
        
        logger.info(f"Enriching {n:,} rows...")
        
        # ================================================================
        # 1. RegimeClassifier outputs
        # ================================================================
        regimes = []
        adj_positions = []
        adj_deltas = []
        adj_dtes = []
        
        for idx in range(n):
            row = df.iloc[idx]
            
            # Get regime
            vix = row.get('vix', 18)
            result = self.regime_clf.classify_by_vix(float(vix))
            regimes.append(result.regime)
            
            # Get adjustments
            adj = self.regime_clf.get_strategy_adjustment(result.regime)
            adj_positions.append(adj['position_size'])
            adj_deltas.append(adj['delta_target'])
            adj_dtes.append(adj['dte_adjustment'])
        
        df['regime_ml'] = regimes
        df['regime_adj_position'] = adj_positions
        df['regime_adj_delta'] = adj_deltas
        df['regime_adj_dte'] = adj_dtes
        
        logger.info("Added regime features")
        
        # ================================================================
        # 2. DTEOptimizer outputs (vectorized)
        # ================================================================
        vix_vals = df['vix'].fillna(18).values
        vix3m_vals = df['vix3m'].fillna(df['vix'] * 1.05).values if 'vix3m' in df.columns else vix_vals * 1.05
        
        optimal_dtes = []
        dte_confidences = []
        
        for vix, vix3m in zip(vix_vals, vix3m_vals):
            result = self.dte_opt.get_optimal_dte(float(vix), float(vix3m))
            optimal_dtes.append(result.dte)
            dte_confidences.append(result.confidence)
        
        df['optimal_dte'] = optimal_dtes
        df['dte_confidence'] = dte_confidences
        
        logger.info("Added DTE features")
        
        # ================================================================
        # 3. TradeSuccessPredictor outputs (batch)
        # ================================================================
        if self.trade_predictor is not None and self.trade_predictor.model is not None:
            try:
                probs = self.trade_predictor.predict_batch(df)
                df['trade_prob'] = probs
                logger.info("Added trade probability")
            except Exception as e:
                logger.warning(f"Could not add trade_prob: {e}")
                df['trade_prob'] = 0.5
        else:
            df['trade_prob'] = 0.5
        
        # ================================================================
        # 4. Derived features for RL
        # ================================================================
        
        # Binary signals
        df['signal_high_prob'] = (df['trade_prob'] >= 0.6).astype(int)
        df['signal_low_vol'] = (df['regime_ml'] == 0).astype(int)
        df['signal_crisis'] = (df['regime_ml'] == 4).astype(int)
        
        # VIX term structure signal
        if 'vix_ratio' in df.columns:
            df['signal_contango'] = (df['vix_ratio'] < 0.95).astype(int)
            df['signal_backwardation'] = (df['vix_ratio'] > 1.05).astype(int)
        
        # ================================================================
        # 5. Earnings features (options behave differently near earnings!)
        # ================================================================
        # Note: For historical data, we estimate based on quarterly patterns
        # Real-time uses actual earnings calendar
        
        if 'timestamp' in df.columns or 'datetime' in df.columns:
            time_col = 'timestamp' if 'timestamp' in df.columns else 'datetime'
            
            # Simulate quarterly earnings (every ~90 days)
            # Most stocks report Jan, Apr, Jul, Oct
            earnings_months = [1, 4, 7, 10]  # Earnings months
            
            days_to_earnings = []
            for idx in range(n):
                ts = pd.Timestamp(df.iloc[idx][time_col])
                month = ts.month
                day = ts.day
                
                # Find next earnings month
                next_earnings = None
                for em in earnings_months:
                    if em > month or (em == month and day < 15):
                        next_earnings = em
                        break
                if next_earnings is None:
                    next_earnings = earnings_months[0]  # January next year
                
                # Approximate days (simplified)
                if next_earnings > month:
                    days = (next_earnings - month) * 30 - day + 15
                else:
                    days = (12 - month + next_earnings) * 30 - day + 15
                
                days_to_earnings.append(max(0, min(days, 90)))
            
            df['days_to_earnings'] = days_to_earnings
            df['is_earnings_week'] = (df['days_to_earnings'] <= 7).astype(int)
            df['is_earnings_month'] = (df['days_to_earnings'] <= 30).astype(int)
            
            # IV typically spikes before earnings (use as multiplier hint)
            # Near earnings (0-7 days): High IV expected
            # Post earnings window: IV crush expected
            df['earnings_iv_multiplier'] = 1.0
            df.loc[df['days_to_earnings'] <= 7, 'earnings_iv_multiplier'] = 1.5
            df.loc[df['days_to_earnings'] <= 3, 'earnings_iv_multiplier'] = 2.0
            df.loc[(df['days_to_earnings'] > 60) & (df['days_to_earnings'] <= 75), 'earnings_iv_multiplier'] = 0.9
            
            logger.info("Added earnings features (days_to_earnings, is_earnings_week, earnings_iv_multiplier)")
        else:
            # No timestamp - use default values
            df['days_to_earnings'] = 45  # Middle of quarter
            df['is_earnings_week'] = 0
            df['is_earnings_month'] = 0
            df['earnings_iv_multiplier'] = 1.0
            logger.warning("No timestamp column - using default earnings features")
        
        logger.info(f"✅ Enriched {n:,} rows with {len(df.columns)} total columns")
        
        return df
    
    def normalize_for_rl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features for RL training.
        
        - Removes metadata columns (timestamp, symbol, OHLCV, timeframe)
        - Normalizes VIX and volume using z-score
        - Normalizes DTE to 0-1 range
        - Ensures all features are comparable across symbols
        
        Args:
            df: Enriched DataFrame
            
        Returns:
            RL-ready DataFrame with normalized features only
        """
        df = df.copy()
        
        # ================================================================
        # 1. Drop metadata columns (not features)
        # ================================================================
        drop_cols = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'symbol', 'timeframe', 'regime_label', 'id'
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        logger.info(f"Dropped metadata columns, {len(df.columns)} remain")
        
        # ================================================================
        # 2. Normalize VIX-related (z-score within dataset)
        # ================================================================
        vix_cols = ['vix', 'vix3m']
        for col in vix_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[f'{col}_norm'] = (df[col] - mean) / std
                else:
                    df[f'{col}_norm'] = 0
                df = df.drop(columns=[col])
        
        # ================================================================
        # 3. Normalize DTE to 0-1 range
        # ================================================================
        if 'optimal_dte' in df.columns:
            df['optimal_dte_norm'] = df['optimal_dte'] / 60.0  # Max DTE is 60
            df = df.drop(columns=['optimal_dte'])
        
        # ================================================================
        # 4. Normalize volume (z-score, often has huge variance)
        # ================================================================
        if 'options_volume' in df.columns:
            mean = df['options_volume'].mean()
            std = df['options_volume'].std()
            if std > 0:
                df['options_volume_norm'] = (df['options_volume'] - mean) / std
            else:
                df['options_volume_norm'] = 0
            df = df.drop(columns=['options_volume'])
        
        # ================================================================
        # 5. Clip outliers (prevent extreme values)
        # ================================================================
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].clip(-10, 10)
        
        # ================================================================
        # 6. Fill any remaining NaN
        # ================================================================
        df = df.fillna(0)
        
        logger.info(f"✅ Normalized: {len(df.columns)} RL features")
        
        return df
    
    def process_file(
        self,
        input_path: str,
        output_path: str = None,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Process single parquet file.
        
        Args:
            input_path: Path to *_vanna.parquet
            output_path: Path to output file (default: *_rl.parquet)
            normalize: Whether to normalize for RL (removes OHLC, normalizes values)
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Not found: {input_path}")
        
        # Load
        logger.info(f"Loading {input_path}...")
        df = pd.read_parquet(input_path)
        
        # Enrich with ML features
        df = self.enrich_dataframe(df)
        
        # Normalize for RL (drop OHLC, normalize values)
        if normalize:
            df = self.normalize_for_rl(df)
        
        # Save
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem.replace('_vanna', '_rl')}.parquet"
        else:
            output_path = Path(output_path)
        
        df.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"💾 Saved to {output_path} ({len(df.columns)} columns)")
        
        return df
    
    def process_all(self) -> Dict[str, int]:
        """Process all *_vanna.parquet files."""
        results = {}
        
        vanna_files = list(self.data_dir.glob("*_1min_vanna.parquet"))
        
        if not vanna_files:
            logger.warning("No *_vanna.parquet files found")
            return results
        
        logger.info(f"Found {len(vanna_files)} files to process")
        
        for filepath in vanna_files:
            try:
                df = self.process_file(str(filepath))
                symbol = filepath.stem.split('_')[0]
                results[symbol] = len(df)
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
        
        return results


def enrich_all_data():
    """Convenience function to enrich all parquet files."""
    enricher = FeatureEnricher()
    return enricher.process_all()


# CLI
if __name__ == "__main__":
    from core.logger import setup_logger
    
    try:
        setup_logger(level="INFO")
    except:
        pass
    
    print("=" * 60)
    print("Feature Enrichment for RL")
    print("=" * 60)
    
    enricher = FeatureEnricher()
    results = enricher.process_all()
    
    print("\nResults:")
    for symbol, rows in results.items():
        print(f"  {symbol}: {rows:,} rows")
    
    print("\n✅ Enrichment complete!")
