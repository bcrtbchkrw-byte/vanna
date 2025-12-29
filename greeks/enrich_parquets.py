#!/usr/bin/env python3
"""
enrich_parquets.py

Greeks & DTE Enrichment Pipeline for Multi-Strategy PPO Training.

Takes raw parquets from jedem9.py and adds:
- Implied Volatility (FAST vectorized Newton-Raphson)
- All Greeks (Delta, Gamma, Theta, Vega, Vanna, Charm, Volga...)
- DTE features for Multi-Strategy trading
- ML-normalized features

Usage:
    python enrich_parquets.py --input data/ --output data/enriched/

The script will process all {SYMBOL}_{YEAR}.parquet files.

CHANGELOG:
- Fixed: Vectorized IV calculation (100x faster)
- Fixed: 0DTE detection (uses original_dte before fractional conversion)
- Fixed: Uses correct greeks_engine.py API
- Added: Progress logging with ETA
- Added: Memory-efficient chunked processing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import logging
import gc
import sys
import time

# Add greeks directory to path
sys.path.insert(0, str(Path(__file__).parent))

from greeks_engine import GreeksEngine, create_engine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParquetEnricher:
    """Enriches raw ThetaData parquets with Greeks and DTE features."""
    
    # Valid rate bounds for sanity checks
    RATE_BOUNDS = {
        'r': (0.0, 0.15),   # Risk-free rate: 0% - 15%
        'q': (0.0, 0.10),   # Dividend yield: 0% - 10%
    }
    
    def __init__(self, use_rates_provider: bool = True, n_workers: int = 1):
        """
        Args:
            use_rates_provider: If True, fetch SOFR rates dynamically.
                               If False, use default rates (faster).
            n_workers: Number of parallel workers (1 = sequential)
        """
        self.n_workers = n_workers
        
        if use_rates_provider:
            try:
                self.engine = create_engine()
                logger.info("GreeksEngine initialized with RatesProvider")
            except Exception as e:
                logger.warning(f"Failed to init RatesProvider: {e}. Using defaults.")
                self.engine = GreeksEngine()
        else:
            self.engine = GreeksEngine()
            logger.info("GreeksEngine initialized with default rates")
    
    def validate_rates(self, r: float, q: float) -> bool:
        """Validate rates are within reasonable bounds.
        
        Returns:
            True if valid, raises ValueError if invalid
        """
        r_min, r_max = self.RATE_BOUNDS['r']
        q_min, q_max = self.RATE_BOUNDS['q']
        
        if not (r_min <= r <= r_max):
            raise ValueError(f"Risk-free rate r={r} outside valid range [{r_min}, {r_max}]")
        
        if not (q_min <= q <= q_max):
            raise ValueError(f"Dividend yield q={q} outside valid range [{q_min}, {q_max}]")
        
        return True
    
    def calculate_dte(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Days To Expiration from timestamp and expiration.
        
        For 0DTE options, uses time-of-day to estimate fractional DTE
        based on hours until market close (4pm ET).
        
        Adds:
            - dte: Fractional DTE for calculations (0DTE = fraction of day)
            - dte_calendar: Original calendar days (for 0DTE detection)
        """
        df = df.copy()
        
        # Parse timestamp if string
        if df['timestamp'].dtype == object:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure expiration is datetime
        if df['expiration'].dtype == object:
            df['expiration'] = pd.to_datetime(df['expiration'])
        
        # Calculate calendar DTE (days between timestamp date and expiration)
        df['dte_calendar'] = (df['expiration'] - df['timestamp'].dt.normalize()).dt.days
        
        # Start with calendar DTE
        df['dte'] = df['dte_calendar'].astype(float)
        
        # Handle 0DTE with intraday time estimation
        # Options expire at 4pm ET, so calculate fractional day remaining
        mask_0dte = df['dte_calendar'] == 0
        if mask_0dte.any():
            # Extract hour from timestamp (assume ET timezone)
            hours = df.loc[mask_0dte, 'timestamp'].dt.hour
            minutes = df.loc[mask_0dte, 'timestamp'].dt.minute
            
            # Market close at 16:00, so calculate hours remaining
            # Add small buffer to avoid T=0 (minimum 0.5 hour = 0.5/24 day ≈ 0.021)
            hours_remaining = np.maximum(16 - hours - minutes/60, 0.5)
            df.loc[mask_0dte, 'dte'] = hours_remaining / 24.0
        
        # Handle edge cases - minimum 0.01 for any non-expired option
        df['dte'] = df['dte'].clip(lower=0.01)
        
        return df
    
    def add_dte_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add DTE-related features for Multi-Strategy PPO."""
        df = df.copy()
        
        # Ensure DTE exists
        if 'dte' not in df.columns:
            df = self.calculate_dte(df)
        
        # Use dte_calendar for bucket classification (if available)
        # This ensures 0DTE is correctly identified even after fractional conversion
        dte_for_buckets = df.get('dte_calendar', df['dte'])
        
        # DTE bucket classification - FIXED: use calendar days
        df['is_0dte'] = (dte_for_buckets == 0).astype(float)
        df['is_weekly'] = ((dte_for_buckets > 0) & (dte_for_buckets <= 7)).astype(float)
        df['is_monthly'] = ((dte_for_buckets > 7) & (dte_for_buckets <= 60)).astype(float)
        df['is_leaps'] = (dte_for_buckets > 60).astype(float)
        
        # DTE bucket as categorical (for stratified analysis)
        def dte_bucket(dte_cal):
            if dte_cal == 0:
                return 0  # 0DTE
            elif dte_cal <= 7:
                return 1  # WEEKLY
            elif dte_cal <= 60:
                return 2  # MONTHLY
            else:
                return 3  # LEAPS
        df['dte_bucket'] = dte_for_buckets.apply(dte_bucket)
        
        # Normalized DTE (0-60 range, clipped) - use fractional dte
        df['dte_norm'] = (df['dte'].clip(upper=60) / 60.0).astype(float)
        
        # Time decay acceleration factor (theta is higher near expiry)
        time_decay = np.where(
            df['dte'] > 0,
            1.0 / np.sqrt(df['dte']),
            3.0  # Max factor for 0DTE
        )
        df['time_decay_factor'] = np.clip(time_decay, 0, 3.0)
        
        return df
    
    def add_option_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add option type (CALL/PUT) features."""
        df = df.copy()
        
        # is_call flag
        if 'right' in df.columns:
            df['is_call'] = df['right'].str.upper().isin(['C', 'CALL']).astype(float)
        elif 'option_type' in df.columns:
            df['is_call'] = df['option_type'].str.upper().isin(['C', 'CALL']).astype(float)
        else:
            logger.warning("No 'right' or 'option_type' column found! Defaulting to call.")
            df['is_call'] = 1.0
        
        # Moneyness
        if 'underlying_price' in df.columns and 'strike' in df.columns:
            df['moneyness'] = df['underlying_price'] / df['strike']
            df['log_moneyness'] = np.log(df['moneyness'])
            
            # ITM/OTM flags
            df['is_itm'] = np.where(
                df['is_call'] == 1,
                (df['underlying_price'] > df['strike']).astype(float),
                (df['underlying_price'] < df['strike']).astype(float)
            )
            df['is_atm'] = ((df['moneyness'] >= 0.97) & (df['moneyness'] <= 1.03)).astype(float)
        
        return df
    
    def calculate_mid_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mid price from OHLC (use close as proxy for mid)."""
        df = df.copy()
        
        # Use close price as option mid price
        if 'close' in df.columns:
            df['option_price'] = df['close']
        elif 'bid' in df.columns and 'ask' in df.columns:
            df['option_price'] = (df['bid'] + df['ask']) / 2
        else:
            logger.warning("No price column found!")
            df['option_price'] = np.nan
        
        return df
    
    def calculate_iv_and_greeks(self, df: pd.DataFrame, r: float = 0.0433, 
                                 q: float = 0.01) -> pd.DataFrame:
        """Calculate IV and all Greeks using FAST vectorized methods."""
        # Validate rates first
        self.validate_rates(r, q)
        
        df = df.copy()
        
        # Prepare arrays
        S = df['underlying_price'].values
        K = df['strike'].values
        T = df['dte'].values / 365.0
        option_price = df['option_price'].values
        is_call = df['is_call'].values.astype(bool)
        
        n = len(df)
        
        # Valid mask for calculation
        valid_mask = (T > 0) & (option_price > 0) & (S > 0) & (K > 0)
        logger.info(f"  Valid rows for IV/Greeks: {valid_mask.sum():,}/{n:,}")
        
        # === FAST VECTORIZED IV CALCULATION ===
        logger.info("  Calculating IV (vectorized Newton-Raphson)...")
        start = time.time()
        
        ivs = self.engine.calculate_iv_vectorized(
            prices=option_price,
            S=S, K=K, T=T,
            r=r, q=q,
            is_call=is_call
        )
        
        iv_time = time.time() - start
        valid_ivs = (~np.isnan(ivs)).sum()
        logger.info(f"  IV done in {iv_time:.1f}s: {valid_ivs:,}/{n:,} valid ({100*valid_ivs/n:.1f}%)")
        
        df['iv'] = ivs
        
        # === VECTORIZED GREEKS CALCULATION ===
        logger.info("  Calculating Greeks (vectorized)...")
        start = time.time()
        
        greeks = self.engine.calculate_greeks_vectorized(
            S=S, K=K, T=T, r=r, q=q, sigma=ivs, is_call=is_call
        )
        
        greeks_time = time.time() - start
        logger.info(f"  Greeks done in {greeks_time:.1f}s")
        
        # Add all Greeks to DataFrame
        for name, values in greeks.items():
            df[name] = values
        
        # Add rates
        df['r'] = r
        df['q'] = q
        
        return df
    
    def add_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ML-normalized features for PPO training."""
        df = df.copy()
        
        # Theta to premium ratio (for sell strategies)
        # Negative theta means option loses value -> good for sellers
        with np.errstate(divide='ignore', invalid='ignore'):
            theta_to_prem = np.where(
                df['option_price'] > 0.01,
                np.abs(df['theta']) / df['option_price'],
                0.0
            )
        df['theta_to_premium'] = np.clip(theta_to_prem, 0, 1.0)
        
        # Gamma exposure (risk metric for 0DTE)
        df['gamma_exposure'] = np.clip(
            df['gamma'] * df['underlying_price'] ** 2 / 100, -10, 10
        )
        
        # Vanna exposure
        df['vanna_exposure'] = np.clip(
            df['vanna'] * df['underlying_price'], -5, 5
        )
        
        # Delta-adjusted moneyness
        df['delta_moneyness'] = df['delta'].abs() * df['log_moneyness']
        
        # Normalized Greeks for NN input (clipped and scaled)
        df['delta_norm'] = np.clip(df['delta'], -1, 1)
        df['gamma_norm'] = np.clip(df['gamma'] * df['underlying_price'], 0, 2) / 2
        df['theta_norm'] = np.clip(df['theta'] / df['underlying_price'] * 100, -5, 0) / 5
        df['vega_norm'] = np.clip(df['vega'] / df['underlying_price'], 0, 1)
        df['iv_norm'] = np.clip(df['iv'], 0, 2) / 2
        
        # Vanna normalized
        df['vanna_norm'] = np.clip(df['vanna'] * df['underlying_price'], -1, 1)
        
        # Charm normalized (per day, relative to delta)
        with np.errstate(divide='ignore', invalid='ignore'):
            charm_rel = np.where(
                np.abs(df['delta']) > 0.01,
                df['charm'] / np.abs(df['delta']),
                0.0
            )
        df['charm_norm'] = np.clip(charm_rel, -0.1, 0.1) * 10
        
        return df
    
    def process_parquet(self, input_path: Path, output_path: Path,
                       r: float = 0.0433, q: float = 0.01,
                       chunk_size: int = 500_000) -> bool:
        """Process a single parquet file with memory-efficient chunking.
        
        For files > 1M rows, processes in chunks to avoid OOM.
        """
        try:
            logger.info(f"Processing {input_path.name}...")
            start_total = time.time()
            
            # Get file size to decide on chunking
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(input_path)
            total_rows = parquet_file.metadata.num_rows
            logger.info(f"  Total rows: {total_rows:,}")
            
            # For small files, process all at once
            if total_rows <= chunk_size:
                return self._process_parquet_simple(input_path, output_path, r, q)
            
            # For large files, process in chunks
            logger.info(f"  Using chunked processing ({chunk_size:,} rows per chunk)")
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            chunks_processed = 0
            writer = None
            
            # Process in chunks using pyarrow batches
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                chunk_df = batch.to_pandas()
                chunks_processed += 1
                logger.info(f"  Chunk {chunks_processed}: {len(chunk_df):,} rows")
                
                # Apply all transformations
                chunk_df = self.calculate_dte(chunk_df)
                chunk_df = self.add_option_type_features(chunk_df)
                chunk_df = self.calculate_mid_price(chunk_df)
                chunk_df = self.add_dte_features(chunk_df)
                chunk_df = self.calculate_iv_and_greeks(chunk_df, r=r, q=q)
                chunk_df = self.add_ml_features(chunk_df)
                
                # Convert to PyArrow table and write
                import pyarrow as pa
                table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                
                if writer is None:
                    # First chunk - create writer with schema
                    writer = pq.ParquetWriter(
                        output_path, 
                        table.schema,
                        compression='zstd'
                    )
                
                writer.write_table(table)
                
                # Free memory
                del chunk_df, table
                gc.collect()
            
            # Close writer
            if writer:
                writer.close()
            
            elapsed = time.time() - start_total
            
            # Verify output
            result = pq.ParquetFile(output_path)
            final_rows = result.metadata.num_rows
            final_cols = len(result.schema)
            
            logger.info(f"  ✅ Saved {final_rows:,} rows to {output_path.name}")
            logger.info(f"  Columns: {final_cols} | Chunks: {chunks_processed} | Time: {elapsed:.1f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {input_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _process_parquet_simple(self, input_path: Path, output_path: Path,
                                r: float = 0.0433, q: float = 0.01) -> bool:
        """Simple processing for small files."""
        df = pd.read_parquet(input_path)
        logger.info(f"  Loaded {len(df):,} rows")
        
        df = self.calculate_dte(df)
        df = self.add_option_type_features(df)
        df = self.calculate_mid_price(df)
        df = self.add_dte_features(df)
        df = self.calculate_iv_and_greeks(df, r=r, q=q)
        df = self.add_ml_features(df)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, compression='zstd', index=False)
        
        elapsed_msg = f"  ✅ Saved {len(df):,} rows, {len(df.columns)} columns"
        logger.info(elapsed_msg)
        
        # Stats
        dte_stats = df.groupby('dte_bucket').size()
        logger.info(f"  DTE: 0DTE={dte_stats.get(0, 0):,}, Weekly={dte_stats.get(1, 0):,}, Monthly={dte_stats.get(2, 0):,}")
        
        del df
        gc.collect()
        return True
    
    def _process_file_wrapper(self, args: tuple) -> tuple:
        """Wrapper for parallel processing."""
        input_path, output_path, r, q = args
        success = self.process_parquet(input_path, output_path, r=r, q=q)
        return (input_path.name, success)
    
    def process_all(self, input_dir: Path, output_dir: Path,
                   r: float = 0.0433, q: float = 0.01,
                   pattern: str = "*_*.parquet") -> dict:
        """Process all parquet files in directory.
        
        If n_workers > 1, uses parallel processing with ProcessPoolExecutor.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        files = sorted(input_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No files matching '{pattern}' in {input_dir}")
            return {'processed': 0, 'failed': 0, 'skipped': 0}
        
        # Validate rates before starting
        self.validate_rates(r, q)
        logger.info(f"Found {len(files)} parquet files to process (r={r:.4f}, q={q:.4f})")
        
        # Filter out already processed files
        files_to_process = []
        skipped = 0
        for file in files:
            output_path = output_dir / file.name
            if output_path.exists():
                skipped += 1
            else:
                files_to_process.append((file, output_path, r, q))
        
        if skipped > 0:
            logger.info(f"Skipping {skipped} already processed files")
        
        if not files_to_process:
            logger.info("All files already processed!")
            return {'processed': 0, 'failed': 0, 'skipped': skipped}
        
        processed = 0
        failed = 0
        start_all = time.time()
        
        # Parallel processing
        if self.n_workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            logger.info(f"Using {self.n_workers} parallel workers")
            
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(self._process_file_wrapper, args): args[0].name 
                          for args in files_to_process}
                
                for i, future in enumerate(as_completed(futures), 1):
                    filename = futures[future]
                    try:
                        name, success = future.result()
                        if success:
                            processed += 1
                        else:
                            failed += 1
                        
                        # Progress
                        elapsed = time.time() - start_all
                        eta = elapsed / i * (len(files_to_process) - i)
                        logger.info(f"[{i}/{len(files_to_process)}] {name} - ETA: {eta/60:.1f}min")
                        
                    except Exception as e:
                        logger.error(f"Worker error for {filename}: {e}")
                        failed += 1
        
        # Sequential processing
        else:
            for i, (file, output_path, r_val, q_val) in enumerate(files_to_process, 1):
                logger.info(f"\n[{i}/{len(files_to_process)}] {file.name}")
                
                if self.process_parquet(file, output_path, r=r_val, q=q_val):
                    processed += 1
                else:
                    failed += 1
                
                # ETA
                elapsed = time.time() - start_all
                if processed + failed > 0:
                    eta = elapsed / (processed + failed) * (len(files_to_process) - i)
                    logger.info(f"  ETA: {eta/60:.1f} minutes remaining")
                
                gc.collect()
        
        total_time = time.time() - start_all
        logger.info(f"\n{'='*60}")
        logger.info(f"DONE: {processed} processed, {failed} failed, {skipped} skipped")
        logger.info(f"Total time: {total_time/60:.1f} minutes ({total_time/(processed+1):.1f}s/file)")
        logger.info(f"{'='*60}")
        
        return {'processed': processed, 'failed': failed, 'skipped': skipped}


def main():
    parser = argparse.ArgumentParser(
        description="Enrich ThetaData parquets with Greeks and DTE features"
    )
    parser.add_argument('--input', '-i', type=str, default='data/raw',
                       help='Input directory with raw parquets')
    parser.add_argument('--output', '-o', type=str, default='data/enriched',
                       help='Output directory for enriched parquets')
    parser.add_argument('--rate', '-r', type=float, default=0.0433,
                       help='Risk-free rate (SOFR, default 4.33%%)')
    parser.add_argument('--dividend', '-q', type=float, default=0.01,
                       help='Dividend yield (default 1%%)')
    parser.add_argument('--pattern', '-p', type=str, default='*_*.parquet',
                       help='File pattern to match')
    parser.add_argument('--workers', '-w', type=int, default=1,
                       help='Number of parallel workers (default 1 = sequential)')
    parser.add_argument('--use-rates-provider', action='store_true',
                       help='Use RatesProvider for dynamic rates')
    
    args = parser.parse_args()
    
    enricher = ParquetEnricher(
        use_rates_provider=args.use_rates_provider,
        n_workers=args.workers
    )
    
    enricher.process_all(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        r=args.rate,
        q=args.dividend,
        pattern=args.pattern
    )


if __name__ == "__main__":
    main()