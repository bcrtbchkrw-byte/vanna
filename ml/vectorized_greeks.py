"""
Vectorized Greeks Calculator for Historical Data

Adds Vanna and other Greeks to historical OHLCV data using vectorized numpy operations.
Used for ML training data preparation.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path
from typing import Optional

from core.logger import get_logger, setup_logger
from ml.data_storage import get_data_storage

logger = get_logger()


class VectorizedGreeksCalculator:
    """
    Vectorized Black-Scholes Greeks calculator.
    
    Calculates Greeks for entire DataFrame columns at once
    instead of row-by-row, providing 100x+ speedup.
    
    Columns added:
    - delta, gamma, theta, vega (first-order)
    - vanna, charm, volga (second-order)
    """
    
    DEFAULT_RISK_FREE_RATE = 0.045
    
    def __init__(self, risk_free_rate: float = None):
        self.r = risk_free_rate or self.DEFAULT_RISK_FREE_RATE
        logger.info(f"VectorizedGreeksCalculator initialized with r={self.r:.4f}")
    
    def add_greeks_to_dataframe(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        iv_col: str = 'iv',
        dte_days: float = 30,
        strike_offset_pct: float = 0.05,  # 5% OTM
        option_type: str = 'put',
        dividend_yield: float = 0.0
    ) -> pd.DataFrame:
        """
        Add Greeks columns to DataFrame using vectorized operations.
        
        Using Generalized Black-Scholes (Merton Model) with Cost of Carry.
        b = r - q (Cost of Carry = risk-free rate - dividend yield)
        
        For historical equity data without options IV, we estimate:
        - IV from VIX (if available) or use default 25%
        - Strike as price ± offset%
        - DTE as fixed assumption
        
        Args:
            df: DataFrame with price data
            price_col: Column name for underlying price
            iv_col: Column name for IV (uses VIX if not exists)
            dte_days: Days to expiration assumption
            strike_offset_pct: Strike distance from ATM (0.05 = 5% OTM)
            option_type: 'call' or 'put' for moneyness direction
            dividend_yield: Annual dividend yield (decimal, e.g. 0.015 for 1.5%)
            
        Returns:
            DataFrame with Greek columns added
        """
        df = df.copy()
        n = len(df)
        
        logger.info(f"Calculating vectorized Greeks for {n:,} rows (div_yield={dividend_yield:.2%})...")
        
        # Get underlying price
        S = df[price_col].values.astype(np.float64)
        
        # Get or estimate IV
        if iv_col in df.columns:
            sigma = df[iv_col].values.astype(np.float64)
        elif 'vix' in df.columns:
            # Use VIX as IV proxy (VIX is annualized % volatility)
            sigma = df['vix'].values.astype(np.float64) / 100.0
            sigma = np.clip(sigma, 0.05, 1.5)  # Limit to 5%-150%
        else:
            # Default 25% IV
            sigma = np.full(n, 0.25)
        
        # Handle NaN in sigma
        sigma = np.nan_to_num(sigma, nan=0.25)
        sigma = np.clip(sigma, 0.01, 2.0)  # Safety bounds
        
        # Calculate strike (OTM puts or calls)
        if option_type.lower() == 'put':
            K = S * (1 - strike_offset_pct)  # 5% below for puts
        else:
            K = S * (1 + strike_offset_pct)  # 5% above for calls
        
        # Time to expiry in years
        T = np.full(n, dte_days / 365.0)
        
        # Risk-free rate and Dividend Yield
        r = self.r
        q = dividend_yield
        b = r - q  # Cost of carry
        
        # ================================================================
        # VECTORIZED GENERALIZED BLACK-SCHOLES (Merton)
        # ================================================================
        
        # d1 and d2
        # d1 = [ln(S/K) + (b + sigma^2/2)T] / (sigma * sqrt(T))
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # PDF and CDF
        phi_d1 = norm.pdf(d1)  # Standard normal PDF
        N_d1 = norm.cdf(d1)    # Standard normal CDF
        N_d2 = norm.cdf(d2)
        
        # Discount factors
        exp_rT = np.exp(-r * T)
        exp_qT = np.exp(-q * T)  # Dividend discount
        
        # ================================================================
        # FIRST-ORDER GREEKS
        # ================================================================
        
        if option_type.lower() == 'call':
            # Delta = e^-qT * N(d1)
            delta = exp_qT * N_d1
            
            # Theta (Merton)
            term1 = -S * exp_qT * phi_d1 * sigma / (2 * sqrt_T)
            term2 = -r * K * exp_rT * N_d2
            term3 = q * S * exp_qT * N_d1
            theta = (term1 + term2 + term3) / 365.0
            
            rho = K * T * exp_rT * N_d2 / 100
            
        else:  # put
            # Delta = e^-qT * (N(d1) - 1)
            delta = exp_qT * (N_d1 - 1)
            
            # Theta (Merton)
            term1 = -S * exp_qT * phi_d1 * sigma / (2 * sqrt_T)
            term2 = r * K * exp_rT * norm.cdf(-d2)
            term3 = -q * S * exp_qT * norm.cdf(-d1)
            theta = (term1 + term2 + term3) / 365.0
            
            rho = -K * T * exp_rT * norm.cdf(-d2) / 100
        
        # Gamma = e^-qT * phi(d1) / (S * sigma * sqrt(T))
        gamma = exp_qT * phi_d1 / (S * sigma * sqrt_T)
        
        # Vega = S * e^-qT * phi(d1) * sqrt(T)
        vega = S * exp_qT * phi_d1 * sqrt_T / 100  # Per 1% IV move
        
        # ================================================================
        # SECOND-ORDER GREEKS (Vanna, Charm, Volga)
        # ================================================================
        
        # Vanna = ∂Δ/∂σ = -φ(d1) × d2 / (S × σ × √T)
        vanna = -(phi_d1 * d2) / (S * sigma * sqrt_T)
        
        # Charm = ∂Δ/∂t = -φ(d1) × (r/(σ√T) - d2/(2T))
        charm = -phi_d1 * (r / (sigma * sqrt_T) - d2 / (2 * T))
        
        # Volga = ∂²V/∂σ² = Vega × d1 × d2 / σ
        volga = vega * d1 * d2 / sigma
        
        # ================================================================
        # Handle edge cases (NaN, Inf)
        # ================================================================
        
        def safe_column(arr, name):
            """Replace inf/nan with 0 and log warnings."""
            bad_mask = ~np.isfinite(arr)
            n_bad = np.sum(bad_mask)
            if n_bad > 0:
                logger.debug(f"{name}: {n_bad} invalid values replaced with 0")
                arr = np.where(bad_mask, 0.0, arr)
            return arr
        
        # Add columns to DataFrame
        df['delta'] = safe_column(delta, 'delta')
        df['gamma'] = safe_column(gamma, 'gamma')
        df['theta'] = safe_column(theta, 'theta')
        df['vega'] = safe_column(vega, 'vega')
        df['vanna'] = safe_column(vanna, 'vanna')
        df['charm'] = safe_column(charm, 'charm')
        df['volga'] = safe_column(volga, 'volga')
        
        logger.info(f"✅ Added 7 Greek columns to {n:,} rows")
        
        return df
    
    def process_parquet_file(
        self,
        input_path: str,
        output_path: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Process a parquet file and add Greeks columns.
        
        Args:
            input_path: Path to input parquet file
            output_path: Path to output file (default: {input}_vanna.parquet)
            **kwargs: Additional args for add_greeks_to_dataframe
            
        Returns:
            Processed DataFrame
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"📂 Loading {input_path}...")
        df = pd.read_parquet(input_path)
        
        logger.info(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # CRITICAL: Apply feature engineering BEFORE Greeks calculation
        # This adds ALL features including new ones (hour_of_day, volume_ratio, etc.)
        from ml.vanna_feature_engineering import VannaFeatureEngineering
        feature_eng = VannaFeatureEngineering()
        df = feature_eng.process_all_features(df)
        logger.info(f"   Features added: {len(df.columns)} total columns")
        
        # Add Greeks
        df = self.add_greeks_to_dataframe(df, **kwargs)
        
        # Save output
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_vanna.parquet"
        else:
            output_path = Path(output_path)
        
        df.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"💾 Saved to {output_path}")
        
        return df


def process_all_historical_data(data_dir: str = "data/vanna_ml"):
    """
    Process all historical parquet files and add Greeks.
    
    Creates *_vanna.parquet files with 7 additional Greek columns.
    Processes all *_1min.parquet files that don't already have _vanna suffix.
    """
    data_dir = Path(data_dir)
    calculator = VectorizedGreeksCalculator()
    
    # Find all 1min parquet files (excluding already processed)
    parquet_files = list(data_dir.glob("*_1min.parquet"))
    parquet_files = [f for f in parquet_files if "_vanna" not in f.stem]
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {data_dir}")
        return
    
    logger.info(f"Found {len(parquet_files)} files to process")
    
    processed = 0
    for filepath in parquet_files:
        output_path = data_dir / f"{filepath.stem}_vanna.parquet"
        
        # Skip if already processed
        if output_path.exists():
            logger.info(f"⏭️ {output_path.name} already exists, skipping")
            continue
        
        try:
            calculator.process_parquet_file(str(filepath), str(output_path))
            processed += 1
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
    
    logger.info(f"✅ Processed {processed} files")


# CLI entry point
if __name__ == "__main__":
    setup_logger(level="INFO")
    logger = get_logger()
    
    import sys
    
    if len(sys.argv) > 1:
        # Process specific file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        calc = VectorizedGreeksCalculator()
        calc.process_parquet_file(input_file, output_file)
    else:
        # Process all files in data/vanna_ml/
        process_all_historical_data()
