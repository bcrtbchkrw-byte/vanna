#!/usr/bin/env python3
"""
Repair Parquet Files - Fix Missing Data

Repairs:
1. Drop options_volume column (use options_volume_norm instead)
2. Fill VIX NaN values with ffill+bfill
3. Recalculate vix_percentile where missing
4. Fill vix3m NaN with VIX * 1.05 (typical contango)

Usage:
    python scripts/repair_parquet_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

def repair_parquet_files():
    """Repair all parquet files with missing data."""
    
    data_dir = Path('data/vanna_ml')
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return
    
    print("=" * 60)
    print("🔧 REPAIRING PARQUET FILES")
    print("=" * 60)
    
    # Process 1min files
    for parquet_file in sorted(data_dir.glob('*_1min.parquet')):
        print(f"\n📁 {parquet_file.name}")
        
        df = pd.read_parquet(parquet_file)
        original_len = len(df)
        modified = False
        
        # 1. Drop options_volume (keep only options_volume_norm)
        if 'options_volume' in df.columns:
            df = df.drop(columns=['options_volume'])
            print("   ✅ Dropped options_volume column")
            modified = True
        
        # 2. Fix VIX NaN values
        if 'vix' in df.columns:
            vix_nan_before = df['vix'].isnull().sum()
            if vix_nan_before > 0:
                # Forward fill, then backward fill for start of data
                df['vix'] = df['vix'].ffill().bfill()
                
                # If still NaN, use default VIX value
                if df['vix'].isnull().any():
                    df['vix'] = df['vix'].fillna(18.0)
                
                vix_nan_after = df['vix'].isnull().sum()
                print(f"   ✅ Fixed VIX: {vix_nan_before} → {vix_nan_after} NaN")
                modified = True
        
        # 3. Fix vix3m NaN values
        if 'vix3m' in df.columns:
            vix3m_nan_before = df['vix3m'].isnull().sum()
            if vix3m_nan_before > 0:
                # First try ffill/bfill
                df['vix3m'] = df['vix3m'].ffill().bfill()
                
                # If still NaN, estimate from VIX (typical contango ~5%)
                if df['vix3m'].isnull().any() and 'vix' in df.columns:
                    df['vix3m'] = df['vix3m'].fillna(df['vix'] * 1.05)
                
                vix3m_nan_after = df['vix3m'].isnull().sum()
                print(f"   ✅ Fixed vix3m: {vix3m_nan_before} → {vix3m_nan_after} NaN")
                modified = True
        
        # 4. Recalculate vix_percentile
        if 'vix' in df.columns:
            vix_pct_nan_before = df['vix_percentile'].isnull().sum() if 'vix_percentile' in df.columns else len(df)
            
            if vix_pct_nan_before > 0:
                # Rolling percentile (20-bar window)
                def rolling_percentile(x):
                    return pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
                
                df['vix_percentile'] = df['vix'].rolling(window=20, min_periods=1).apply(
                    rolling_percentile, raw=False
                )
                
                vix_pct_nan_after = df['vix_percentile'].isnull().sum()
                print(f"   ✅ Fixed vix_percentile: {vix_pct_nan_before} → {vix_pct_nan_after} NaN")
                modified = True
        
        # 5. Recalculate vix_ratio if needed
        if 'vix' in df.columns and 'vix3m' in df.columns:
            if 'vix_ratio' not in df.columns or df['vix_ratio'].isnull().any():
                df['vix_ratio'] = df['vix'] / df['vix3m'].replace(0, np.nan)
                df['vix_ratio'] = df['vix_ratio'].fillna(1.0)
                print(f"   ✅ Recalculated vix_ratio")
                modified = True
        
        # 6. Ensure options_volume_norm exists
        if 'options_volume_norm' not in df.columns:
            if 'vix' in df.columns:
                df['options_volume_norm'] = 0.5 + (df['vix'] - 20) * 0.01
                df['options_volume_norm'] = df['options_volume_norm'].clip(0.2, 1.0)
            else:
                df['options_volume_norm'] = 0.5
            print(f"   ✅ Added options_volume_norm")
            modified = True
        
        # Save if modified
        if modified:
            df.to_parquet(parquet_file, index=False, compression='snappy')
            print(f"   💾 Saved ({len(df)} rows)")
        else:
            print(f"   ⏭️ No changes needed")
    
    # Process 1day files (lighter touch)
    print("\n" + "=" * 60)
    print("📅 CHECKING 1DAY FILES")
    print("=" * 60)
    
    for parquet_file in sorted(data_dir.glob('*_1day.parquet')):
        df = pd.read_parquet(parquet_file)
        missing = df.isnull().sum()
        cols_with_missing = missing[missing > 0]
        
        if len(cols_with_missing) > 0:
            print(f"\n📁 {parquet_file.name}")
            for col, count in cols_with_missing.items():
                pct = (count / len(df)) * 100
                print(f"   ⚠️ {col}: {count} NaN ({pct:.1f}%)")
        else:
            print(f"✅ {parquet_file.name} - OK")
    
    print("\n" + "=" * 60)
    print("✅ REPAIR COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    repair_parquet_files()
