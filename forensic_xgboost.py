import pandas as pd
import numpy as np
import joblib
# import seaborn as sns # Removed
# import matplotlib.pyplot as plt # Removed

def forensic_audit():
    print("🕵️ FORENSIC AUDIT: XGBoost Data Leak Investigation")
    print("==================================================")
    
    # 1. Load Data (SPY)
    try:
        df = pd.read_parquet('data/vanna_ml/SPY_1min_vanna.parquet')
        print(f"✅ Loaded SPY Data: {len(df)} rows")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # 2. Re-create Labels (Triple Barrier) exactly as in training
    # Standard Triple Barrier logic from training script
    # Target 1 if High hits Upper Barrier before Low hits Lower Barrier
    vol = df['close'].pct_change().rolling(window=20).std()
    upper = df['close'] * (1 + vol * 2)
    lower = df['close'] * (1 - vol * 1)
    
    labels = []
    future_window = 30
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    ub = upper.values
    lb = lower.values
    
    # Iterate with look-ahead
    for i in range(len(df) - future_window):
        outcome = 0
        for j in range(1, future_window + 1):
            if highs[i+j] >= ub[i]:
                outcome = 1 # Profit Take
                break
            if lows[i+j] <= lb[i]:
                outcome = 0 # Stop Loss
                break
        labels.append(outcome)
    
    # Pad labels
    labels = labels + [0] * future_window
    df['audit_target'] = labels
    
    # 3. Load Model Features list
    try:
        model_data = joblib.load('data/models/trade_success_predictor.pkl')
        try:
             model = model_data.get('model')
             feature_names = model_data.get('features')
        except:
             # Direct model load fallback
             model = model_data
             feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
             
        print(f"✅ Loaded Model. Features: {len(feature_names)}")
    except Exception as e:
        print(f"Model load error: {e}. Using fallback list.")
        # Hardcoded from inspect of file
        feature_names = [
            'vix', 'vix_ratio', 'vix_change_1d', 'vix_percentile',
            'sin_time', 'cos_time', 'sin_dow', 'cos_dow',
            'return_1m', 'return_5m', 'volatility_20', 'momentum_20',
            'high_low_range', 
            'delta', 'gamma', 'theta', 'vega', 'vanna', 'charm', 'volga', 
            'iv_atm', 'volume', 'put_call_ratio',
            'volatility_20_lag1'
        ]

    # 4. Check Correlations
    print("\n🔍 CORRELATION SCAN (Feature vs Target)")
    print("---------------------------------------")
    
    # Ensure all features exist
    available_features = [f for f in feature_names if f in df.columns]
    missing = set(feature_names) - set(available_features)
    if missing:
        print(f"Warning: Missing features in parquet: {missing}")
    
    audit_df = df[available_features + ['audit_target']].copy().fillna(0)
    
    correlations = audit_df.corr()['audit_target'].sort_values(ascending=False)
    
    suspicious_found = False
    print(f"\nTop Positive Correlations:")
    for feat, corr in correlations.items():
        if feat == 'audit_target': continue
        if corr > 0.05:
            print(f"  {feat}: {corr:.4f}")
        
    print(f"\nTop Negative Correlations:")
    for feat, corr in correlations.sort_values().items():
        if feat == 'audit_target': continue
        if corr < -0.05:
            print(f"  {feat}: {corr:.4f}")

    # Check for LEAK
    max_corr = correlations.drop('audit_target').abs().max()
    print(f"\n🚨 MAXIMUM CORRELATION DETECTED: {max_corr:.4f}")
    
    if max_corr > 0.8:
        print("❌ CRITICAL: LEAK DETECTED (>0.8)!!!")
    elif max_corr > 0.5:
        print("⚠️ WARNING: High correlation (>0.5). Check feature logic.")
    else:
        print("✅ CLEAN: No suspiciously high linear correlation found.")

    # 5. Feature Importance Check
    if hasattr(model, 'feature_importances_'):
         print("\n📊 MODEL FEATURE IMPORTANCE")
         imps = model.feature_importances_
         indices = np.argsort(imps)[::-1]
         for i in range(min(10, len(feature_names))):
             idx = indices[i]
             if idx < len(feature_names):
                print(f"{i+1}. {feature_names[idx]}: {imps[idx]:.4f}")

if __name__ == "__main__":
    forensic_audit()
