import pandas as pd
import numpy as np
import joblib
import os
from ml.trade_success_predictor import TradeSuccessPredictor
from ml.regime_classifier import RegimeClassifier
from rl.trading_env import TradingEnvironment

def audit_features(feature_list, model_name):
    print(f"\n--- Audit: {model_name} ---")
    suspicious = ['future', 'target', 'label', 'next', 'profit', 'drawdown']
    leaks = []
    for f in feature_list:
        for s in suspicious:
            if s in f.lower() and 'lag' not in f.lower(): # 'lag' is safe
                leaks.append(f)
    
    if leaks:
        print(f"⚠️ POTENTIAL LEAK DETECTED: {leaks}")
    else:
        print(f"✅ No suspicious keywords found in {len(feature_list)} features.")
        
    print(f"Sample Features: {feature_list[:5]}...")

def audit_xgboost():
    path = "data/models/trade_success_predictor.pkl"
    if not os.path.exists(path):
        print("XGBoost model not found.")
        return

    print(f"\nLoading {path}...")
    model_data = joblib.load(path)
    # Check if it's the dict format or raw model
    if isinstance(model_data, dict) and 'features' in model_data:
        features = model_data['features']
        audit_features(features, "XGBoost TradePredictor")
        
        # Feature Importance if available
        if 'model' in model_data and hasattr(model_data['model'], 'feature_importances_'):
            imps = model_data['model'].feature_importances_
            indices = np.argsort(imps)[::-1]
            print("\nTop 10 Features (XGBoost):")
            for i in range(10):
                if i < len(features):
                    print(f"{i+1}. {features[indices[i]]}: {imps[indices[i]]:.4f}")
    else:
        # Fallback to class features
        predictor = TradeSuccessPredictor()
        audit_features(predictor.FEATURES, "XGBoost TradePredictor (Class Def)")

def audit_lstm():
    # LSTM features are hardcoded in class
    features = RegimeClassifier.FEATURES
    audit_features(features, "LSTM RegimeClassifier")

def audit_ppo():
    # PPO features are in TradingEnvironment
    features = TradingEnvironment.MARKET_FEATURES + TradingEnvironment.POSITION_FEATURES
    audit_features(features, "PPO TradingEnvironment")

if __name__ == "__main__":
    print("==================================================")
    print("🔍 DEEP AUDIT: MODEL INTEGRITY CHECK")
    print("==================================================")
    
    audit_xgboost()
    audit_lstm()
    audit_ppo()
    
    print("\n==================================================")
    print("✅ AUDIT COMPLETE")
