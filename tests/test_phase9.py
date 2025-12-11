"""
Phase 9 Test - Machine Learning

Tests:
- Feature Engineering (RSI, Volatility calc)
- Trade Predictor (Interface check)

Run: python tests/test_phase9.py
"""
import asyncio
import os
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import setup_logger
from ml.features import FeatureEngineer
from ml.predictor import TradePredictor


def test_feature_engineering():
    print("\n--- Testing Feature Engineering ---")
    fe = FeatureEngineer()
    
    # Generate dummy price data (30 days)
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    # Random walk
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(50))
    
    df = pd.DataFrame({'close': prices}, index=dates)
    
    # Compute
    features = fe.compute_features(df)
    
    # Check Columns
    expected_cols = ['rsi', 'hist_vol', 'sma_20', 'dist_sma_20']
    missing = [c for c in expected_cols if c not in features.columns]
    
    if missing:
        print(f"❌ Missing features: {missing}")
        return False
        
    print(f"✅ Features generated: {list(features.columns)}")
    
    # Check RSI values
    last_rsi = features['rsi'].iloc[-1]
    if not (0 <= last_rsi <= 100):
        print(f"❌ Invalid RSI: {last_rsi}")
        return False
        
    print(f"✅ RSI Valid: {last_rsi:.2f}")
    return True

def test_predictor():
    print("\n--- Testing Trade Predictor ---")
    predictor = TradePredictor()
    
    # Test High VIX case (Expect high prob)
    setup_high_vix = {
        'strategy': 'IRON_CONDOR',
        'market_data': {'vix': 25}
    }
    
    prob_high = predictor.predict_probability(setup_high_vix)
    print(f"Prediction (VIX 25): {prob_high:.1%}")
    
    if prob_high < 0.60:
        print("❌ Prediction logic failed for High VIX")
        return False
        
    # Test Low VIX case (Expect low prob)
    setup_low_vix = {
        'strategy': 'IRON_CONDOR',
        'market_data': {'vix': 10}
    }
    
    prob_low = predictor.predict_probability(setup_low_vix)
    print(f"Prediction (VIX 10): {prob_low:.1%}")
    
    if prob_low > 0.55:
        print("❌ Prediction logic failed for Low VIX")
        return False
        
    print("✅ Predictor logic valid")
    return True

async def run_tests():
    setup_logger()
    
    results = []
    # Using run_in_executor if heavy? No, these are light.
    results.append(test_feature_engineering())
    results.append(test_predictor())
    
    if all(results):
        print("\n✅ ALL PHASE 9 TESTS PASSED")
        return 0
    else:
        print("\n❌ TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(run_tests()))
