
import numpy as np
import pandas as pd

from ml.features import FeatureEngineer
from ml.trade_success_predictor import TradePredictor


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
    
    assert not missing, f"Missing features: {missing}"
        
    print(f"✅ Features generated: {list(features.columns)}")
    
    # Check RSI values
    last_rsi = features['rsi'].iloc[-1]
    assert 0 <= last_rsi <= 100, f"Invalid RSI: {last_rsi}"
        
    print(f"✅ RSI Valid: {last_rsi:.2f}")

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
    
    assert prob_high >= 0.60, "Prediction logic failed for High VIX"
        
    # Test Low VIX case (Expect low prob)
    setup_low_vix = {
        'strategy': 'IRON_CONDOR',
        'market_data': {'vix': 10}
    }
    
    prob_low = predictor.predict_probability(setup_low_vix)
    print(f"Prediction (VIX 10): {prob_low:.1%}")
    
    assert prob_low <= 0.55, "Prediction logic failed for Low VIX"
        
    print("✅ Predictor logic valid")
