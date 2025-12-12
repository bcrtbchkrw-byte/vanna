"""
Tests for Vanna Feature Engineering

Validates feature extraction for ML training.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestVannaFeatureEngineering:
    """Test suite for VannaFeatureEngineering."""
    
    @pytest.fixture
    def feature_eng(self):
        """Create feature engineering instance."""
        from ml.vanna_feature_engineering import VannaFeatureEngineering
        return VannaFeatureEngineering()
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        dates = pd.date_range(
            start='2024-01-02 09:30:00',
            periods=100,
            freq='1min'
        )
        return pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(500, 510, 100),
            'high': np.random.uniform(505, 515, 100),
            'low': np.random.uniform(495, 505, 100),
            'close': np.random.uniform(500, 510, 100),
            'volume': np.random.randint(1000, 10000, 100),
            'vix': np.random.uniform(15, 25, 100),
            'vix3m': np.random.uniform(17, 27, 100)
        })
    
    def test_add_time_features(self, feature_eng, sample_df):
        """Test time feature extraction."""
        result = feature_eng.add_time_features(sample_df)
        
        # Check new columns exist
        assert 'sin_time' in result.columns
        assert 'cos_time' in result.columns
        assert 'sin_dow' in result.columns
        assert 'cos_dow' in result.columns
        assert 'sin_doy' in result.columns
        assert 'cos_doy' in result.columns
        
        # Check sin/cos range
        assert result['sin_time'].min() >= -1
        assert result['sin_time'].max() <= 1
        assert result['cos_time'].min() >= -1
        assert result['cos_time'].max() <= 1
    
    def test_sin_cos_cyclic_property(self, feature_eng):
        """Test that sin²+cos² = 1 for time encoding."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-02 09:30', periods=390, freq='1min')
        })
        
        result = feature_eng.add_time_features(df)
        
        # sin² + cos² should be ~1 for proper encoding
        sum_squares = result['sin_time']**2 + result['cos_time']**2
        np.testing.assert_array_almost_equal(sum_squares.values, np.ones(390), decimal=10)
    
    def test_add_vix_features(self, feature_eng, sample_df):
        """Test VIX feature extraction."""
        result = feature_eng.add_vix_features(sample_df)
        
        # Check new columns
        assert 'vix_ratio' in result.columns
        assert 'vix_change_1d' in result.columns
        assert 'vix_in_contango' in result.columns
        
        # VIX ratio should be positive
        assert (result['vix_ratio'] > 0).all()
    
    def test_vix_ratio_calculation(self, feature_eng):
        """Test VIX/VIX3M ratio correctness."""
        df = pd.DataFrame({
            'vix': [15.0, 20.0, 25.0],
            'vix3m': [18.0, 20.0, 22.0]
        })
        
        result = feature_eng.add_vix_features(df)
        
        expected_ratios = [15.0/18.0, 20.0/20.0, 25.0/22.0]
        np.testing.assert_array_almost_equal(
            result['vix_ratio'].values, 
            expected_ratios, 
            decimal=5
        )
        
        # Contango check
        assert result['vix_in_contango'].iloc[0] == 1  # 15/18 < 1
        assert result['vix_in_contango'].iloc[1] == 0  # 20/20 = 1
        assert result['vix_in_contango'].iloc[2] == 0  # 25/22 > 1
    
    def test_add_regime_labels(self, feature_eng):
        """Test market regime classification."""
        df = pd.DataFrame({
            'vix': [12.0, 20.0, 30.0, 14.9, 15.0, 24.9, 25.0]
        })
        
        result = feature_eng.add_regime_labels(df)
        
        assert 'regime' in result.columns
        assert 'regime_label' in result.columns
        
        expected_regimes = [0, 1, 2, 0, 1, 1, 2]  # Based on thresholds 15, 25
        assert list(result['regime']) == expected_regimes
    
    def test_add_price_features(self, feature_eng, sample_df):
        """Test price-based feature extraction."""
        result = feature_eng.add_price_features(sample_df)
        
        # Check columns
        assert 'return_1m' in result.columns
        assert 'return_5m' in result.columns
        assert 'volatility_20' in result.columns
        assert 'momentum_20' in result.columns
    
    def test_process_all_features(self, feature_eng, sample_df):
        """Test full feature pipeline."""
        result = feature_eng.process_all_features(sample_df)
        
        # Should have many more columns than input
        assert len(result.columns) > len(sample_df.columns)
        
        # Key features should exist
        assert 'sin_time' in result.columns
        assert 'vix_ratio' in result.columns
        assert 'regime' in result.columns
        assert 'return_1m' in result.columns
    
    def test_get_feature_columns(self, feature_eng):
        """Test feature column list."""
        columns = feature_eng.get_feature_columns()
        
        assert isinstance(columns, list)
        assert len(columns) > 0
        assert 'sin_time' in columns
        assert 'vix_ratio' in columns
        assert 'regime' in columns
    
    def test_handles_missing_columns_gracefully(self, feature_eng):
        """Test graceful handling of missing columns."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-02', periods=10, freq='1min'),
            'close': [100] * 10
        })
        
        # Should not raise, should add placeholders
        result = feature_eng.add_vix_features(df)
        result = feature_eng.add_regime_labels(result)
        
        # Regime should default
        assert 'regime' in result.columns


class TestFeatureEngineeringSingleton:
    """Test singleton behavior."""
    
    def test_get_feature_engineering_returns_instance(self):
        """Test singleton returns instance."""
        from ml.vanna_feature_engineering import get_feature_engineering
        
        eng1 = get_feature_engineering()
        eng2 = get_feature_engineering()
        
        assert eng1 is not None
        assert eng2 is not None
