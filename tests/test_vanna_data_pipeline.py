"""
Tests for Vanna Data Pipeline

Integration tests for data fetching and storage.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile


class TestVannaDataStorage:
    """Test suite for VannaDataStorage."""
    
    @pytest.fixture
    def storage(self):
        """Create storage with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from ml.data_storage import VannaDataStorage
            yield VannaDataStorage(
                data_dir=tmpdir,
                db_path=f"{tmpdir}/test_vanna.db"
            )
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-02 09:30', periods=100, freq='1min'),
            'symbol': ['SPY'] * 100,
            'open': np.random.uniform(500, 510, 100),
            'high': np.random.uniform(505, 515, 100),
            'low': np.random.uniform(495, 505, 100),
            'close': np.random.uniform(500, 510, 100),
            'volume': np.random.randint(1000, 10000, 100),
            'vix': np.random.uniform(15, 25, 100),
            'vix3m': np.random.uniform(17, 27, 100),
            'vix_ratio': np.random.uniform(0.8, 1.2, 100),
            'regime': np.random.randint(0, 3, 100),
            'sin_time': np.sin(np.linspace(0, np.pi, 100)),
            'cos_time': np.cos(np.linspace(0, np.pi, 100))
        })
    
    def test_save_load_parquet(self, storage, sample_df):
        """Test Parquet save and load."""
        # Save
        filepath = storage.save_historical_parquet(sample_df, 'SPY', '1min')
        assert filepath.exists()
        
        # Load
        loaded = storage.load_historical_parquet('SPY', '1min')
        assert loaded is not None
        assert len(loaded) == len(sample_df)
    
    def test_load_nonexistent_parquet(self, storage):
        """Test loading non-existent file returns None."""
        result = storage.load_historical_parquet('NONEXISTENT', '1min')
        assert result is None
    
    def test_save_live_bar(self, storage):
        """Test saving live bar to SQLite."""
        bar_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'SPY',
            'timeframe': '1min',
            'open': 505.0,
            'high': 506.0,
            'low': 504.0,
            'close': 505.5,
            'volume': 5000,
            'vix': 18.5,
            'vix3m': 20.0,
            'vix_ratio': 0.925,
            'regime': 1,
            'sin_time': 0.5,
            'cos_time': 0.866
        }
        
        result = storage.save_live_bar(bar_data)
        assert result is True
    
    def test_save_vanna_calculation(self, storage):
        """Test saving Vanna calculation."""
        vanna_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'SPY',
            'strike': 500.0,
            'expiry': '2024-02-16',
            'option_type': 'call',
            'vanna': 0.015,
            'delta': 0.45,
            'gamma': 0.02,
            'vega': 0.35,
            'charm': -0.001,
            'volga': 0.02,
            'underlying_price': 502.0,
            'iv': 0.18,
            'risk_free_rate': 0.045
        }
        
        result = storage.save_vanna_calculation(vanna_data)
        assert result is True
    
    def test_get_storage_stats(self, storage, sample_df):
        """Test storage statistics."""
        # Save some data first
        storage.save_historical_parquet(sample_df, 'SPY', '1min')
        
        stats = storage.get_storage_stats()
        
        assert 'parquet_files' in stats
        assert 'sqlite_tables' in stats
        assert len(stats['parquet_files']) >= 1


class TestVannaDataPipelineUnit:
    """Unit tests for VannaDataPipeline (no IBKR required)."""
    
    def test_pipeline_initialization(self):
        """Test pipeline can be imported and instantiated."""
        # This tests the import only, not connection
        from ml.vanna_data_pipeline import VannaDataPipeline
        
        # Check class attributes
        assert 'SPY' in VannaDataPipeline.SYMBOLS
        assert 'QQQ' in VannaDataPipeline.SYMBOLS
        assert 'VIX' in VannaDataPipeline.VIX_SYMBOLS
    
    def test_symbols_list(self):
        """Test configured symbols."""
        from ml.vanna_data_pipeline import VannaDataPipeline
        
        expected = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
        assert VannaDataPipeline.SYMBOLS == expected


class TestDataStorageSingleton:
    """Test singleton behavior."""
    
    def test_get_data_storage_with_temp_path(self):
        """Test storage creation with custom path."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from ml.data_storage import VannaDataStorage
            storage = VannaDataStorage(
                data_dir=tmpdir,
                db_path=f"{tmpdir}/test.db"
            )
            assert storage is not None
