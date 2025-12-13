"""
Unit tests for RL Trading Environment.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path


class TestTradingEnvironment:
    """Tests for TradingEnvironment."""
    
    @pytest.fixture
    def mock_parquet_data(self, tmp_path):
        """Create mock parquet data for testing."""
        # Create minimal RL-ready data
        n_rows = 500
        data = {
            'sin_time': np.random.uniform(-1, 1, n_rows),
            'cos_time': np.random.uniform(-1, 1, n_rows),
            'sin_dow': np.random.uniform(-1, 1, n_rows),
            'cos_dow': np.random.uniform(-1, 1, n_rows),
            'sin_doy': np.random.uniform(-1, 1, n_rows),
            'cos_doy': np.random.uniform(-1, 1, n_rows),
            'vix_ratio': np.random.uniform(0.8, 1.2, n_rows),
            'vix_in_contango': np.random.randint(0, 2, n_rows),
            'vix_change_1d': np.random.uniform(-0.1, 0.1, n_rows),
            'vix_change_5d': np.random.uniform(-0.1, 0.1, n_rows),
            'vix_percentile': np.random.uniform(0, 1, n_rows),
            'vix_zscore': np.random.uniform(-2, 2, n_rows),
            'vix_norm': np.random.uniform(-2, 2, n_rows),
            'vix3m_norm': np.random.uniform(-2, 2, n_rows),
            'regime': np.random.randint(0, 5, n_rows),
            'options_iv_atm': np.zeros(n_rows),
            'options_put_call_ratio': np.zeros(n_rows),
            'options_volume_norm': np.zeros(n_rows),
            'return_1m': np.random.uniform(-0.01, 0.01, n_rows),
            'return_5m': np.random.uniform(-0.02, 0.02, n_rows),
            'volatility_20': np.random.uniform(0.1, 0.3, n_rows),
            'momentum_20': np.random.uniform(-0.05, 0.05, n_rows),
            'range_pct': np.random.uniform(0, 0.02, n_rows),
            'delta': np.random.uniform(-0.5, 0.5, n_rows),
            'gamma': np.random.uniform(0, 0.1, n_rows),
            'theta': np.random.uniform(-0.1, 0, n_rows),
            'vega': np.random.uniform(0, 0.5, n_rows),
            'vanna': np.random.uniform(-0.1, 0.1, n_rows),
            'charm': np.random.uniform(-0.01, 0.01, n_rows),
            'volga': np.random.uniform(0, 0.05, n_rows),
            'regime_ml': np.random.randint(0, 5, n_rows),
            'regime_adj_position': np.random.uniform(0.5, 1.2, n_rows),
            'regime_adj_delta': np.random.uniform(-0.2, -0.1, n_rows),
            'regime_adj_dte': np.random.randint(-10, 10, n_rows),
            'dte_confidence': np.random.uniform(0.5, 1.0, n_rows),
            'optimal_dte_norm': np.random.uniform(0.2, 1.0, n_rows),
            'trade_prob': np.random.uniform(0.3, 0.7, n_rows),
            'signal_high_prob': np.random.randint(0, 2, n_rows),
            'signal_low_vol': np.random.randint(0, 2, n_rows),
            'signal_crisis': np.zeros(n_rows, dtype=int),
            'signal_contango': np.random.randint(0, 2, n_rows),
            'signal_backwardation': np.random.randint(0, 2, n_rows),
        }
        
        df = pd.DataFrame(data)
        
        # Save to temp dir
        data_dir = tmp_path / "vanna_ml"
        data_dir.mkdir()
        df.to_parquet(data_dir / "TEST_1min_rl.parquet", index=False)
        
        return str(data_dir)
    
    def test_env_creation(self, mock_parquet_data):
        """Test environment can be created."""
        from rl.trading_env import TradingEnvironment
        
        env = TradingEnvironment(
            data_dir=mock_parquet_data,
            symbols=['TEST']
        )
        
        assert env is not None
        assert env.observation_space.shape == (49,)
        assert env.action_space.n == 5
    
    def test_reset(self, mock_parquet_data):
        """Test environment reset."""
        from rl.trading_env import TradingEnvironment
        
        env = TradingEnvironment(
            data_dir=mock_parquet_data,
            symbols=['TEST']
        )
        
        obs, info = env.reset()
        
        assert obs.shape == (49,)
        assert 'capital' in info
        assert info['capital'] == 10000.0
        assert 'symbol' in info
    
    def test_step(self, mock_parquet_data):
        """Test environment step."""
        from rl.trading_env import TradingEnvironment
        
        env = TradingEnvironment(
            data_dir=mock_parquet_data,
            symbols=['TEST']
        )
        
        env.reset()
        
        # Test HOLD action
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (49,)
        assert isinstance(reward, float)
        
        # Test OPEN action
        obs, reward, terminated, truncated, info = env.step(1)
        assert info['position_size'] == 1
        
        # Test CLOSE action
        obs, reward, terminated, truncated, info = env.step(2)
        assert info['position_size'] == 0
    
    def test_episode_completes(self, mock_parquet_data):
        """Test episode runs to completion."""
        from rl.trading_env import TradingEnvironment
        
        env = TradingEnvironment(
            data_dir=mock_parquet_data,
            symbols=['TEST'],
            episode_length=100
        )
        
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            assert steps <= 150  # Safety limit
        
        assert steps >= 100


class TestRegimeClassifier:
    """Tests for RegimeClassifier."""
    
    def test_classify_by_vix(self):
        """Test VIX-based classification."""
        from ml.regime_classifier import RegimeClassifier
        
        clf = RegimeClassifier()
        
        # Low volatility
        result = clf.classify_by_vix(12)
        assert result.regime == 0
        assert result.regime_name == 'low_vol'
        
        # Normal
        result = clf.classify_by_vix(17)
        assert result.regime == 1
        
        # Crisis
        result = clf.classify_by_vix(40)
        assert result.regime == 4
        assert result.regime_name == 'crisis'
    
    def test_strategy_adjustment(self):
        """Test strategy adjustments per regime."""
        from ml.regime_classifier import RegimeClassifier
        
        clf = RegimeClassifier()
        
        # Crisis should reduce position size
        adj = clf.get_strategy_adjustment(4)
        assert adj['position_size'] < 0.5
        assert adj['delta_target'] > -0.1


class TestDTEOptimizer:
    """Tests for DTEOptimizer."""
    
    def test_contango_long_dte(self):
        """Test contango recommends longer DTE."""
        from ml.dte_optimizer import DTEOptimizer
        
        opt = DTEOptimizer()
        
        # Contango (VIX < VIX3M)
        result = opt.get_optimal_dte(vix=15, vix3m=18)
        assert result.dte >= 35  # Should be longer DTE
        assert 'contango' in result.reason.lower()
    
    def test_backwardation_short_dte(self):
        """Test backwardation recommends shorter DTE."""
        from ml.dte_optimizer import DTEOptimizer
        
        opt = DTEOptimizer()
        
        # Backwardation (VIX > VIX3M)
        result = opt.get_optimal_dte(vix=25, vix3m=20)
        assert result.dte <= 35  # Should be shorter DTE
        assert 'backwardation' in result.reason.lower()


class TestVecEnv:
    """Tests for vectorized environment."""
    
    def test_get_available_symbols(self, tmp_path):
        """Test symbol detection."""
        # Create mock files
        data_dir = tmp_path / "vanna_ml"
        data_dir.mkdir()
        (data_dir / "SPY_1min_rl.parquet").touch()
        (data_dir / "QQQ_1min_rl.parquet").touch()
        
        with patch('rl.vec_env.Path') as mock_path:
            mock_path.return_value = data_dir
            # This would need more mocking to fully test
            pass
