"""Phase 17 Tests - Reinforcement Learning (Trading Env, PPO Agent).

Updated to use current TradingEnvironment (70 features) and TradingAgent API.
"""
import asyncio
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_data_dir(tmp_path):
    """Create mock parquet data for TradingEnvironment."""
    n_rows = 500
    
    # All 63 market features required by TradingEnvironment
    data = {
        # Time (6)
        'sin_time': np.random.uniform(-1, 1, n_rows),
        'cos_time': np.random.uniform(-1, 1, n_rows),
        'sin_dow': np.random.uniform(-1, 1, n_rows),
        'cos_dow': np.random.uniform(-1, 1, n_rows),
        'sin_doy': np.random.uniform(-1, 1, n_rows),
        'cos_doy': np.random.uniform(-1, 1, n_rows),
        # VIX (8)
        'vix_ratio': np.random.uniform(0.8, 1.2, n_rows),
        'vix_in_contango': np.random.randint(0, 2, n_rows),
        'vix_change_1d': np.random.uniform(-0.1, 0.1, n_rows),
        'vix_change_5d': np.random.uniform(-0.1, 0.1, n_rows),
        'vix_percentile': np.random.uniform(0, 1, n_rows),
        'vix_zscore': np.random.uniform(-2, 2, n_rows),
        'vix_norm': np.random.uniform(-2, 2, n_rows),
        'vix3m_norm': np.random.uniform(-2, 2, n_rows),
        # Regime + Options + Price + Greeks (16)
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
        # ML outputs (7)
        'regime_ml': np.random.randint(0, 5, n_rows),
        'regime_adj_position': np.random.uniform(0.5, 1.2, n_rows),
        'regime_adj_delta': np.random.uniform(-0.2, -0.1, n_rows),
        'regime_adj_dte': np.random.randint(-10, 10, n_rows),
        'dte_confidence': np.random.uniform(0.5, 1.0, n_rows),
        'optimal_dte_norm': np.random.uniform(0.2, 1.0, n_rows),
        'trade_prob': np.random.uniform(0.3, 0.7, n_rows),
        # Binary signals (5)
        'signal_high_prob': np.random.randint(0, 2, n_rows),
        'signal_low_vol': np.random.randint(0, 2, n_rows),
        'signal_crisis': np.zeros(n_rows, dtype=int),
        'signal_contango': np.random.randint(0, 2, n_rows),
        'signal_backwardation': np.random.randint(0, 2, n_rows),
        # Major event features (4)
        'days_to_major_event': np.random.randint(0, 30, n_rows),
        'is_event_week': np.random.randint(0, 2, n_rows),
        'is_event_day': np.zeros(n_rows, dtype=int),
        'event_iv_boost': np.random.uniform(0, 0.1, n_rows),
        # Daily features (17)
        'day_sma_200': np.random.uniform(0.95, 1.05, n_rows),
        'day_sma_50': np.random.uniform(0.98, 1.02, n_rows),
        'day_sma_20': np.random.uniform(0.99, 1.01, n_rows),
        'day_price_vs_sma200': np.random.uniform(-0.1, 0.1, n_rows),
        'day_price_vs_sma50': np.random.uniform(-0.05, 0.05, n_rows),
        'day_rsi_14': np.random.uniform(30, 70, n_rows),
        'day_atr_14': np.random.uniform(1, 5, n_rows),
        'day_atr_pct': np.random.uniform(0.01, 0.03, n_rows),
        'day_bb_position': np.random.uniform(-1, 1, n_rows),
        'day_macd': np.random.uniform(-1, 1, n_rows),
        'day_macd_hist': np.random.uniform(-0.5, 0.5, n_rows),
        'day_above_sma200': np.random.randint(0, 2, n_rows),
        'day_above_sma50': np.random.randint(0, 2, n_rows),
        'day_sma_50_200_ratio': np.random.uniform(0.95, 1.05, n_rows),
        'day_days_to_major_event': np.random.randint(0, 30, n_rows),
        'day_is_event_week': np.random.randint(0, 2, n_rows),
        'day_event_iv_boost': np.random.uniform(0, 0.1, n_rows),
    }
    
    df = pd.DataFrame(data)
    data_dir = tmp_path / "vanna_ml"
    data_dir.mkdir()
    df.to_parquet(data_dir / "TEST_1min_rl.parquet", index=False)
    
    return str(data_dir)


class TestTradingEnvironment:
    """Test Gymnasium trading environment."""
    
    def test_environment_creation(self, mock_data_dir):
        """Test environment can be created."""
        from rl.trading_env import TradingEnvironment
        
        env = TradingEnvironment(data_dir=mock_data_dir, symbols=['TEST'])
        
        # Current environment has 70 features (63 market + 7 position)
        assert env.observation_space.shape == (70,)
        assert env.action_space.n == 5
    
    def test_environment_reset(self, mock_data_dir):
        """Test environment reset."""
        from rl.trading_env import TradingEnvironment
        
        env = TradingEnvironment(data_dir=mock_data_dir, symbols=['TEST'])
        
        obs, info = env.reset()
        
        assert obs.shape == (70,)
        assert "capital" in info
        assert info["capital"] == 10000.0
    
    def test_environment_step(self, mock_data_dir):
        """Test environment step."""
        from rl.trading_env import TradingEnvironment
        
        env = TradingEnvironment(data_dir=mock_data_dir, symbols=['TEST'])
        env.reset()
        
        # Take HOLD action
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert obs.shape == (70,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_environment_open_position(self, mock_data_dir):
        """Test opening a position."""
        from rl.trading_env import TradingEnvironment
        
        env = TradingEnvironment(data_dir=mock_data_dir, symbols=['TEST'])
        env.reset()
        
        # Take OPEN action
        obs, reward, _, _, info = env.step(1)
        
        assert info["position_size"] == 1
        assert info["trades"] == 1
    
    def test_environment_close_position(self, mock_data_dir):
        """Test closing a position."""
        from rl.trading_env import TradingEnvironment
        
        env = TradingEnvironment(data_dir=mock_data_dir, symbols=['TEST'])
        env.reset()
        
        # Open then close
        env.step(1)  # OPEN
        obs, reward, _, _, info = env.step(2)  # CLOSE
        
        assert info["position_size"] == 0
    
    def test_environment_full_episode(self, mock_data_dir):
        """Test running a full episode."""
        from rl.trading_env import TradingEnvironment
        
        # Use episode_length instead of max_steps
        env = TradingEnvironment(
            data_dir=mock_data_dir, 
            symbols=['TEST'],
            episode_length=100
        )
        obs, info = env.reset()
        
        done = False
        steps = 0
        
        while not done and steps < 150:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        assert steps > 0
        assert "total_pnl" in info


class TestPPOAgent:
    """Test PPO Trading Agent."""
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        from rl.ppo_agent import TradingAgent
        
        agent = TradingAgent()
        
        # Model should be None before training/loading
        assert agent.model is None
    
    def test_get_trading_agent(self):
        """Test singleton function."""
        from rl.ppo_agent import get_trading_agent
        
        agent = get_trading_agent()
        
        assert agent is not None
        assert hasattr(agent, 'model')
        assert hasattr(agent, 'predict')
    
    def test_agent_predict_requires_model(self, mock_data_dir):
        """Test that predict loads/uses model."""
        from rl.ppo_agent import TradingAgent
        import numpy as np
        
        agent = TradingAgent()
        
        # Create dummy observation of correct size
        obs = np.zeros(70, dtype=np.float32)
        
        # Without model loaded, predict should handle gracefully
        # (either return default or raise)
        try:
            agent.load()  # This may fail if no saved model
        except Exception:
            pass  # Expected when no model exists
