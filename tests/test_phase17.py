"""Phase 17 Tests - Reinforcement Learning (Trading Env, PPO Agent)."""
import asyncio

import pytest


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class TestTradingEnvironment:
    """Test Gymnasium trading environment."""
    
    def test_environment_creation(self):
        """Test environment can be created."""
        from rl.trading_env import TradingEnvironment
        
        env = TradingEnvironment()
        
        assert env.observation_space.shape == (6,)
        assert env.action_space.n == 5
    
    def test_environment_reset(self):
        """Test environment reset."""
        from rl.trading_env import make_trading_env
        
        env = make_trading_env()
        
        obs, info = env.reset()
        
        assert obs.shape == (6,)
        assert "capital" in info
        assert info["capital"] == 10000.0
    
    def test_environment_step(self):
        """Test environment step."""
        from rl.trading_env import TradingEnvironment
        
        env = TradingEnvironment()
        env.reset()
        
        # Take HOLD action
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_environment_open_position(self):
        """Test opening a position."""
        from rl.trading_env import TradingEnvironment
        
        env = TradingEnvironment()
        env.reset()
        
        # Take OPEN action
        obs, reward, _, _, info = env.step(1)
        
        assert info["position_size"] == 1
        assert info["trades"] == 1
    
    def test_environment_close_position(self):
        """Test closing a position."""
        from rl.trading_env import TradingEnvironment
        
        env = TradingEnvironment()
        env.reset()
        
        # Open then close
        env.step(1)  # OPEN
        obs, reward, _, _, info = env.step(2)  # CLOSE
        
        assert info["position_size"] == 0
    
    def test_environment_full_episode(self):
        """Test running a full episode."""
        from rl.trading_env import TradingEnvironment
        
        env = TradingEnvironment(max_steps=100)
        obs, info = env.reset()
        
        done = False
        steps = 0
        
        while not done and steps < 100:
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
        from rl.ppo_agent import PPOTradingAgent
        
        agent = PPOTradingAgent()
        
        # Should be in heuristic mode (no trained model)
        assert agent._model_loaded is False
    
    def test_heuristic_predict_no_position(self):
        """Test heuristic prediction without position."""
        from rl.ppo_agent import PPOTradingAgent
        
        agent = PPOTradingAgent()
        
        # Observation: [price_change, vix_norm, delta, pnl, days, position]
        # Good VIX (20), no position
        obs = [0.01, 0.2, -0.15, 0.0, 0.0, 0.0]
        
        action, confidence = agent.predict(obs)
        
        assert action == 1  # Should suggest OPEN
        assert confidence > 0.5
    
    def test_heuristic_predict_with_profit(self):
        """Test heuristic prediction with profitable position."""
        from rl.ppo_agent import PPOTradingAgent
        
        agent = PPOTradingAgent()
        
        # Has position with 60% profit
        obs = [0.01, 0.2, -0.15, 0.6, 0.5, 1.0]
        
        action, confidence = agent.predict(obs)
        
        assert action == 2  # Should suggest CLOSE
    
    def test_heuristic_predict_with_loss(self):
        """Test heuristic prediction with losing position."""
        from rl.ppo_agent import get_ppo_agent
        
        agent = get_ppo_agent()
        
        # Has position with 35% loss
        obs = [0.01, 0.2, -0.15, -0.35, 0.5, 1.0]
        
        action, confidence = agent.predict(obs)
        
        assert action == 2  # Should suggest CLOSE (stop loss)
    
    def test_action_name_conversion(self):
        """Test action number to name conversion."""
        from rl.ppo_agent import PPOTradingAgent
        
        agent = PPOTradingAgent()
        
        assert agent.get_action_name(0) == "HOLD"
        assert agent.get_action_name(1) == "OPEN"
        assert agent.get_action_name(2) == "CLOSE"
        assert agent.get_action_name(3) == "INCREASE"
        assert agent.get_action_name(4) == "DECREASE"
