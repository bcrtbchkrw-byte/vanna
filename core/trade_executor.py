import asyncio
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from core.logger import get_logger
from ml.regime_classifier import get_regime_classifier
from ml.trade_success_predictor import get_trade_success_predictor
from ml.live_feature_builder import get_live_feature_builder

logger = get_logger()

class Action(Enum):
    HOLD = 0
    OPEN = 1
    CLOSE = 2
    INCREASE = 3
    DECREASE = 4

@dataclass
class TradeDecision:
    action: Action
    symbol: str
    quantity: int = 0
    confidence: float = 0.0
    reason: str = ""
    model_outputs: Dict[str, Any] = None

class TradeExecutor:
    """
    Central Brain of the Trading Bot.
    Integrates signals from PPO, LSTM, and XGBoost to make final trading decisions.
    Applies hard risk management rules (Theta/Vega protection).
    """

    def __init__(self, agent=None):
        self._regime_classifier = get_regime_classifier()
        self._trade_predictor = get_trade_success_predictor()
        self._ppo_agent = agent # Use passed agent if available
        self._feature_builder = get_live_feature_builder()
        
        # Configuration
        self.MIN_PROBABILITY_THRESHOLD = 0.55  # Below this, XGBoost VETO
        self.HIGH_PROBABILITY_THRESHOLD = 0.75 # Above this, position size boost
        
        logger.info("🧠 TradeExecutor initialized")

    def _get_ppo_agent(self):
        """Lazy load PPO agent to avoid circular imports."""
        if self._ppo_agent is None:
            from rl.ppo_agent import TradingAgent
            # Use standard path
            self._ppo_agent = TradingAgent(model_path="data/models/ppo_trading_agent")
            if not self._ppo_agent.load():
                logger.error("❌ Failed to load PPO Agent! Will default to HOLD.")
            else:
                logger.info("🧠 PPO Agent loaded successfully")
        return self._ppo_agent

    async def get_trade_decision(self, symbol: str, market_data: Dict[str, Any], current_position: Optional[Dict] = None) -> TradeDecision:
        """
        The Main Decision Loop.
        
        Args:
            symbol: Ticker symbol
            market_data: Live features dict (must match standard format)
            current_position: Dict with current position details or None
            
        Returns:
            TradeDecision object
        """
        response_reason = []
        model_outputs = {}
        
        # 1. Get Market Regime (LSTM) 🌡️
        regime_val = market_data.get('regime', 2) # Default to 2 (Normal)
        # TODO: Decode regime integer to readable label (0=Low, 4=Crisis)
        regime_label = "NORMAL"
        if regime_val == 0: regime_label = "LOW_VOL"
        elif regime_val == 4: regime_label = "CRISIS"
        
        response_reason.append(f"Regime: {regime_label} ({regime_val})")
        
        # 2. Get PPO Agent (Already loaded in init)
        agent = self._get_ppo_agent()

        # 3. Get Trade Quality (XGBoost) 🛡️
        try:
            success_prob = self._trade_predictor.predict(market_data)
        except Exception as e:
            logger.error(f"XGBoost Prediction failed: {e}")
            success_prob = 0.5 # Neutral fallback
            
        model_outputs['xgboost_prob'] = success_prob
        response_reason.append(f"Prob: {success_prob:.2%}")
        
        # 4. Get Strategy Signal (PPO) 🎯
        # PPO Re-enabled - fixing stability via Healthcheck config, not disabling.
        
        if agent is None:
             # Fallback/Error - cannot decide without PPO
             # logger.warning("PPO Agent disabled/unavailable. Defaulting to HOLD signal from PPO.")
             action = Action.HOLD
             ppo_conf = 0.0
        else:
            try:
                # Force garbage collection to free memory before heavy inference
                import gc
                gc.collect()
                
                obs = self._feature_builder.to_observation_vector(market_data)
                
                # CRITICAL: PPO Inference re-enabled per user request.
                # Relying on OMP_NUM_THREADS=1 and torch.set_num_threads(1) to prevent crashes.
                action_int, ppo_conf = agent.predict_with_confidence(obs, deterministic=True)
                action = Action(action_int)
                
                # logger.warning("PPO Inference skipped impacting stability. Defaulting to HOLD.")
                # action = Action.HOLD
                # ppo_conf = 0.0
                
            except Exception as e:
                logger.error(f"PPO Inference Error: {e}")
                action = Action.HOLD
                ppo_conf = 0.0
        
        model_outputs['ppo_action'] = action.name
        model_outputs['ppo_confidence'] = ppo_conf
        response_reason.append(f"PPO: {action.name} ({ppo_conf:.2f})")
        
        # ==============================================================================
        # 4. THE COUNCIL LOGIC (VETO & FILTERS)
        # ==============================================================================
        
        final_action = action
        final_reason = ", ".join(response_reason)
        
        # A. XGBoost Veto (Only applies to OPEN and INCREASE)
        if action in [Action.OPEN, Action.INCREASE]:
            if success_prob < self.MIN_PROBABILITY_THRESHOLD:
                final_action = Action.HOLD
                final_reason = f"⛔ VETO by XGBoost (Prob {success_prob:.2f} < {self.MIN_PROBABILITY_THRESHOLD}). PPO wanted {action.name}."
                return TradeDecision(final_action, symbol, 0, success_prob, final_reason, model_outputs)

        # B. Regime Veto (Theta Protection)
        # If Regime is LOW_VOLatility (0) and we want to OPEN Long Option -> Danger
        # Assuming we are long-only options bot for now (or strategy is implied)
        if action == Action.OPEN and regime_val == 0:
             # Check if we have high probability to override
             if success_prob < 0.70: # Only allowed if really high conviction
                 final_action = Action.HOLD
                 final_reason = f"⛔ VETO by Regime (Low Vol = Theta Burn). Prob {success_prob:.2f} not high enough to risk it."
                 return TradeDecision(final_action, symbol, 0, success_prob, final_reason, model_outputs)

        # C. Crisis Protection
        # If Regime is CRISIS (4) -> Reduce existing positions, Don't OPEN new ones (unless PPO is specifically trained for it)
        # For safety, let's say we trust PPO in crisis, but maybe cut size
        
        # D. Open Interest Check (Sentiment)
        # We can use the 'put_call_oi_ratio' from market_data
        pc_ratio = market_data.get('put_call_oi_ratio', 1.0)
        if action == Action.OPEN:
            # If PC Ratio is extremely high (> 2.5), it might be a bottom (bullish) or crash (bearish).
            # If PC Ratio is extremely low (< 0.5), it might be a top (bearish) or rally (bullish).
            # Let's rely on XGBoost to interpret this, but log it.
            pass

        # 5. Position Sizing
        # Determine quantity based on confidence and regime
        quantity = 1 # Default
        
        if final_action in [Action.OPEN, Action.INCREASE]:
            if regime_val == 2: # Normal
                quantity = 1
            elif regime_val == 1 or regime_val == 3: # Transitions
                quantity = 1
            
            # Boost size if XGBoost is very confident
            if success_prob > self.HIGH_PROBABILITY_THRESHOLD:
                quantity = 2 # Double down
                final_reason += " (Size Boosted by High Prob)"
        
        # If Close, quantity is usually "ALL" or partial. For now assume full close or handled by OrderManager
        if final_action == Action.CLOSE:
            quantity = -1 # Signal to close all
            
        return TradeDecision(
            action=final_action,
            symbol=symbol,
            quantity=quantity,
            confidence=float(success_prob),
            reason=final_reason,
            model_outputs=model_outputs
        )
