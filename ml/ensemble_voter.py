"""
Ensemble Voting System - Combined ML Decision Making

Part of Multi-Model Trading Architecture.
Combines outputs from XGBoost, LSTM, and PPO for final trading decisions.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


from core.logger import get_logger

logger = get_logger()


@dataclass
class EnsembleDecision:
    """Final decision from ensemble voting."""
    
    # Action
    action: str             # OPEN, CLOSE, HOLD, ROLL
    strategy: str           # BUY_CALL, SELL_PUT, etc.
    side: str              # BUY, SELL, HOLD
    dte_bucket: int        # 0=0DTE, 1=WEEKLY, 2=MONTHLY
    
    # Sizing
    size_pct: float        # Position size (0-1 of max allowed)
    
    # Confidence
    confidence: float      # Overall confidence (0-1)
    agreement: float       # Model agreement level (0-1)
    
    # Source data
    xgb_output: dict = field(default_factory=dict)
    lstm_output: dict = field(default_factory=dict)
    ppo_output: dict = field(default_factory=dict)
    
    # Reasoning
    reasoning: str = ""
    
    def should_execute(self, threshold: float = 0.5) -> bool:
        """Should this trade be executed?"""
        return (
            self.action in ["OPEN", "CLOSE", "ROLL"] and
            self.confidence >= threshold and
            self.agreement >= 0.5
        )


class EnsembleVoter:
    """
    Combines multiple model outputs into final trading decision.
    
    Voting Rules:
    1. XGBoost provides strategy recommendation
    2. LSTM provides directional confidence
    3. PPO provides execution timing
    4. Final decision requires majority agreement
    """
    
    # Model weights
    DEFAULT_WEIGHTS = {
        'xgb': 0.4,    # Strategy expert
        'lstm': 0.3,   # Direction expert
        'ppo': 0.3,    # Timing expert
    }
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize voter.
        
        Args:
            weights: Model weights for voting
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        # Load models lazily
        self._strategy_advisor = None
        self._price_forecaster = None
    
    @property
    def strategy_advisor(self):
        """Lazy load StrategyAdvisor."""
        if self._strategy_advisor is None:
            try:
                from ml.strategy_advisor import get_strategy_advisor
                self._strategy_advisor = get_strategy_advisor()
            except Exception as e:
                logger.warning(f"Could not load StrategyAdvisor: {e}")
        return self._strategy_advisor
    
    @property
    def price_forecaster(self):
        """Lazy load PriceForecaster."""
        if self._price_forecaster is None:
            try:
                from ml.price_forecaster import get_price_forecaster
                self._price_forecaster = get_price_forecaster()
            except Exception as e:
                logger.warning(f"Could not load PriceForecaster: {e}")
        return self._price_forecaster
    
    def vote(self, 
             features: Dict[str, Any],
             ppo_output: Dict[str, Any] = None,
             history: list = None) -> EnsembleDecision:
        """
        Get ensemble decision from all models.
        
        Args:
            features: Current market features
            ppo_output: PPO agent output (if available)
            history: Historical data for LSTM (60+ bars)
            
        Returns:
            EnsembleDecision with final recommendation
        """
        # Get XGBoost recommendation
        xgb_output = self._get_xgb_output(features)
        
        # Get LSTM forecast
        lstm_output = self._get_lstm_output(features, history)
        
        # Use provided PPO output or default
        ppo_output = ppo_output or {'action': 'HOLD', 'confidence': 0.5}
        
        # Calculate agreement
        agreement = self._calculate_agreement(xgb_output, lstm_output, ppo_output)
        
        # Combine for final decision
        decision = self._combine_decisions(xgb_output, lstm_output, ppo_output, agreement)
        
        return decision
    
    def _get_xgb_output(self, features: Dict[str, Any]) -> dict:
        """Get XGBoost strategy recommendation."""
        if self.strategy_advisor is None:
            return {'strategy': 'CASH', 'side': 'HOLD', 'confidence': 0.5, 'regime': 1}
        
        try:
            rec = self.strategy_advisor.recommend(features)
            return {
                'strategy': rec.strategy.value,
                'side': rec.side,
                'dte_bucket': rec.dte_bucket,
                'confidence': rec.confidence,
                'regime': rec.regime.value,
                'win_probability': rec.win_probability,
            }
        except Exception as e:
            logger.warning(f"XGBoost failed: {e}")
            return {'strategy': 'CASH', 'side': 'HOLD', 'confidence': 0.5, 'regime': 1}
    
    def _get_lstm_output(self, features: Dict[str, Any], history: list = None) -> dict:
        """Get LSTM price forecast."""
        if self.price_forecaster is None or history is None:
            return {'direction': 'FLAT', 'confidence': 0.1, 'return_5min': 0}
        
        try:
            forecast = self.price_forecaster.predict_from_features(features, history)
            return {
                'direction': forecast.direction,
                'return_5min': forecast.return_5min,
                'return_15min': forecast.return_15min,
                'vol_forecast': forecast.vol_forecast,
                'confidence': forecast.confidence,
            }
        except Exception as e:
            logger.warning(f"LSTM failed: {e}")
            return {'direction': 'FLAT', 'confidence': 0.1, 'return_5min': 0}
    
    def _calculate_agreement(self, xgb: dict, lstm: dict, ppo: dict) -> float:
        """Calculate how much models agree."""
        agreements = []
        
        # Direction agreement
        xgb_bullish = xgb.get('side') == 'BUY' or 'CALL' in xgb.get('strategy', '')
        lstm_bullish = lstm.get('direction') == 'UP'
        ppo_bullish = ppo.get('action') == 'OPEN' and ppo.get('side', '') == 'BUY'
        
        direction_votes = [xgb_bullish, lstm_bullish]
        if ppo.get('action') != 'HOLD':
            direction_votes.append(ppo_bullish)
        
        # Agreement = how many agree / total
        if len(direction_votes) >= 2:
            majority = sum(direction_votes) / len(direction_votes)
            agreements.append(max(majority, 1 - majority))  # Agreement with majority
        
        # Action agreement (should we trade?)
        xgb_trade = xgb.get('strategy', 'CASH') != 'CASH'
        lstm_trade = abs(lstm.get('return_5min', 0)) > 0.001
        ppo_trade = ppo.get('action') != 'HOLD'
        
        trade_votes = [xgb_trade, lstm_trade, ppo_trade]
        trade_agreement = sum(trade_votes) / len(trade_votes)
        agreements.append(max(trade_agreement, 1 - trade_agreement))
        
        return np.mean(agreements) if agreements else 0.5
    
    def _combine_decisions(self, xgb: dict, lstm: dict, ppo: dict, 
                          agreement: float) -> EnsembleDecision:
        """Combine model outputs into final decision."""
        
        # Action from PPO (timing expert)
        ppo_action = ppo.get('action', 'HOLD')
        if ppo_action == 'HOLD' and xgb.get('strategy') == 'CASH':
            action = 'HOLD'
        elif ppo_action in ['OPEN', 'CLOSE', 'ROLL']:
            action = ppo_action
        elif xgb.get('strategy') != 'CASH':
            action = 'OPEN'  # XGBoost says trade, PPO says wait - wait
        else:
            action = 'HOLD'
        
        # Strategy from XGBoost
        strategy = xgb.get('strategy', 'CASH')
        side = xgb.get('side', 'HOLD')
        dte_bucket = xgb.get('dte_bucket', 1)
        
        # Adjust based on LSTM
        if lstm.get('direction') == 'DOWN' and 'CALL' in strategy:
            # LSTM bearish but XGBoost says call - reduce confidence
            action = 'HOLD'  # Conflict - wait
        elif lstm.get('direction') == 'UP' and 'PUT' in strategy:
            action = 'HOLD'  # Conflict - wait
        
        # Size from agreement and confidence
        avg_confidence = (
            xgb.get('confidence', 0.5) * self.weights['xgb'] +
            lstm.get('confidence', 0.5) * self.weights['lstm'] +
            ppo.get('confidence', 0.5) * self.weights['ppo']
        )
        
        # Size: higher agreement = larger position
        size_pct = min(agreement * avg_confidence, 0.5)  # Max 50%
        
        # Overall confidence
        confidence = avg_confidence * agreement
        
        # Reasoning
        reasoning = self._generate_reasoning(xgb, lstm, ppo, action, agreement)
        
        return EnsembleDecision(
            action=action,
            strategy=strategy,
            side=side,
            dte_bucket=dte_bucket,
            size_pct=size_pct,
            confidence=confidence,
            agreement=agreement,
            xgb_output=xgb,
            lstm_output=lstm,
            ppo_output=ppo,
            reasoning=reasoning,
        )
    
    def _generate_reasoning(self, xgb: dict, lstm: dict, ppo: dict,
                           action: str, agreement: float) -> str:
        """Generate human-readable reasoning."""
        parts = [
            f"XGB: {xgb.get('strategy', '?')} ({xgb.get('confidence', 0):.0%})",
            f"LSTM: {lstm.get('direction', '?')} ({lstm.get('return_5min', 0)*100:.2f}%)",
            f"PPO: {ppo.get('action', '?')} ({ppo.get('confidence', 0):.0%})",
            f"Agreement: {agreement:.0%}",
            f"→ {action}",
        ]
        return " | ".join(parts)


# Singleton
_voter: Optional[EnsembleVoter] = None

def get_ensemble_voter() -> EnsembleVoter:
    """Get singleton EnsembleVoter."""
    global _voter
    if _voter is None:
        _voter = EnsembleVoter()
    return _voter


# Quick test
if __name__ == "__main__":
    from core.logger import setup_logger
    setup_logger(level="INFO")
    
    voter = EnsembleVoter()
    
    # Test with sample features
    features = {
        'vix': 18,
        'close': 100,
        'sma50': 98,
        'sma200': 95,
        'rsi': 55,
    }
    
    ppo_output = {
        'action': 'OPEN',
        'confidence': 0.65,
    }
    
    decision = voter.vote(features, ppo_output)
    
    print(f"\n=== Ensemble Decision ===")
    print(f"Action: {decision.action}")
    print(f"Strategy: {decision.strategy}")
    print(f"Side: {decision.side}")
    print(f"DTE: {decision.dte_bucket}")
    print(f"Size: {decision.size_pct:.0%}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Agreement: {decision.agreement:.2f}")
    print(f"\nReasoning: {decision.reasoning}")
    print(f"\nShould Execute: {decision.should_execute()}")
