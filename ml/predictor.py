"""
Trade Predictor Module

Wrapper around TradeSuccessPredictor for backward compatibility.
Uses real XGBoost model instead of random heuristics.
"""
from typing import Any, Dict, Optional
import numpy as np

from loguru import logger

from ml.trade_success_predictor import get_trade_success_predictor


class TradePredictor:
    """
    Predicts trade success probability using trained ML model.
    
    This is a facade over TradeSuccessPredictor for simpler interface.
    """
    
    def __init__(self):
        self._predictor = None
        self._init_predictor()
        
    def _init_predictor(self):
        """Initialize the real predictor."""
        try:
            self._predictor = get_trade_success_predictor()
            if self._predictor.model is not None:
                logger.info("TradePredictor: Using trained XGBoost model")
            else:
                logger.warning("TradePredictor: Model not trained, using rule-based fallback")
        except Exception as e:
            logger.warning(f"TradePredictor init error: {e}")
            self._predictor = None
        
    def load_model(self, path: str):
        """Load trained model from path."""
        if self._predictor is not None:
            self._predictor.load(path)
        
    def predict_probability(self, trade_setup: Dict[str, Any]) -> float:
        """
        Predict probability of profit.
        
        Args:
            trade_setup: Dict with 'symbol', 'strategy', 'market_data'
            
        Returns:
            Float 0.0 to 1.0 (Probability of Success)
        """
        market = trade_setup.get('market_data', {})
        vix = market.get('vix', 18)
        iv_rank = market.get('iv_rank', 0.5)
        volume = market.get('volume', 0)
        strategy = trade_setup.get('strategy', 'UNKNOWN')
        symbol = trade_setup.get('symbol', 'UNKNOWN')
        
        # Try using trained model
        if self._predictor is not None and self._predictor.model is not None:
            try:
                # Build feature dict for predictor
                features = {
                    'vix': vix,
                    'iv_rank': iv_rank * 100 if iv_rank <= 1 else iv_rank,
                    'volume_ratio': min(volume / 1000000, 5) if volume > 0 else 1.0,
                    'rsi_14': market.get('rsi_14', 50),
                    'macd_hist': market.get('macd_hist', 0),
                    'bb_position': market.get('bb_position', 0.5),
                    'atr_pct': market.get('atr_pct', 0.02),
                }
                
                prob = self._predictor.predict_proba(features)
                logger.debug(f"🤖 ML Prediction ({symbol}/{strategy}): {prob:.1%}")
                return float(prob)
                
            except Exception as e:
                logger.debug(f"Model prediction failed, using fallback: {e}")
        
        # Rule-based fallback (no random!)
        prob = self._calculate_rule_based_prob(vix, iv_rank, strategy)
        logger.debug(f"📐 Rule-based Prediction ({symbol}/{strategy}): {prob:.1%}")
        return prob
    
    def _calculate_rule_based_prob(
        self, 
        vix: float, 
        iv_rank: float,
        strategy: str
    ) -> float:
        """
        Rule-based probability calculation.
        
        NO RANDOM! Uses deterministic rules based on market conditions.
        """
        base_prob = 0.50  # Neutral baseline
        
        # VIX adjustments for credit strategies
        if strategy in ['BULL_PUT_SPREAD', 'BEAR_CALL_SPREAD', 'IRON_CONDOR', 
                        'IRON_BUTTERFLY', 'JADE_LIZARD']:
            # High VIX = better for premium selling
            if vix > 25:
                base_prob += 0.15
            elif vix > 20:
                base_prob += 0.10
            elif vix < 15:
                base_prob -= 0.10
            
            # High IV rank = sell premium
            if iv_rank > 0.7:
                base_prob += 0.10
            elif iv_rank > 0.5:
                base_prob += 0.05
        
        # Debit strategies prefer low IV
        elif strategy in ['PUT_DEBIT_SPREAD', 'CALL_DEBIT_SPREAD', 
                          'POOR_MANS_COVERED_CALL']:
            if iv_rank < 0.3:
                base_prob += 0.10
            elif iv_rank > 0.7:
                base_prob -= 0.10
        
        # Clamp to valid range
        return max(0.20, min(0.85, base_prob))


# Singleton
_predictor: Optional[TradePredictor] = None


def get_trade_predictor() -> TradePredictor:
    """Get or create trade predictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = TradePredictor()
    return _predictor
