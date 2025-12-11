"""
Neural Gatekeeper for Vanna Trading Bot.

A neural network-based trade signal filter that predicts
the probability of trade success based on market conditions,
Greeks, and historical performance patterns.

Uses TensorFlow for inference (trained model loaded from disk).
Falls back to heuristic scoring if no trained model is available.
"""
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.logger import get_logger


@dataclass
class GatekeeperInput:
    """Input features for the gatekeeper model."""
    symbol: str
    strategy: str
    vix: float
    delta: float
    theta: float
    iv_rank: float
    days_to_expiry: int
    credit: float
    width: float
    current_price: float
    sma_distance: float  # Distance from SMA50 as %


@dataclass
class GatekeeperResult:
    """Result from gatekeeper analysis."""
    approved: bool
    probability: float
    reason: str
    confidence: str


class NeuralGatekeeper:
    """
    Neural network gatekeeper for filtering trade signals.
    
    Uses a trained model (if available) or falls back to
    rule-based heuristics for trade approval.
    """
    
    # Approval thresholds
    MIN_PROBABILITY = 0.55  # Minimum 55% success probability
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    
    def __init__(self, model_path: str | None = None) -> None:
        self.logger = get_logger()
        self.model_path = model_path or "models/gatekeeper.keras"
        self._model: Any = None
        self._model_loaded = False
        
        self._try_load_model()
    
    def _try_load_model(self) -> bool:
        """Attempt to load the trained TensorFlow model."""
        if not os.path.exists(self.model_path):
            self.logger.info(
                f"No trained model at {self.model_path}, using heuristic mode"
            )
            return False
        
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(self.model_path)
            self._model_loaded = True
            self.logger.info(f"Loaded gatekeeper model from {self.model_path}")
            return True
        except ImportError:
            self.logger.warning(
                "TensorFlow not installed - using heuristic mode. "
                "Install with: pip install tensorflow"
            )
            return False
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def _prepare_features(self, inp: GatekeeperInput) -> np.ndarray:
        """Convert input to feature array for model inference."""
        # Strategy encoding (one-hot style score)
        strategy_score = {
            "BULL_PUT": 0.7,
            "BEAR_CALL": 0.7,
            "IRON_CONDOR": 0.8,
            "JADE_LIZARD": 0.6,
            "PMCC": 0.5,
            "CREDIT_SPREAD": 0.7
        }.get(inp.strategy.upper(), 0.5)
        
        features = np.array([
            inp.vix / 50.0,  # Normalize VIX
            abs(inp.delta),  # Delta magnitude
            inp.theta * 10,  # Scale theta
            inp.iv_rank / 100.0,  # IV rank as decimal
            inp.days_to_expiry / 45.0,  # Normalize DTE
            inp.credit / inp.width if inp.width > 0 else 0,  # ROI
            inp.sma_distance,  # SMA distance
            strategy_score  # Strategy suitability
        ], dtype=np.float32)
        
        return features.reshape(1, -1)
    
    def _heuristic_score(self, inp: GatekeeperInput) -> float:
        """
        Calculate probability using rule-based heuristics.
        Used when no trained model is available.
        """
        score = 0.5  # Base score
        
        # VIX scoring - higher VIX better for premium selling
        if 15 <= inp.vix <= 25:
            score += 0.1  # Optimal range
        elif inp.vix > 30:
            score -= 0.1  # Too high, risky
        elif inp.vix < 12:
            score -= 0.05  # Low premium
        
        # Delta scoring - prefer lower delta for safety
        delta_abs = abs(inp.delta)
        if delta_abs <= 0.20:
            score += 0.1
        elif delta_abs <= 0.30:
            score += 0.05
        elif delta_abs > 0.40:
            score -= 0.1
        
        # Theta scoring - positive theta is good
        if inp.theta > 0:
            score += min(inp.theta * 5, 0.1)
        
        # IV Rank scoring - higher IV rank better for selling
        if inp.iv_rank >= 50:
            score += 0.1
        elif inp.iv_rank >= 30:
            score += 0.05
        elif inp.iv_rank < 20:
            score -= 0.1
        
        # DTE scoring - prefer 30-45 DTE
        if 30 <= inp.days_to_expiry <= 45:
            score += 0.1
        elif 21 <= inp.days_to_expiry < 30:
            score += 0.05
        elif inp.days_to_expiry < 14:
            score -= 0.1
        
        # ROI scoring
        roi = (inp.credit / inp.width * 100) if inp.width > 0 else 0
        if roi >= 15:
            score += 0.1
        elif roi >= 10:
            score += 0.05
        elif roi < 5:
            score -= 0.05
        
        # SMA distance scoring - prefer near SMA
        if abs(inp.sma_distance) <= 0.02:
            score += 0.05
        elif abs(inp.sma_distance) > 0.05:
            score -= 0.05
        
        return max(0.1, min(0.95, score))  # Clamp to valid range
    
    def evaluate(self, inp: GatekeeperInput) -> GatekeeperResult:
        """
        Evaluate a trade setup and return approval decision.
        
        Args:
            inp: GatekeeperInput with trade setup details
            
        Returns:
            GatekeeperResult with approval and probability
        """
        if self._model_loaded and self._model is not None:
            # Use neural network
            features = self._prepare_features(inp)
            probability = float(self._model.predict(features, verbose=0)[0][0])
        else:
            # Use heuristics
            probability = self._heuristic_score(inp)
        
        # Determine approval
        approved = probability >= self.MIN_PROBABILITY
        
        # Confidence level
        if probability >= self.HIGH_CONFIDENCE_THRESHOLD:
            confidence = "HIGH"
        elif probability >= self.MIN_PROBABILITY:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Generate reason
        if approved:
            reason = f"Trade approved with {probability:.0%} success probability"
        else:
            reason = f"Trade rejected: {probability:.0%} < {self.MIN_PROBABILITY:.0%} threshold"
        
        result = GatekeeperResult(
            approved=approved,
            probability=probability,
            reason=reason,
            confidence=confidence
        )
        
        self.logger.info(
            f"Gatekeeper: {inp.symbol} {inp.strategy} -> "
            f"{'✅' if approved else '❌'} {probability:.0%}"
        )
        
        return result
    
    def batch_evaluate(
        self, inputs: list[GatekeeperInput]
    ) -> list[GatekeeperResult]:
        """Evaluate multiple trade setups."""
        return [self.evaluate(inp) for inp in inputs]


# Singleton
_gatekeeper: NeuralGatekeeper | None = None


def get_neural_gatekeeper() -> NeuralGatekeeper:
    """Get global neural gatekeeper instance."""
    global _gatekeeper
    if _gatekeeper is None:
        _gatekeeper = NeuralGatekeeper()
    return _gatekeeper
