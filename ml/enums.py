#!/usr/bin/env python3
"""
ml/enums.py

CENTRÁLNÍ DEFINICE ENUMŮ PRO CELÝ VANNA PROJEKT

Všechny enumy jsou definovány ZDE a importovány odjinud.
Tím se zabrání duplicitám a nekonzistencím.

Usage:
    from ml.enums import Strategy, MarketRegime, DTEBucket, Action
"""

from enum import Enum, auto
from typing import List


# =============================================================================
# MARKET REGIME
# =============================================================================

class MarketRegime(Enum):
    """
    Klasifikace tržního režimu podle VIX.
    
    VIX Thresholds:
    - CALM: VIX < 15
    - NORMAL: 15 <= VIX < 20
    - ELEVATED: 20 <= VIX < 25
    - HIGH_VOL: 25 <= VIX < 35
    - CRISIS: VIX >= 35
    """
    CALM = 0        # VIX < 15 - Ideal pro theta strategies
    NORMAL = 1      # VIX 15-20 - Standard trading
    ELEVATED = 2    # VIX 20-25 - Caution
    HIGH_VOL = 3    # VIX 25-35 - Defined risk only
    CRISIS = 4      # VIX > 35 - Cash or hedges only
    
    @classmethod
    def from_vix(cls, vix: float) -> 'MarketRegime':
        """Classify regime from VIX value."""
        if vix < 15:
            return cls.CALM
        elif vix < 20:
            return cls.NORMAL
        elif vix < 25:
            return cls.ELEVATED
        elif vix < 35:
            return cls.HIGH_VOL
        else:
            return cls.CRISIS
    
    @property
    def allows_new_trades(self) -> bool:
        """Can we open new positions in this regime?"""
        return self in (MarketRegime.CALM, MarketRegime.NORMAL, MarketRegime.ELEVATED)
    
    @property
    def max_position_size_pct(self) -> float:
        """Maximum position size as % of account in this regime."""
        return {
            MarketRegime.CALM: 0.25,      # 25%
            MarketRegime.NORMAL: 0.20,    # 20%
            MarketRegime.ELEVATED: 0.15,  # 15%
            MarketRegime.HIGH_VOL: 0.10,  # 10%
            MarketRegime.CRISIS: 0.05,    # 5%
        }[self]


# =============================================================================
# STRATEGY
# =============================================================================

class Strategy(Enum):
    """
    Všechny podporované opční strategie.
    
    Categories:
    - Premium Selling (credit): IRON_CONDOR, PUT_CREDIT_SPREAD, CALL_CREDIT_SPREAD, SHORT_STRANGLE
    - Directional Debit: BULL_CALL_SPREAD, BEAR_PUT_SPREAD, BULL_PUT_SPREAD, BEAR_CALL_SPREAD
    - Volatility: LONG_STRADDLE, LONG_STRANGLE, IRON_BUTTERFLY
    - Time Spreads: CALENDAR_SPREAD, DIAGONAL_SPREAD
    - No Trade: CASH
    """
    # No position
    CASH = "CASH"
    
    # Premium Selling (Credit Strategies)
    IRON_CONDOR = "IRON_CONDOR"
    IRON_BUTTERFLY = "IRON_BUTTERFLY"
    PUT_CREDIT_SPREAD = "PUT_CREDIT_SPREAD"
    CALL_CREDIT_SPREAD = "CALL_CREDIT_SPREAD"
    SHORT_STRANGLE = "SHORT_STRANGLE"
    SHORT_STRADDLE = "SHORT_STRADDLE"
    
    # Directional (Debit Strategies)
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"
    BULL_PUT_SPREAD = "BULL_PUT_SPREAD"      # Same as PUT_CREDIT_SPREAD
    BEAR_CALL_SPREAD = "BEAR_CALL_SPREAD"    # Same as CALL_CREDIT_SPREAD
    
    # Volatility Strategies
    LONG_STRADDLE = "LONG_STRADDLE"
    LONG_STRANGLE = "LONG_STRANGLE"
    
    # Time Spreads
    CALENDAR_SPREAD = "CALENDAR_SPREAD"
    DIAGONAL_SPREAD = "DIAGONAL_SPREAD"
    
    # Single Leg (rare)
    NAKED_PUT = "NAKED_PUT"
    NAKED_CALL = "NAKED_CALL"
    COVERED_CALL = "COVERED_CALL"
    CASH_SECURED_PUT = "CASH_SECURED_PUT"
    
    @property
    def is_credit(self) -> bool:
        """Is this a credit (premium selling) strategy?"""
        return self in (
            Strategy.IRON_CONDOR,
            Strategy.IRON_BUTTERFLY,
            Strategy.PUT_CREDIT_SPREAD,
            Strategy.CALL_CREDIT_SPREAD,
            Strategy.BULL_PUT_SPREAD,
            Strategy.BEAR_CALL_SPREAD,
            Strategy.SHORT_STRANGLE,
            Strategy.SHORT_STRADDLE,
            Strategy.NAKED_PUT,
            Strategy.NAKED_CALL,
            Strategy.COVERED_CALL,
            Strategy.CASH_SECURED_PUT,
        )
    
    @property
    def is_debit(self) -> bool:
        """Is this a debit strategy?"""
        return self in (
            Strategy.BULL_CALL_SPREAD,
            Strategy.BEAR_PUT_SPREAD,
            Strategy.LONG_STRADDLE,
            Strategy.LONG_STRANGLE,
            Strategy.CALENDAR_SPREAD,
            Strategy.DIAGONAL_SPREAD,
        )
    
    @property
    def is_defined_risk(self) -> bool:
        """Does this strategy have defined/limited risk?"""
        undefined_risk = (
            Strategy.SHORT_STRANGLE,
            Strategy.SHORT_STRADDLE,
            Strategy.NAKED_PUT,
            Strategy.NAKED_CALL,
        )
        return self not in undefined_risk and self != Strategy.CASH
    
    @property
    def num_legs(self) -> int:
        """Number of option legs in this strategy."""
        return {
            Strategy.CASH: 0,
            Strategy.NAKED_PUT: 1,
            Strategy.NAKED_CALL: 1,
            Strategy.COVERED_CALL: 1,
            Strategy.CASH_SECURED_PUT: 1,
            Strategy.PUT_CREDIT_SPREAD: 2,
            Strategy.CALL_CREDIT_SPREAD: 2,
            Strategy.BULL_PUT_SPREAD: 2,
            Strategy.BEAR_CALL_SPREAD: 2,
            Strategy.BULL_CALL_SPREAD: 2,
            Strategy.BEAR_PUT_SPREAD: 2,
            Strategy.SHORT_STRANGLE: 2,
            Strategy.SHORT_STRADDLE: 2,
            Strategy.LONG_STRADDLE: 2,
            Strategy.LONG_STRANGLE: 2,
            Strategy.CALENDAR_SPREAD: 2,
            Strategy.DIAGONAL_SPREAD: 2,
            Strategy.IRON_CONDOR: 4,
            Strategy.IRON_BUTTERFLY: 4,
        }.get(self, 2)
    
    @classmethod
    def credit_strategies(cls) -> List['Strategy']:
        """Get all credit strategies."""
        return [s for s in cls if s.is_credit]
    
    @classmethod
    def debit_strategies(cls) -> List['Strategy']:
        """Get all debit strategies."""
        return [s for s in cls if s.is_debit]
    
    @classmethod
    def defined_risk_strategies(cls) -> List['Strategy']:
        """Get all defined risk strategies."""
        return [s for s in cls if s.is_defined_risk]


# =============================================================================
# DTE BUCKET
# =============================================================================

class DTEBucket(Enum):
    """
    Days To Expiration categories.
    
    0DTE: Same day expiration (SPX, SPY)
    WEEKLY: 1-7 days
    MONTHLY: 8-45 days (sweet spot for theta)
    LEAPS: 46+ days
    """
    ZERO_DTE = 0    # 0DTE - intraday
    WEEKLY = 1      # 1-7 days
    MONTHLY = 2     # 8-45 days
    LEAPS = 3       # 46+ days
    
    @classmethod
    def from_dte(cls, dte: int) -> 'DTEBucket':
        """Classify DTE into bucket."""
        if dte == 0:
            return cls.ZERO_DTE
        elif dte <= 7:
            return cls.WEEKLY
        elif dte <= 45:
            return cls.MONTHLY
        else:
            return cls.LEAPS
    
    @property
    def typical_dte_range(self) -> tuple:
        """Get typical DTE range for this bucket."""
        return {
            DTEBucket.ZERO_DTE: (0, 0),
            DTEBucket.WEEKLY: (1, 7),
            DTEBucket.MONTHLY: (21, 45),
            DTEBucket.LEAPS: (60, 180),
        }[self]
    
    @property
    def theta_decay_rate(self) -> str:
        """Relative theta decay rate."""
        return {
            DTEBucket.ZERO_DTE: "EXTREME",
            DTEBucket.WEEKLY: "HIGH",
            DTEBucket.MONTHLY: "MODERATE",
            DTEBucket.LEAPS: "LOW",
        }[self]


# =============================================================================
# ACTION
# =============================================================================

class Action(Enum):
    """
    Trading actions that PPO/Brain can take.
    """
    HOLD = 0              # Do nothing
    OPEN_POSITION = 1     # Open new position
    CLOSE_POSITION = 2    # Close existing position
    ROLL_POSITION = 3     # Roll to different expiration
    HEDGE = 4             # Add protective hedge
    REDUCE_SIZE = 5       # Partial close
    ADD_TO_POSITION = 6   # Scale in


# =============================================================================
# OPTION LEG TYPE
# =============================================================================

class LegType(Enum):
    """Type of option leg in a spread."""
    LONG_CALL = "LONG_CALL"
    SHORT_CALL = "SHORT_CALL"
    LONG_PUT = "LONG_PUT"
    SHORT_PUT = "SHORT_PUT"


class OptionRight(Enum):
    """Option right (call or put)."""
    CALL = "C"
    PUT = "P"


# =============================================================================
# SIGNAL TYPES (for unusual activity)
# =============================================================================

class SignalType(Enum):
    """Types of unusual activity signals."""
    VOLUME_SPIKE = "VOLUME_SPIKE"
    OI_CHANGE = "OI_CHANGE"
    WHALE_TRADE = "WHALE_TRADE"
    SWEEP = "SWEEP"
    BLOCK_TRADE = "BLOCK_TRADE"
    PUT_CALL_SKEW = "PUT_CALL_SKEW"
    IV_SPIKE = "IV_SPIKE"
    SMART_MONEY = "SMART_MONEY"


class SignalDirection(Enum):
    """Direction implied by the signal."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# ALERT LEVELS (for circuit breaker)
# =============================================================================

class AlertLevel(Enum):
    """Circuit breaker alert levels."""
    NORMAL = 0      # All systems go
    CAUTION = 1     # Reduce size
    WARNING = 2     # New positions only
    DANGER = 3      # Close only
    CRITICAL = 4    # Emergency liquidation


class EmergencyAction(Enum):
    """Emergency actions for circuit breaker."""
    NONE = "NONE"
    REDUCE_EXPOSURE = "REDUCE_EXPOSURE"
    STOP_NEW_TRADES = "STOP_NEW_TRADES"
    CLOSE_LOSING = "CLOSE_LOSING"
    HEDGE_ALL = "HEDGE_ALL"
    LIQUIDATE = "LIQUIDATE"


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    'MarketRegime',
    'Strategy',
    'DTEBucket',
    'Action',
    'LegType',
    'OptionRight',
    'SignalType',
    'SignalDirection',
    'AlertLevel',
    'EmergencyAction',
]
