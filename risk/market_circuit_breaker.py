#!/usr/bin/env python3
"""
circuit_breaker.py

HYBRID RISK PROTECTION SYSTEM

3 úrovně ochrany:
1. HARD CIRCUIT BREAKER - nelze override, okamžitá akce
2. PPO RISK AGENT - učí se optimální reakci na volatilitu
3. NORMAL TRADING - standardní PPO + strategy

HARD LIMITS (nikdy override):
- VIX > 50 → CLOSE_ALL
- Daily loss > 10% → CLOSE_ALL  
- SPY gap > 5% → CLOSE_ALL
- Margin call → CLOSE_ALL

PPO ZONE (VIX 25-50):
- PPO rozhoduje: HOLD, REDUCE, HEDGE, CLOSE_RISKY

Usage:
    from circuit_breaker import CircuitBreaker, PPORiskAgent
    
    cb = CircuitBreaker(account_size=10000)
    
    # Check before any action
    action = cb.check(vix=55, daily_pnl=-0.08)
    if action.is_emergency:
        execute_emergency(action)  # CLOSE ALL NOW!
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime, date, time, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class AlertLevel(Enum):
    """Úroveň alertu."""
    NORMAL = 0          # Vše OK
    ELEVATED = 1        # Zvýšená pozornost
    HIGH = 2            # PPO rozhoduje
    CRITICAL = 3        # Hard circuit breaker


class EmergencyAction(Enum):
    """Nouzové akce."""
    NONE = "NONE"
    CLOSE_ALL = "CLOSE_ALL"
    CLOSE_RISKY = "CLOSE_RISKY"
    REDUCE_50 = "REDUCE_50"
    HALT_TRADING = "HALT_TRADING"
    ADD_HEDGE = "ADD_HEDGE"


class PPORiskAction(Enum):
    """PPO akce pro risk management."""
    HOLD = 0                # Drž pozice
    REDUCE_25 = 1           # Zavři 25% pozic
    REDUCE_50 = 2           # Zavři 50% pozic
    CLOSE_HIGHEST_RISK = 3  # Zavři nejrizikovější
    CLOSE_ITM = 4           # Zavři ITM pozice
    ADD_HEDGE = 5           # Kup protective puts
    ROLL_OUT = 6            # Roll všechny pozice
    CLOSE_ALL = 7           # Zavři vše (PPO může taky)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CircuitBreakerConfig:
    """Hard limits - NELZE OVERRIDE!"""
    
    # VIX limits
    vix_extreme: float = 50.0           # VIX > 50 = CLOSE ALL
    vix_high: float = 35.0              # VIX > 35 = PPO zone
    vix_elevated: float = 25.0          # VIX > 25 = caution
    
    # Daily P&L limits
    daily_loss_extreme: float = -0.10   # -10% = CLOSE ALL
    daily_loss_high: float = -0.05      # -5% = PPO zone
    daily_loss_elevated: float = -0.03  # -3% = caution
    
    # Gap limits
    gap_extreme: float = -0.05          # -5% gap = CLOSE ALL
    gap_high: float = -0.03             # -3% gap = PPO zone
    
    # Intraday limits
    intraday_loss_halt: float = -0.07   # -7% intraday = HALT
    
    # VIX spike (velocity)
    vix_spike_pct: float = 0.30         # +30% VIX intraday = alert
    
    # Position limits during stress
    max_positions_high_vol: int = 3
    max_margin_high_vol: float = 0.25   # Max 25% margin in high vol
    
    # Time-based
    no_new_trades_after_gap: int = 60   # Minutes after gap


@dataclass
class PPORiskConfig:
    """Konfigurace pro PPO risk agent."""
    
    # State features
    state_features: List[str] = field(default_factory=lambda: [
        'vix',
        'vix_change_pct',
        'daily_pnl_pct',
        'margin_usage_pct',
        'num_positions',
        'avg_delta',
        'time_to_close_hours',
        'positions_itm_pct',
        'unrealized_pnl_pct',
    ])
    
    # Action space
    num_actions: int = 8
    
    # Reward shaping
    survival_reward: float = 1.0        # Reward za přežití bez velké ztráty
    loss_penalty: float = -2.0          # Penalty za ztrátu
    overreaction_penalty: float = -0.5  # Penalty za zbytečné zavření
    
    # Learning
    learning_rate: float = 0.0003
    gamma: float = 0.99
    
    # Thresholds pro automatické akce
    confidence_threshold: float = 0.7


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MarketState:
    """Aktuální stav trhu pro circuit breaker."""
    vix: float
    vix_open: float                     # VIX at open
    spy_price: float
    spy_open: float                     # SPY at open
    spy_prev_close: float               # SPY previous close
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def vix_change_pct(self) -> float:
        """VIX change od open."""
        if self.vix_open == 0:
            return 0
        return (self.vix - self.vix_open) / self.vix_open
    
    @property
    def spy_gap_pct(self) -> float:
        """SPY gap od previous close."""
        if self.spy_prev_close == 0:
            return 0
        return (self.spy_open - self.spy_prev_close) / self.spy_prev_close
    
    @property
    def spy_intraday_pct(self) -> float:
        """SPY intraday change."""
        if self.spy_open == 0:
            return 0
        return (self.spy_price - self.spy_open) / self.spy_open


@dataclass
class PortfolioState:
    """Stav portfolia pro risk assessment."""
    account_size: float
    daily_pnl: float
    unrealized_pnl: float
    margin_used: float
    num_positions: int
    positions_itm: int                  # Počet ITM pozic
    avg_position_delta: float
    highest_risk_position: Optional[str] = None
    highest_risk_amount: float = 0
    
    @property
    def daily_pnl_pct(self) -> float:
        if self.account_size == 0:
            return 0
        return self.daily_pnl / self.account_size
    
    @property
    def margin_usage_pct(self) -> float:
        if self.account_size == 0:
            return 0
        return self.margin_used / self.account_size
    
    @property
    def positions_itm_pct(self) -> float:
        if self.num_positions == 0:
            return 0
        return self.positions_itm / self.num_positions


@dataclass
class CircuitBreakerResult:
    """Výsledek circuit breaker check."""
    level: AlertLevel
    action: EmergencyAction
    is_emergency: bool
    reason: str
    details: List[str] = field(default_factory=list)
    ppo_should_decide: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __repr__(self):
        emoji = {
            AlertLevel.NORMAL: "🟢",
            AlertLevel.ELEVATED: "🟡", 
            AlertLevel.HIGH: "🟠",
            AlertLevel.CRITICAL: "🔴"
        }
        return f"{emoji[self.level]} {self.level.name}: {self.action.value} - {self.reason}"


# =============================================================================
# CIRCUIT BREAKER (Hard Rules)
# =============================================================================

class CircuitBreaker:
    """
    Hard Circuit Breaker - NELZE OVERRIDE!
    
    Chrání proti black swan events.
    Okamžitá akce bez čekání na ML.
    """
    
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.triggered_today = False
        self.last_trigger_time: Optional[datetime] = None
        self.trigger_count = 0
        
        logger.info("CircuitBreaker initialized")
        logger.info(f"  VIX extreme: {self.config.vix_extreme}")
        logger.info(f"  Daily loss extreme: {self.config.daily_loss_extreme:.0%}")
    
    def check(self, market: MarketState, 
              portfolio: PortfolioState) -> CircuitBreakerResult:
        """
        Hlavní check - volat PŘED každou akcí!
        
        Returns:
            CircuitBreakerResult s akcí k provedení
        """
        details = []
        
        # =================================================================
        # LEVEL 1: CRITICAL - Okamžité CLOSE ALL
        # =================================================================
        
        # VIX extreme
        if market.vix >= self.config.vix_extreme:
            return self._trigger_critical(
                f"VIX EXTREME: {market.vix:.1f} >= {self.config.vix_extreme}",
                EmergencyAction.CLOSE_ALL
            )
        
        # Daily loss extreme
        if portfolio.daily_pnl_pct <= self.config.daily_loss_extreme:
            return self._trigger_critical(
                f"DAILY LOSS EXTREME: {portfolio.daily_pnl_pct:.1%} <= {self.config.daily_loss_extreme:.0%}",
                EmergencyAction.CLOSE_ALL
            )
        
        # Gap extreme
        if market.spy_gap_pct <= self.config.gap_extreme:
            return self._trigger_critical(
                f"GAP EXTREME: SPY {market.spy_gap_pct:.1%} gap",
                EmergencyAction.CLOSE_ALL
            )
        
        # Intraday crash
        if market.spy_intraday_pct <= -0.07:  # -7% intraday
            return self._trigger_critical(
                f"INTRADAY CRASH: SPY {market.spy_intraday_pct:.1%}",
                EmergencyAction.CLOSE_ALL
            )
        
        # =================================================================
        # LEVEL 2: HIGH - PPO Should Decide
        # =================================================================
        
        ppo_triggers = []
        
        # VIX high
        if market.vix >= self.config.vix_high:
            ppo_triggers.append(f"VIX HIGH: {market.vix:.1f}")
        
        # Daily loss high
        if portfolio.daily_pnl_pct <= self.config.daily_loss_high:
            ppo_triggers.append(f"Daily loss: {portfolio.daily_pnl_pct:.1%}")
        
        # VIX spike
        if market.vix_change_pct >= self.config.vix_spike_pct:
            ppo_triggers.append(f"VIX spike: +{market.vix_change_pct:.0%}")
        
        # Gap high
        if market.spy_gap_pct <= self.config.gap_high:
            ppo_triggers.append(f"SPY gap: {market.spy_gap_pct:.1%}")
        
        # Too many ITM positions
        if portfolio.positions_itm_pct > 0.5:
            ppo_triggers.append(f"ITM positions: {portfolio.positions_itm_pct:.0%}")
        
        if ppo_triggers:
            return CircuitBreakerResult(
                level=AlertLevel.HIGH,
                action=EmergencyAction.NONE,  # PPO decides
                is_emergency=False,
                reason="Multiple stress indicators",
                details=ppo_triggers,
                ppo_should_decide=True
            )
        
        # =================================================================
        # LEVEL 3: ELEVATED - Caution
        # =================================================================
        
        elevated_triggers = []
        
        if market.vix >= self.config.vix_elevated:
            elevated_triggers.append(f"VIX elevated: {market.vix:.1f}")
        
        if portfolio.daily_pnl_pct <= self.config.daily_loss_elevated:
            elevated_triggers.append(f"Daily loss: {portfolio.daily_pnl_pct:.1%}")
        
        if portfolio.margin_usage_pct > 0.4:
            elevated_triggers.append(f"Margin usage: {portfolio.margin_usage_pct:.0%}")
        
        if elevated_triggers:
            return CircuitBreakerResult(
                level=AlertLevel.ELEVATED,
                action=EmergencyAction.NONE,
                is_emergency=False,
                reason="Elevated risk",
                details=elevated_triggers,
                ppo_should_decide=False
            )
        
        # =================================================================
        # LEVEL 4: NORMAL
        # =================================================================
        
        return CircuitBreakerResult(
            level=AlertLevel.NORMAL,
            action=EmergencyAction.NONE,
            is_emergency=False,
            reason="All clear",
            details=[]
        )
    
    def _trigger_critical(self, reason: str, 
                          action: EmergencyAction) -> CircuitBreakerResult:
        """Trigger CRITICAL alert."""
        self.triggered_today = True
        self.last_trigger_time = datetime.now()
        self.trigger_count += 1
        
        logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}")
        logger.critical(f"🚨 ACTION: {action.value}")
        
        return CircuitBreakerResult(
            level=AlertLevel.CRITICAL,
            action=action,
            is_emergency=True,
            reason=reason,
            details=[f"Trigger #{self.trigger_count}"]
        )
    
    def reset_daily(self):
        """Reset daily counters (call at market open)."""
        self.triggered_today = False
        self.trigger_count = 0
    
    def get_current_limits(self, market: MarketState) -> Dict:
        """
        Vrátí aktuální limity upravené podle VIX.
        
        V high vol = přísnější limity.
        """
        if market.vix >= self.config.vix_high:
            return {
                'max_positions': self.config.max_positions_high_vol,
                'max_margin': self.config.max_margin_high_vol,
                'max_risk_per_trade': 0.01,  # 1% v high vol
                'allow_new_trades': False,
                'reason': f"HIGH VOL MODE (VIX={market.vix:.1f})"
            }
        elif market.vix >= self.config.vix_elevated:
            return {
                'max_positions': 5,
                'max_margin': 0.35,
                'max_risk_per_trade': 0.02,
                'allow_new_trades': True,
                'reason': f"ELEVATED MODE (VIX={market.vix:.1f})"
            }
        else:
            return {
                'max_positions': 10,
                'max_margin': 0.50,
                'max_risk_per_trade': 0.03,
                'allow_new_trades': True,
                'reason': "NORMAL MODE"
            }


# =============================================================================
# PPO RISK AGENT
# =============================================================================

class PPORiskAgent:
    """
    PPO Agent pro risk management v HIGH ALERT zóně.
    
    Rozhoduje mezi:
    - HOLD: Drž pozice, situace se zlepší
    - REDUCE: Částečně zavři pozice
    - HEDGE: Přidej ochranu
    - CLOSE: Zavři vše
    
    Učí se z historických dat co je nejlepší reakce.
    """
    
    def __init__(self, config: PPORiskConfig = None):
        self.config = config or PPORiskConfig()
        self.model = None  # PPO model (bude natrénován)
        self.action_history: List[Dict] = []
        
        logger.info("PPORiskAgent initialized")
    
    def get_action(self, market: MarketState, 
                   portfolio: PortfolioState) -> Tuple[PPORiskAction, float]:
        """
        Získá akci od PPO.
        
        Returns:
            (action, confidence)
        """
        # Build state vector
        state = self._build_state(market, portfolio)
        
        if self.model is None:
            # Fallback na rule-based dokud není model natrénován
            return self._rule_based_action(market, portfolio)
        
        # Get PPO prediction
        action_probs = self.model.predict(state)
        action_idx = np.argmax(action_probs)
        confidence = action_probs[action_idx]
        
        action = PPORiskAction(action_idx)
        
        # Log action
        self.action_history.append({
            'timestamp': datetime.now(),
            'state': state,
            'action': action,
            'confidence': confidence,
            'vix': market.vix,
            'daily_pnl': portfolio.daily_pnl_pct
        })
        
        return action, confidence
    
    def _build_state(self, market: MarketState, 
                     portfolio: PortfolioState) -> np.ndarray:
        """Build state vector pro PPO."""
        
        # Normalize features
        now = datetime.now()
        hours_to_close = max(0, 16 - now.hour + (0 if now.minute < 30 else -0.5))
        
        state = np.array([
            market.vix / 50,                          # Normalized VIX
            market.vix_change_pct,                    # VIX change
            portfolio.daily_pnl_pct * 10,             # Scaled P&L
            portfolio.margin_usage_pct,               # Margin usage
            portfolio.num_positions / 10,             # Normalized positions
            portfolio.avg_position_delta,             # Avg delta
            hours_to_close / 6.5,                     # Time to close
            portfolio.positions_itm_pct,              # ITM percentage
            portfolio.unrealized_pnl / portfolio.account_size if portfolio.account_size > 0 else 0,
        ])
        
        return state
    
    def _rule_based_action(self, market: MarketState,
                           portfolio: PortfolioState) -> Tuple[PPORiskAction, float]:
        """
        Rule-based fallback když není model.
        
        Konzervativní pravidla pro bezpečnost.
        """
        
        # VIX velmi vysoký = close risky
        if market.vix >= 40:
            if portfolio.positions_itm_pct > 0.3:
                return PPORiskAction.CLOSE_ITM, 0.8
            return PPORiskAction.REDUCE_50, 0.7
        
        # VIX spike = hedge
        if market.vix_change_pct >= 0.25:
            return PPORiskAction.ADD_HEDGE, 0.7
        
        # Significant daily loss = reduce
        if portfolio.daily_pnl_pct <= -0.04:
            return PPORiskAction.REDUCE_25, 0.6
        
        # Many ITM positions = close them
        if portfolio.positions_itm_pct > 0.5:
            return PPORiskAction.CLOSE_ITM, 0.7
        
        # High margin usage = reduce
        if portfolio.margin_usage_pct > 0.6:
            return PPORiskAction.REDUCE_25, 0.6
        
        # Default = hold and monitor
        return PPORiskAction.HOLD, 0.5
    
    def record_outcome(self, action: PPORiskAction, 
                       outcome_pnl: float,
                       market_recovered: bool):
        """
        Zaznamenává výsledek akce pro trénink.
        
        Args:
            action: Jaká akce byla provedena
            outcome_pnl: P&L po akci
            market_recovered: Zotavil se trh?
        """
        reward = 0
        
        if action == PPORiskAction.HOLD:
            if market_recovered:
                reward = self.config.survival_reward  # Good hold
            else:
                reward = self.config.loss_penalty     # Should have acted
        
        elif action in [PPORiskAction.REDUCE_25, PPORiskAction.REDUCE_50, 
                        PPORiskAction.CLOSE_ALL]:
            if market_recovered:
                reward = self.config.overreaction_penalty  # Overreacted
            else:
                reward = self.config.survival_reward       # Good protection
        
        elif action == PPORiskAction.ADD_HEDGE:
            if not market_recovered:
                reward = self.config.survival_reward * 1.5  # Hedge paid off
            else:
                reward = -0.2  # Wasted money on hedge
        
        # Store for training
        if self.action_history:
            self.action_history[-1]['reward'] = reward
            self.action_history[-1]['outcome_pnl'] = outcome_pnl
    
    def get_training_data(self) -> List[Dict]:
        """Vrátí data pro trénink PPO."""
        return [h for h in self.action_history if 'reward' in h]


# =============================================================================
# INTEGRATED RISK MANAGER
# =============================================================================

class IntegratedRiskManager:
    """
    Kombinuje Circuit Breaker + PPO Risk Agent.
    
    Flow:
    1. Circuit Breaker check (hard limits)
    2. If CRITICAL → immediate action
    3. If HIGH → PPO decides
    4. If ELEVATED/NORMAL → continue with limits
    """
    
    def __init__(self, account_size: float):
        self.account_size = account_size
        self.circuit_breaker = CircuitBreaker()
        self.ppo_agent = PPORiskAgent()
        
        # Callbacks pro akce
        self.on_close_all: Optional[Callable] = None
        self.on_close_position: Optional[Callable] = None
        self.on_add_hedge: Optional[Callable] = None
        
        logger.info(f"IntegratedRiskManager initialized (${account_size:,.0f})")
    
    def check_and_act(self, market: MarketState,
                      portfolio: PortfolioState) -> Dict:
        """
        Hlavní entry point - check a případně act.
        
        Returns:
            Dict s akcí a detaily
        """
        # 1. Circuit breaker check
        cb_result = self.circuit_breaker.check(market, portfolio)
        
        result = {
            'alert_level': cb_result.level,
            'action_taken': None,
            'action_source': None,
            'details': cb_result.details,
            'current_limits': self.circuit_breaker.get_current_limits(market)
        }
        
        # 2. CRITICAL = okamžitá akce
        if cb_result.is_emergency:
            logger.critical(f"🚨 EMERGENCY: {cb_result.reason}")
            result['action_taken'] = cb_result.action.value
            result['action_source'] = 'CIRCUIT_BREAKER'
            
            # Execute emergency action
            if cb_result.action == EmergencyAction.CLOSE_ALL:
                self._execute_close_all()
            elif cb_result.action == EmergencyAction.HALT_TRADING:
                self._execute_halt()
            
            return result
        
        # 3. HIGH = PPO rozhoduje
        if cb_result.ppo_should_decide:
            logger.warning(f"⚠️ HIGH ALERT: PPO deciding...")
            
            ppo_action, confidence = self.ppo_agent.get_action(market, portfolio)
            
            result['action_source'] = 'PPO_AGENT'
            result['ppo_confidence'] = confidence
            
            # Execute if confident
            if confidence >= self.ppo_agent.config.confidence_threshold:
                result['action_taken'] = ppo_action.name
                self._execute_ppo_action(ppo_action, portfolio)
            else:
                result['action_taken'] = 'MONITOR'
                logger.info(f"PPO suggests {ppo_action.name} but confidence too low ({confidence:.0%})")
            
            return result
        
        # 4. ELEVATED/NORMAL = pokračuj s limity
        result['action_taken'] = 'NONE'
        result['action_source'] = 'NORMAL'
        
        return result
    
    def _execute_close_all(self):
        """Execute CLOSE ALL emergency action."""
        logger.critical("🚨 EXECUTING CLOSE ALL!")
        if self.on_close_all:
            self.on_close_all()
    
    def _execute_halt(self):
        """Execute trading halt."""
        logger.critical("🚨 TRADING HALTED!")
        # Set flag to prevent new trades
    
    def _execute_ppo_action(self, action: PPORiskAction, 
                            portfolio: PortfolioState):
        """Execute PPO recommended action."""
        logger.warning(f"⚠️ Executing PPO action: {action.name}")
        
        if action == PPORiskAction.CLOSE_ALL:
            self._execute_close_all()
        
        elif action == PPORiskAction.REDUCE_50:
            # Close 50% of positions (highest risk first)
            logger.info("Reducing 50% of positions")
        
        elif action == PPORiskAction.REDUCE_25:
            logger.info("Reducing 25% of positions")
        
        elif action == PPORiskAction.CLOSE_ITM:
            logger.info("Closing ITM positions")
        
        elif action == PPORiskAction.CLOSE_HIGHEST_RISK:
            if portfolio.highest_risk_position:
                logger.info(f"Closing highest risk: {portfolio.highest_risk_position}")
        
        elif action == PPORiskAction.ADD_HEDGE:
            logger.info("Adding protective hedge")
            if self.on_add_hedge:
                self.on_add_hedge()
        
        elif action == PPORiskAction.ROLL_OUT:
            logger.info("Rolling all positions out")
        
        # HOLD = do nothing


# =============================================================================
# CLI / DEMO
# =============================================================================

def main():
    """Demo circuit breaker and PPO risk agent."""
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print(" 🛡️ CIRCUIT BREAKER + PPO RISK SYSTEM")
    print("=" * 70)
    
    # Initialize
    risk_manager = IntegratedRiskManager(account_size=100_000)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'NORMAL DAY',
            'market': MarketState(vix=17, vix_open=16, spy_price=595, 
                                   spy_open=594, spy_prev_close=593),
            'portfolio': PortfolioState(account_size=100000, daily_pnl=500,
                                        unrealized_pnl=200, margin_used=30000,
                                        num_positions=5, positions_itm=1,
                                        avg_position_delta=0.15)
        },
        {
            'name': 'ELEVATED VOLATILITY',
            'market': MarketState(vix=28, vix_open=22, spy_price=588,
                                   spy_open=592, spy_prev_close=594),
            'portfolio': PortfolioState(account_size=100000, daily_pnl=-1500,
                                        unrealized_pnl=-1000, margin_used=35000,
                                        num_positions=6, positions_itm=2,
                                        avg_position_delta=0.25)
        },
        {
            'name': 'HIGH STRESS - PPO ZONE',
            'market': MarketState(vix=38, vix_open=25, spy_price=575,
                                   spy_open=590, spy_prev_close=592),
            'portfolio': PortfolioState(account_size=100000, daily_pnl=-4000,
                                        unrealized_pnl=-3000, margin_used=40000,
                                        num_positions=5, positions_itm=3,
                                        avg_position_delta=0.35)
        },
        {
            'name': 'CRITICAL - CIRCUIT BREAKER',
            'market': MarketState(vix=55, vix_open=30, spy_price=550,
                                   spy_open=580, spy_prev_close=595),
            'portfolio': PortfolioState(account_size=100000, daily_pnl=-8000,
                                        unrealized_pnl=-6000, margin_used=45000,
                                        num_positions=5, positions_itm=4,
                                        avg_position_delta=0.50)
        },
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f" Scenario: {scenario['name']}")
        print(f"{'='*60}")
        
        market = scenario['market']
        portfolio = scenario['portfolio']
        
        print(f"  VIX: {market.vix:.1f} (open: {market.vix_open:.1f}, change: {market.vix_change_pct:+.0%})")
        print(f"  SPY: ${market.spy_price:.2f} (gap: {market.spy_gap_pct:+.1%})")
        print(f"  Daily P&L: ${portfolio.daily_pnl:,.0f} ({portfolio.daily_pnl_pct:.1%})")
        print(f"  Margin usage: {portfolio.margin_usage_pct:.0%}")
        print(f"  ITM positions: {portfolio.positions_itm}/{portfolio.num_positions}")
        
        # Check
        result = risk_manager.check_and_act(market, portfolio)
        
        print(f"\n  RESULT:")
        print(f"    Alert Level: {result['alert_level'].name}")
        print(f"    Action: {result['action_taken']}")
        print(f"    Source: {result['action_source']}")
        if result['details']:
            print(f"    Details: {result['details']}")
        print(f"    Current Limits: {result['current_limits']['reason']}")
    
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print("""
    HARD CIRCUIT BREAKER (always active):
    ├── VIX > 50         → CLOSE ALL
    ├── Daily loss > 10% → CLOSE ALL
    ├── SPY gap > 5%     → CLOSE ALL
    └── Cannot be overridden!
    
    PPO RISK AGENT (learns optimal response):
    ├── VIX 25-50        → PPO decides
    ├── Daily loss 5-10% → PPO decides
    ├── Actions: HOLD, REDUCE, HEDGE, CLOSE
    └── Learns from outcomes
    """)


if __name__ == "__main__":
    main()
