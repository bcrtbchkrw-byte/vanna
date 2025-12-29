#!/usr/bin/env python3
"""
trading_brain.py

UNIFIED TRADING SYSTEM - 10-15% Monthly Target

ORCHESTRÁTOR který kombinuje:
- strategy_advisor.py (výběr strategie + XGBoost win probability)
- circuit_breaker.py (hard limits + PPO risk management)
- Money Printer konfigurace (DTE buckets)
- Position Sizing (procentuální)

Škáluje na JAKÝKOLIV účet: $500 i $500,000
Vše je v PROCENTECH, ne v pevných částkách.

Usage:
    from trading_brain import TradingBrain
    
    brain = TradingBrain(account_size=10000)
    decision = brain.analyze(market_data)
    
    if decision.should_execute():
        execute_trade(decision.trade_spec)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, date, time, timedelta
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# IMPORTS - External modules (with fallbacks)
# =============================================================================

# Strategy Advisor (XGBoost-based strategy selection)
try:
    from ml.strategy_advisor import (
        StrategyAdvisor, 
        get_strategy_advisor,
        MarketRegime,
        Strategy as AdvisorStrategy,
        StrategyRecommendation
    )
    HAS_STRATEGY_ADVISOR = True
    logger.info("✅ StrategyAdvisor loaded")
except ImportError:
    HAS_STRATEGY_ADVISOR = False
    logger.warning("⚠️ StrategyAdvisor not available, using fallback")

# Circuit Breaker (hard limits + PPO risk)
try:
    from risk.market_circuit_breaker import (
        CircuitBreaker,
        PPORiskAgent,
        IntegratedRiskManager,
        MarketState,
        PortfolioState,
        AlertLevel,
        EmergencyAction
    )
    HAS_CIRCUIT_BREAKER = True
    logger.info("✅ CircuitBreaker loaded")
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    logger.warning("⚠️ CircuitBreaker not available, using fallback")

# Trade Success Predictor (XGBoost win probability)
try:
    from ml.trade_success_predictor import get_trade_success_predictor
    HAS_WIN_PREDICTOR = True
    logger.info("✅ TradeSuccessPredictor loaded")
except ImportError:
    HAS_WIN_PREDICTOR = False
    logger.warning("⚠️ TradeSuccessPredictor not available")


# =============================================================================
# ENUMS (import from central ml/enums.py)
# =============================================================================

from ml.enums import MarketRegime, Strategy, DTEBucket, Action


# =============================================================================
# MONEY PRINTER CONFIGURATION (DTE Buckets)
# =============================================================================

@dataclass
class DTEConfig:
    """Konfigurace pro DTE bucket - vše procentuálně."""
    bucket: DTEBucket
    name: str
    
    # Portfolio allocation (% of total)
    allocation_pct: float
    
    # Delta targeting
    target_delta: float         # Short strike delta (0.10 = 90% POP)
    
    # DTE range
    min_dte: int
    max_dte: int
    
    # Management (% of credit received)
    profit_target_pct: float    # Close at X% of max profit
    stop_loss_pct: float        # Stop at X% of credit lost
    
    # Timing
    entry_start: time
    entry_end: time
    
    # Trade frequency
    max_trades_per_day: int
    max_trades_per_week: int
    
    @property
    def pop(self) -> float:
        """Probability of Profit."""
        return 1 - self.target_delta


# DTE Bucket Configurations (Money Printer)
DTE_CONFIGS = {
    DTEBucket.ZERO_DTE: DTEConfig(
        bucket=DTEBucket.ZERO_DTE,
        name="0DTE",
        allocation_pct=0.40,
        target_delta=0.10,          # 90% POP
        min_dte=0,
        max_dte=0,
        profit_target_pct=0.50,
        stop_loss_pct=1.50,
        entry_start=time(9, 45),
        entry_end=time(14, 0),
        max_trades_per_day=3,
        max_trades_per_week=15,
    ),
    DTEBucket.WEEKLY: DTEConfig(
        bucket=DTEBucket.WEEKLY,
        name="WEEKLY",
        allocation_pct=0.40,
        target_delta=0.14,          # 86% POP
        min_dte=1,
        max_dte=7,
        profit_target_pct=0.50,
        stop_loss_pct=2.00,
        entry_start=time(9, 45),
        entry_end=time(15, 30),
        max_trades_per_day=4,
        max_trades_per_week=8,
    ),
    DTEBucket.MONTHLY: DTEConfig(
        bucket=DTEBucket.MONTHLY,
        name="MONTHLY",
        allocation_pct=0.20,
        target_delta=0.12,          # 88% POP
        min_dte=21,
        max_dte=45,
        profit_target_pct=0.50,
        stop_loss_pct=2.00,
        entry_start=time(9, 45),
        entry_end=time(15, 30),
        max_trades_per_day=2,
        max_trades_per_week=4,
    ),
}


# =============================================================================
# RISK CONFIGURATION
# =============================================================================

@dataclass
class RiskConfig:
    """Risk limity - VŠE V PROCENTECH!"""
    
    # Position sizing
    max_risk_per_trade_pct: float = 0.03    # 3% max risk per trade
    max_margin_per_trade_pct: float = 0.10  # 10% max margin per trade
    
    # Daily limits
    max_daily_loss_pct: float = 0.05        # -5% = STOP trading
    max_daily_trades: int = 8
    
    # Weekly limits
    max_weekly_loss_pct: float = 0.10       # -10% = REDUCE size
    
    # Portfolio limits
    max_drawdown_pct: float = 0.20          # -20% = PAUSE system
    max_margin_usage_pct: float = 0.50      # Max 50% margin used
    max_delta_exposure: float = 0.30        # Max 30% delta exposure
    max_positions: int = 10
    
    # Correlation
    max_positions_per_symbol: int = 2
    max_correlated_positions: int = 4


@dataclass
class SmallAccountConfig:
    """Speciální konfigurace pro malé účty (<$2000)."""
    
    min_for_spreads: float = 200
    min_for_iron_condor: float = 400
    min_for_multi_position: float = 1000
    max_positions_small: int = 2
    prefer_single_leg: bool = True


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OptionLeg:
    """Jedna noha opční strategie."""
    action: str             # BUY nebo SELL
    right: str              # CALL nebo PUT
    strike: float
    expiry: date
    delta: float = 0.0
    price: float = 0.0
    quantity: int = 1


@dataclass
class TradeSpec:
    """Specifikace obchodu."""
    symbol: str
    strategy: Strategy
    dte_bucket: DTEBucket
    legs: List[OptionLeg]
    
    # Pricing
    underlying_price: float = 0.0
    credit_received: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    
    # Position size
    contracts: int = 1
    total_margin: float = 0.0
    total_max_profit: float = 0.0
    total_max_loss: float = 0.0
    
    # Risk metrics
    pop: float = 0.0
    win_probability: float = 0.0    # From XGBoost
    
    # Management rules
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 2.00
    roll_at_dte: int = 0
    
    def calculate_totals(self):
        """Přepočítá totals podle počtu kontraktů."""
        self.total_margin = self.max_loss * self.contracts
        self.total_max_profit = self.max_profit * self.contracts
        self.total_max_loss = self.max_loss * self.contracts


@dataclass
class MarketData:
    """Market data pro analýzu."""
    symbol: str
    underlying_price: float
    
    # Volatility
    vix: float = 18.0
    vix3m: float = 20.0
    iv_rank: float = 50.0
    
    # Trend
    trend: str = "NEUTRAL"
    trend_strength: float = 0.5
    
    # Technical
    rsi: float = 50.0
    sma50: float = 0.0
    sma200: float = 0.0
    atr: float = 0.0
    
    # Greeks (for XGBoost)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    vanna: float = 0.0
    charm: float = 0.0
    volga: float = 0.0
    
    # Events
    days_to_earnings: Optional[int] = None
    days_to_fomc: Optional[int] = None
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_features_dict(self) -> Dict[str, Any]:
        """Convert to features dict for strategy_advisor."""
        return {
            'vix': self.vix,
            'vix3m': self.vix3m,
            'close': self.underlying_price,
            'sma50': self.sma50 or self.underlying_price,
            'sma200': self.sma200 or self.underlying_price,
            'rsi': self.rsi,
            'iv_rank': self.iv_rank,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'vanna': self.vanna,
            'charm': self.charm,
            'volga': self.volga,
        }


@dataclass
class Portfolio:
    """Stav portfolia."""
    account_size: float
    cash: float
    margin_used: float = 0.0
    
    # Greeks
    net_delta: float = 0.0
    net_theta: float = 0.0
    
    # Positions
    open_positions: int = 0
    positions_by_symbol: Dict[str, int] = field(default_factory=dict)
    positions_itm: int = 0
    
    # P&L tracking
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Trades tracking
    trades_today: int = 0
    trades_this_week: int = 0
    
    @property
    def margin_available(self) -> float:
        return max(0, self.cash - self.margin_used)
    
    @property
    def margin_usage_pct(self) -> float:
        if self.account_size == 0:
            return 1.0
        return self.margin_used / self.account_size
    
    @property
    def daily_pnl_pct(self) -> float:
        if self.account_size == 0:
            return 0
        return self.daily_pnl / self.account_size
    
    def to_circuit_breaker_state(self) -> 'PortfolioState':
        """Convert to PortfolioState for circuit_breaker."""
        if HAS_CIRCUIT_BREAKER:
            return PortfolioState(
                account_size=self.account_size,
                daily_pnl=self.daily_pnl,
                unrealized_pnl=self.unrealized_pnl,
                margin_used=self.margin_used,
                num_positions=self.open_positions,
                positions_itm=self.positions_itm,
                avg_position_delta=self.net_delta / max(1, self.open_positions)
            )
        return None


@dataclass
class BrainDecision:
    """Výstup z TradingBrain."""
    action: Action
    trade_spec: Optional[TradeSpec] = None
    
    # Confidence
    confidence: float = 0.0
    
    # Context
    regime: MarketRegime = None
    dte_bucket: DTEBucket = DTEBucket.WEEKLY
    
    # From StrategyAdvisor
    advisor_recommendation: Optional[Any] = None
    
    # Circuit Breaker
    circuit_breaker_level: str = "NORMAL"
    circuit_breaker_action: str = "NONE"
    
    # Reasoning
    reasoning: List[str] = field(default_factory=list)
    
    # Risk check results
    risk_approved: bool = False
    risk_rejection_reason: str = ""
    
    def should_execute(self) -> bool:
        return (
            self.action == Action.OPEN and
            self.trade_spec is not None and
            self.risk_approved and
            self.confidence >= 0.5 and
            self.circuit_breaker_level not in ["CRITICAL", "HIGH"]
        )


# =============================================================================
# TRADING BRAIN - ORCHESTRATOR
# =============================================================================

class TradingBrain:
    """
    Unified Trading Brain - ORCHESTRATOR.
    
    Kombinuje:
    - StrategyAdvisor (XGBoost strategy selection)
    - CircuitBreaker (hard limits + PPO risk)
    - Money Printer (DTE buckets)
    - Position sizing (procentuální)
    
    Škáluje na jakýkoliv účet ($500 - $500,000).
    """
    
    def __init__(self,
                 account_size: float,
                 risk_config: RiskConfig = None,
                 small_account_config: SmallAccountConfig = None):
        """
        Args:
            account_size: Velikost účtu v $
        """
        self.account_size = account_size
        self.risk = risk_config or RiskConfig()
        self.small_config = small_account_config or SmallAccountConfig()
        
        # DTE configs (Money Printer)
        self.dte_configs = DTE_CONFIGS
        
        # Portfolio state
        self.portfolio = Portfolio(
            account_size=account_size,
            cash=account_size
        )
        
        # Initialize external modules
        self._init_strategy_advisor()
        self._init_circuit_breaker()
        
        # Log initialization
        self._log_config()
    
    def _init_strategy_advisor(self):
        """Initialize StrategyAdvisor."""
        if HAS_STRATEGY_ADVISOR:
            try:
                self.strategy_advisor = get_strategy_advisor()
                logger.info("StrategyAdvisor connected")
            except Exception as e:
                logger.warning(f"Could not init StrategyAdvisor: {e}")
                self.strategy_advisor = None
        else:
            self.strategy_advisor = None
    
    def _init_circuit_breaker(self):
        """Initialize CircuitBreaker."""
        if HAS_CIRCUIT_BREAKER:
            try:
                self.circuit_breaker = IntegratedRiskManager(self.account_size)
                logger.info("CircuitBreaker connected")
            except Exception as e:
                logger.warning(f"Could not init CircuitBreaker: {e}")
                self.circuit_breaker = None
        else:
            self.circuit_breaker = None
    
    def _log_config(self):
        """Loguje konfiguraci."""
        logger.info("=" * 60)
        logger.info("TRADING BRAIN INITIALIZED (ORCHESTRATOR)")
        logger.info(f"Account size: ${self.account_size:,.2f}")
        logger.info("=" * 60)
        
        # Modules status
        logger.info("MODULES:")
        logger.info(f"  StrategyAdvisor: {'✅' if self.strategy_advisor else '❌ (fallback)'}")
        logger.info(f"  CircuitBreaker: {'✅' if self.circuit_breaker else '❌ (fallback)'}")
        logger.info(f"  WinPredictor: {'✅' if HAS_WIN_PREDICTOR else '❌'}")
        
        # Risk limits
        logger.info("\nRISK LIMITS:")
        logger.info(f"  Max risk/trade: {self.risk.max_risk_per_trade_pct:.0%} "
                   f"(${self.account_size * self.risk.max_risk_per_trade_pct:,.2f})")
        logger.info(f"  Max daily loss: {self.risk.max_daily_loss_pct:.0%} "
                   f"(${self.account_size * self.risk.max_daily_loss_pct:,.2f})")
        
        # DTE Buckets
        logger.info("\nDTE BUCKETS (Money Printer):")
        for bucket, config in self.dte_configs.items():
            allocated = self.account_size * config.allocation_pct
            logger.info(f"  {config.name}: {config.allocation_pct:.0%} (${allocated:,.2f})")
        
        if self.account_size < 1000:
            logger.warning("\n⚠️ SMALL ACCOUNT MODE (<$1000)")
        
        logger.info("=" * 60)
    
    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================
    
    def analyze(self, market: MarketData, 
                dte_bucket: DTEBucket = None) -> BrainDecision:
        """
        Hlavní analýza - rozhodne co obchodovat.
        
        Flow:
        1. Circuit Breaker check (FIRST!)
        2. Strategy Advisor (regime, strategy, win prob)
        3. DTE bucket selection
        4. Trade spec creation
        5. Position sizing
        6. Final risk check
        """
        reasoning = []
        
        # =====================================================================
        # 1. CIRCUIT BREAKER CHECK (FIRST!)
        # =====================================================================
        cb_level, cb_action, cb_details = self._check_circuit_breaker(market)
        reasoning.append(f"Circuit Breaker: {cb_level}")
        
        if cb_level == "CRITICAL":
            reasoning.append(f"🚨 EMERGENCY: {cb_action}")
            return BrainDecision(
                action=Action.EMERGENCY_CLOSE,
                confidence=1.0,
                circuit_breaker_level=cb_level,
                circuit_breaker_action=cb_action,
                reasoning=reasoning,
                risk_approved=False,
                risk_rejection_reason="Circuit breaker triggered"
            )
        
        if cb_level == "HIGH":
            reasoning.append(f"⚠️ HIGH ALERT - PPO managing risk")
            # PPO rozhoduje, ale můžeme pokračovat s analýzou
        
        # =====================================================================
        # 2. STRATEGY ADVISOR (Regime, Strategy, Win Probability)
        # =====================================================================
        regime, strategy, win_prob, advisor_rec = self._get_strategy_recommendation(market)
        reasoning.append(f"Regime: {regime.name} (VIX={market.vix:.1f})")
        reasoning.append(f"Strategy: {strategy.value}")
        reasoning.append(f"Win Probability: {win_prob:.0%}")
        
        if strategy == Strategy.CASH:
            return BrainDecision(
                action=Action.HOLD,
                regime=regime,
                advisor_recommendation=advisor_rec,
                circuit_breaker_level=cb_level,
                reasoning=reasoning + ["No trade signal from advisor"],
                risk_approved=False
            )
        
        # =====================================================================
        # 3. DTE BUCKET SELECTION
        # =====================================================================
        if dte_bucket is None:
            dte_bucket = self._select_dte_bucket(regime, advisor_rec)
        
        config = self.dte_configs.get(dte_bucket)
        if config is None:
            config = self.dte_configs[DTEBucket.WEEKLY]
        
        reasoning.append(f"DTE Bucket: {config.name}")
        
        # Check time window
        if not self._check_time_window(config):
            reasoning.append(f"Outside trading window ({config.entry_start}-{config.entry_end})")
            return BrainDecision(
                action=Action.HOLD,
                regime=regime,
                dte_bucket=dte_bucket,
                advisor_recommendation=advisor_rec,
                circuit_breaker_level=cb_level,
                reasoning=reasoning,
                risk_approved=False
            )
        
        # =====================================================================
        # 4. TRADE SPEC CREATION
        # =====================================================================
        trade_spec = self._create_trade_spec(
            symbol=market.symbol,
            strategy=strategy,
            dte_bucket=dte_bucket,
            config=config,
            market=market,
            win_probability=win_prob
        )
        
        if trade_spec is None:
            reasoning.append("Could not create trade spec")
            return BrainDecision(
                action=Action.HOLD,
                regime=regime,
                dte_bucket=dte_bucket,
                reasoning=reasoning,
                risk_approved=False
            )
        
        # =====================================================================
        # 5. POSITION SIZING
        # =====================================================================
        trade_spec = self._size_position(trade_spec, config)
        reasoning.append(f"Position: {trade_spec.contracts} contracts")
        reasoning.append(f"Margin: ${trade_spec.total_margin:.2f}")
        
        # =====================================================================
        # 6. FINAL RISK CHECK
        # =====================================================================
        approved, risk_reason = self._check_risk(trade_spec, config)
        
        if not approved:
            reasoning.append(f"❌ Risk rejected: {risk_reason}")
            return BrainDecision(
                action=Action.HOLD,
                trade_spec=trade_spec,
                regime=regime,
                dte_bucket=dte_bucket,
                advisor_recommendation=advisor_rec,
                circuit_breaker_level=cb_level,
                reasoning=reasoning,
                risk_approved=False,
                risk_rejection_reason=risk_reason
            )
        
        reasoning.append("✅ Risk approved")
        
        # Calculate final confidence
        confidence = self._calculate_confidence(regime, market, trade_spec, win_prob)
        
        return BrainDecision(
            action=Action.OPEN,
            trade_spec=trade_spec,
            confidence=confidence,
            regime=regime,
            dte_bucket=dte_bucket,
            advisor_recommendation=advisor_rec,
            circuit_breaker_level=cb_level,
            circuit_breaker_action=cb_action,
            reasoning=reasoning,
            risk_approved=True
        )
    
    # =========================================================================
    # CIRCUIT BREAKER INTEGRATION
    # =========================================================================
    
    def _check_circuit_breaker(self, market: MarketData) -> Tuple[str, str, List[str]]:
        """
        Check circuit breaker status.
        
        Returns:
            (level, action, details)
        """
        if not self.circuit_breaker:
            # Fallback - basic VIX check
            if market.vix >= 50:
                return "CRITICAL", "CLOSE_ALL", ["VIX extreme (fallback)"]
            elif market.vix >= 35:
                return "HIGH", "REDUCE", ["VIX high (fallback)"]
            elif market.vix >= 25:
                return "ELEVATED", "NONE", ["VIX elevated (fallback)"]
            return "NORMAL", "NONE", []
        
        try:
            # Build MarketState for circuit breaker
            market_state = MarketState(
                vix=market.vix,
                vix_open=market.vix,  # Would need real open value
                spy_price=market.underlying_price,
                spy_open=market.underlying_price,
                spy_prev_close=market.underlying_price
            )
            
            portfolio_state = self.portfolio.to_circuit_breaker_state()
            
            result = self.circuit_breaker.check_and_act(market_state, portfolio_state)
            
            return (
                result['alert_level'].name,
                result['action_taken'] or "NONE",
                result.get('details', [])
            )
        except Exception as e:
            logger.error(f"Circuit breaker error: {e}")
            return "NORMAL", "NONE", [f"Error: {e}"]
    
    # =========================================================================
    # STRATEGY ADVISOR INTEGRATION
    # =========================================================================
    
    def _get_strategy_recommendation(self, market: MarketData) -> Tuple[MarketRegime, Strategy, float, Any]:
        """
        Get strategy recommendation from StrategyAdvisor.
        
        Returns:
            (regime, strategy, win_probability, full_recommendation)
        """
        if self.strategy_advisor:
            try:
                features = market.to_features_dict()
                rec = self.strategy_advisor.recommend(features)
                
                # Map strategy from advisor to our Strategy enum
                strategy = self._map_advisor_strategy(rec.strategy)
                
                return (
                    rec.regime,
                    strategy,
                    rec.win_probability,
                    rec
                )
            except Exception as e:
                logger.warning(f"StrategyAdvisor error: {e}, using fallback")
        
        # Fallback - basic rule-based
        regime = self._classify_regime_fallback(market.vix)
        strategy = self._select_strategy_fallback(regime, market)
        win_prob = 0.5  # Neutral without XGBoost
        
        return regime, strategy, win_prob, None
    
    def _map_advisor_strategy(self, advisor_strategy) -> Strategy:
        """Map strategy from advisor to our enum."""
        mapping = {
            'CASH': Strategy.CASH,
            'BUY_CALL': Strategy.BUY_CALL,
            'BUY_PUT': Strategy.BUY_PUT,
            'SELL_PUT': Strategy.PUT_CREDIT_SPREAD,  # Map to spread
            'SELL_CALL': Strategy.CALL_CREDIT_SPREAD,
            'CALL_SPREAD': Strategy.CALL_CREDIT_SPREAD,
            'PUT_SPREAD': Strategy.PUT_CREDIT_SPREAD,
            'IRON_CONDOR': Strategy.IRON_CONDOR,
            'STRADDLE': Strategy.STRADDLE,
        }
        
        strategy_name = advisor_strategy.value if hasattr(advisor_strategy, 'value') else str(advisor_strategy)
        return mapping.get(strategy_name, Strategy.CASH)
    
    def _classify_regime_fallback(self, vix: float) -> MarketRegime:
        """Fallback regime classification (matches strategy_advisor thresholds)."""
        if vix < 15:
            return MarketRegime.CALM
        elif vix < 20:
            return MarketRegime.NORMAL
        elif vix < 25:
            return MarketRegime.ELEVATED
        elif vix < 35:
            return MarketRegime.HIGH_VOL
        else:
            return MarketRegime.CRISIS
    
    def _select_strategy_fallback(self, regime: MarketRegime, market: MarketData) -> Strategy:
        """Fallback strategy selection."""
        if regime == MarketRegime.CRISIS:
            return Strategy.CASH
        
        if regime == MarketRegime.HIGH_VOL:
            if market.trend == "BULLISH":
                return Strategy.PUT_CREDIT_SPREAD
            elif market.trend == "BEARISH":
                return Strategy.CALL_CREDIT_SPREAD
            return Strategy.CASH
        
        # Small account
        if self.account_size < self.small_config.min_for_iron_condor:
            return Strategy.PUT_CREDIT_SPREAD
        
        if market.trend == "NEUTRAL" or market.trend_strength < 0.3:
            return Strategy.IRON_CONDOR
        elif market.trend == "BULLISH":
            return Strategy.PUT_CREDIT_SPREAD
        else:
            return Strategy.CALL_CREDIT_SPREAD
    
    # =========================================================================
    # DTE BUCKET SELECTION
    # =========================================================================
    
    def _select_dte_bucket(self, regime: MarketRegime, advisor_rec) -> DTEBucket:
        """Select DTE bucket based on regime and advisor."""
        
        # Use advisor's recommendation if available
        if advisor_rec and hasattr(advisor_rec, 'dte_bucket'):
            bucket_map = {
                0: DTEBucket.ZERO_DTE,
                1: DTEBucket.WEEKLY,
                2: DTEBucket.MONTHLY,
                3: DTEBucket.LEAPS
            }
            return bucket_map.get(advisor_rec.dte_bucket, DTEBucket.WEEKLY)
        
        # Fallback rules
        if regime in [MarketRegime.HIGH_VOL, MarketRegime.CRISIS]:
            return DTEBucket.MONTHLY
        
        if regime == MarketRegime.CALM:
            return DTEBucket.ZERO_DTE
        
        return DTEBucket.WEEKLY
    
    def _check_time_window(self, config: DTEConfig) -> bool:
        """Check if we're in trading window."""
        now = datetime.now().time()
        return config.entry_start <= now <= config.entry_end
    
    # =========================================================================
    # TRADE SPEC CREATION
    # =========================================================================
    
    def _create_trade_spec(self, symbol: str, strategy: Strategy,
                           dte_bucket: DTEBucket, config: DTEConfig,
                           market: MarketData, win_probability: float) -> Optional[TradeSpec]:
        """Create trade specification."""
        
        underlying = market.underlying_price
        delta = config.target_delta
        otm_distance_pct = delta * 2
        
        if strategy == Strategy.IRON_CONDOR:
            put_short = underlying * (1 - otm_distance_pct)
            put_long = put_short - (underlying * 0.01)
            call_short = underlying * (1 + otm_distance_pct)
            call_long = call_short + (underlying * 0.01)
            
            legs = [
                OptionLeg("BUY", "PUT", round(put_long, 0), date.today(), delta=-0.05),
                OptionLeg("SELL", "PUT", round(put_short, 0), date.today(), delta=-delta),
                OptionLeg("SELL", "CALL", round(call_short, 0), date.today(), delta=delta),
                OptionLeg("BUY", "CALL", round(call_long, 0), date.today(), delta=0.05),
            ]
            
            wing_width = abs(put_short - put_long)
            credit = wing_width * 0.30
            max_loss = wing_width - credit
            
        elif strategy in [Strategy.PUT_CREDIT_SPREAD, Strategy.BULL_PUT_SPREAD, Strategy.SELL_PUT]:
            put_short = underlying * (1 - otm_distance_pct)
            put_long = put_short - (underlying * 0.01)
            
            legs = [
                OptionLeg("BUY", "PUT", round(put_long, 0), date.today(), delta=-0.05),
                OptionLeg("SELL", "PUT", round(put_short, 0), date.today(), delta=-delta),
            ]
            
            wing_width = abs(put_short - put_long)
            credit = wing_width * 0.35
            max_loss = wing_width - credit
            
        elif strategy in [Strategy.CALL_CREDIT_SPREAD, Strategy.BEAR_CALL_SPREAD, Strategy.SELL_CALL]:
            call_short = underlying * (1 + otm_distance_pct)
            call_long = call_short + (underlying * 0.01)
            
            legs = [
                OptionLeg("SELL", "CALL", round(call_short, 0), date.today(), delta=delta),
                OptionLeg("BUY", "CALL", round(call_long, 0), date.today(), delta=0.05),
            ]
            
            wing_width = abs(call_long - call_short)
            credit = wing_width * 0.35
            max_loss = wing_width - credit
        
        else:
            return None
        
        return TradeSpec(
            symbol=symbol,
            strategy=strategy,
            dte_bucket=dte_bucket,
            legs=legs,
            underlying_price=underlying,
            credit_received=credit * 100,
            max_profit=credit * 100,
            max_loss=max_loss * 100,
            contracts=1,
            pop=config.pop,
            win_probability=win_probability,
            profit_target_pct=config.profit_target_pct,
            stop_loss_pct=config.stop_loss_pct,
        )
    
    # =========================================================================
    # POSITION SIZING
    # =========================================================================
    
    def _size_position(self, trade: TradeSpec, config: DTEConfig) -> TradeSpec:
        """Position sizing - PROCENTUÁLNÍ!"""
        
        max_risk_dollars = self.account_size * self.risk.max_risk_per_trade_pct
        max_margin_dollars = self.account_size * self.risk.max_margin_per_trade_pct
        bucket_allocation = self.account_size * config.allocation_pct
        available_in_bucket = bucket_allocation * 0.5
        
        margin_per_contract = trade.max_loss
        
        if margin_per_contract <= 0:
            trade.contracts = 1
            trade.calculate_totals()
            return trade
        
        by_risk = int(max_risk_dollars / margin_per_contract)
        by_margin = int(max_margin_dollars / margin_per_contract)
        by_bucket = int(available_in_bucket / margin_per_contract)
        by_available = int(self.portfolio.margin_available / margin_per_contract)
        
        contracts = min(by_risk, by_margin, by_bucket, by_available)
        contracts = max(1, contracts)
        
        if self.account_size < self.small_config.min_for_multi_position:
            contracts = min(contracts, 2)
        
        trade.contracts = contracts
        trade.calculate_totals()
        
        return trade
    
    # =========================================================================
    # RISK MANAGEMENT
    # =========================================================================
    
    def _check_risk(self, trade: TradeSpec, config: DTEConfig) -> Tuple[bool, str]:
        """Final risk check - VETO právo!"""
        
        # Account size checks
        if trade.strategy == Strategy.IRON_CONDOR:
            if self.account_size < self.small_config.min_for_iron_condor:
                return False, f"Account too small for IC (need ${self.small_config.min_for_iron_condor})"
        
        if self.account_size < self.small_config.min_for_spreads:
            return False, f"Account too small for spreads"
        
        # Daily loss limit
        if self.portfolio.daily_pnl <= -self.account_size * self.risk.max_daily_loss_pct:
            return False, f"Daily loss limit reached ({self.portfolio.daily_pnl_pct:.1%})"
        
        # Weekly loss limit
        weekly_pnl_pct = self.portfolio.weekly_pnl / self.account_size if self.account_size > 0 else 0
        if weekly_pnl_pct <= -self.risk.max_weekly_loss_pct:
            return False, f"Weekly loss limit reached ({weekly_pnl_pct:.1%})"
        
        # Margin usage
        new_margin_usage = (self.portfolio.margin_used + trade.total_margin) / self.account_size
        if new_margin_usage > self.risk.max_margin_usage_pct:
            return False, f"Would exceed margin limit ({new_margin_usage:.0%})"
        
        # Position count
        max_positions = self.risk.max_positions
        if self.account_size < self.small_config.min_for_multi_position:
            max_positions = self.small_config.max_positions_small
        
        if self.portfolio.open_positions >= max_positions:
            return False, f"Max positions reached ({self.portfolio.open_positions})"
        
        # Daily trades
        if self.portfolio.trades_today >= self.risk.max_daily_trades:
            return False, f"Max daily trades reached"
        
        # Symbol concentration
        symbol_positions = self.portfolio.positions_by_symbol.get(trade.symbol, 0)
        if symbol_positions >= self.risk.max_positions_per_symbol:
            return False, f"Max positions for {trade.symbol}"
        
        # Trade size
        if trade.total_margin > self.account_size * 0.25:
            return False, f"Trade too large (${trade.total_margin:.0f})"
        
        # Win probability check (from XGBoost)
        if trade.win_probability < 0.4:
            return False, f"Win probability too low ({trade.win_probability:.0%})"
        
        return True, "Approved"
    
    def _calculate_confidence(self, regime: MarketRegime, market: MarketData,
                              trade: TradeSpec, win_prob: float) -> float:
        """Calculate final confidence score."""
        confidence = 0.5
        
        # Regime bonus
        regime_bonus = {
            MarketRegime.CALM: 0.15,
            MarketRegime.NORMAL: 0.10,
            MarketRegime.ELEVATED: 0.0,
            MarketRegime.HIGH_VOL: -0.10,
            MarketRegime.CRISIS: -0.20,
        }
        confidence += regime_bonus.get(regime, 0)
        
        # Win probability bonus (from XGBoost)
        confidence += (win_prob - 0.5) * 0.4
        
        # IV rank bonus
        if market.iv_rank > 50:
            confidence += 0.05
        
        return min(max(confidence, 0.1), 0.95)
    
    # =========================================================================
    # PORTFOLIO MANAGEMENT
    # =========================================================================
    
    def record_trade(self, trade: TradeSpec):
        """Record trade opening."""
        self.portfolio.margin_used += trade.total_margin
        self.portfolio.open_positions += 1
        self.portfolio.trades_today += 1
        self.portfolio.trades_this_week += 1
        
        symbol = trade.symbol
        self.portfolio.positions_by_symbol[symbol] = \
            self.portfolio.positions_by_symbol.get(symbol, 0) + 1
    
    def record_close(self, trade: TradeSpec, pnl: float):
        """Record trade closing."""
        self.portfolio.margin_used = max(0, self.portfolio.margin_used - trade.total_margin)
        self.portfolio.open_positions = max(0, self.portfolio.open_positions - 1)
        
        self.portfolio.daily_pnl += pnl
        self.portfolio.weekly_pnl += pnl
        self.portfolio.monthly_pnl += pnl
        
        self.portfolio.account_size += pnl
        self.portfolio.cash += pnl
        self.account_size = self.portfolio.account_size
        
        symbol = trade.symbol
        if symbol in self.portfolio.positions_by_symbol:
            self.portfolio.positions_by_symbol[symbol] -= 1
    
    def reset_daily(self):
        """Reset daily counters."""
        self.portfolio.daily_pnl = 0.0
        self.portfolio.trades_today = 0
    
    def reset_weekly(self):
        """Reset weekly counters."""
        self.portfolio.weekly_pnl = 0.0
        self.portfolio.trades_this_week = 0
    
    def reset_monthly(self):
        """Reset monthly counters."""
        self.portfolio.monthly_pnl = 0.0
    
    def get_status(self) -> Dict:
        """Get current status."""
        return {
            'account_size': self.account_size,
            'cash': self.portfolio.cash,
            'margin_used': self.portfolio.margin_used,
            'margin_available': self.portfolio.margin_available,
            'margin_usage_pct': self.portfolio.margin_usage_pct * 100,
            'open_positions': self.portfolio.open_positions,
            'daily_pnl': self.portfolio.daily_pnl,
            'daily_pnl_pct': self.portfolio.daily_pnl_pct * 100,
            'weekly_pnl': self.portfolio.weekly_pnl,
            'monthly_pnl': self.portfolio.monthly_pnl,
            'trades_today': self.portfolio.trades_today,
        }
    
    def get_daily_plan(self, vix: float = 18.0) -> Dict:
        """Get daily trading plan."""
        regime = self._classify_regime_fallback(vix)
        
        plan = {
            'date': date.today(),
            'regime': regime.name,
            'vix': vix,
            'account_size': self.account_size,
            'modules': {
                'strategy_advisor': self.strategy_advisor is not None,
                'circuit_breaker': self.circuit_breaker is not None,
            },
            'buckets': {}
        }
        
        for bucket, config in self.dte_configs.items():
            if bucket == DTEBucket.ZERO_DTE and regime in [MarketRegime.HIGH_VOL, MarketRegime.CRISIS]:
                continue
            
            allocation = self.account_size * config.allocation_pct
            max_risk = self.account_size * self.risk.max_risk_per_trade_pct
            
            plan['buckets'][config.name] = {
                'allocation': allocation,
                'target_delta': config.target_delta,
                'pop': config.pop,
                'max_trades': config.max_trades_per_day,
                'entry_window': f"{config.entry_start} - {config.entry_end}",
                'max_risk_per_trade': max_risk,
            }
        
        return plan


# =============================================================================
# DEMO
# =============================================================================

def main():
    """Demo Trading Brain."""
    
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 70)
    print(" TRADING BRAIN - ORCHESTRATOR")
    print("=" * 70)
    
    # Test with different account sizes
    for size in [500, 10000, 100000]:
        print(f"\n{'='*60}")
        print(f" ACCOUNT: ${size:,}")
        print("=" * 60)
        
        brain = TradingBrain(account_size=size)
        
        # Simulate market data
        market = MarketData(
            symbol="SPY",
            underlying_price=595.0,
            vix=17.5,
            iv_rank=45,
            trend="NEUTRAL",
        )
        
        # Get decision
        decision = brain.analyze(market)
        
        print(f"\nCircuit Breaker: {decision.circuit_breaker_level}")
        print(f"Decision: {decision.action.value}")
        
        if decision.trade_spec:
            print(f"Strategy: {decision.trade_spec.strategy.value}")
            print(f"Contracts: {decision.trade_spec.contracts}")
            print(f"Margin: ${decision.trade_spec.total_margin:.2f}")
            print(f"Win Prob: {decision.trade_spec.win_probability:.0%}")
        
        print(f"Confidence: {decision.confidence:.0%}")
        print(f"Should Execute: {decision.should_execute()}")
        
        print("\nReasoning:")
        for r in decision.reasoning:
            print(f"  • {r}")


if __name__ == "__main__":
    main()
