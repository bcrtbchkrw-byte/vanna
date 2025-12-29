#!/usr/bin/env python3
"""
strategy_configurator.py

Strategy Configurator pro 80%+ Win Rate Options Trading.

Klíčový insight: Win rate závisí na DELTA short striků:
- Delta 0.10 = 90% POP (probability of profit)
- Delta 0.16 = 84% POP  ← SWEET SPOT pro 80% win rate
- Delta 0.20 = 80% POP
- Delta 0.30 = 70% POP
- Delta 0.50 = 50% POP (ATM)

Trade-off: Vyšší POP = menší premium = nižší ROI per trade
Řešení: Volume - více trades s vysokou POP

Usage:
    from strategy_configurator import StrategyConfigurator, Strategy
    
    configurator = StrategyConfigurator()
    trade = configurator.configure(Strategy.IRON_CONDOR, market_state)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple
from datetime import datetime, date, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

from ml.enums import Strategy, LegType, OptionRight


@dataclass
class OptionLeg:
    """Jedna noha opční strategie."""
    action: LegType
    right: OptionRight
    strike: float
    expiry: date
    delta: float = 0.0
    price: float = 0.0
    iv: float = 0.0
    quantity: int = 1
    
    @property
    def is_short(self) -> bool:
        return self.action == LegType.SELL
    
    def __repr__(self):
        return f"{self.action.value} {self.right.value} {self.strike} @ {self.expiry}"


@dataclass
class TradeSpec:
    """Kompletní specifikace obchodu."""
    strategy: Strategy
    symbol: str
    underlying_price: float
    legs: List[OptionLeg]
    
    # Calculated fields
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven_low: float = 0.0
    breakeven_high: float = 0.0
    pop: float = 0.0  # Probability of profit
    expected_value: float = 0.0
    
    # Risk metrics
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0  # Positive = collecting theta
    net_vega: float = 0.0
    
    # Management rules
    profit_target_pct: float = 0.50  # Close at 50% max profit
    stop_loss_pct: float = 2.0  # Close at 200% of credit received
    dte_exit: int = 21  # Roll/close at 21 DTE
    
    @property
    def risk_reward_ratio(self) -> float:
        if self.max_profit == 0:
            return 0
        return self.max_loss / self.max_profit
    
    @property
    def credit_received(self) -> float:
        """Pro credit spreads."""
        return sum(leg.price * (1 if leg.is_short else -1) * leg.quantity * 100 
                   for leg in self.legs)
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"TRADE: {self.strategy.value} on {self.symbol}",
            f"{'='*60}",
            f"Underlying: ${self.underlying_price:.2f}",
            f"",
            "LEGS:",
        ]
        for leg in self.legs:
            lines.append(f"  {leg.action.value:4} {leg.right.value:4} "
                        f"${leg.strike:>7.2f} | Δ={leg.delta:+.3f} | "
                        f"${leg.price:.2f}")
        
        lines.extend([
            "",
            "P&L PROFILE:",
            f"  Max Profit:  ${self.max_profit:>8.2f}",
            f"  Max Loss:    ${self.max_loss:>8.2f}",
            f"  Risk/Reward: {self.risk_reward_ratio:.1f}:1",
            f"  POP:         {self.pop*100:.1f}%",
            f"  Expected:    ${self.expected_value:>8.2f}",
            "",
            "BREAKEVENS:",
            f"  Low:  ${self.breakeven_low:.2f}",
            f"  High: ${self.breakeven_high:.2f}",
            "",
            "GREEKS (position):",
            f"  Delta: {self.net_delta:+.2f}",
            f"  Gamma: {self.net_gamma:+.4f}",
            f"  Theta: {self.net_theta:+.2f}/day",
            f"  Vega:  {self.net_vega:+.2f}",
            "",
            "MANAGEMENT:",
            f"  Take profit: {self.profit_target_pct*100:.0f}% of max",
            f"  Stop loss:   {self.stop_loss_pct*100:.0f}% of credit",
            f"  Exit at:     {self.dte_exit} DTE",
            f"{'='*60}",
        ])
        return "\n".join(lines)


@dataclass 
class OptionChainRow:
    """Jeden řádek z option chainu."""
    strike: float
    expiry: date
    right: OptionRight
    bid: float
    ask: float
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float
    volume: int = 0
    open_interest: int = 0
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        if self.mid == 0:
            return float('inf')
        return self.spread / self.mid


@dataclass
class MarketState:
    """Aktuální stav trhu pro konfiguraci."""
    symbol: str
    underlying_price: float
    option_chain: List[OptionChainRow]
    
    # Volatility
    iv_rank: float  # 0-100
    vix: float
    
    # Technical
    atr_20: float  # 20-day ATR pro wing width
    
    # Time
    current_time: datetime = field(default_factory=datetime.now)


# =============================================================================
# WIN RATE CONFIGURATION
# =============================================================================

@dataclass
class WinRateConfig:
    """
    Konfigurace pro cílovou win rate.
    
    Vztah delta → POP (Probability of Profit):
    - Aproximace: POP ≈ 1 - |delta| pro OTM opce
    - Pro credit spread: POP ≈ 1 - |delta_short|
    
    Pro 80% win rate potřebujeme delta ≈ 0.16-0.20
    """
    
    # Target deltas pro různé strategie
    target_delta_short: float = 0.16  # 84% POP → ~80% realized win rate
    
    # DTE targeting
    min_dte: int = 30
    max_dte: int = 45
    target_dte: int = 35  # Sweet spot pro theta decay
    
    # Wing width (pro defined risk strategies)
    min_wing_width: float = 2.0  # Minimálně $2 wide
    max_wing_width: float = 10.0  # Max $10 wide
    wing_width_atr_mult: float = 0.5  # Wing = 0.5 * ATR
    
    # Liquidity filters
    min_open_interest: int = 100
    max_spread_pct: float = 0.10  # Max 10% bid-ask spread
    
    # Management rules pro vysokou win rate
    profit_target_pct: float = 0.50  # Bere 50% max profit BRZO
    stop_loss_multiplier: float = 2.0  # Stop at 2x credit
    roll_at_dte: int = 21  # Roll nebo close at 21 DTE
    
    # Position sizing
    max_portfolio_risk_pct: float = 0.02  # Max 2% portfolio per trade
    max_contracts: int = 10  # Safety limit


# =============================================================================
# DTE BUCKET CONFIGURATIONS
# =============================================================================
#
# Bucket 0: 0DTE      (0 days)     - Intraday gamma scalping
# Bucket 1: WEEKLY    (1-14 days)  - Short-term theta harvest  
# Bucket 2: MONTHLY   (15-45 days) - Standard theta strategies ← DEFAULT
# Bucket 3: LEAPS     (46+ days)   - Long-term, conservative
# =============================================================================

# 0DTE Configuration - INTRADAY THETA HARVEST
# Requires: Intraday data, tight spreads, liquid underlyings (SPY/SPX/QQQ)
ZERO_DTE_CONFIG = WinRateConfig(
    target_delta_short=0.10,      # 90% POP - WIDER strikes for fast moves
    min_dte=0,
    max_dte=0,
    target_dte=0,
    profit_target_pct=0.50,       # Take 50% quickly
    stop_loss_multiplier=1.5,     # Tighter stop - moves are fast
    roll_at_dte=0,                # Can't roll 0DTE
    min_wing_width=1.0,           # Tighter wings OK for 0DTE
    max_wing_width=5.0,           # Not too wide
    wing_width_atr_mult=0.3,      # Smaller wings
    min_open_interest=500,        # Higher liquidity requirement
    max_spread_pct=0.05,          # Tighter bid-ask required
    max_portfolio_risk_pct=0.01,  # Smaller position size (1%)
)

# WEEKLY Configuration (1-14 DTE) - SHORT-TERM AGGRESSIVE
WEEKLY_CONFIG = WinRateConfig(
    target_delta_short=0.20,      # 80% POP
    min_dte=1,
    max_dte=14,
    target_dte=7,
    profit_target_pct=0.40,       # Take profits at 40%
    stop_loss_multiplier=2.0,
    roll_at_dte=2,                # Roll at 2 DTE or close
    min_wing_width=2.0,
    max_wing_width=7.0,
    wing_width_atr_mult=0.4,
    min_open_interest=200,
    max_spread_pct=0.08,
    max_portfolio_risk_pct=0.015, # 1.5% per trade
)

# MONTHLY Configuration (15-45 DTE) - BALANCED DEFAULT
# This is the SWEET SPOT for 80% win rate theta strategies
MONTHLY_CONFIG = WinRateConfig(
    target_delta_short=0.16,      # 84% POP - OPTIMAL for 80% WR
    min_dte=15,
    max_dte=45,
    target_dte=30,                # 30 DTE = best theta/gamma ratio
    profit_target_pct=0.50,       # Standard 50% profit target
    stop_loss_multiplier=2.0,
    roll_at_dte=14,               # Roll at 14 DTE (2 weeks before expiry)
    min_wing_width=2.0,
    max_wing_width=10.0,
    wing_width_atr_mult=0.5,
    min_open_interest=100,
    max_spread_pct=0.10,
    max_portfolio_risk_pct=0.02,  # 2% per trade
)

# LEAPS Configuration (46+ DTE) - CONSERVATIVE
LEAPS_CONFIG = WinRateConfig(
    target_delta_short=0.10,      # 90% POP - very safe
    min_dte=46,
    max_dte=90,
    target_dte=60,
    profit_target_pct=0.25,       # Take profits early
    stop_loss_multiplier=1.5,
    roll_at_dte=30,               # Roll at 30 DTE
    min_wing_width=5.0,
    max_wing_width=15.0,
    wing_width_atr_mult=0.6,
    min_open_interest=50,
    max_spread_pct=0.15,          # Wider spreads OK for LEAPS
    max_portfolio_risk_pct=0.02,
)

# Backward compatibility aliases
CONSERVATIVE_CONFIG = LEAPS_CONFIG
BALANCED_CONFIG = MONTHLY_CONFIG
AGGRESSIVE_CONFIG = WEEKLY_CONFIG

# =============================================================================
# DTE BUCKET MAPPING
# =============================================================================

DTE_BUCKET_TO_CONFIG = {
    0: ZERO_DTE_CONFIG,    # 0DTE - Intraday
    1: WEEKLY_CONFIG,      # WEEKLY (1-14 DTE)
    2: MONTHLY_CONFIG,     # MONTHLY (15-45 DTE) ← DEFAULT
    3: LEAPS_CONFIG,       # LEAPS (46+ DTE)
}

def get_config_for_dte(dte: int) -> WinRateConfig:
    """Get appropriate config based on DTE."""
    if dte == 0:
        return ZERO_DTE_CONFIG
    elif dte <= 14:
        return WEEKLY_CONFIG
    elif dte <= 45:
        return MONTHLY_CONFIG
    else:
        return LEAPS_CONFIG

def get_config_for_bucket(bucket: int) -> WinRateConfig:
    """Get config for DTE bucket (0-3)."""
    return DTE_BUCKET_TO_CONFIG.get(bucket, MONTHLY_CONFIG)


# =============================================================================
# STRATEGY CONFIGURATOR
# =============================================================================

class StrategyConfigurator:
    """
    Konfiguruje konkrétní opční strategie pro cílovou win rate.
    
    Používá delta targeting pro výběr striků.
    Podporuje všechny DTE buckety včetně 0DTE.
    """
    
    # High liquidity symbols (no warning)
    HIGH_LIQUIDITY_SYMBOLS = {'SPY', 'SPX', 'QQQ', 'IWM', 'XSP', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'AMZN', 'META', 'GOOGL', 'MSFT'}
    
    # 0DTE Entry windows (avoid open volatility and gamma explosion at close)
    ZERO_DTE_ENTRY_START = 9 * 60 + 45   # 9:45 AM ET (minutes from midnight)
    ZERO_DTE_ENTRY_END = 14 * 60         # 2:00 PM ET
    
    def __init__(self, config: WinRateConfig = None, dte_bucket: int = None):
        """
        Args:
            config: WinRateConfig (if None, uses dte_bucket to select)
            dte_bucket: 0-3 bucket to auto-select config
        """
        if config is not None:
            self.config = config
        elif dte_bucket is not None:
            self.config = get_config_for_bucket(dte_bucket)
        else:
            self.config = MONTHLY_CONFIG  # Default
        
        self.dte_bucket = dte_bucket
    
    def configure(self, strategy: Strategy, state: MarketState, 
                  dte_bucket: int = None) -> Optional[TradeSpec]:
        """
        Hlavní entry point - nakonfiguruje strategii.
        
        Args:
            strategy: Strategy to configure
            state: Current market state
            dte_bucket: Override DTE bucket (0-3)
        """
        
        if strategy == Strategy.CASH:
            return None
        
        # Select config based on bucket if provided
        if dte_bucket is not None:
            self.config = get_config_for_bucket(dte_bucket)
        
        # 0DTE specific checks (time window only, no symbol restriction)
        if self.config == ZERO_DTE_CONFIG:
            if not self._validate_zero_dte(state):
                return None
        
        # Najdi správnou expiraci
        expiry = self._find_expiry(state)
        if expiry is None:
            logger.warning(f"No suitable expiry found for {state.symbol}")
            return None
        
        # Filtruj chain na tuto expiraci
        chain = [opt for opt in state.option_chain if opt.expiry == expiry]
        
        if not chain:
            logger.warning(f"No options for expiry {expiry}")
            return None
        
        # Dispatch na správný konfigurátor
        configurators = {
            Strategy.IRON_CONDOR: self._configure_iron_condor,
            Strategy.PUT_CREDIT_SPREAD: self._configure_put_credit_spread,
            Strategy.CALL_CREDIT_SPREAD: self._configure_call_credit_spread,
            Strategy.BULL_PUT_SPREAD: self._configure_put_credit_spread,  # Alias
            Strategy.BEAR_CALL_SPREAD: self._configure_call_credit_spread,  # Alias
            Strategy.SHORT_STRANGLE: self._configure_short_strangle,
            Strategy.IRON_BUTTERFLY: self._configure_iron_butterfly,
            Strategy.LONG_STRADDLE: self._configure_long_straddle,
            Strategy.LONG_STRANGLE: self._configure_long_strangle,
            Strategy.BULL_CALL_SPREAD: self._configure_bull_call_spread,
            Strategy.BEAR_PUT_SPREAD: self._configure_bear_put_spread,
        }
        
        if strategy not in configurators:
            logger.error(f"Strategy {strategy} not implemented")
            return None
        
        return configurators[strategy](state, chain, expiry)
    
    def _validate_zero_dte(self, state: MarketState) -> bool:
        """Validate that 0DTE trade is allowed."""
        
        # Info about liquidity (not blocking, just info)
        if state.symbol not in self.HIGH_LIQUIDITY_SYMBOLS:
            logger.info(f"0DTE on {state.symbol} - check bid-ask spreads for liquidity")
        
        # Check time window (this IS blocking for safety)
        current_minutes = state.current_time.hour * 60 + state.current_time.minute
        
        if current_minutes < self.ZERO_DTE_ENTRY_START:
            logger.warning(f"0DTE: Too early ({state.current_time.strftime('%H:%M')}), wait until 9:45 AM")
            return False
        
        if current_minutes > self.ZERO_DTE_ENTRY_END:
            logger.warning(f"0DTE: Too late ({state.current_time.strftime('%H:%M')}), gamma risk too high")
            return False
        
        return True
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _find_expiry(self, state: MarketState) -> Optional[date]:
        """Najde expiraci v target DTE range."""
        today = state.current_time.date()
        
        # Získej unikátní expirace
        expiries = sorted(set(opt.expiry for opt in state.option_chain))
        
        # 0DTE special handling
        if self.config.target_dte == 0:
            # Look for today's expiration
            if today in expiries:
                return today
            else:
                logger.warning(f"No 0DTE expiration available for {today}")
                return None
        
        best_expiry = None
        best_diff = float('inf')
        
        for exp in expiries:
            dte = (exp - today).days
            
            if self.config.min_dte <= dte <= self.config.max_dte:
                diff = abs(dte - self.config.target_dte)
                if diff < best_diff:
                    best_diff = diff
                    best_expiry = exp
        
        return best_expiry
    
    def _find_strike_by_delta(self, chain: List[OptionChainRow], 
                               target_delta: float, 
                               right: OptionRight) -> Optional[OptionChainRow]:
        """Najde strike nejblíž target delta."""
        
        # Filter by right
        options = [opt for opt in chain if opt.right == right]
        
        if not options:
            return None
        
        # Pro puts je delta záporné, ale target_delta je abs hodnota
        if right == OptionRight.PUT:
            target_delta = -abs(target_delta)
        else:
            target_delta = abs(target_delta)
        
        # Najdi nejbližší
        best = min(options, key=lambda o: abs(o.delta - target_delta))
        
        # Liquidity check
        if best.open_interest < self.config.min_open_interest:
            logger.warning(f"Low OI for {best.strike}: {best.open_interest}")
        
        if best.spread_pct > self.config.max_spread_pct:
            logger.warning(f"Wide spread for {best.strike}: {best.spread_pct:.1%}")
        
        return best
    
    def _find_strike_by_offset(self, chain: List[OptionChainRow],
                                base_strike: float,
                                offset: float,
                                right: OptionRight) -> Optional[OptionChainRow]:
        """Najde strike s daným offsetem od base strike."""
        target_strike = base_strike + offset
        
        options = [opt for opt in chain if opt.right == right]
        
        if not options:
            return None
        
        return min(options, key=lambda o: abs(o.strike - target_strike))
    
    def _calculate_wing_width(self, state: MarketState) -> float:
        """Vypočítá wing width na základě ATR."""
        # Wing = ATR * multiplier, ale v rámci limitů
        width = state.atr_20 * self.config.wing_width_atr_mult
        
        # Round to standard strikes ($1, $2.5, $5)
        if width < 3:
            width = round(width)  # $1 increments
        elif width < 7.5:
            width = round(width / 2.5) * 2.5  # $2.5 increments
        else:
            width = round(width / 5) * 5  # $5 increments
        
        return max(self.config.min_wing_width, 
                   min(self.config.max_wing_width, width))
    
    def _create_leg(self, opt: OptionChainRow, action: LegType, 
                    quantity: int = 1) -> OptionLeg:
        """Vytvoří OptionLeg z chain row."""
        return OptionLeg(
            action=action,
            right=opt.right,
            strike=opt.strike,
            expiry=opt.expiry,
            delta=opt.delta,
            price=opt.mid,
            iv=opt.iv,
            quantity=quantity
        )
    
    def _calculate_position_greeks(self, legs: List[OptionLeg]) -> Dict[str, float]:
        """Vypočítá net Greeks pro pozici."""
        net_delta = 0
        net_gamma = 0
        net_theta = 0
        net_vega = 0
        
        for leg in legs:
            multiplier = -1 if leg.is_short else 1
            # Poznámka: Tady bychom potřebovali plné Greeks z chain
            # Pro teď používáme delta z leg
            net_delta += leg.delta * multiplier * leg.quantity * 100
        
        return {
            'delta': net_delta,
            'gamma': net_gamma,
            'theta': net_theta,
            'vega': net_vega
        }
    
    # =========================================================================
    # STRATEGY CONFIGURATORS - PREMIUM SELLING (HIGH WIN RATE)
    # =========================================================================
    
    def _configure_iron_condor(self, state: MarketState, 
                                chain: List[OptionChainRow],
                                expiry: date) -> Optional[TradeSpec]:
        """
        Iron Condor - KRÁL 80% win rate strategií.
        
        Structure:
        - BUY  PUT  (long put wing)
        - SELL PUT  (short put @ 16 delta)
        - SELL CALL (short call @ 16 delta)  
        - BUY  CALL (long call wing)
        
        Profit zone: Between short strikes
        Max profit: Net credit received
        Max loss: Wing width - credit
        """
        
        # 1. Najdi short strikes (16 delta = 84% POP každá strana)
        short_put = self._find_strike_by_delta(
            chain, self.config.target_delta_short, OptionRight.PUT
        )
        short_call = self._find_strike_by_delta(
            chain, self.config.target_delta_short, OptionRight.CALL
        )
        
        if not short_put or not short_call:
            logger.error("Could not find short strikes")
            return None
        
        # 2. Vypočítej wing width
        wing_width = self._calculate_wing_width(state)
        
        # 3. Najdi long strikes (wings)
        long_put = self._find_strike_by_offset(
            chain, short_put.strike, -wing_width, OptionRight.PUT
        )
        long_call = self._find_strike_by_offset(
            chain, short_call.strike, wing_width, OptionRight.CALL
        )
        
        if not long_put or not long_call:
            logger.error("Could not find wing strikes")
            return None
        
        # 4. Vytvoř nohy
        legs = [
            self._create_leg(long_put, LegType.BUY),
            self._create_leg(short_put, LegType.SELL),
            self._create_leg(short_call, LegType.SELL),
            self._create_leg(long_call, LegType.BUY),
        ]
        
        # 5. Vypočítej P&L
        credit = (short_put.mid + short_call.mid - long_put.mid - long_call.mid)
        max_profit = credit * 100
        max_loss = (wing_width - credit) * 100
        
        # 6. Breakevens
        breakeven_low = short_put.strike - credit
        breakeven_high = short_call.strike + credit
        
        # 7. POP - aproximace: střed mezi oběma short strikes
        # Přesnější: použij normální distribuci
        pop = self._calculate_iron_condor_pop(
            state.underlying_price,
            short_put.strike,
            short_call.strike,
            short_put.iv,
            (expiry - state.current_time.date()).days
        )
        
        # 8. Expected value
        expected_value = pop * max_profit - (1 - pop) * max_loss
        
        trade = TradeSpec(
            strategy=Strategy.IRON_CONDOR,
            symbol=state.symbol,
            underlying_price=state.underlying_price,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_low=breakeven_low,
            breakeven_high=breakeven_high,
            pop=pop,
            expected_value=expected_value,
            profit_target_pct=self.config.profit_target_pct,
            stop_loss_pct=self.config.stop_loss_multiplier,
            dte_exit=self.config.roll_at_dte,
        )
        
        # 9. Position Greeks
        greeks = self._calculate_position_greeks(legs)
        trade.net_delta = greeks['delta']
        trade.net_gamma = greeks['gamma']
        trade.net_theta = greeks['theta']
        trade.net_vega = greeks['vega']
        
        return trade
    
    def _calculate_iron_condor_pop(self, S: float, put_strike: float, 
                                    call_strike: float, iv: float, 
                                    dte: int) -> float:
        """
        Vypočítá POP pro iron condor.
        
        POP = P(put_strike < S_T < call_strike)
        """
        from scipy.stats import norm
        
        T = dte / 365.0
        if T <= 0 or iv <= 0:
            return 0.5
        
        # Log-normal distribution
        # P(S_T > K) = N(d2) for call
        # P(S_T < K) = N(-d2) for put
        r = 0.05  # Risk-free rate approximation
        
        def d2(K):
            return (np.log(S/K) + (r - 0.5*iv**2)*T) / (iv*np.sqrt(T))
        
        # P(S_T > put_strike) = N(d2_put)
        prob_above_put = norm.cdf(d2(put_strike))
        
        # P(S_T < call_strike) = N(-d2_call) = 1 - N(d2_call)
        prob_below_call = 1 - norm.cdf(d2(call_strike))
        
        # POP = P(above put) - P(above call) = P(between strikes)
        pop = prob_above_put - (1 - prob_below_call)
        
        return max(0, min(1, pop))
    
    def _configure_put_credit_spread(self, state: MarketState,
                                      chain: List[OptionChainRow],
                                      expiry: date) -> Optional[TradeSpec]:
        """
        Put Credit Spread (Bull Put Spread).
        
        Structure:
        - SELL PUT (short @ 16 delta)
        - BUY  PUT (long, lower strike)
        
        Bullish to neutral bias.
        """
        
        short_put = self._find_strike_by_delta(
            chain, self.config.target_delta_short, OptionRight.PUT
        )
        
        if not short_put:
            return None
        
        wing_width = self._calculate_wing_width(state)
        
        long_put = self._find_strike_by_offset(
            chain, short_put.strike, -wing_width, OptionRight.PUT
        )
        
        if not long_put:
            return None
        
        legs = [
            self._create_leg(long_put, LegType.BUY),
            self._create_leg(short_put, LegType.SELL),
        ]
        
        credit = short_put.mid - long_put.mid
        max_profit = credit * 100
        max_loss = (wing_width - credit) * 100
        breakeven = short_put.strike - credit
        
        # POP = P(S_T > short_strike)
        pop = 1 - abs(short_put.delta)  # Aproximace
        
        return TradeSpec(
            strategy=Strategy.PUT_CREDIT_SPREAD,
            symbol=state.symbol,
            underlying_price=state.underlying_price,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_low=breakeven,
            breakeven_high=float('inf'),
            pop=pop,
            expected_value=pop * max_profit - (1-pop) * max_loss,
            profit_target_pct=self.config.profit_target_pct,
            stop_loss_pct=self.config.stop_loss_multiplier,
            dte_exit=self.config.roll_at_dte,
        )
    
    def _configure_call_credit_spread(self, state: MarketState,
                                       chain: List[OptionChainRow],
                                       expiry: date) -> Optional[TradeSpec]:
        """
        Call Credit Spread (Bear Call Spread).
        
        Structure:
        - SELL CALL (short @ 16 delta)
        - BUY  CALL (long, higher strike)
        
        Bearish to neutral bias.
        """
        
        short_call = self._find_strike_by_delta(
            chain, self.config.target_delta_short, OptionRight.CALL
        )
        
        if not short_call:
            return None
        
        wing_width = self._calculate_wing_width(state)
        
        long_call = self._find_strike_by_offset(
            chain, short_call.strike, wing_width, OptionRight.CALL
        )
        
        if not long_call:
            return None
        
        legs = [
            self._create_leg(short_call, LegType.SELL),
            self._create_leg(long_call, LegType.BUY),
        ]
        
        credit = short_call.mid - long_call.mid
        max_profit = credit * 100
        max_loss = (wing_width - credit) * 100
        breakeven = short_call.strike + credit
        
        pop = 1 - abs(short_call.delta)
        
        return TradeSpec(
            strategy=Strategy.CALL_CREDIT_SPREAD,
            symbol=state.symbol,
            underlying_price=state.underlying_price,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_low=0,
            breakeven_high=breakeven,
            pop=pop,
            expected_value=pop * max_profit - (1-pop) * max_loss,
            profit_target_pct=self.config.profit_target_pct,
            stop_loss_pct=self.config.stop_loss_multiplier,
            dte_exit=self.config.roll_at_dte,
        )
    
    def _configure_short_strangle(self, state: MarketState,
                                   chain: List[OptionChainRow],
                                   expiry: date) -> Optional[TradeSpec]:
        """
        Short Strangle - UNDEFINED RISK!
        
        Structure:
        - SELL PUT  (OTM @ 16 delta)
        - SELL CALL (OTM @ 16 delta)
        
        Vyžaduje margin a risk management!
        """
        
        short_put = self._find_strike_by_delta(
            chain, self.config.target_delta_short, OptionRight.PUT
        )
        short_call = self._find_strike_by_delta(
            chain, self.config.target_delta_short, OptionRight.CALL
        )
        
        if not short_put or not short_call:
            return None
        
        legs = [
            self._create_leg(short_put, LegType.SELL),
            self._create_leg(short_call, LegType.SELL),
        ]
        
        credit = short_put.mid + short_call.mid
        max_profit = credit * 100
        # Max loss je teoreticky unlimited!
        # Pro sizing použijeme "expected" max loss = 2-3x credit
        max_loss = credit * 100 * 3  # Conservative estimate
        
        breakeven_low = short_put.strike - credit
        breakeven_high = short_call.strike + credit
        
        pop = self._calculate_iron_condor_pop(
            state.underlying_price,
            short_put.strike,
            short_call.strike,
            short_put.iv,
            (expiry - state.current_time.date()).days
        )
        
        return TradeSpec(
            strategy=Strategy.SHORT_STRANGLE,
            symbol=state.symbol,
            underlying_price=state.underlying_price,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,  # UNDEFINED - toto je estimate
            breakeven_low=breakeven_low,
            breakeven_high=breakeven_high,
            pop=pop,
            expected_value=pop * max_profit - (1-pop) * max_loss * 0.5,
            profit_target_pct=self.config.profit_target_pct,
            stop_loss_pct=self.config.stop_loss_multiplier,
            dte_exit=self.config.roll_at_dte,
        )
    
    def _configure_iron_butterfly(self, state: MarketState,
                                   chain: List[OptionChainRow],
                                   expiry: date) -> Optional[TradeSpec]:
        """
        Iron Butterfly - ATM short strikes.
        
        Structure:
        - BUY  PUT  (OTM wing)
        - SELL PUT  (ATM)
        - SELL CALL (ATM, same strike)
        - BUY  CALL (OTM wing)
        
        Max profit at exactly ATM. Lower POP than IC.
        """
        
        # ATM strike
        atm_strike = round(state.underlying_price)
        
        atm_put = self._find_strike_by_offset(
            chain, atm_strike, 0, OptionRight.PUT
        )
        atm_call = self._find_strike_by_offset(
            chain, atm_strike, 0, OptionRight.CALL
        )
        
        if not atm_put or not atm_call:
            return None
        
        wing_width = self._calculate_wing_width(state)
        
        long_put = self._find_strike_by_offset(
            chain, atm_strike, -wing_width, OptionRight.PUT
        )
        long_call = self._find_strike_by_offset(
            chain, atm_strike, wing_width, OptionRight.CALL
        )
        
        if not long_put or not long_call:
            return None
        
        legs = [
            self._create_leg(long_put, LegType.BUY),
            self._create_leg(atm_put, LegType.SELL),
            self._create_leg(atm_call, LegType.SELL),
            self._create_leg(long_call, LegType.BUY),
        ]
        
        credit = atm_put.mid + atm_call.mid - long_put.mid - long_call.mid
        max_profit = credit * 100
        max_loss = (wing_width - credit) * 100
        
        # POP je nižší - potřebuje zůstat blízko ATM
        pop = 0.40  # Rough estimate - butterfly má narrow profit zone
        
        return TradeSpec(
            strategy=Strategy.IRON_BUTTERFLY,
            symbol=state.symbol,
            underlying_price=state.underlying_price,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_low=atm_strike - credit,
            breakeven_high=atm_strike + credit,
            pop=pop,
            expected_value=pop * max_profit - (1-pop) * max_loss,
            profit_target_pct=0.25,  # Take profits earlier on butterfly
            stop_loss_pct=self.config.stop_loss_multiplier,
            dte_exit=self.config.roll_at_dte,
        )
    
    # =========================================================================
    # STRATEGY CONFIGURATORS - PREMIUM BUYING (LOW WIN RATE)
    # =========================================================================
    
    def _configure_long_straddle(self, state: MarketState,
                                  chain: List[OptionChainRow],
                                  expiry: date) -> Optional[TradeSpec]:
        """
        Long Straddle - bet on volatility.
        
        Structure:
        - BUY PUT  (ATM)
        - BUY CALL (ATM, same strike)
        
        Low win rate, high reward potential.
        """
        
        atm_strike = round(state.underlying_price)
        
        atm_put = self._find_strike_by_offset(chain, atm_strike, 0, OptionRight.PUT)
        atm_call = self._find_strike_by_offset(chain, atm_strike, 0, OptionRight.CALL)
        
        if not atm_put or not atm_call:
            return None
        
        legs = [
            self._create_leg(atm_put, LegType.BUY),
            self._create_leg(atm_call, LegType.BUY),
        ]
        
        debit = atm_put.mid + atm_call.mid
        max_loss = debit * 100
        max_profit = float('inf')  # Unlimited on both sides
        
        breakeven_low = atm_strike - debit
        breakeven_high = atm_strike + debit
        
        pop = 0.35  # Straddles have low win rate
        
        return TradeSpec(
            strategy=Strategy.LONG_STRADDLE,
            symbol=state.symbol,
            underlying_price=state.underlying_price,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_low=breakeven_low,
            breakeven_high=breakeven_high,
            pop=pop,
            expected_value=0,  # Hard to estimate
            profit_target_pct=1.0,  # Let it run
            stop_loss_pct=0.50,  # Tight stop
            dte_exit=7,  # Exit early due to theta
        )
    
    def _configure_long_strangle(self, state: MarketState,
                                  chain: List[OptionChainRow],
                                  expiry: date) -> Optional[TradeSpec]:
        """
        Long Strangle - cheaper vol bet.
        
        Structure:
        - BUY PUT  (OTM)
        - BUY CALL (OTM)
        """
        
        # OTM strikes at ~25 delta
        long_put = self._find_strike_by_delta(chain, 0.25, OptionRight.PUT)
        long_call = self._find_strike_by_delta(chain, 0.25, OptionRight.CALL)
        
        if not long_put or not long_call:
            return None
        
        legs = [
            self._create_leg(long_put, LegType.BUY),
            self._create_leg(long_call, LegType.BUY),
        ]
        
        debit = long_put.mid + long_call.mid
        max_loss = debit * 100
        
        breakeven_low = long_put.strike - debit
        breakeven_high = long_call.strike + debit
        
        pop = 0.25  # Even lower than straddle
        
        return TradeSpec(
            strategy=Strategy.LONG_STRANGLE,
            symbol=state.symbol,
            underlying_price=state.underlying_price,
            legs=legs,
            max_profit=float('inf'),
            max_loss=max_loss,
            breakeven_low=breakeven_low,
            breakeven_high=breakeven_high,
            pop=pop,
            expected_value=0,
            profit_target_pct=1.5,  # Need bigger move
            stop_loss_pct=0.50,
            dte_exit=7,
        )
    
    # =========================================================================
    # STRATEGY CONFIGURATORS - DIRECTIONAL
    # =========================================================================
    
    def _configure_bull_call_spread(self, state: MarketState,
                                     chain: List[OptionChainRow],
                                     expiry: date) -> Optional[TradeSpec]:
        """
        Bull Call Spread - debit spread.
        
        Structure:
        - BUY  CALL (ATM or slightly ITM)
        - SELL CALL (OTM)
        """
        
        # Long slightly ITM or ATM (40-50 delta)
        long_call = self._find_strike_by_delta(chain, 0.45, OptionRight.CALL)
        
        if not long_call:
            return None
        
        wing_width = self._calculate_wing_width(state)
        
        short_call = self._find_strike_by_offset(
            chain, long_call.strike, wing_width, OptionRight.CALL
        )
        
        if not short_call:
            return None
        
        legs = [
            self._create_leg(long_call, LegType.BUY),
            self._create_leg(short_call, LegType.SELL),
        ]
        
        debit = long_call.mid - short_call.mid
        max_loss = debit * 100
        max_profit = (wing_width - debit) * 100
        breakeven = long_call.strike + debit
        
        pop = abs(long_call.delta)  # Rough estimate
        
        return TradeSpec(
            strategy=Strategy.BULL_CALL_SPREAD,
            symbol=state.symbol,
            underlying_price=state.underlying_price,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_low=breakeven,
            breakeven_high=float('inf'),
            pop=pop,
            expected_value=pop * max_profit - (1-pop) * max_loss,
            profit_target_pct=0.75,
            stop_loss_pct=0.50,
            dte_exit=self.config.roll_at_dte,
        )
    
    def _configure_bear_put_spread(self, state: MarketState,
                                    chain: List[OptionChainRow],
                                    expiry: date) -> Optional[TradeSpec]:
        """
        Bear Put Spread - debit spread.
        
        Structure:
        - BUY  PUT (ATM or slightly ITM)
        - SELL PUT (OTM)
        """
        
        long_put = self._find_strike_by_delta(chain, 0.45, OptionRight.PUT)
        
        if not long_put:
            return None
        
        wing_width = self._calculate_wing_width(state)
        
        short_put = self._find_strike_by_offset(
            chain, long_put.strike, -wing_width, OptionRight.PUT
        )
        
        if not short_put:
            return None
        
        legs = [
            self._create_leg(long_put, LegType.BUY),
            self._create_leg(short_put, LegType.SELL),
        ]
        
        debit = long_put.mid - short_put.mid
        max_loss = debit * 100
        max_profit = (wing_width - debit) * 100
        breakeven = long_put.strike - debit
        
        pop = abs(long_put.delta)
        
        return TradeSpec(
            strategy=Strategy.BEAR_PUT_SPREAD,
            symbol=state.symbol,
            underlying_price=state.underlying_price,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_low=0,
            breakeven_high=breakeven,
            pop=pop,
            expected_value=pop * max_profit - (1-pop) * max_loss,
            profit_target_pct=0.75,
            stop_loss_pct=0.50,
            dte_exit=self.config.roll_at_dte,
        )


# =============================================================================
# DEMO / TEST
# =============================================================================

def create_mock_option_chain(underlying: float, 
                              expiry: date,
                              iv: float = 0.20) -> List[OptionChainRow]:
    """Vytvoří mock option chain pro testování."""
    from scipy.stats import norm
    
    chain = []
    
    # Strikes od -20% do +20% od underlying
    strikes = np.arange(
        round(underlying * 0.80),
        round(underlying * 1.20),
        1.0 if underlying < 100 else 5.0
    )
    
    T = (expiry - date.today()).days / 365.0
    r = 0.05
    
    for K in strikes:
        for right in [OptionRight.CALL, OptionRight.PUT]:
            # Simplified BS delta
            d1 = (np.log(underlying/K) + (r + 0.5*iv**2)*T) / (iv*np.sqrt(T))
            
            if right == OptionRight.CALL:
                delta = norm.cdf(d1)
                # Simplified price
                price = max(0.05, underlying * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d1 - iv*np.sqrt(T)))
            else:
                delta = norm.cdf(d1) - 1
                price = max(0.05, K * np.exp(-r*T) * norm.cdf(-(d1 - iv*np.sqrt(T))) - underlying * norm.cdf(-d1))
            
            gamma = norm.pdf(d1) / (underlying * iv * np.sqrt(T))
            theta = -underlying * norm.pdf(d1) * iv / (2 * np.sqrt(T)) / 365
            vega = underlying * norm.pdf(d1) * np.sqrt(T) * 0.01
            
            # Add some spread
            spread = max(0.05, price * 0.02)
            
            chain.append(OptionChainRow(
                strike=K,
                expiry=expiry,
                right=right,
                bid=price - spread/2,
                ask=price + spread/2,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                iv=iv,
                volume=1000,
                open_interest=5000
            ))
    
    return chain


def main():
    """Demo všech strategií."""
    
    # Mock market state
    underlying = 595.0
    expiry = date.today() + timedelta(days=35)
    
    chain = create_mock_option_chain(underlying, expiry, iv=0.18)
    
    state = MarketState(
        symbol="SPY",
        underlying_price=underlying,
        option_chain=chain,
        iv_rank=45,
        vix=16.5,
        atr_20=8.5,
    )
    
    # Test různé konfigurace
    configs = [
        ("CONSERVATIVE (85% WR)", CONSERVATIVE_CONFIG),
        ("BALANCED (80% WR)", BALANCED_CONFIG),
        ("AGGRESSIVE (70% WR)", AGGRESSIVE_CONFIG),
    ]
    
    strategies_to_test = [
        Strategy.IRON_CONDOR,
        Strategy.PUT_CREDIT_SPREAD,
        Strategy.CALL_CREDIT_SPREAD,
    ]
    
    for config_name, config in configs:
        print(f"\n{'#'*70}")
        print(f"# {config_name}")
        print(f"# Target delta: {config.target_delta_short} | DTE: {config.target_dte}")
        print(f"{'#'*70}")
        
        configurator = StrategyConfigurator(config)
        
        for strategy in strategies_to_test:
            trade = configurator.configure(strategy, state)
            
            if trade:
                print(trade.summary())
            else:
                print(f"\n{strategy.value}: Failed to configure")


if __name__ == "__main__":
    main()
