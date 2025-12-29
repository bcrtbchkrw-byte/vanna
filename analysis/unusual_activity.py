#!/usr/bin/env python3
"""
unusual_activity.py

WHALE DETECTOR - Detekce neobvyklé aktivity z IBKR dat

Detekuje:
1. Volume Spikes - neobvyklý objem opcí
2. OI Changes - velké změny open interest
3. Block Trades - velké single transakce
4. Put/Call Ratio - sentiment shift
5. Support/Resistance z option flow - kde jsou velké pozice

ZDARMA - používá pouze IBKR data, žádné placené služby.

Usage:
    from unusual_activity import UnusualActivityDetector
    
    detector = UnusualActivityDetector()
    signals = await detector.scan_symbol('SPY')
    
    if signals.has_unusual_activity:
        print(f"⚠️ WHALE ALERT: {signals.summary}")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class SignalType(Enum):
    """Typ unusual activity signálu."""
    VOLUME_SPIKE = "VOLUME_SPIKE"
    OI_SURGE = "OI_SURGE"
    BLOCK_TRADE = "BLOCK_TRADE"
    PUT_CALL_SKEW = "PUT_CALL_SKEW"
    SWEEP_ORDER = "SWEEP_ORDER"
    WHALE_ACCUMULATION = "WHALE_ACCUMULATION"
    SUPPORT_LEVEL = "SUPPORT_LEVEL"
    RESISTANCE_LEVEL = "RESISTANCE_LEVEL"


class SignalDirection(Enum):
    """Směr signálu - bullish/bearish."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


# Thresholds pro detekci
THRESHOLDS = {
    'volume_spike_multiplier': 3.0,      # Volume > 3x average
    'oi_change_pct': 0.50,               # OI change > 50%
    'block_trade_min': 500,              # Min contracts for block
    'put_call_ratio_high': 1.5,          # Bearish threshold
    'put_call_ratio_low': 0.5,           # Bullish threshold
    'sweep_time_window_sec': 60,         # Sweeps within 60 sec
    'whale_oi_threshold': 10000,         # OI > 10k = significant level
    'whale_volume_threshold': 5000,      # Volume > 5k = whale activity
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OptionFlow:
    """Single option flow entry."""
    symbol: str
    strike: float
    expiry: date
    right: str                  # CALL or PUT
    volume: int
    open_interest: int
    oi_change: int              # Change from previous day
    avg_volume_20d: int
    last_trade_size: int
    last_trade_price: float
    bid: float
    ask: float
    underlying_price: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def volume_ratio(self) -> float:
        """Volume vs 20-day average."""
        if self.avg_volume_20d == 0:
            return 0
        return self.volume / self.avg_volume_20d
    
    @property
    def oi_change_pct(self) -> float:
        """OI change as percentage."""
        if self.open_interest == 0:
            return 0
        return self.oi_change / (self.open_interest - self.oi_change)
    
    @property
    def is_otm(self) -> bool:
        """Is option out of the money?"""
        if self.right == "CALL":
            return self.strike > self.underlying_price
        else:
            return self.strike < self.underlying_price
    
    @property
    def moneyness_pct(self) -> float:
        """How far OTM/ITM as percentage."""
        return (self.strike - self.underlying_price) / self.underlying_price


@dataclass
class UnusualSignal:
    """Jeden unusual activity signál."""
    type: SignalType
    direction: SignalDirection
    symbol: str
    strike: float
    expiry: date
    right: str
    strength: float             # 0-1, jak silný signál
    details: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __repr__(self):
        arrow = "🟢" if self.direction == SignalDirection.BULLISH else "🔴" if self.direction == SignalDirection.BEARISH else "⚪"
        return f"{arrow} {self.type.value}: {self.symbol} {self.strike}{self.right[0]} - {self.details}"


@dataclass  
class SupportResistance:
    """Support/Resistance level z option flow."""
    strike: float
    type: str                   # "SUPPORT" or "RESISTANCE"
    strength: float             # 0-1, based on OI/volume
    total_oi: int
    total_volume: int
    put_oi: int
    call_oi: int
    source: str                 # "PUT_WALL", "CALL_WALL", "GAMMA_LEVEL"
    
    def __repr__(self):
        emoji = "🟢" if self.type == "SUPPORT" else "🔴"
        return f"{emoji} {self.type} @ ${self.strike:.0f} (OI: {self.total_oi:,}, Strength: {self.strength:.0%})"


@dataclass
class UnusualActivityResult:
    """Výsledek scanu pro jeden symbol."""
    symbol: str
    underlying_price: float
    timestamp: datetime
    
    # Signals
    signals: List[UnusualSignal] = field(default_factory=list)
    
    # Support/Resistance levels
    support_levels: List[SupportResistance] = field(default_factory=list)
    resistance_levels: List[SupportResistance] = field(default_factory=list)
    
    # Aggregated metrics
    total_put_volume: int = 0
    total_call_volume: int = 0
    total_put_oi: int = 0
    total_call_oi: int = 0
    
    # Overall sentiment
    put_call_ratio: float = 0.0
    sentiment: SignalDirection = SignalDirection.NEUTRAL
    
    @property
    def has_unusual_activity(self) -> bool:
        return len(self.signals) > 0
    
    @property
    def bullish_signals(self) -> List[UnusualSignal]:
        return [s for s in self.signals if s.direction == SignalDirection.BULLISH]
    
    @property
    def bearish_signals(self) -> List[UnusualSignal]:
        return [s for s in self.signals if s.direction == SignalDirection.BEARISH]
    
    @property
    def strongest_support(self) -> Optional[SupportResistance]:
        if not self.support_levels:
            return None
        return max(self.support_levels, key=lambda x: x.strength)
    
    @property
    def strongest_resistance(self) -> Optional[SupportResistance]:
        if not self.resistance_levels:
            return None
        return max(self.resistance_levels, key=lambda x: x.strength)
    
    @property
    def summary(self) -> str:
        parts = []
        if self.bullish_signals:
            parts.append(f"{len(self.bullish_signals)} bullish")
        if self.bearish_signals:
            parts.append(f"{len(self.bearish_signals)} bearish")
        parts.append(f"P/C ratio: {self.put_call_ratio:.2f}")
        if self.strongest_support:
            parts.append(f"Support: ${self.strongest_support.strike:.0f}")
        if self.strongest_resistance:
            parts.append(f"Resistance: ${self.strongest_resistance.strike:.0f}")
        return " | ".join(parts)


# =============================================================================
# UNUSUAL ACTIVITY DETECTOR
# =============================================================================

class UnusualActivityDetector:
    """
    Detekuje unusual activity z IBKR dat.
    
    Sleduje:
    - Volume spikes (> 3x average)
    - OI changes (> 50%)
    - Block trades (> 500 contracts)
    - Put/Call ratio shifts
    - Support/Resistance z option walls
    """
    
    def __init__(self, ibkr_client=None):
        """
        Args:
            ibkr_client: IBKR client pro data (pokud None, použije mock)
        """
        self.ibkr = ibkr_client
        self.thresholds = THRESHOLDS.copy()
        
        # Cache
        self._flow_cache: Dict[str, List[OptionFlow]] = {}
        self._last_scan: Dict[str, datetime] = {}
        
        logger.info("UnusualActivityDetector initialized")
    
    async def scan_symbol(self, symbol: str, 
                          expiry_range_days: int = 45) -> UnusualActivityResult:
        """
        Skenuje symbol pro unusual activity.
        
        Args:
            symbol: Ticker symbol
            expiry_range_days: Jak daleko dopředu hledat expirace
            
        Returns:
            UnusualActivityResult s detekovanými signály
        """
        logger.info(f"Scanning {symbol} for unusual activity...")
        
        # Get option chain data
        flows = await self._get_option_flows(symbol, expiry_range_days)
        
        if not flows:
            logger.warning(f"No option flow data for {symbol}")
            return UnusualActivityResult(
                symbol=symbol,
                underlying_price=0,
                timestamp=datetime.now()
            )
        
        underlying_price = flows[0].underlying_price if flows else 0
        
        # Initialize result
        result = UnusualActivityResult(
            symbol=symbol,
            underlying_price=underlying_price,
            timestamp=datetime.now()
        )
        
        # Calculate aggregated metrics
        result.total_put_volume = sum(f.volume for f in flows if f.right == "PUT")
        result.total_call_volume = sum(f.volume for f in flows if f.right == "CALL")
        result.total_put_oi = sum(f.open_interest for f in flows if f.right == "PUT")
        result.total_call_oi = sum(f.open_interest for f in flows if f.right == "CALL")
        
        if result.total_call_volume > 0:
            result.put_call_ratio = result.total_put_volume / result.total_call_volume
        
        # Determine sentiment
        if result.put_call_ratio > self.thresholds['put_call_ratio_high']:
            result.sentiment = SignalDirection.BEARISH
        elif result.put_call_ratio < self.thresholds['put_call_ratio_low']:
            result.sentiment = SignalDirection.BULLISH
        
        # Detect signals
        result.signals.extend(self._detect_volume_spikes(flows))
        result.signals.extend(self._detect_oi_changes(flows))
        result.signals.extend(self._detect_block_trades(flows))
        result.signals.extend(self._detect_put_call_skew(flows, result.put_call_ratio))
        
        # Find support/resistance levels
        result.support_levels, result.resistance_levels = self._find_support_resistance(
            flows, underlying_price
        )
        
        # Add support/resistance as signals
        for level in result.support_levels:
            if level.strength > 0.7:
                result.signals.append(UnusualSignal(
                    type=SignalType.SUPPORT_LEVEL,
                    direction=SignalDirection.BULLISH,
                    symbol=symbol,
                    strike=level.strike,
                    expiry=date.today(),
                    right="PUT",
                    strength=level.strength,
                    details=f"Strong PUT wall: {level.total_oi:,} OI"
                ))
        
        for level in result.resistance_levels:
            if level.strength > 0.7:
                result.signals.append(UnusualSignal(
                    type=SignalType.RESISTANCE_LEVEL,
                    direction=SignalDirection.BEARISH,
                    symbol=symbol,
                    strike=level.strike,
                    expiry=date.today(),
                    right="CALL",
                    strength=level.strength,
                    details=f"Strong CALL wall: {level.total_oi:,} OI"
                ))
        
        logger.info(f"Scan complete: {len(result.signals)} signals, "
                   f"{len(result.support_levels)} supports, "
                   f"{len(result.resistance_levels)} resistances")
        
        return result
    
    async def _get_option_flows(self, symbol: str, 
                                 expiry_range_days: int) -> List[OptionFlow]:
        """Získá option flow data z IBKR nebo mock."""
        
        if self.ibkr:
            # Real IBKR implementation
            return await self._fetch_ibkr_flows(symbol, expiry_range_days)
        else:
            # Mock data pro testování
            return self._generate_mock_flows(symbol, expiry_range_days)
    
    async def _fetch_ibkr_flows(self, symbol: str, 
                                 expiry_range_days: int) -> List[OptionFlow]:
        """Fetch real data from IBKR."""
        # TODO: Implement real IBKR data fetching
        # This would use:
        # - reqMktData for real-time quotes
        # - reqHistoricalData for historical volume
        # - reqSecDefOptParams for option chain
        raise NotImplementedError("IBKR integration pending")
    
    def _generate_mock_flows(self, symbol: str, 
                              expiry_range_days: int) -> List[OptionFlow]:
        """Generuje mock data pro testování."""
        import random
        
        # Estimate underlying price
        price_estimates = {
            'SPY': 595, 'QQQ': 520, 'AAPL': 250, 'TSLA': 420,
            'NVDA': 140, 'AMD': 120, 'PLTR': 80, 'SOFI': 15,
        }
        underlying = price_estimates.get(symbol, 100)
        
        flows = []
        today = date.today()
        
        # Generate strikes around current price
        if underlying > 100:
            strike_step = 5
        elif underlying > 50:
            strike_step = 2.5
        else:
            strike_step = 1
        
        strikes = [underlying + i * strike_step for i in range(-10, 11)]
        
        # Generate a few expiries
        expiries = [
            today,                              # 0DTE
            today + timedelta(days=7),          # Weekly
            today + timedelta(days=30),         # Monthly
        ]
        
        for expiry in expiries:
            for strike in strikes:
                for right in ["PUT", "CALL"]:
                    # Random but realistic-ish data
                    base_oi = random.randint(100, 5000)
                    base_volume = random.randint(50, 2000)
                    
                    # Add some "unusual" activity randomly
                    if random.random() < 0.1:  # 10% chance of unusual
                        base_volume *= random.randint(3, 10)
                        base_oi += random.randint(1000, 5000)
                    
                    # More activity near ATM
                    atm_factor = max(0.1, 1 - abs(strike - underlying) / underlying)
                    base_oi = int(base_oi * atm_factor * 2)
                    base_volume = int(base_volume * atm_factor * 2)
                    
                    flows.append(OptionFlow(
                        symbol=symbol,
                        strike=strike,
                        expiry=expiry,
                        right=right,
                        volume=base_volume,
                        open_interest=base_oi,
                        oi_change=random.randint(-500, 1000),
                        avg_volume_20d=max(100, base_volume // 3),
                        last_trade_size=random.randint(1, 100),
                        last_trade_price=random.uniform(0.5, 10),
                        bid=random.uniform(0.4, 9),
                        ask=random.uniform(0.6, 11),
                        underlying_price=underlying,
                    ))
        
        return flows
    
    def _detect_volume_spikes(self, flows: List[OptionFlow]) -> List[UnusualSignal]:
        """Detekuje volume spikes."""
        signals = []
        threshold = self.thresholds['volume_spike_multiplier']
        
        for flow in flows:
            if flow.volume_ratio >= threshold:
                direction = SignalDirection.BULLISH if flow.right == "CALL" else SignalDirection.BEARISH
                
                # OTM calls = very bullish, OTM puts = very bearish
                if flow.is_otm:
                    strength = min(1.0, flow.volume_ratio / 10)
                else:
                    strength = min(1.0, flow.volume_ratio / 15)
                
                signals.append(UnusualSignal(
                    type=SignalType.VOLUME_SPIKE,
                    direction=direction,
                    symbol=flow.symbol,
                    strike=flow.strike,
                    expiry=flow.expiry,
                    right=flow.right,
                    strength=strength,
                    details=f"Volume {flow.volume:,} = {flow.volume_ratio:.1f}x average"
                ))
        
        return signals
    
    def _detect_oi_changes(self, flows: List[OptionFlow]) -> List[UnusualSignal]:
        """Detekuje velké změny Open Interest."""
        signals = []
        threshold = self.thresholds['oi_change_pct']
        
        for flow in flows:
            if abs(flow.oi_change_pct) >= threshold and flow.oi_change > 500:
                # Positive OI change = new positions opened
                if flow.oi_change > 0:
                    direction = SignalDirection.BULLISH if flow.right == "CALL" else SignalDirection.BEARISH
                else:
                    # Negative OI change = positions closed (opposite signal)
                    direction = SignalDirection.BEARISH if flow.right == "CALL" else SignalDirection.BULLISH
                
                strength = min(1.0, abs(flow.oi_change_pct))
                
                signals.append(UnusualSignal(
                    type=SignalType.OI_SURGE,
                    direction=direction,
                    symbol=flow.symbol,
                    strike=flow.strike,
                    expiry=flow.expiry,
                    right=flow.right,
                    strength=strength,
                    details=f"OI change: {flow.oi_change:+,} ({flow.oi_change_pct:+.0%})"
                ))
        
        return signals
    
    def _detect_block_trades(self, flows: List[OptionFlow]) -> List[UnusualSignal]:
        """Detekuje block trades (velké single transakce)."""
        signals = []
        threshold = self.thresholds['block_trade_min']
        
        for flow in flows:
            if flow.last_trade_size >= threshold:
                direction = SignalDirection.BULLISH if flow.right == "CALL" else SignalDirection.BEARISH
                
                # Bought vs sold - check if trade was at ask (bought) or bid (sold)
                mid = (flow.bid + flow.ask) / 2
                if flow.last_trade_price > mid:
                    # Bought at ask = aggressive buyer
                    strength = min(1.0, flow.last_trade_size / 2000)
                else:
                    # Sold at bid = aggressive seller (flip direction)
                    direction = SignalDirection.BEARISH if flow.right == "CALL" else SignalDirection.BULLISH
                    strength = min(1.0, flow.last_trade_size / 2000) * 0.8
                
                signals.append(UnusualSignal(
                    type=SignalType.BLOCK_TRADE,
                    direction=direction,
                    symbol=flow.symbol,
                    strike=flow.strike,
                    expiry=flow.expiry,
                    right=flow.right,
                    strength=strength,
                    details=f"Block: {flow.last_trade_size:,} contracts @ ${flow.last_trade_price:.2f}"
                ))
        
        return signals
    
    def _detect_put_call_skew(self, flows: List[OptionFlow], 
                               pc_ratio: float) -> List[UnusualSignal]:
        """Detekuje unusual put/call skew."""
        signals = []
        
        if pc_ratio > self.thresholds['put_call_ratio_high']:
            signals.append(UnusualSignal(
                type=SignalType.PUT_CALL_SKEW,
                direction=SignalDirection.BEARISH,
                symbol=flows[0].symbol if flows else "UNKNOWN",
                strike=0,
                expiry=date.today(),
                right="PUT",
                strength=min(1.0, (pc_ratio - 1) / 2),
                details=f"High P/C ratio: {pc_ratio:.2f} (bearish)"
            ))
        elif pc_ratio < self.thresholds['put_call_ratio_low']:
            signals.append(UnusualSignal(
                type=SignalType.PUT_CALL_SKEW,
                direction=SignalDirection.BULLISH,
                symbol=flows[0].symbol if flows else "UNKNOWN",
                strike=0,
                expiry=date.today(),
                right="CALL",
                strength=min(1.0, (1 - pc_ratio) / 0.5),
                details=f"Low P/C ratio: {pc_ratio:.2f} (bullish)"
            ))
        
        return signals
    
    def _find_support_resistance(self, flows: List[OptionFlow],
                                   underlying: float) -> Tuple[List[SupportResistance], 
                                                               List[SupportResistance]]:
        """
        Najde support/resistance levels z option flow.
        
        PUT WALL = Support (market makers hedge by buying stock)
        CALL WALL = Resistance (market makers hedge by selling stock)
        """
        # Aggregate OI by strike
        strike_data: Dict[float, Dict] = {}
        
        for flow in flows:
            strike = flow.strike
            if strike not in strike_data:
                strike_data[strike] = {
                    'put_oi': 0, 'call_oi': 0,
                    'put_volume': 0, 'call_volume': 0,
                }
            
            if flow.right == "PUT":
                strike_data[strike]['put_oi'] += flow.open_interest
                strike_data[strike]['put_volume'] += flow.volume
            else:
                strike_data[strike]['call_oi'] += flow.open_interest
                strike_data[strike]['call_volume'] += flow.volume
        
        supports = []
        resistances = []
        
        # Find max OI for normalization
        max_oi = max(
            max(d['put_oi'], d['call_oi']) 
            for d in strike_data.values()
        ) if strike_data else 1
        
        whale_threshold = self.thresholds['whale_oi_threshold']
        
        for strike, data in strike_data.items():
            total_oi = data['put_oi'] + data['call_oi']
            total_volume = data['put_volume'] + data['call_volume']
            
            # SUPPORT: Large PUT OI below current price
            if strike < underlying and data['put_oi'] > whale_threshold:
                strength = data['put_oi'] / max_oi
                supports.append(SupportResistance(
                    strike=strike,
                    type="SUPPORT",
                    strength=strength,
                    total_oi=total_oi,
                    total_volume=total_volume,
                    put_oi=data['put_oi'],
                    call_oi=data['call_oi'],
                    source="PUT_WALL"
                ))
            
            # RESISTANCE: Large CALL OI above current price
            if strike > underlying and data['call_oi'] > whale_threshold:
                strength = data['call_oi'] / max_oi
                resistances.append(SupportResistance(
                    strike=strike,
                    type="RESISTANCE",
                    strength=strength,
                    total_oi=total_oi,
                    total_volume=total_volume,
                    put_oi=data['put_oi'],
                    call_oi=data['call_oi'],
                    source="CALL_WALL"
                ))
        
        # Sort by strength
        supports.sort(key=lambda x: x.strength, reverse=True)
        resistances.sort(key=lambda x: x.strength, reverse=True)
        
        # Keep top 5
        return supports[:5], resistances[:5]
    
    def get_whale_strikes(self, result: UnusualActivityResult) -> Dict[str, List[float]]:
        """
        Vrátí strikes kde jsou velké pozice (whale levels).
        
        Tyto strikes jsou dobré pro:
        - Short strike selection (vyhni se těmto)
        - Profit target levels
        - Stop loss levels
        """
        return {
            'support_strikes': [s.strike for s in result.support_levels],
            'resistance_strikes': [r.strike for r in result.resistance_levels],
            'avoid_strikes': [
                s.strike for s in result.signals 
                if s.type in [SignalType.WHALE_ACCUMULATION, SignalType.BLOCK_TRADE]
                and s.strength > 0.7
            ]
        }


# =============================================================================
# INTEGRATION WITH TRADING BRAIN
# =============================================================================

class WhaleAwareStrategyFilter:
    """
    Filtruje strategie podle whale activity.
    
    Použití v trading_brain.py:
    - Vyhni se short strikes na whale levels
    - Adjust delta když unusual activity detected
    - Alert při extreme signals
    """
    
    def __init__(self, detector: UnusualActivityDetector = None):
        self.detector = detector or UnusualActivityDetector()
    
    async def filter_trade(self, symbol: str, 
                           proposed_short_strike: float,
                           direction: str) -> Tuple[bool, str, Optional[float]]:
        """
        Zkontroluje jestli navrhovaný trade nekoliduje s whale activity.
        
        Args:
            symbol: Ticker
            proposed_short_strike: Navrhovaný short strike
            direction: "PUT" nebo "CALL"
            
        Returns:
            (approved, reason, suggested_alternative_strike)
        """
        result = await self.detector.scan_symbol(symbol)
        
        whale_strikes = self.detector.get_whale_strikes(result)
        
        # Check if proposed strike is on a whale level
        if direction == "PUT":
            # For put credit spread, check support levels
            for support in result.support_levels:
                if abs(proposed_short_strike - support.strike) < 2:
                    # Too close to support wall!
                    suggested = support.strike - 5  # Move further OTM
                    return (
                        False, 
                        f"Short strike ${proposed_short_strike} too close to support wall @ ${support.strike} ({support.total_oi:,} OI)",
                        suggested
                    )
        else:
            # For call credit spread, check resistance levels
            for resistance in result.resistance_levels:
                if abs(proposed_short_strike - resistance.strike) < 2:
                    suggested = resistance.strike + 5
                    return (
                        False,
                        f"Short strike ${proposed_short_strike} too close to resistance wall @ ${resistance.strike} ({resistance.total_oi:,} OI)",
                        suggested
                    )
        
        # Check for bearish signals when selling puts
        if direction == "PUT" and len(result.bearish_signals) > 2:
            return (
                False,
                f"Multiple bearish signals detected ({len(result.bearish_signals)}), avoid selling puts",
                None
            )
        
        # Check for bullish signals when selling calls
        if direction == "CALL" and len(result.bullish_signals) > 2:
            return (
                False,
                f"Multiple bullish signals detected ({len(result.bullish_signals)}), avoid selling calls",
                None
            )
        
        return (True, "No whale conflicts", None)


# =============================================================================
# CLI / DEMO
# =============================================================================

async def main():
    """Demo unusual activity detection."""
    
    print("=" * 70)
    print(" 🐋 WHALE DETECTOR - Unusual Activity Scanner")
    print("=" * 70)
    
    detector = UnusualActivityDetector()
    
    # Scan a few symbols
    symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA']
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f" Scanning {symbol}...")
        print("=" * 50)
        
        result = await detector.scan_symbol(symbol)
        
        print(f"\nUnderlying: ${result.underlying_price:.2f}")
        print(f"Put/Call Ratio: {result.put_call_ratio:.2f}")
        print(f"Sentiment: {result.sentiment.value}")
        
        if result.signals:
            print(f"\n📊 SIGNALS ({len(result.signals)}):")
            for signal in result.signals[:5]:  # Top 5
                print(f"  {signal}")
        
        if result.support_levels:
            print(f"\n🟢 SUPPORT LEVELS:")
            for level in result.support_levels[:3]:
                print(f"  {level}")
        
        if result.resistance_levels:
            print(f"\n🔴 RESISTANCE LEVELS:")
            for level in result.resistance_levels[:3]:
                print(f"  {level}")
        
        # Get whale strikes
        whale_strikes = detector.get_whale_strikes(result)
        print(f"\n🐋 WHALE STRIKES:")
        print(f"  Support: {whale_strikes['support_strikes'][:3]}")
        print(f"  Resistance: {whale_strikes['resistance_strikes'][:3]}")
    
    print("\n" + "=" * 70)
    print(" INTEGRATION EXAMPLE")
    print("=" * 70)
    
    # Test whale-aware filter
    whale_filter = WhaleAwareStrategyFilter(detector)
    
    print("\nTesting trade filter for SPY put credit spread @ $580...")
    approved, reason, alt = await whale_filter.filter_trade('SPY', 580, 'PUT')
    print(f"  Approved: {approved}")
    print(f"  Reason: {reason}")
    if alt:
        print(f"  Suggested alternative: ${alt}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
