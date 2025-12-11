"""
Vertical Credit Spread Strategy

Implements Bull Put Spreads and Bear Call Spreads.
Target: High probability income (Short Delta ~0.20).
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from strategies.base_strategy import AbstractStrategy, StrategySignal


class VerticalCreditSpread(AbstractStrategy):
    
    def __init__(self):
        super().__init__("VerticalCreditSpread")
        self.target_delta = 0.20
        self.wing_width_pct = 0.01  # Spread width as % of spot price (approx)
        
    async def analyze_market(
        self, 
        symbol: str, 
        market_data: Dict[str, Any], 
        options_chain: Optional[Any] = None
    ) -> StrategySignal:
        """
        Determine if market is suitable for Credit Spreads.
        """
        # Simple Logic:
        # VIX 20-30 = Good
        # IV Rank > 50 = Good
        # Trend check
        
        vix = market_data.get('vix', 0)
        # vix = market_data.get('vix', 0)
        # iv_rank = market_data.get('iv_rank', 0)
        # price = market_data.get('price', 0)
        
        setup_quality = 5.0
        reasoning = []
        direction = "NEUTRAL"
        
        if vix > 20:
            setup_quality += 2.0
            reasoning.append("High VIX (>20)")
        elif vix < 12:
            setup_quality -= 2.0
            reasoning.append("Low VIX (<12)")
            
        # Determine direction based on simple moving average or trend provided
        trend = market_data.get('trend', 'NEUTRAL')
        if trend == 'BULLISH':
            direction = 'BULLISH' # Sell Puts
        elif trend == 'BEARISH':
            direction = 'BEARISH' # Sell Calls
        else:
            direction = 'NEUTRAL' # Iron Condor territory, but here we pick side based on premium?

        return StrategySignal(
            signal_id=f"VCS-{symbol}-{datetime.now().strftime('%H%M')}",
            symbol=symbol,
            strategy_name=self.name,
            direction=direction,
            setup_quality=min(10.0, max(0.0, setup_quality)),
            reasoning="; ".join(reasoning)
        )

    async def find_execution_candidates(
        self,
        symbol: str,
        chain: Any,
        risk_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find credit spread legs.
        """
        candidates = []
        
        # We need expirations. Let's assume chain provides access to flat list of contracts
        # In ib_insync, we usually filter by expiry first
        # Here we assume 'chain' is a list of ContractData objects with greeks
        
        if not chain:
            return []
            
        # Filter for 30-45 DTE ideally (simplified here to "all provided in chain")
        # Separate Calls and Puts
        calls = [c for c in chain if c['right'] == 'C']
        puts = [c for c in chain if c['right'] == 'P']
        
        # Sort by strike
        calls.sort(key=lambda x: x['strike'])
        puts.sort(key=lambda x: x['strike'])
        
        # BULL PUT SPREAD (Sell Put, Buy Lower Put)
        # Look for Short Put around Delta 0.20
        candidates.extend(self._find_spreads_in_chain(puts, 'P', risk_profile, is_bullish=True))
        
        # BEAR CALL SPREAD (Sell Call, Buy Higher Call)
        # Look for Short Call around Delta 0.20
        candidates.extend(self._find_spreads_in_chain(calls, 'C', risk_profile, is_bullish=False))
        
        return candidates

    def _find_spreads_in_chain(
        self, 
        legs: List[Dict], 
        right: str, 
        risk_profile: Dict,
        is_bullish: bool
    ) -> List[Dict]:
        found = []
        max_risk = risk_profile.get('max_risk_per_trade', 1000)
        
        # Find Short Leg (Delta ~ target)
        # Put Delta is negative (-0.20). Call Delta is positive (0.20).
        # target_delta_val = -self.target_delta if right == 'P' else self.target_delta
        
        # Find closest leg to target delta
        # This is O(N^2) naive implementation for finding pairs
        
        for i, short_leg in enumerate(legs):
            # Check if delta is close enough (e.g. 0.15 to 0.25)
            d = short_leg.get('delta', 0)
            if not (0.15 <= abs(d) <= 0.25):
                continue
                
            # Now find Long Leg (Protection)
            # Bull Put: Buy Lower Strike (Index < i)
            # Bear Call: Buy Higher Strike (Index > i)
            
            start_j = 0 if is_bullish else i + 1
            end_j = i if is_bullish else len(legs)
            
            potential_long_legs = legs[start_j:end_j]
            
            for long_leg in potential_long_legs:
                width = abs(short_leg['strike'] - long_leg['strike'])
                
                if width == 0:
                    continue
                
                # Credit calculation
                # Sell Short (Bid), Buy Long (Ask)
                credit = short_leg['bid'] - long_leg['ask']
                
                if credit <= 0:
                    continue # No credit, skip
                    
                max_loss = (width * 100) - (credit * 100)
                
                if max_loss > max_risk:
                    continue # Exceeds risk per contract
                    
                # Min ROI check (e.g. 10%)
                roi = (credit * 100) / max_loss
                if roi < 0.10: 
                    continue
                    
                found.append({
                    'strategy': 'BULL_PUT' if is_bullish else 'BEAR_CALL',
                    'short_leg': short_leg,
                    'long_leg': long_leg,
                    'net_credit': credit,
                    'max_loss': max_loss,
                    'roi_pct': roi * 100,
                    'width': width,
                    'short_delta': d
                })
                
        # Return top 2 by credit
        found.sort(key=lambda x: x['net_credit'], reverse=True)
        return found[:2]

