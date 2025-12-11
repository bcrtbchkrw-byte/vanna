"""
Iron Condor Strategy

Combines OTM Bull Put Spread and OTM Bear Call Spread.
Target: Neutral market, harvesting Theta from both sides.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from strategies.base_strategy import AbstractStrategy, StrategySignal
from strategies.credit_spreads import VerticalCreditSpread


class IronCondor(AbstractStrategy):
    
    def __init__(self):
        super().__init__("IronCondor")
        # Reuse logic from Vertical Spreads
        self.spread_generator = VerticalCreditSpread()
        
    async def analyze_market(
        self, 
        symbol: str, 
        market_data: Dict[str, Any], 
        options_chain: Optional[Any] = None
    ) -> StrategySignal:
        """
        Determine if market is suitable for Iron Condors.
        """
        vix = market_data.get('vix', 0)
        trend = market_data.get('trend', 'NEUTRAL')
        
        setup_quality = 5.0
        reasoning = []
        
        # Iron Condors best in high IV, neutral trend
        if vix > 20:
            setup_quality += 2.0
            reasoning.append("High VIX (>20)")
            
        if trend == 'NEUTRAL':
            setup_quality += 3.0
            reasoning.append("Neutral Trend")
        else:
            setup_quality -= 2.0
            reasoning.append(f"Trend is {trend} (Prefer Neutral)")
            
        return StrategySignal(
            signal_id=f"IC-{symbol}-{datetime.now().strftime('%H%M')}",
            symbol=symbol,
            strategy_name=self.name,
            direction="NEUTRAL",
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
        Find Iron Condor candidates by combining Put and Call spreads.
        """
        # 1. Get Bull Put candidates
        # For IC, we might want slightly lower delta (e.g. 0.16) than directional spread (0.20-0.25)
        # But for simplicity, reuse the same logic
        put_spreads = await self.spread_generator.find_execution_candidates(symbol, chain, risk_profile)
        bull_puts = [s for s in put_spreads if s['strategy'] == 'BULL_PUT']
        
        # 2. Get Bear Call candidates
        call_spreads = await self.spread_generator.find_execution_candidates(symbol, chain, risk_profile)
        bear_calls = [s for s in call_spreads if s['strategy'] == 'BEAR_CALL']
        
        candidates = []
        
        # 3. Combine them
        for bp in bull_puts:
            for bc in bear_calls:
                # Ensure strikes ensure distance
                # Put Short Strike < Call Short Strike
                if bp['short_leg']['strike'] >= bc['short_leg']['strike']:
                    continue # Inverted or overlapping
                    
                total_credit = bp['net_credit'] + bc['net_credit']
                
                # Risk in IC is usually the wider of the two wings minus total credit
                # (Assuming price can't blow through both sides at once)
                max_wing_width = max(bp['width'], bc['width'])
                max_loss = (max_wing_width * 100) - (total_credit * 100)
                
                if max_loss <= 0:
                    continue # Arbitrage? Unlikely data error
                
                roi = (total_credit * 100) / max_loss
                
                candidates.append({
                    'strategy': 'IRON_CONDOR',
                    'bull_put': bp,
                    'bear_call': bc,
                    'net_credit': total_credit,
                    'max_loss': max_loss,
                    'roi_pct': roi * 100,
                    'short_put_strike': bp['short_leg']['strike'],
                    'short_call_strike': bc['short_leg']['strike']
                })
                
        # Sort by ROI
        candidates.sort(key=lambda x: x['roi_pct'], reverse=True)
        return candidates[:2]
