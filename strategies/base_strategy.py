"""
Base Strategy Interface

Defines the contract for all trading strategies.
Strategies must implement market analysis and leg generation.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class StrategySignal:
    """Standardized output from strategy analysis."""
    signal_id: str
    symbol: str
    strategy_name: str
    direction: str  # BULLISH, BEARISH, NEUTRAL
    setup_quality: float  # 0.0 to 10.0
    reasoning: str
    timestamp: datetime = datetime.now()

class AbstractStrategy(ABC):
    """
    Abstract base class for trading strategies.
    """
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    async def analyze_market(
        self, 
        symbol: str, 
        market_data: Dict[str, Any], 
        options_chain: Optional[Any] = None
    ) -> StrategySignal:
        """
        Analyze market conditions to generate a signal.
        
        Args:
            symbol: Ticker symbol
            market_data: Price, VIX, trend info
            options_chain: Full option chain if needed for deep scan
            
        Returns:
            StrategySignal object
        """
        pass

    @abstractmethod
    async def find_execution_candidates(
        self,
        symbol: str,
        chain: Any,
        risk_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find specific option legs to execute the strategy.
        
        Args:
            symbol: Ticker symbol
            chain: DataFetcher option chain object
            risk_profile: Account risk limits (max_risk, etc)
            
        Returns:
            List of trade setups (dicts containing legs, credit/debit, metrics)
        """
        pass
        
    def calculate_risk_reward(self, entry_price: float, max_loss: float) -> float:
        """Helper to calculate R:R ratio."""
        if max_loss == 0:
            return 0.0
        return entry_price / max_loss
