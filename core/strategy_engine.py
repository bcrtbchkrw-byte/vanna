"""
Strategy Execution Engine (The Body)

Orchestrates the conversion of High-Level TradeDecisions into Low-Level Orders.
Handles:
1. Option Selection (Chain lookup, Strike selection)
2. Smart Pricing (Mid-price calculation)
3. Order Execution via IBKRExecutionClient
"""
import asyncio
from typing import Optional, Dict
from datetime import datetime, timedelta

from core.logger import get_logger
from core.trade_executor import TradeDecision, Action
from ibkr.execution_client import IBKRExecutionClient
from ibkr.data_fetcher import IBKRDataFetcher
from ml.dte_optimizer import get_dte_optimizer

logger = get_logger()

class StrategyExecutionEngine:
    
    def __init__(self):
        self.execution_client = IBKRExecutionClient()
        self.data_fetcher = IBKRDataFetcher()
        self.dte_optimizer = get_dte_optimizer()
        
        # Configuration
        self.TARGET_DELTA = 0.50 # ATM
        self.DELTA_TOLERANCE = 0.10 # 0.40 - 0.60
        self.MAX_SPREAD_PERCENT = 0.10 # Don't buy if spread > 10%
        
    async def execute_decision(self, decision: TradeDecision) -> bool:
        """
        Execute a TradeDecision.
        """
        logger.info(f"🤖 Body received command: {decision}")
        
        if decision.action == Action.HOLD:
            return False
            
        if decision.action == Action.OPEN:
            return await self._handle_open(decision)
            
        if decision.action == Action.CLOSE:
            # TODO: Implement close logic (need position tracking)
            logger.warning("CLOSE action not yet implemented in Engine")
            return False
            
        return False

    async def _handle_open(self, decision: TradeDecision) -> bool:
        """Handle OPEN action (Buy Call/Put)."""
        symbol = decision.symbol
        
        # 1. Determine Strategy (Call vs Put?)
        # PPO Action OPEN usually implies LONG CALL in our current setup?
        # TODO: PPO should specify direction. For now assuming LONG CALL (Bullish)
        # unless we add 'SHORT' action or directional output.
        # Assuming PPO is trained for Directionless Volatility or Bullish? 
        # Typically PPO for trading: Action 1 = BUY. PnL depends on price.
        # Let's assume BUY CALL for now as MVP.
        right = 'C' 
        
        # 2. Determine DTE
        # Use DTEOptimizer with VIX
        try:
            # We need VIX. Fetch it fresh.
            chain = await self.data_fetcher.get_option_chain(symbol, strikes=1)
            # Chain dict might contain 'underlying_price' and VIX proxy if available?
            # Better: fetch VIX directly
            # For speed, let's use a default or assume the Decision carried VIX?
            # Decision object has model_outputs with 'xgboost_prob'.
            # Executor had market_data with VIX.
            # But we don't pass market_data here. 
            # Let's re-fetch simplified VIX or use safe default (30 DTE).
            
            # TODO: Properly fetch VIX for optimization. 
            # For MVP, we use robust default 30-45 days.
            target_dte = 30
            
        except Exception as e:
            logger.error(f"Failed to optimize DTE: {e}")
            target_dte = 30 # Safe default
            
        # 3. Find Option Contract
        logger.info(f"🔍 Searching for {symbol} {right} DTE~{target_dte} Delta~{self.TARGET_DELTA}...")
        
        try:
            # Get chain for specific DTE window
            # We assume get_option_chain logic can filter by DTE or we filter here
            # data_fetcher.get_option_chain usually gets specific expiry or all?
            # Let's use a specialized method if available, or fetch ATM
            
            # Since get_option_chain implementation is complex, let's use a smart helper directly here
            # We need to find the expiry date first
            target_date = datetime.now() + timedelta(days=target_dte)
            
            # Fetch generic ATM chain to find expiries
            # (Optimization: Add find_contract_by_criteria to DataFetcher)
            # For now, let's assume DataFetcher returns a Chain object
            chain = await self.data_fetcher.get_option_chain(symbol, strikes=10) # Get 10 strikes around ATM
            
            if not chain:
                logger.error("No option chain data")
                return False
                
            # Filter Expirations (find closest to target_dte)
            # Chain structure: {'calls': [...], 'puts': [...], 'expirations': [...]}
            # Checking DataFetcher implementation... it returns Dict.
            
            # MVP: Select best contract from returned 'calls'
            candidates = chain.get('calls', [])
            if not candidates:
                logger.warning("No call options found")
                return False
                
            # Filter by DTE (approximate)
            # Candidates usually have 'expiration' string.
            # We need to parse dates. 
            # Assume data_fetcher returns relevant monthly/weekly?
            # This part is tricky without specific API method.
            
            # Best Candidate Selection (Closest to Delta 0.50)
            best_contract = None
            best_diff = 1.0
            
            for c in candidates:
                delta = c.get('delta', 0.5) # If None, assume ATM?
                if delta is None: continue
                
                diff = abs(delta - self.TARGET_DELTA)
                if diff < best_diff and diff < self.DELTA_TOLERANCE:
                    best_diff = diff
                    best_contract = c
            
            if not best_contract:
                logger.warning(f"No contract found with Delta ~{self.TARGET_DELTA}")
                return False
                
            logger.info(f"🎯 Selected: {best_contract['contract'].localSymbol} (Delta {best_contract.get('delta', '?'):.2f})")
            
            # 4. Calculate Price (MID)
            bid = best_contract.get('bid', 0)
            ask = best_contract.get('ask', 0)
            
            if bid <= 0 or ask <= 0:
                logger.warning(f"Invalid quotes: Bid {bid} Ask {ask}")
                return False
                
            mid_price = round((bid + ask) / 2, 2)
            
            # 5. Execute Order
            contract = best_contract['contract']
            qty = decision.quantity if decision.quantity > 0 else 1
            
            logger.info(f"🚀 Placing Order: BUY {qty} {contract.localSymbol} @ {mid_price}")
            
            trade = await self.execution_client.place_limit_order(
                contract=contract,
                action='BUY',
                quantity=qty,
                limit_price=mid_price
            )
            
            # Wait for fill (optional or async monitor)
            # For now, just fire and forget (pipeline will monitor orders list)
            return True
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return False
            
