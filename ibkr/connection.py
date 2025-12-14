"""
Vanna IBKR Connection Module

Manages connection to Interactive Brokers Gateway.
Features:
- Automatic reconnection with exponential backoff
- Health check for Docker healthcheck
- Clean disconnect handling
- Circuit breaker protection
"""

import asyncio
from typing import Optional

from ib_insync import IB

from core.logger import get_logger
from core.circuit_breaker import get_ibkr_circuit_breaker
from core.exceptions import IBKRConnectionError, IBKRDisconnectedError

logger = get_logger()

# Singleton instance
_connection: Optional['IBKRConnection'] = None


class IBKRConnection:
    """
    IBKR Connection Manager.
    
    Usage:
        conn = await get_ibkr_connection()
        await conn.connect()
        
        # Use connection
        account = conn.account_info()
        
        # Cleanup
        await conn.disconnect()
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 99,
        account: str = ""
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account = account
        
        self._ib: Optional[IB] = None
        self._connected = False
        self._connection_attempts = 0
        self._max_retries = 10
        
        # Circuit breaker for connection resilience
        self._circuit_breaker = get_ibkr_circuit_breaker()
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        return self._ib is not None and self._ib.isConnected()
    
    @property
    def ib(self) -> IB:
        """Get the underlying IB instance."""
        if self._ib is None:
            raise IBKRDisconnectedError("Not connected to IBKR. Call connect() first.")
        return self._ib
    
    async def connect(self) -> bool:
        """
        Connect to IBKR Gateway with retry logic.
        
        Returns:
            True if connected successfully, False otherwise
        """
        if self.is_connected:
            logger.info("Already connected to IBKR")
            return True
        
        self._ib = IB()
        
        # Set up event handlers
        self._ib.disconnectedEvent += self._on_disconnect
        self._ib.errorEvent += self._on_error
        
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.info(f"🔌 Connecting to IBKR (Attempt {attempt}/{self._max_retries})...")
                logger.info(f"   Target: {self.host}:{self.port} (Client ID: {self.client_id})")
                
                await self._ib.connectAsync(
                    host=self.host,
                    port=self.port,
                    clientId=self.client_id,
                    timeout=30,
                    readonly=False
                )
                
                # Verify connection
                if self._ib.isConnected():
                    accounts = self._ib.managedAccounts()
                    logger.info("✅ Connected to IBKR successfully!")
                    logger.info(f"   Accounts: {accounts}")
                    
                    # Set account if not specified
                    if not self.account and accounts:
                        self.account = accounts[0]
                        logger.info(f"   Using account: {self.account}")
                    
                    self._connected = True
                    self._connection_attempts = 0
                    return True
                    
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Connection timeout (attempt {attempt})")
            except ConnectionRefusedError:
                logger.warning(f"🚫 Connection refused (attempt {attempt})")
            except Exception as e:
                logger.warning(f"❌ Connection error: {type(e).__name__}: {e}")
            
            # Exponential backoff
            if attempt < self._max_retries:
                wait_time = min(2 ** attempt, 60)  # Max 60 seconds
                logger.info(f"   Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        logger.error("❌ Failed to connect to IBKR after all retries")
        return False
    
    async def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._ib is not None:
            try:
                self._ib.disconnect()
                logger.info("🔌 Disconnected from IBKR")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self._ib = None
                self._connected = False
    
    async def reconnect(self) -> bool:
        """Reconnect to IBKR (disconnect first if needed)."""
        await self.disconnect()
        await asyncio.sleep(2)
        return await self.connect()
    
    def _on_disconnect(self) -> None:
        """Handle disconnection event."""
        logger.warning("⚠️ IBKR connection lost!")
        self._connected = False
    
    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract) -> None:
        """Handle error events."""
        # Ignore non-critical errors
        ignored_codes = [
            2104, 2106, 2158,  # Market data farm messages
            10276,  # News feed not allowed (account limitation, not error)
        ]
        if errorCode in ignored_codes:
            return
        
        if errorCode >= 2000:  # Warnings
            logger.warning(f"IBKR Warning [{errorCode}]: {errorString}")
        else:
            logger.error(f"IBKR Error [{errorCode}]: {errorString}")
    
    # =========================================================================
    # Account Methods
    # =========================================================================
    
    async def get_account_summary(self) -> dict:
        """
        Get account summary.
        
        Returns:
            Dict with NetLiquidation, AvailableFunds, BuyingPower, etc.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to IBKR")
        
        summary = {}
        account_values = await self.ib.accountSummaryAsync(self.account)
        
        for av in account_values:
            if av.tag in ['NetLiquidation', 'AvailableFunds', 'BuyingPower', 
                          'TotalCashValue', 'GrossPositionValue']:
                summary[av.tag] = float(av.value)
        
        return summary
    
    def get_positions(self) -> list:
        """Get current positions."""
        if not self.is_connected:
            raise RuntimeError("Not connected to IBKR")
        
        return self.ib.positions(self.account)
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    def health_check(self) -> dict:
        """
        Health check for Docker healthcheck.
        
        Returns:
            Dict with status and details
        """
        return {
            "connected": self.is_connected,
            "host": self.host,
            "port": self.port,
            "account": self.account,
            "client_id": self.client_id
        }


async def get_ibkr_connection(
        host: Optional[str] = None, 
        port: Optional[int] = None, 
        client_id: Optional[int] = None,
        account: Optional[str] = None
) -> IBKRConnection:
    """
    Get the global IBKR connection instance.
    
    Args:
        host: IBKR Gateway host (default from config)
        port: IBKR Gateway port (default from config)
        client_id: Client ID (default from config)
        account: Account ID (default from config)
    
    Returns:
        IBKRConnection instance
    """
    global _connection
    
    if _connection is None:
        from config import get_config
        config = get_config()
        
        _connection = IBKRConnection(
            host=host or config.ibkr.host,
            port=port or config.ibkr.port,
            client_id=client_id or config.ibkr.client_id,
            account=account or config.ibkr.account
        )
    
    return _connection
