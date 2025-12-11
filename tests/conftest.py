"""
IBKR Mock Fixtures for CI/CD
Provides mock objects for tests that require IBKR connection.
Used when real IBKR gateway is not available (e.g., GitHub Actions).
"""
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def is_ci_environment() -> bool:
    """Check if running in CI environment (no IBKR available)."""
    return os.environ.get("CI", "false").lower() == "true" or \
           os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"


class MockIBKRConnection:
    """Mock IBKR connection for testing."""
    
    def __init__(self) -> None:
        self.ib = MagicMock()
        self.is_connected = True
        self._account = "DU1234567"
        
        # Mock IB methods
        self.ib.isConnected.return_value = True
        self.ib.accountSummary.return_value = [
            MagicMock(tag="NetLiquidation", value="100000"),
            MagicMock(tag="AvailableFunds", value="50000"),
            MagicMock(tag="BuyingPower", value="200000"),
        ]
        self.ib.positions.return_value = []
        self.ib.openOrders.return_value = []
        self.ib.reqContractDetails = AsyncMock(return_value=[])
        self.ib.qualifyContractsAsync = AsyncMock()
        self.ib.reqSecDefOptParamsAsync = AsyncMock(return_value=[])
        self.ib.reqHistoricalDataAsync = AsyncMock(return_value=[])
        self.ib.reqMktData = MagicMock()
        self.ib.placeOrder = MagicMock()
        
    async def connect(self) -> bool:
        return True
    
    async def disconnect(self) -> None:
        pass
    
    def get_account_value(self, tag: str = "NetLiquidation") -> float:
        if tag == "NetLiquidation":
            return 100000.0
        elif tag == "AvailableFunds":
            return 50000.0
        elif tag == "BuyingPower":
            return 200000.0
        return 0.0


class MockDataFetcher:
    """Mock data fetcher for testing."""
    
    def __init__(self) -> None:
        self._connection = MockIBKRConnection()
    
    async def _get_connection(self) -> MockIBKRConnection:
        return self._connection
    
    async def get_stock_quote(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "last": 150.0,
            "bid": 149.95,
            "ask": 150.05,
            "volume": 1000000,
        }
    
    async def get_vix(self) -> float:
        return 18.5
    
    async def get_option_chain(
        self, symbol: str, expiry: str | None = None
    ) -> list[dict[str, Any]]:
        return [
            {"strike": 145, "right": "P", "delta": -0.15, "bid": 1.00, "ask": 1.10},
            {"strike": 150, "right": "P", "delta": -0.30, "bid": 2.50, "ask": 2.60},
            {"strike": 155, "right": "C", "delta": 0.30, "bid": 2.50, "ask": 2.60},
            {"strike": 160, "right": "C", "delta": 0.15, "bid": 1.00, "ask": 1.10},
        ]


@pytest.fixture
def mock_ibkr_connection() -> MockIBKRConnection:
    """Pytest fixture for mock IBKR connection."""
    return MockIBKRConnection()


@pytest.fixture
def mock_data_fetcher() -> MockDataFetcher:
    """Pytest fixture for mock data fetcher."""
    return MockDataFetcher()


# Singleton instances for module-level mocking
_mock_connection: MockIBKRConnection | None = None
_mock_fetcher: MockDataFetcher | None = None


async def get_mock_ibkr_connection() -> MockIBKRConnection:
    """Get singleton mock IBKR connection."""
    global _mock_connection
    if _mock_connection is None:
        _mock_connection = MockIBKRConnection()
    return _mock_connection


def get_mock_data_fetcher() -> MockDataFetcher:
    """Get singleton mock data fetcher."""
    global _mock_fetcher
    if _mock_fetcher is None:
        _mock_fetcher = MockDataFetcher()
    return _mock_fetcher
