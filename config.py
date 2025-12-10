"""
Vanna Configuration Module

Centralized configuration management using environment variables.
All config is loaded lazily to avoid import-time failures.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


@dataclass
class IBKRConfig:
    """IBKR connection settings."""
    host: str
    port: int
    client_id: int
    account: str
    username: str
    password: str
    trading_mode: str


@dataclass
class AIConfig:
    """AI API settings."""
    gemini_api_key: str
    anthropic_api_key: str
    daily_cost_limit: float = 5.0


@dataclass
class TradingConfig:
    """Trading parameters."""
    max_risk_per_trade: float
    max_allocation_percent: float
    paper_trading: bool
    earnings_blackout_hours: int
    vix_panic_threshold: float


@dataclass
class LogConfig:
    """Logging settings."""
    level: str
    rotation: str


class Config:
    """
    Main configuration class.
    
    Usage:
        config = Config()
        print(config.ibkr.host)
        print(config.trading.max_risk_per_trade)
    """
    
    _instance: Optional['Config'] = None
    
    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._load_config()
    
    def _load_config(self):
        """Load all configuration from environment."""
        self.ibkr = IBKRConfig(
            host=os.getenv('IBKR_HOST', 'ib-gateway'),
            port=int(os.getenv('IBKR_PORT', '4002')),
            client_id=int(os.getenv('IBKR_CLIENT_ID', '1')),
            account=os.getenv('IBKR_ACCOUNT', ''),
            username=os.getenv('IBKR_USERNAME', ''),
            password=os.getenv('IBKR_PASSWORD', ''),
            trading_mode=os.getenv('TRADING_MODE', 'paper'),
        )
        
        self.ai = AIConfig(
            gemini_api_key=os.getenv('GEMINI_API_KEY', ''),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY', ''),
            daily_cost_limit=float(os.getenv('AI_DAILY_COST_LIMIT', '5.0')),
        )
        
        self.trading = TradingConfig(
            max_risk_per_trade=float(os.getenv('MAX_RISK_PER_TRADE', '120')),
            max_allocation_percent=float(os.getenv('MAX_ALLOCATION_PERCENT', '25')),
            paper_trading=os.getenv('PAPER_TRADING', 'true').lower() == 'true',
            earnings_blackout_hours=int(os.getenv('EARNINGS_BLACKOUT_HOURS', '48')),
            vix_panic_threshold=float(os.getenv('VIX_PANIC_THRESHOLD', '30')),
        )
        
        self.log = LogConfig(
            level=os.getenv('LOG_LEVEL', 'INFO'),
            rotation=os.getenv('LOG_ROTATION', '100 MB'),
        )
    
    def validate(self) -> list[str]:
        """
        Validate required configuration.
        
        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []
        
        if not self.ibkr.account:
            errors.append("IBKR_ACCOUNT is required")
        
        if not self.ai.gemini_api_key:
            errors.append("GEMINI_API_KEY is required")
        
        if not self.ai.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required")
        
        return errors


def get_config() -> Config:
    """Get the global config instance."""
    return Config()
