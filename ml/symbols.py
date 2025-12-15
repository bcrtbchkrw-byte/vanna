"""
Symbol Configuration - Single Source of Truth

All modules should import symbols from here to avoid circular imports.
"""

# Core ETFs
CORE_ETFS = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']

# Individual Stocks (User requested + High Volatility)
STOCKS = ['AAPL', 'AMD', 'TSLA', 'NVDA', 'MSFT', 'AMZN', 'COIN', 'JPM', 'SMCI']

# Complete list for ML/RL training
TRAINING_SYMBOLS = CORE_ETFS + STOCKS

# VIX indices
VIX_SYMBOLS = ['VIX', 'VIX3M']

# Alias for backwards compatibility
SYMBOLS = TRAINING_SYMBOLS
