# Vanna - AI Options Trading Bot

**Vanna** is an AI-powered options trading bot designed for automated trading with Interactive Brokers.

## 🎯 Features

- **IBKR Integration** - Real-time market data, Greeks, order execution
- **AI Analysis** - Gemini (fundamental) + Claude (Greeks/strategy)
- **VIX Regime** - Automatic strategy selection based on volatility
- **Risk Management** - Position sizing, Greeks validation, portfolio limits
- **Docker Deployment** - Consistent Python 3.11 environment

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- IBKR Account with API access
- API Keys: Google Gemini, Anthropic Claude

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/vanna.git
cd vanna

# Configure
cp .env.example .env
nano .env  # Fill in your credentials

# Start
docker compose up -d

# Check logs
docker compose logs -f trader
```

## 📁 Project Structure

```
vanna/
├── main.py              # Entry point
├── config.py            # Configuration
├── core/                # Logger, Database
├── ibkr/                # IBKR connection & data
├── analysis/            # VIX, Earnings, Screener
├── ai/                  # Gemini, Claude clients
├── risk/                # Position sizing, Greeks
├── strategies/          # Credit spreads, Iron condors
├── execution/           # Order management
├── automation/          # Scheduler, Watchdog
└── tests/               # Phase tests
```

## 🔧 Configuration

Key settings in `.env`:

```bash
# IBKR
IBKR_HOST=ib-gateway
IBKR_PORT=4002
IBKR_ACCOUNT=DU123456
IBKR_USERNAME=your_username
IBKR_PASSWORD=your_password
TRADING_MODE=paper

# AI
GEMINI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# Trading
MAX_RISK_PER_TRADE=120
MAX_ALLOCATION_PERCENT=25
```

## 🛡️ Safety Features

- VIX Panic Mode (VIX > 30 = no new trades)
- Earnings Blackout (48h window)
- Position Limits (max 25% account)
- Greeks Validation (Delta, Vega, Theta limits)

## ⚠️ Disclaimer

This software is provided "AS IS". Options trading is risky. Never risk money you can't afford to lose.

## 📜 License

MIT License
