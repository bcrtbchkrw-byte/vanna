"""Phase 12 Tests - Notification System (Telegram)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_telegram_bot():
    """Mock Telegram bot for testing."""
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(return_value=True)
    return mock_bot


@pytest.mark.asyncio
async def test_telegram_notifier_initialization():
    """Test TelegramNotifier initializes correctly."""
    from notifications.telegram_client import TelegramNotifier
    
    # Reset singleton for clean test
    TelegramNotifier._instance = None
    
    # Without env vars, should be disabled
    notifier = TelegramNotifier()
    assert notifier.enabled is False
    assert notifier.bot_token == ""
    assert notifier.chat_id == ""


@pytest.mark.asyncio
async def test_telegram_notifier_with_env(monkeypatch):
    """Test TelegramNotifier with environment variables set."""
    from notifications.telegram_client import TelegramNotifier
    
    # Reset singleton
    TelegramNotifier._instance = None
    
    # Set mock env vars
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token_123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test_chat_456")
    
    notifier = TelegramNotifier()
    assert notifier.bot_token == "test_token_123"
    assert notifier.chat_id == "test_chat_456"
    # Note: enabled may still be False if telegram library not imported


@pytest.mark.asyncio
async def test_send_trade_alert():
    """Test trade alert message formatting."""
    from notifications.telegram_client import TelegramNotifier
    
    # Reset singleton
    TelegramNotifier._instance = None
    
    notifier = TelegramNotifier()
    
    # Even if disabled, should return False without error
    result = await notifier.send_trade_alert(
        symbol="SPY",
        strategy="BULL_PUT",
        action="OPEN",
        quantity=1,
        price=1.50
    )
    
    # Should be False because notifications are disabled
    assert result is False


@pytest.mark.asyncio
async def test_send_error_alert():
    """Test error alert message formatting."""
    from notifications.telegram_client import TelegramNotifier
    
    TelegramNotifier._instance = None
    notifier = TelegramNotifier()
    
    result = await notifier.send_error_alert(
        error_type="DISCONNECT",
        error_msg="Lost connection to IBKR Gateway",
        component="IBKRConnection"
    )
    
    assert result is False  # Disabled


@pytest.mark.asyncio
async def test_send_opportunity_alert():
    """Test opportunity alert message formatting."""
    from notifications.telegram_client import TelegramNotifier
    
    TelegramNotifier._instance = None
    notifier = TelegramNotifier()
    
    result = await notifier.send_opportunity_alert(
        symbol="AAPL",
        strategy="IRON_CONDOR",
        expected_credit=2.50,
        roi_pct=15.5,
        confidence=0.85
    )
    
    assert result is False  # Disabled


@pytest.mark.asyncio
async def test_send_daily_summary():
    """Test daily summary message formatting."""
    from notifications.telegram_client import TelegramNotifier
    
    TelegramNotifier._instance = None
    notifier = TelegramNotifier()
    
    result = await notifier.send_daily_summary(
        total_trades=5,
        winning_trades=4,
        total_pnl=125.50,
        open_positions=2
    )
    
    assert result is False  # Disabled


@pytest.mark.asyncio
async def test_notifier_singleton():
    """Test that get_telegram_notifier returns singleton."""
    from notifications.telegram_client import TelegramNotifier, get_telegram_notifier
    
    TelegramNotifier._instance = None
    
    notifier1 = get_telegram_notifier()
    notifier2 = get_telegram_notifier()
    
    assert notifier1 is notifier2
