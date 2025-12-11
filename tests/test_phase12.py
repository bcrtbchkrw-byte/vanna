from unittest.mock import AsyncMock, MagicMock

import pytest

from notifications.telegram_client import get_telegram_client


@pytest.fixture
def mock_telegram(monkeypatch):
    """Mock Telegram Bot and Config."""
    # Mock config.get_config
    mock_config = MagicMock()
    mock_config.telegram.token = "mock_token"
    mock_config.telegram.chat_id = "mock_chat_id"
    monkeypatch.setattr("config.get_config", MagicMock(return_value=mock_config))
    
    # Mock Bot class
    mock_bot = MagicMock()
    mock_bot.get_me = AsyncMock(return_value=MagicMock(username="mock_bot"))
    mock_bot.send_message = AsyncMock()
    
    monkeypatch.setattr("notifications.telegram_client.Bot", MagicMock(return_value=mock_bot))
    
    return mock_bot

@pytest.mark.asyncio
async def test_telegram_client_init(mock_telegram):
    """Test standard initialization."""
    # Reset singleton
    import notifications.telegram_client
    notifications.telegram_client._client_instance = None
    
    client = get_telegram_client()
    await client.initialize()
    
    assert client._initialized
    assert client.bot is not None
    
@pytest.mark.asyncio
async def test_telegram_send_message(mock_telegram):
    import notifications.telegram_client
    notifications.telegram_client._client_instance = None
    
    client = get_telegram_client()
    await client.initialize()
    
    # client.bot is the mock instance due to fixture
    await client.send_message("Test Message")
    
    client.bot.send_message.assert_called_with(chat_id="mock_chat_id", text="Test Message")

@pytest.mark.asyncio
async def test_telegram_missing_config(monkeypatch):
    """Test graceful failure when config is missing."""
    # Mock empty config
    mock_config = MagicMock()
    mock_config.telegram.token = ""
    mock_config.telegram.chat_id = ""
    monkeypatch.setattr("config.get_config", MagicMock(return_value=mock_config))
    
    import notifications.telegram_client
    notifications.telegram_client._client_instance = None
    
    client = get_telegram_client()
    await client.initialize()
    
    assert not client._initialized
    assert client.bot is None
