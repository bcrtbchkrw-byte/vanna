from typing import Optional

from telegram import Bot
from telegram.error import TelegramError

import config
from core.logger import get_logger

logger = get_logger()

class TelegramClient:
    """
    Async Telegram Client for sending notifications.
    """
    def __init__(self):
        cfg = config.get_config()
        self.token = cfg.telegram.token
        self.chat_id = cfg.telegram.chat_id
        self.bot: Optional[Bot] = None
        self._initialized = False

    async def initialize(self):
        """Initialize the bot if token is present."""
        if not self.token or not self.chat_id:
            logger.warning("Telegram settings missing. Notifications disabled.")
            return

        try:
            self.bot = Bot(token=self.token)
            me = await self.bot.get_me()
            logger.info(f"Telegram Bot initialized: @{me.username}")
            self._initialized = True
        except TelegramError as e:
            logger.error(f"Failed to initialize Telegram Bot: {e}")
            self._initialized = False

    async def send_message(self, message: str):
        """Send a text message to the configured chat."""
        if not self._initialized or not self.bot:
            logger.debug("Telegram not initialized, skipping message.")
            return

        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
            logger.info("Notification sent via Telegram")
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")

_client_instance = None

def get_telegram_client() -> TelegramClient:
    """Get singleton instance of TelegramClient."""
    global _client_instance
    if _client_instance is None:
        _client_instance = TelegramClient()
    return _client_instance
