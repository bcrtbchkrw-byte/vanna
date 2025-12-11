"""
Telegram Notification Client for Vanna Trading Bot.

Provides real-time alerts via Telegram for:
- Trade execution notifications
- Error/disconnect alerts
- New opportunity alerts
"""
import os
from typing import Any

from core.logger import get_logger


class TelegramNotifier:
    """
    Telegram notification client for trading alerts.
    
    Sends formatted messages to a configured Telegram chat.
    Uses python-telegram-bot library for async messaging.
    """
    
    _instance: "TelegramNotifier | None" = None
    _initialized: bool = False

    
    def __new__(cls) -> "TelegramNotifier":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        
        self.logger = get_logger()
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)
        self._bot: Any = None
        
        if not self.enabled:
            self.logger.warning(
                "Telegram notifications disabled - missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"
            )
    
    async def _get_bot(self) -> Any:
        """Lazy-load the Telegram bot."""
        if self._bot is None and self.enabled:
            try:
                from telegram import Bot
                self._bot = Bot(token=self.bot_token)
            except ImportError:
                self.logger.error(
                    "python-telegram-bot not installed. "
                    "Run: pip install python-telegram-bot"
                )
                self.enabled = False
        return self._bot
    
    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to the configured Telegram chat.
        
        Args:
            message: The message text to send
            parse_mode: Message formatting mode (HTML or Markdown)
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.enabled:
            self.logger.debug(f"[TELEGRAM DISABLED] {message}")
            return False
        
        try:
            bot = await self._get_bot()
            if bot:
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=parse_mode
                )
                self.logger.info(f"Telegram sent: {message[:50]}...")
                return True
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
        return False
    
    async def send_trade_alert(
        self,
        symbol: str,
        strategy: str,
        action: str,
        quantity: int,
        price: float,
        pnl: float | None = None
    ) -> bool:
        """
        Send a trade execution alert.
        
        Args:
            symbol: Trading symbol (e.g., "SPY")
            strategy: Strategy name (e.g., "BULL_PUT")
            action: Trade action ("OPEN" or "CLOSE")
            quantity: Number of contracts
            price: Entry/exit price
            pnl: Profit/Loss (for CLOSE actions)
        """
        emoji = "🟢" if action == "OPEN" else ("🟢" if pnl and pnl > 0 else "🔴")
        
        message = (
            f"{emoji} <b>Trade {action}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 Symbol: <code>{symbol}</code>\n"
            f"📈 Strategy: <code>{strategy}</code>\n"
            f"📦 Quantity: <code>{quantity}</code>\n"
            f"💵 Price: <code>${price:.2f}</code>"
        )
        
        if pnl is not None:
            pnl_emoji = "✅" if pnl > 0 else "❌"
            message += f"\n{pnl_emoji} P&L: <code>${pnl:+.2f}</code>"
        
        return await self.send_message(message)
    
    async def send_error_alert(
        self,
        error_type: str,
        error_msg: str,
        component: str = "System"
    ) -> bool:
        """
        Send an error/disconnect alert.
        
        Args:
            error_type: Type of error (e.g., "DISCONNECT", "CRASH")
            error_msg: Error message details
            component: Component that triggered the error
        """
        message = (
            f"🚨 <b>ALERT: {error_type}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🔧 Component: <code>{component}</code>\n"
            f"⚠️ Details: {error_msg}"
        )
        
        return await self.send_message(message)
    
    async def send_opportunity_alert(
        self,
        symbol: str,
        strategy: str,
        expected_credit: float,
        roi_pct: float,
        confidence: float
    ) -> bool:
        """
        Send a new trading opportunity alert.
        
        Args:
            symbol: Trading symbol
            strategy: Recommended strategy
            expected_credit: Expected credit/premium
            roi_pct: Expected return on investment percentage
            confidence: AI confidence score (0-1)
        """
        confidence_bar = "🟢" * int(confidence * 5) + "⚪" * (5 - int(confidence * 5))
        
        message = (
            f"💡 <b>New Opportunity</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 Symbol: <code>{symbol}</code>\n"
            f"📈 Strategy: <code>{strategy}</code>\n"
            f"💵 Expected Credit: <code>${expected_credit:.2f}</code>\n"
            f"📊 ROI: <code>{roi_pct:.1f}%</code>\n"
            f"🎯 Confidence: {confidence_bar} ({confidence:.0%})"
        )
        
        return await self.send_message(message)
    
    async def send_daily_summary(
        self,
        total_trades: int,
        winning_trades: int,
        total_pnl: float,
        open_positions: int
    ) -> bool:
        """
        Send end-of-day trading summary.
        
        Args:
            total_trades: Number of trades today
            winning_trades: Number of winning trades
            total_pnl: Total P&L for the day
            open_positions: Number of open positions
        """
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        
        message = (
            f"📋 <b>Daily Summary</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 Trades: <code>{total_trades}</code>\n"
            f"✅ Win Rate: <code>{win_rate:.1f}%</code>\n"
            f"{pnl_emoji} P&L: <code>${total_pnl:+.2f}</code>\n"
            f"📦 Open Positions: <code>{open_positions}</code>"
        )
        
        return await self.send_message(message)


# Singleton accessor
_notifier: TelegramNotifier | None = None


def get_telegram_notifier() -> TelegramNotifier:
    """Get the global Telegram notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
