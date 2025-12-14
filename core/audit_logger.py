"""
Audit Logger for Trade Compliance

Provides structured JSON logging for all trade operations.
Essential for audit trail and compliance.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from core.logger import get_logger

logger = get_logger()


class AuditLogger:
    """
    Structured audit logging for trades and orders.
    
    Logs to:
    - data/audit/trades_YYYY-MM.jsonl (JSON Lines format)
    - Each line is a complete JSON object
    
    Usage:
        audit = get_audit_logger()
        audit.log_order_attempt(symbol="SPY", action="BUY", ...)
        audit.log_order_result(correlation_id, success=True, ...)
    """
    
    def __init__(self, audit_dir: str = "data/audit"):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_audit_file(self) -> Path:
        """Get current month's audit file."""
        now = datetime.now(timezone.utc)
        filename = f"trades_{now.strftime('%Y-%m')}.jsonl"
        return self.audit_dir / filename
    
    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Write a single audit entry to file."""
        audit_file = self._get_audit_file()
        try:
            with open(audit_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}")
    
    def log_order_attempt(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: Optional[float],
        strategy: str,
        order_type: str = "SPREAD",
        extra: Optional[dict] = None
    ) -> str:
        """
        Log an order attempt BEFORE placement.
        
        Returns:
            correlation_id for matching with result
        """
        correlation_id = str(uuid4())[:8]
        
        entry = {
            "event": "ORDER_ATTEMPT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "strategy": strategy,
            "order_type": order_type,
            "paper_mode": os.getenv("PAPER_TRADING", "true").lower() == "true"
        }
        
        if extra:
            entry["extra"] = extra
        
        self._write_entry(entry)
        logger.info(f"📋 Audit: ORDER_ATTEMPT {correlation_id} - {action} {quantity} {symbol}")
        
        return correlation_id
    
    def log_order_result(
        self,
        correlation_id: str,
        success: bool,
        order_id: Optional[str] = None,
        fill_price: Optional[float] = None,
        error: Optional[str] = None,
        extra: Optional[dict] = None
    ) -> None:
        """Log the result of an order AFTER placement attempt."""
        entry = {
            "event": "ORDER_RESULT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "success": success,
            "order_id": order_id,
            "fill_price": fill_price,
            "error": error
        }
        
        if extra:
            entry["extra"] = extra
        
        self._write_entry(entry)
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"📋 Audit: ORDER_RESULT {correlation_id} - {status}")
    
    def log_position_change(
        self,
        symbol: str,
        change_type: str,  # OPEN, CLOSE, ADJUST
        quantity: int,
        pnl: Optional[float] = None,
        extra: Optional[dict] = None
    ) -> None:
        """Log position changes."""
        entry = {
            "event": "POSITION_CHANGE",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "change_type": change_type,
            "quantity": quantity,
            "pnl": pnl
        }
        
        if extra:
            entry["extra"] = extra
        
        self._write_entry(entry)
    
    def log_validation_error(
        self,
        symbol: str,
        error_type: str,
        details: str
    ) -> None:
        """Log validation errors (rejected orders)."""
        entry = {
            "event": "VALIDATION_ERROR",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "error_type": error_type,
            "details": details
        }
        
        self._write_entry(entry)
        logger.warning(f"📋 Audit: VALIDATION_ERROR - {error_type}: {details}")


# ============================================================
# Singleton
# ============================================================

_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
