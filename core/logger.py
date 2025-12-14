"""
Vanna Logger Module

Configures loguru for structured logging with JSON support.
Features:
- Console output (colored)
- File logging (text + JSON)
- Correlation ID support
- Trade-specific logs
- Auto-initialization for subprocesses
"""

import sys
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from contextvars import ContextVar

from loguru import logger

# Track if we've set up in this process
_logger_initialized = False
_process_id = None

# Context variable for correlation ID (async-safe)
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


def set_correlation_id(corr_id: str) -> None:
    """Set correlation ID for current async context."""
    correlation_id_var.set(corr_id)


def get_correlation_id() -> Optional[str]:
    """Get correlation ID for current async context."""
    return correlation_id_var.get()


def _json_sink(message):
    """Custom sink that outputs JSON format."""
    record = message.record
    
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
        "correlation_id": get_correlation_id(),
    }
    
    # Add extra fields if present
    if record["extra"]:
        log_entry["extra"] = record["extra"]
    
    print(json.dumps(log_entry), file=sys.stdout, flush=True)


def _ensure_minimal_logger():
    """Ensure at least a minimal logger is configured (for subprocesses)."""
    global _logger_initialized, _process_id
    
    current_pid = os.getpid()
    
    # If we're in a new process (multiprocessing), reset state
    if _process_id is not None and _process_id != current_pid:
        _logger_initialized = False
    
    _process_id = current_pid
    
    if not _logger_initialized:
        # Remove any default handlers
        try:
            logger.remove()
        except ValueError:
            pass  # Already removed
        
        # Add minimal console handler
        logger.add(
            sys.stderr,
            level="INFO",
            format="{time:HH:mm:ss} | {level: <8} | {message}",
            enqueue=True,  # Thread-safe for multiprocessing
        )
        _logger_initialized = True


def setup_logger(
    level: str = "INFO",
    log_dir: str = "logs",
    rotation: str = "100 MB",
    json_output: bool = False
) -> None:
    """
    Initialize the logger with console and file handlers.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        rotation: Log rotation size
        json_output: If True, use JSON format for console output
    
    Example:
        from core.logger import setup_logger, get_logger
        
        setup_logger(level="DEBUG", json_output=True)
        logger = get_logger()
        logger.info("Application started")
    """
    global _logger_initialized, _process_id
    
    current_pid = os.getpid()
    
    # Skip if already initialized in this process with full config
    if _logger_initialized and _process_id == current_pid:
        return
    
    # Remove any existing handlers
    try:
        logger.remove()
    except ValueError:
        pass
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Console handler
    if json_output:
        logger.add(
            _json_sink,
            level=level,
            enqueue=True,
        )
    else:
        logger.add(
            sys.stdout,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
                   "<level>{message}</level>",
            colorize=True,
            enqueue=True,
        )
    
    # Main log file (text format for readability)
    logger.add(
        log_path / "vanna_{time:YYYY-MM-DD}.log",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        rotation=rotation,
        retention="7 days",
        compression="gz",
        enqueue=True,
    )
    
    # JSON log file (for machine parsing)
    logger.add(
        log_path / "vanna_{time:YYYY-MM-DD}.jsonl",
        level=level,
        format="{message}",
        serialize=True,  # Loguru's built-in JSON serialization
        rotation=rotation,
        retention="7 days",
        enqueue=True,
    )
    
    # Error log file
    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        rotation=rotation,
        retention="30 days",
        enqueue=True,
    )
    
    # Trade execution log
    logger.add(
        log_path / "trades_{time:YYYY-MM-DD}.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        filter=lambda record: "TRADE" in record["message"],
        rotation=rotation,
        retention="90 days",
        enqueue=True,
    )
    
    _logger_initialized = True
    _process_id = current_pid
    logger.info("Logger initialized successfully")


def get_logger():
    """
    Get the configured logger instance.
    
    Returns:
        loguru.Logger: The configured logger
    
    Note:
        Auto-initializes with minimal config in subprocesses.
    """
    _ensure_minimal_logger()
    return logger


def log_with_context(level: str, message: str, **kwargs):
    """
    Log with automatic correlation ID.
    
    Usage:
        log_with_context("info", "Processing order", symbol="SPY", quantity=5)
    """
    corr_id = get_correlation_id()
    if corr_id:
        message = f"[{corr_id}] {message}"
    
    log_func = getattr(logger, level.lower())
    log_func(message, **kwargs)
