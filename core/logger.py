"""
Vanna Logger Module

Configures loguru for structured logging.
IMPORTANT: Logger is NOT auto-initialized. Call setup_logger() explicitly.
"""

import sys
from pathlib import Path

from loguru import logger

# Remove default handler
logger.remove()

_logger_initialized = False


def setup_logger(
    level: str = "INFO",
    log_dir: str = "logs",
    rotation: str = "100 MB"
) -> None:
    """
    Initialize the logger with console and file handlers.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        rotation: Log rotation size
    
    Example:
        from core.logger import setup_logger, get_logger
        
        setup_logger(level="DEBUG")
        logger = get_logger()
        logger.info("Application started")
    """
    global _logger_initialized
    
    if _logger_initialized:
        return
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Console handler (stdout)
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
    )
    
    # Main log file
    logger.add(
        log_path / "vanna_{time:YYYY-MM-DD}.log",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        rotation=rotation,
        retention="7 days",
        compression="gz",
    )
    
    # Error log file
    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        rotation=rotation,
        retention="30 days",
    )
    
    # Trade execution log
    logger.add(
        log_path / "trades_{time:YYYY-MM-DD}.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        filter=lambda record: "TRADE" in record["message"],
        rotation=rotation,
        retention="90 days",
    )
    
    _logger_initialized = True
    logger.info("Logger initialized successfully")


def get_logger():
    """
    Get the configured logger instance.
    
    Returns:
        loguru.Logger: The configured logger
    
    Note:
        If logger is not initialized, a minimal console handler is added.
    """
    global _logger_initialized
    
    if not _logger_initialized:
        # Auto-init with minimal config (just console)
        logger.add(
            sys.stderr, 
            level="INFO", 
            format="{time:HH:mm:ss} | {level: <8} | {message}"
        )
        _logger_initialized = True
    
    return logger
