"""
Logging — configure once, import `logger` everywhere.
Uses loguru for clean console output and structlog for JSON production logs.
"""

import sys
from utils.settings import settings
from loguru import logger


def setup_logging() -> None:
    logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=fmt,
        level=settings.log_level,
        colorize=True,
    )

    logger.add(
        "logs/findoc_{time:YYYY-MM-DD}.log",
        rotation="00:00",       # new file each day
        retention="14 days",
        compression="gz",
        level="DEBUG",
        format="{time} | {level} | {name}:{function}:{line} — {message}",
    )

    logger.info("Logging initialised at level={}", settings.log_level)


__all__ = ["logger", "setup_logging"]
