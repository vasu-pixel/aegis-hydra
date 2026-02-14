"""
aegis_hydra.utils.logger — Structured Logging

Configures loguru for structured, color-coded logging with
rotation and optional JSON output for production.

Dependencies: loguru
"""

from __future__ import annotations

import sys
from typing import Optional

from loguru import logger


def get_logger(
    name: str = "aegis_hydra",
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    rotation: str = "100 MB",
    retention: str = "7 days",
) -> "logger":
    """
    Configure and return a loguru logger instance.

    Parameters
    ----------
    name : str
        Logger name (appears in log prefix).
    level : str
        Minimum log level: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_file : str, optional
        Path to log file. If None, logs only to stderr.
    json_format : bool
        If True, output structured JSON logs (for production).
    rotation : str
        Log file rotation policy.
    retention : str
        How long to keep old log files.

    Returns
    -------
    loguru.Logger
    """
    # Remove default handler
    logger.remove()

    # Console handler
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        f"<cyan>{name}</cyan>:<cyan>{{function}}</cyan>:<cyan>{{line}}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, format=fmt, level=level, colorize=True)

    # File handler
    if log_file:
        if json_format:
            logger.add(
                log_file,
                serialize=True,  # JSON output
                rotation=rotation,
                retention=retention,
                level=level,
            )
        else:
            logger.add(
                log_file,
                format=fmt,
                rotation=rotation,
                retention=retention,
                level=level,
            )

    return logger
