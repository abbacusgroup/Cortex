"""Structured logging for Cortex.

Provides a JSON-formatted logger that includes timestamp, level, module, and message.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class JSONFormatter(logging.Formatter):
    """Outputs log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(*, level: str = "INFO", json_output: bool = True) -> logging.Logger:
    """Configure and return the root cortex logger.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, use JSON formatter. Otherwise, use simple text.

    Returns:
        The configured 'cortex' logger.
    """
    logger = logging.getLogger("cortex")

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(module)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))

    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the cortex namespace."""
    return logging.getLogger(f"cortex.{name}")
