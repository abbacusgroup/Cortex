"""Structured logging for Cortex.

Provides a JSON-formatted logger that includes timestamp, level, module, and message.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any


# Bundle 9 / F.4: third-party loggers that flood stderr with per-request
# INFO chatter when Cortex runs as ``cortex serve --transport mcp-http``.
# Each session lifecycle event (Created new transport / Terminating
# session / Processing request of type ListToolsRequest, etc.) writes
# multiple lines, totaling ~1 MB/day on a moderately-used server. We
# bump them to WARNING by default so the log files stay useful and
# bounded. Set ``CORTEX_DEBUG_MCP_SDK=1`` to opt back into INFO.
_NOISY_THIRD_PARTY_LOGGERS = (
    "mcp.server.streamable_http",
    "mcp.client.streamable_http",
    "mcp.server.streamable_http_manager",
    "mcp.client.streamable_http_manager",
    "mcp.server.lowlevel.server",
)


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


def _quiet_noisy_loggers() -> None:
    """Bundle 9 / F.4: bump noisy third-party loggers to WARNING.

    Skipped if ``CORTEX_DEBUG_MCP_SDK`` is set (any non-empty value), so
    operators can opt back into the chatty INFO logs when debugging
    transport-level issues. Idempotent — safe to call repeatedly.
    """
    if os.environ.get("CORTEX_DEBUG_MCP_SDK", "").strip():
        return
    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def setup_logging(*, level: str = "INFO", json_output: bool = True) -> logging.Logger:
    """Configure and return the root cortex logger.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, use JSON formatter. Otherwise, use simple text.

    Returns:
        The configured 'cortex' logger.
    """
    logger = logging.getLogger("cortex")

    # Bundle 9 / F.4: silence the noisy MCP SDK loggers on every call.
    # We do this even when our own handler is already attached so that a
    # later test that imports the SDK after our logger is set up still
    # gets the WARNING level applied.
    _quiet_noisy_loggers()

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
