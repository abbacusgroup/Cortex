"""Structured logging for Cortex.

Provides a JSON-formatted logger that includes timestamp, level, module, and message.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

# Bundle 9 / F.4: third-party MCP SDK loggers that flood stderr with
# per-request INFO chatter when Cortex runs as ``cortex serve
# --transport mcp-http``. Each session lifecycle event (Created new
# transport / Terminating session / Processing request of type
# ListToolsRequest, etc.) writes multiple lines. We bump them to
# WARNING by default so the log files stay useful and bounded. Set
# ``CORTEX_DEBUG_MCP_SDK=1`` to opt back into INFO.
_NOISY_THIRD_PARTY_LOGGERS = (
    "mcp.server.streamable_http",
    "mcp.client.streamable_http",
    "mcp.server.streamable_http_manager",
    "mcp.client.streamable_http_manager",
    "mcp.server.lowlevel.server",
)

# Bundle 10.7 / F.4: HTTP-layer loggers that drive most of the log-file
# growth on a production LaunchAgent install. Empirically measured at
# ~33 MB/day combined across ``~/.cortex/mcp-http.{log,err}``:
#   * ``uvicorn.access`` — one INFO line per HTTP request (~13.5 MB/day,
#     the single biggest growth source, because FastMCP's streamable-http
#     transport starts uvicorn internally with access_log=True by default)
#   * ``httpx`` / ``httpcore.*`` — one INFO line per outbound HTTP call
#     made by sentence-transformers during embedding model warmup
#     (~19 MB/day surge at startup + periodic chatter)
#   * ``uvicorn`` — the startup banner
# ``uvicorn.error`` is intentionally NOT in this list — we keep error
# signal flowing. Set ``CORTEX_DEBUG_HTTP=1`` to opt back into the
# verbose INFO logs when debugging HTTP-level issues.
_NOISY_HTTP_LOGGERS = (
    "uvicorn",
    "uvicorn.access",
    "httpx",
    "httpcore",
    "httpcore.http11",
    "httpcore.connection",
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


class _MinLevelFilter(logging.Filter):
    """Drop log records below ``min_level``.

    Attached to the ``uvicorn`` and ``uvicorn.access`` logger instances
    to survive uvicorn's own ``configure_logging`` pass — filters are
    preserved across ``logging.config.dictConfig`` (when
    ``disable_existing_loggers=False``) AND across the explicit
    ``setLevel`` call uvicorn makes on ``uvicorn.access`` right after
    ``dictConfig``. Plain ``setLevel`` alone does NOT stick because of
    that second override.
    """

    def __init__(self, min_level: int) -> None:
        super().__init__()
        self._min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self._min_level


def _patch_uvicorn_logging_config() -> None:
    """Silence ``uvicorn`` and ``uvicorn.access`` at WARNING.

    We use logger-attached filters instead of ``setLevel`` because
    uvicorn's ``Server.configure_logging`` unconditionally re-applies
    the config-supplied ``log_level`` to ``uvicorn.access`` AFTER
    ``dictConfig`` runs, which would overwrite any level we set. A
    filter on the logger is immune: ``dictConfig`` with
    ``disable_existing_loggers=False`` doesn't touch filters, and
    explicit ``setLevel`` doesn't either. Records with
    ``levelno < WARNING`` are dropped before they reach uvicorn's
    stream handler.

    No-op if uvicorn isn't installed (stdio transport doesn't need it)
    or if ``CORTEX_DEBUG_HTTP`` is set. Idempotent — guards against
    stacking multiple filter instances on repeat calls.

    ``uvicorn.error`` is intentionally NOT filtered; errors must keep
    flowing.
    """
    if os.environ.get("CORTEX_DEBUG_HTTP", "").strip():
        return
    try:
        import uvicorn.config  # noqa: F401 — availability probe only
    except ImportError:
        return
    for name in ("uvicorn", "uvicorn.access"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, _MinLevelFilter) for f in logger.filters):
            logger.addFilter(_MinLevelFilter(logging.WARNING))


def _quiet_noisy_loggers() -> None:
    """Bundles 9 & 10.7 / F.4: bump noisy third-party loggers to WARNING.

    Two independent groups, each with its own escape hatch env var:

    * MCP SDK loggers (Bundle 9) — gated by ``CORTEX_DEBUG_MCP_SDK``.
    * HTTP-layer loggers (Bundle 10.7) — gated by ``CORTEX_DEBUG_HTTP``.

    Each env var, when set to any non-empty value, leaves its own group
    alone so operators can opt back into the chatty INFO logs when
    debugging. The two escape hatches are independent: setting one does
    NOT re-enable the other. Idempotent — safe to call repeatedly.

    Also patches ``uvicorn.config.LOGGING_CONFIG`` in place so uvicorn's
    own ``dictConfig`` call at startup honors the WARNING level for
    ``uvicorn.access`` (which otherwise overrides a plain ``setLevel``).
    """
    if not os.environ.get("CORTEX_DEBUG_MCP_SDK", "").strip():
        for name in _NOISY_THIRD_PARTY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
    if not os.environ.get("CORTEX_DEBUG_HTTP", "").strip():
        for name in _NOISY_HTTP_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
    _patch_uvicorn_logging_config()


def setup_logging(*, level: str = "INFO", json_output: bool = True) -> logging.Logger:
    """Configure and return the root cortex logger.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, use JSON formatter. Otherwise, use simple text.

    Returns:
        The configured 'cortex' logger.
    """
    logger = logging.getLogger("cortex")

    # Bundles 9 & 10.7 / F.4: silence the noisy MCP SDK + HTTP loggers on
    # every call. We do this even when our own handler is already attached
    # so that a later test which imports the SDK (or uvicorn/httpx) after
    # our logger is set up still gets the WARNING level applied.
    _quiet_noisy_loggers()

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s [%(module)s] %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )

    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the cortex namespace."""
    return logging.getLogger(f"cortex.{name}")
