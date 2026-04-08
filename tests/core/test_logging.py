"""Tests for cortex.core.logging."""

from __future__ import annotations

import json
import logging

from cortex.core.logging import JSONFormatter, get_logger, setup_logging


class TestJSONFormatter:
    def test_produces_valid_json(self):
        formatter = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord(
            name="cortex.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello world"
        assert "ts" in parsed
        assert "module" in parsed

    def test_includes_exception(self):
        formatter = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="cortex.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="failed",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


class TestSetupLogging:
    def setup_method(self):
        # Clear handlers between tests
        logger = logging.getLogger("cortex")
        logger.handlers.clear()

    def test_returns_logger(self):
        logger = setup_logging(level="DEBUG")
        assert logger.name == "cortex"
        assert logger.level == logging.DEBUG

    def test_json_handler(self):
        logger = setup_logging(json_output=True)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_text_handler(self):
        logger = setup_logging(json_output=False)
        assert len(logger.handlers) == 1
        assert not isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_no_duplicate_handlers(self):
        setup_logging()
        setup_logging()
        logger = logging.getLogger("cortex")
        assert len(logger.handlers) == 1

    def test_all_levels(self, capsys):
        logger = setup_logging(level="DEBUG", json_output=True)
        for level in ("debug", "info", "warning", "error", "critical"):
            getattr(logger, level)(f"test {level}")
        # Logs go to stderr
        captured = capsys.readouterr()
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            assert level in captured.err

    def test_unicode_messages(self):
        logger = setup_logging(level="DEBUG", json_output=True)
        # Should not raise
        logger.info("emoji: 🧠 CJK: 知識 RTL: مرحبا")


class TestGetLogger:
    def test_child_namespace(self):
        child = get_logger("db")
        assert child.name == "cortex.db"

    def test_nested_child(self):
        child = get_logger("db.graph")
        assert child.name == "cortex.db.graph"


class TestQuietNoisyLoggers:
    """Bundle 9 / F.4: third-party MCP SDK loggers should be set to
    WARNING after ``setup_logging`` runs, unless ``CORTEX_DEBUG_MCP_SDK``
    is set to opt back into the chatty INFO logs.
    """

    NOISY = (
        "mcp.server.streamable_http",
        "mcp.client.streamable_http",
        "mcp.server.streamable_http_manager",
        "mcp.client.streamable_http_manager",
        "mcp.server.lowlevel.server",
    )

    def setup_method(self):
        logging.getLogger("cortex").handlers.clear()
        for name in self.NOISY:
            logging.getLogger(name).setLevel(logging.NOTSET)

    def test_noisy_loggers_are_quieted_by_default(self, monkeypatch):
        monkeypatch.delenv("CORTEX_DEBUG_MCP_SDK", raising=False)
        setup_logging()
        for name in self.NOISY:
            assert logging.getLogger(name).level == logging.WARNING, name

    def test_debug_env_var_disables_quieting(self, monkeypatch):
        monkeypatch.setenv("CORTEX_DEBUG_MCP_SDK", "1")
        for name in self.NOISY:
            logging.getLogger(name).setLevel(logging.NOTSET)
        setup_logging()
        for name in self.NOISY:
            # Untouched — left at NOTSET so the SDK's own configured
            # level (or the root logger's INFO) is honored.
            assert logging.getLogger(name).level == logging.NOTSET, name

    def test_quieting_is_idempotent(self, monkeypatch):
        monkeypatch.delenv("CORTEX_DEBUG_MCP_SDK", raising=False)
        setup_logging()
        # Manually drop the level to verify the second call re-applies.
        logging.getLogger("mcp.server.streamable_http").setLevel(logging.DEBUG)
        setup_logging()  # second call — should re-quiet the logger
        assert (
            logging.getLogger("mcp.server.streamable_http").level
            == logging.WARNING
        )
