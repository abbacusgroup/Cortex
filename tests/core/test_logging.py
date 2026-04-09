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
        monkeypatch.delenv("CORTEX_DEBUG_HTTP", raising=False)
        setup_logging()
        for name in self.NOISY:
            assert logging.getLogger(name).level == logging.WARNING, name

    def test_debug_env_var_disables_quieting(self, monkeypatch):
        monkeypatch.setenv("CORTEX_DEBUG_MCP_SDK", "1")
        monkeypatch.delenv("CORTEX_DEBUG_HTTP", raising=False)
        for name in self.NOISY:
            logging.getLogger(name).setLevel(logging.NOTSET)
        setup_logging()
        for name in self.NOISY:
            # Untouched — left at NOTSET so the SDK's own configured
            # level (or the root logger's INFO) is honored.
            assert logging.getLogger(name).level == logging.NOTSET, name

    def test_quieting_is_idempotent(self, monkeypatch):
        monkeypatch.delenv("CORTEX_DEBUG_MCP_SDK", raising=False)
        monkeypatch.delenv("CORTEX_DEBUG_HTTP", raising=False)
        setup_logging()
        # Manually drop the level to verify the second call re-applies.
        logging.getLogger("mcp.server.streamable_http").setLevel(logging.DEBUG)
        setup_logging()  # second call — should re-quiet the logger
        assert (
            logging.getLogger("mcp.server.streamable_http").level
            == logging.WARNING
        )


class TestQuietHttpLoggers:
    """Bundle 10.7 / F.4: HTTP-layer loggers (uvicorn.access, httpx,
    httpcore) should be set to WARNING after ``setup_logging`` runs,
    unless ``CORTEX_DEBUG_HTTP`` is set. Independent from the MCP SDK
    escape hatch.
    """

    HTTP_NOISY = (
        "uvicorn",
        "uvicorn.access",
        "httpx",
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
    )
    MCP_NOISY = (
        "mcp.server.streamable_http",
        "mcp.client.streamable_http",
        "mcp.server.streamable_http_manager",
        "mcp.client.streamable_http_manager",
        "mcp.server.lowlevel.server",
    )

    def setup_method(self):
        logging.getLogger("cortex").handlers.clear()
        for name in self.HTTP_NOISY + self.MCP_NOISY:
            logging.getLogger(name).setLevel(logging.NOTSET)

    def test_http_loggers_are_quieted_by_default(self, monkeypatch):
        monkeypatch.delenv("CORTEX_DEBUG_MCP_SDK", raising=False)
        monkeypatch.delenv("CORTEX_DEBUG_HTTP", raising=False)
        setup_logging()
        for name in self.HTTP_NOISY:
            assert logging.getLogger(name).level == logging.WARNING, name

    def test_uvicorn_error_is_not_quieted(self, monkeypatch):
        """``uvicorn.error`` is intentionally left at its default so
        server-side error signal keeps flowing to stderr.
        """
        monkeypatch.delenv("CORTEX_DEBUG_MCP_SDK", raising=False)
        monkeypatch.delenv("CORTEX_DEBUG_HTTP", raising=False)
        logging.getLogger("uvicorn.error").setLevel(logging.NOTSET)
        setup_logging()
        assert logging.getLogger("uvicorn.error").level == logging.NOTSET

    def test_debug_http_env_var_disables_http_quieting(self, monkeypatch):
        monkeypatch.setenv("CORTEX_DEBUG_HTTP", "1")
        monkeypatch.delenv("CORTEX_DEBUG_MCP_SDK", raising=False)
        for name in self.HTTP_NOISY:
            logging.getLogger(name).setLevel(logging.NOTSET)
        setup_logging()
        for name in self.HTTP_NOISY:
            # Untouched — left at NOTSET so uvicorn/httpx default levels
            # (or the root logger's INFO) are honored.
            assert logging.getLogger(name).level == logging.NOTSET, name

    def test_escape_hatches_are_independent(self, monkeypatch):
        """Setting ``CORTEX_DEBUG_HTTP`` must NOT re-enable the MCP SDK
        loggers, and vice versa.
        """
        # Case 1: HTTP hatch set, MCP hatch unset — HTTP loggers free,
        # MCP loggers quieted.
        monkeypatch.setenv("CORTEX_DEBUG_HTTP", "1")
        monkeypatch.delenv("CORTEX_DEBUG_MCP_SDK", raising=False)
        for name in self.HTTP_NOISY + self.MCP_NOISY:
            logging.getLogger(name).setLevel(logging.NOTSET)
        logging.getLogger("cortex").handlers.clear()
        setup_logging()
        for name in self.HTTP_NOISY:
            assert logging.getLogger(name).level == logging.NOTSET, name
        for name in self.MCP_NOISY:
            assert logging.getLogger(name).level == logging.WARNING, name

        # Case 2: MCP hatch set, HTTP hatch unset — inverse.
        monkeypatch.delenv("CORTEX_DEBUG_HTTP", raising=False)
        monkeypatch.setenv("CORTEX_DEBUG_MCP_SDK", "1")
        for name in self.HTTP_NOISY + self.MCP_NOISY:
            logging.getLogger(name).setLevel(logging.NOTSET)
        logging.getLogger("cortex").handlers.clear()
        setup_logging()
        for name in self.HTTP_NOISY:
            assert logging.getLogger(name).level == logging.WARNING, name
        for name in self.MCP_NOISY:
            assert logging.getLogger(name).level == logging.NOTSET, name

    def test_http_quieting_is_idempotent(self, monkeypatch):
        monkeypatch.delenv("CORTEX_DEBUG_MCP_SDK", raising=False)
        monkeypatch.delenv("CORTEX_DEBUG_HTTP", raising=False)
        setup_logging()
        # Manually drop the level to verify the second call re-applies.
        logging.getLogger("uvicorn.access").setLevel(logging.DEBUG)
        setup_logging()  # second call — should re-quiet the logger
        assert (
            logging.getLogger("uvicorn.access").level == logging.WARNING
        )

    def test_uvicorn_access_has_min_level_filter(self, monkeypatch):
        """``setup_logging`` must attach a ``_MinLevelFilter`` to the
        ``uvicorn`` and ``uvicorn.access`` loggers. Plain ``setLevel``
        doesn't survive uvicorn's own ``configure_logging`` pass, but a
        filter on the logger instance does.
        """
        from cortex.core.logging import _MinLevelFilter

        monkeypatch.delenv("CORTEX_DEBUG_HTTP", raising=False)
        # Clear any pre-existing filters to simulate a fresh process
        for name in ("uvicorn", "uvicorn.access"):
            logger = logging.getLogger(name)
            for f in list(logger.filters):
                logger.removeFilter(f)

        setup_logging()

        for name in ("uvicorn", "uvicorn.access"):
            logger = logging.getLogger(name)
            filters = [
                f for f in logger.filters if isinstance(f, _MinLevelFilter)
            ]
            assert len(filters) == 1, f"{name} should have exactly 1 filter"
            assert filters[0]._min_level == logging.WARNING

    def test_uvicorn_filter_drops_info_records(self, monkeypatch):
        """End-to-end: an INFO record sent to ``uvicorn.access`` must be
        rejected by the attached filter (regardless of the logger's
        own level, which uvicorn may reset later).
        """
        monkeypatch.delenv("CORTEX_DEBUG_HTTP", raising=False)
        for name in ("uvicorn", "uvicorn.access"):
            logger = logging.getLogger(name)
            for f in list(logger.filters):
                logger.removeFilter(f)

        setup_logging()

        access_logger = logging.getLogger("uvicorn.access")
        # Simulate uvicorn resetting the level back to INFO after
        # dictConfig — the filter should still drop INFO records.
        access_logger.setLevel(logging.INFO)
        info_record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="x.py",
            lineno=1,
            msg='127.0.0.1:0 - "POST /mcp" 200 OK',
            args=(),
            exc_info=None,
        )
        warning_record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.WARNING,
            pathname="x.py",
            lineno=1,
            msg="slow request",
            args=(),
            exc_info=None,
        )
        # Our filters are attached to the logger, so walk and apply them.
        assert all(f.filter(warning_record) for f in access_logger.filters)
        assert any(not f.filter(info_record) for f in access_logger.filters)

    def test_uvicorn_filter_idempotent(self, monkeypatch):
        """Repeat ``setup_logging`` calls must not stack filters."""
        from cortex.core.logging import _MinLevelFilter

        monkeypatch.delenv("CORTEX_DEBUG_HTTP", raising=False)
        access_logger = logging.getLogger("uvicorn.access")
        for f in list(access_logger.filters):
            access_logger.removeFilter(f)

        setup_logging()
        setup_logging()
        setup_logging()

        cortex_filters = [
            f for f in access_logger.filters
            if isinstance(f, _MinLevelFilter)
        ]
        assert len(cortex_filters) == 1

    def test_uvicorn_filter_respects_debug_http(self, monkeypatch):
        """With ``CORTEX_DEBUG_HTTP=1`` set, no filter is attached."""
        from cortex.core.logging import _MinLevelFilter

        monkeypatch.setenv("CORTEX_DEBUG_HTTP", "1")
        for name in ("uvicorn", "uvicorn.access"):
            logger = logging.getLogger(name)
            for f in list(logger.filters):
                logger.removeFilter(f)

        setup_logging()

        for name in ("uvicorn", "uvicorn.access"):
            logger = logging.getLogger(name)
            assert not any(
                isinstance(f, _MinLevelFilter) for f in logger.filters
            )
