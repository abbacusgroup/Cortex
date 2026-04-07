"""Tests for cortex.core.errors."""

from __future__ import annotations

import json

from cortex.core.errors import (
    AuthenticationError,
    ClassificationError,
    ConfigError,
    ConfigNotFoundError,
    ConfigPermissionError,
    CortexError,
    LLMError,
    NotFoundError,
    OntologyError,
    PipelineError,
    StoreError,
    StoreLockedError,
    SyncError,
    TransportError,
    ValidationError,
)

import pytest


class TestCortexError:
    def test_basic(self):
        err = CortexError("something broke")
        assert str(err) == "something broke"
        assert err.code == "CORTEX_ERROR"
        assert err.message == "something broke"
        assert err.context == {}

    def test_with_context(self):
        err = CortexError("bad", context={"file": "foo.db"})
        assert err.context == {"file": "foo.db"}

    def test_to_dict(self):
        err = CortexError("msg", context={"key": "val"})
        d = err.to_dict()
        assert d["code"] == "CORTEX_ERROR"
        assert d["message"] == "msg"
        assert d["context"]["key"] == "val"
        assert "cause" not in d

    def test_to_json(self):
        err = CortexError("msg")
        parsed = json.loads(err.to_json())
        assert parsed["code"] == "CORTEX_ERROR"

    def test_chained_cortex_errors(self):
        inner = StoreError("db failed", context={"db": "sqlite"})
        outer = SyncError("sync failed", cause=inner)
        d = outer.to_dict()
        assert d["cause"]["code"] == "CORTEX_STORE_ERROR"
        assert d["cause"]["message"] == "db failed"

    def test_chained_external_error(self):
        inner = OSError("disk full")
        outer = ConfigError("cannot write", cause=inner)
        d = outer.to_dict()
        assert d["cause"]["code"] == "EXTERNAL"
        assert "disk full" in d["cause"]["message"]

    def test_none_context(self):
        err = CortexError("msg", context=None)
        assert err.context == {}

    def test_error_recovery(self):
        """After catching a CortexError, execution continues."""
        result = None
        try:
            raise NotFoundError("missing", context={"id": "42"})
        except CortexError:
            result = "recovered"
        assert result == "recovered"


class TestErrorHierarchy:
    """Verify the inheritance chain so except CortexError catches everything."""

    def test_config_errors(self):
        assert issubclass(ConfigError, CortexError)
        assert issubclass(ConfigNotFoundError, ConfigError)
        assert issubclass(ConfigPermissionError, ConfigError)

    def test_store_errors(self):
        assert issubclass(StoreError, CortexError)
        assert issubclass(NotFoundError, StoreError)
        assert issubclass(SyncError, StoreError)
        assert issubclass(StoreLockedError, StoreError)

    def test_pipeline_errors(self):
        assert issubclass(PipelineError, CortexError)
        assert issubclass(ClassificationError, PipelineError)
        assert issubclass(LLMError, PipelineError)

    def test_transport_errors(self):
        assert issubclass(TransportError, CortexError)
        assert issubclass(AuthenticationError, TransportError)

    def test_other_errors(self):
        assert issubclass(OntologyError, CortexError)
        assert issubclass(ValidationError, CortexError)


class TestErrorCodes:
    """Each error type has a unique code."""

    def test_unique_codes(self):
        error_classes = [
            CortexError, ConfigError, ConfigNotFoundError, ConfigPermissionError,
            StoreError, NotFoundError, SyncError, StoreLockedError,
            OntologyError, ValidationError,
            PipelineError, ClassificationError, LLMError,
            TransportError, AuthenticationError,
        ]
        codes = [cls.code for cls in error_classes]
        assert len(codes) == len(set(codes)), f"Duplicate codes: {codes}"


class TestStoreLockedError:
    """StoreLockedError carries holder PID/cmdline and produces actionable messages."""

    def test_basic_with_holder(self):
        err = StoreLockedError(
            "Graph DB is locked.",
            holder_pid=12345,
            holder_cmdline="cortex serve --transport stdio",
        )
        assert err.holder_pid == 12345
        assert err.holder_cmdline == "cortex serve --transport stdio"
        assert err.is_stale is False
        assert err.code == "CORTEX_STORE_LOCKED"

    def test_str_includes_pid_and_cmdline(self):
        err = StoreLockedError(
            "Graph DB is locked.",
            holder_pid=12345,
            holder_cmdline="cortex serve --transport stdio",
        )
        s = str(err)
        assert "12345" in s
        assert "cortex serve --transport stdio" in s
        assert "Stop the conflicting process" in s or "kill PID 12345" in s

    def test_str_when_holder_unknown(self):
        err = StoreLockedError("Graph DB is locked.")
        s = str(err)
        assert "unknown" in s.lower()
        assert "PID" not in s.replace("unknown", "")  # no PID phrasing when none

    def test_stale_flag_in_message(self):
        err = StoreLockedError(
            "Graph DB is locked.",
            holder_pid=99999,
            holder_cmdline="cortex serve",
            is_stale=True,
        )
        assert err.is_stale is True
        s = str(err)
        assert "stale" in s.lower()
        assert "99999" in s

    def test_isinstance_store_error(self):
        err = StoreLockedError("locked")
        assert isinstance(err, StoreError)
        assert isinstance(err, CortexError)
        assert isinstance(err, Exception)

    def test_to_dict_includes_holder_in_context(self):
        err = StoreLockedError(
            "locked",
            holder_pid=42,
            holder_cmdline="foo",
            is_stale=False,
        )
        d = err.to_dict()
        assert d["code"] == "CORTEX_STORE_LOCKED"
        assert d["context"]["holder_pid"] == 42
        assert d["context"]["holder_cmdline"] == "foo"
        assert d["context"]["is_stale"] is False

    def test_rejects_non_int_pid(self):
        with pytest.raises(TypeError, match="holder_pid must be int"):
            StoreLockedError("locked", holder_pid="abc")  # type: ignore[arg-type]

    def test_accepts_none_pid(self):
        err = StoreLockedError("locked", holder_pid=None)
        assert err.holder_pid is None

    def test_can_be_raised_and_caught_as_store_error(self):
        """Code that catches StoreError must also catch StoreLockedError."""
        caught = False
        try:
            raise StoreLockedError("locked", holder_pid=1)
        except StoreError:
            caught = True
        assert caught

    def test_extra_context_preserved(self):
        err = StoreLockedError(
            "locked",
            holder_pid=1,
            holder_cmdline="x",
            context={"path": "/tmp/g.db"},
        )
        assert err.context["path"] == "/tmp/g.db"
        assert err.context["holder_pid"] == 1

    def test_chained_with_oserror_cause(self):
        cause = OSError("lock hold by current process")
        err = StoreLockedError("locked", holder_pid=42, cause=cause)
        d = err.to_dict()
        assert d["cause"]["code"] == "EXTERNAL"
        assert "lock hold" in d["cause"]["message"]
