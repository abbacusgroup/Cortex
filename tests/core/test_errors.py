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
    SyncError,
    TransportError,
    ValidationError,
)


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
            StoreError, NotFoundError, SyncError, OntologyError, ValidationError,
            PipelineError, ClassificationError, LLMError,
            TransportError, AuthenticationError,
        ]
        codes = [cls.code for cls in error_classes]
        assert len(codes) == len(set(codes)), f"Duplicate codes: {codes}"
