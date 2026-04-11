"""Tests for cortex.core.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.core.config import CortexConfig, load_config
from cortex.core.constants import DEFAULT_DATA_DIR, DEFAULT_HOST, DEFAULT_PORT
from cortex.core.errors import ConfigError, ConfigPermissionError


class TestCortexConfig:
    """Tests for the CortexConfig dataclass."""

    def test_defaults(self):
        cfg = CortexConfig()
        assert cfg.data_dir == DEFAULT_DATA_DIR
        assert cfg.host == DEFAULT_HOST
        assert cfg.port == DEFAULT_PORT
        assert cfg.log_level == "INFO"
        assert cfg.mcp_server_url == "http://127.0.0.1:1314/mcp"

    def test_derived_paths(self):
        cfg = CortexConfig(data_dir=Path("/tmp/cortex-test"))
        assert cfg.graph_db_path == Path("/tmp/cortex-test/graph.db")
        assert cfg.sqlite_db_path == Path("/tmp/cortex-test/cortex.db")
        assert cfg.ontology_dir == Path("/tmp/cortex-test/ontology")

    def test_frozen(self):
        cfg = CortexConfig()
        with pytest.raises(AttributeError):
            cfg.port = 9999  # type: ignore[misc]


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_loads_with_defaults(self, tmp_path: Path):
        cfg = load_config(data_dir=tmp_path / "data")
        assert cfg.data_dir == (tmp_path / "data").resolve()
        assert cfg.host == DEFAULT_HOST

    def test_creates_data_dir(self, tmp_path: Path):
        data = tmp_path / "new" / "nested" / "data"
        assert not data.exists()
        cfg = load_config(data_dir=data)
        assert data.exists()
        assert data.is_dir()
        assert cfg.data_dir == data.resolve()

    def test_env_var_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CORTEX_PORT", "9999")
        monkeypatch.setenv("CORTEX_HOST", "0.0.0.0")
        cfg = load_config(data_dir=tmp_path)
        assert cfg.port == 9999
        assert cfg.host == "0.0.0.0"

    def test_env_file_loading(self, tmp_path: Path):
        env_file = tmp_path / ".env"
        env_file.write_text("CORTEX_LOG_LEVEL=DEBUG\nCORTEX_HOST=10.0.0.1\n")
        cfg = load_config(data_dir=tmp_path / "data", env_file=env_file)
        assert cfg.log_level == "DEBUG"
        assert cfg.host == "10.0.0.1"

    def test_env_var_beats_env_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        env_file = tmp_path / ".env"
        env_file.write_text("CORTEX_PORT=1111\n")
        monkeypatch.setenv("CORTEX_PORT", "2222")
        cfg = load_config(data_dir=tmp_path / "data", env_file=env_file)
        assert cfg.port == 2222

    def test_unwritable_data_dir(self, tmp_path: Path):
        readonly = tmp_path / "readonly"
        readonly.mkdir()
        readonly.chmod(0o444)
        try:
            with pytest.raises(ConfigPermissionError):
                load_config(data_dir=readonly)
        finally:
            readonly.chmod(0o755)

    def test_invalid_port_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CORTEX_PORT", "not_a_number")
        cfg = load_config(data_dir=tmp_path)
        assert cfg.port == DEFAULT_PORT

    def test_malformed_env_file(self, tmp_path: Path):
        env_file = tmp_path / ".env"
        env_file.write_text("GARBAGENONSENSE\nCORTEX_HOST=ok\n=bad\n")
        cfg = load_config(data_dir=tmp_path / "data", env_file=env_file)
        assert cfg.host == "ok"

    def test_unrecognized_keys_ignored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CORTEX_UNKNOWN_KEY", "whatever")
        cfg = load_config(data_dir=tmp_path)
        assert cfg.host == DEFAULT_HOST  # no crash

    def test_shell_injection_literal(self, tmp_path: Path):
        env_file = tmp_path / ".env"
        env_file.write_text("CORTEX_HOST=$(rm -rf /)\n")
        cfg = load_config(data_dir=tmp_path / "data", env_file=env_file)
        assert cfg.host == "$(rm -rf /)"  # treated as literal string

    def test_log_json_parsing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        for truthy in ("true", "True", "1", "yes"):
            monkeypatch.setenv("CORTEX_LOG_JSON", truthy)
            cfg = load_config(data_dir=tmp_path)
            assert cfg.log_json is True

        for falsy in ("false", "0", "no"):
            monkeypatch.setenv("CORTEX_LOG_JSON", falsy)
            cfg = load_config(data_dir=tmp_path)
            assert cfg.log_json is False


class TestMcpServerUrl:
    """Tests for the mcp_server_url config field added in Phase 2.F."""

    def test_default_url(self, tmp_path: Path):
        cfg = load_config(data_dir=tmp_path)
        assert cfg.mcp_server_url == "http://127.0.0.1:1314/mcp"

    def test_env_var_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CORTEX_MCP_SERVER_URL", "http://example.com:9999/mcp")
        cfg = load_config(data_dir=tmp_path)
        assert cfg.mcp_server_url == "http://example.com:9999/mcp"

    def test_https_url_accepted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CORTEX_MCP_SERVER_URL", "https://secure.example/mcp")
        cfg = load_config(data_dir=tmp_path)
        assert cfg.mcp_server_url == "https://secure.example/mcp"

    def test_invalid_scheme_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CORTEX_MCP_SERVER_URL", "gopher://example/mcp")
        with pytest.raises(ConfigError, match="http://"):
            load_config(data_dir=tmp_path)

    def test_no_scheme_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CORTEX_MCP_SERVER_URL", "127.0.0.1:1314/mcp")
        with pytest.raises(ConfigError):
            load_config(data_dir=tmp_path)

    def test_path_preserved(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CORTEX_MCP_SERVER_URL", "http://localhost:1234/v1/mcp")
        cfg = load_config(data_dir=tmp_path)
        assert cfg.mcp_server_url == "http://localhost:1234/v1/mcp"
