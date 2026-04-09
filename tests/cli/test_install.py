"""Tests for cortex.cli.install — platform-aware service installation."""

from __future__ import annotations

import plistlib
from pathlib import Path
from unittest.mock import patch

import pytest

from cortex.cli.install import (
    _DASHBOARD_LABEL,
    _MCP_LABEL,
    detect_cortex_binary,
    detect_platform,
    render_dashboard_plist,
    render_dashboard_unit,
    render_mcp_plist,
    render_mcp_unit,
)
from cortex.core.config import CortexConfig

# ---------------------------------------------------------------------------
# detect_cortex_binary
# ---------------------------------------------------------------------------


class TestDetectBinary:
    def test_finds_binary_next_to_python(self, tmp_path):
        fake_bin = tmp_path / "cortex"
        fake_bin.touch()
        with patch("cortex.cli.install.sys") as mock_sys:
            mock_sys.executable = str(tmp_path / "python3")
            result = detect_cortex_binary()
        assert result == str(fake_bin.resolve())

    def test_falls_back_to_shutil_which(self, tmp_path):
        with (
            patch("cortex.cli.install.sys") as mock_sys,
            patch("cortex.cli.install.shutil") as mock_shutil,
        ):
            # No binary next to python
            mock_sys.executable = str(tmp_path / "python3")
            mock_shutil.which.return_value = "/usr/local/bin/cortex"
            result = detect_cortex_binary()
        assert "cortex" in result

    def test_raises_if_not_found(self, tmp_path):
        with (
            patch("cortex.cli.install.sys") as mock_sys,
            patch("cortex.cli.install.shutil") as mock_shutil,
        ):
            mock_sys.executable = str(tmp_path / "python3")
            mock_shutil.which.return_value = None
            with pytest.raises(FileNotFoundError, match="Cannot find"):
                detect_cortex_binary()


# ---------------------------------------------------------------------------
# detect_platform
# ---------------------------------------------------------------------------


class TestDetectPlatform:
    def test_darwin_returns_macos(self):
        with patch("cortex.cli.install.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert detect_platform() == "macos"

    def test_linux_returns_linux(self):
        with patch("cortex.cli.install.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert detect_platform() == "linux"

    def test_unsupported_raises(self):
        with patch("cortex.cli.install.sys") as mock_sys:
            mock_sys.platform = "win32"
            with pytest.raises(RuntimeError, match="Unsupported"):
                detect_platform()


# ---------------------------------------------------------------------------
# macOS plist rendering
# ---------------------------------------------------------------------------


class TestMacosPlist:
    @pytest.fixture()
    def config(self, tmp_path: Path) -> CortexConfig:
        return CortexConfig(data_dir=tmp_path, host="127.0.0.1", port=1314)

    def test_mcp_plist_is_valid_xml(self, config):
        content = render_mcp_plist(config, "/usr/local/bin/cortex")
        # plistlib.loads validates XML structure
        parsed = plistlib.loads(content.encode())
        assert parsed["Label"] == _MCP_LABEL
        assert parsed["KeepAlive"] is True
        assert parsed["RunAtLoad"] is True

    def test_mcp_plist_has_correct_binary(self, config):
        content = render_mcp_plist(config, "/opt/cortex/bin/cortex")
        parsed = plistlib.loads(content.encode())
        assert parsed["ProgramArguments"][0] == "/opt/cortex/bin/cortex"
        assert "serve" in parsed["ProgramArguments"]
        assert "mcp-http" in parsed["ProgramArguments"]

    def test_mcp_plist_has_correct_port(self, config):
        content = render_mcp_plist(config, "/usr/local/bin/cortex")
        parsed = plistlib.loads(content.encode())
        assert "1314" in parsed["ProgramArguments"]

    def test_dashboard_plist_uses_port_plus_one(self, config):
        content = render_dashboard_plist(config, "/usr/local/bin/cortex")
        parsed = plistlib.loads(content.encode())
        assert parsed["Label"] == _DASHBOARD_LABEL
        assert "1315" in parsed["ProgramArguments"]

    def test_dashboard_plist_has_dashboard_command(self, config):
        content = render_dashboard_plist(config, "/usr/local/bin/cortex")
        parsed = plistlib.loads(content.encode())
        assert "dashboard" in parsed["ProgramArguments"]

    def test_plist_data_dir_in_log_paths(self, config):
        content = render_mcp_plist(config, "/usr/local/bin/cortex")
        parsed = plistlib.loads(content.encode())
        assert str(config.data_dir) in parsed["StandardOutPath"]
        assert str(config.data_dir) in parsed["StandardErrorPath"]


# ---------------------------------------------------------------------------
# Linux systemd unit rendering
# ---------------------------------------------------------------------------


class TestLinuxUnit:
    @pytest.fixture()
    def config(self, tmp_path: Path) -> CortexConfig:
        return CortexConfig(data_dir=tmp_path, host="127.0.0.1", port=1314)

    def test_mcp_unit_has_required_sections(self, config):
        content = render_mcp_unit(config, "/usr/local/bin/cortex")
        assert "[Unit]" in content
        assert "[Service]" in content
        assert "[Install]" in content

    def test_mcp_unit_has_correct_exec(self, config):
        content = render_mcp_unit(config, "/opt/bin/cortex")
        assert "ExecStart=/opt/bin/cortex serve --transport mcp-http" in content

    def test_mcp_unit_has_restart(self, config):
        content = render_mcp_unit(config, "/usr/local/bin/cortex")
        assert "Restart=always" in content

    def test_dashboard_unit_uses_port_plus_one(self, config):
        content = render_dashboard_unit(config, "/usr/local/bin/cortex")
        assert "--port 1315" in content

    def test_dashboard_unit_has_dashboard_command(self, config):
        content = render_dashboard_unit(config, "/usr/local/bin/cortex")
        assert "ExecStart=" in content
        assert "dashboard" in content

    def test_unit_has_data_dir_env(self, config):
        content = render_mcp_unit(config, "/usr/local/bin/cortex")
        assert f"CORTEX_DATA_DIR={config.data_dir}" in content
