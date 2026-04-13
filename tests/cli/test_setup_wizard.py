"""Tests for cortex.cli.env_writer and cortex.cli.setup_wizard."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cortex.cli.env_writer import read_env, write_env


# ---------------------------------------------------------------------------
# env_writer — read_env
# ---------------------------------------------------------------------------


class TestReadEnv:
    def test_reads_key_value_pairs(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=bar\nBAZ=qux\n")
        assert read_env(env_file) == {"FOO": "bar", "BAZ": "qux"}

    def test_ignores_comments_and_blanks(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nFOO=bar\n# another\n")
        assert read_env(env_file) == {"FOO": "bar"}

    def test_returns_empty_for_missing_file(self, tmp_path):
        assert read_env(tmp_path / "nonexistent") == {}

    def test_handles_values_with_equals(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("URL=http://host:1234/path?a=b\n")
        assert read_env(env_file) == {"URL": "http://host:1234/path?a=b"}


# ---------------------------------------------------------------------------
# env_writer — write_env
# ---------------------------------------------------------------------------


class TestWriteEnv:
    def test_creates_new_file(self, tmp_path):
        env_file = tmp_path / ".env"
        write_env(env_file, {"KEY": "value"})
        assert env_file.read_text() == "KEY=value\n"

    def test_merges_into_existing(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING=keep\nUPDATE=old\n")
        write_env(env_file, {"UPDATE": "new", "ADDED": "fresh"})
        content = env_file.read_text()
        assert "EXISTING=keep" in content
        assert "UPDATE=new" in content
        assert "ADDED=fresh" in content
        assert "UPDATE=old" not in content

    def test_preserves_comments(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# Header comment\nFOO=bar\n# Footer\n")
        write_env(env_file, {"FOO": "baz"})
        lines = env_file.read_text().splitlines()
        assert lines[0] == "# Header comment"
        assert lines[1] == "FOO=baz"
        assert lines[2] == "# Footer"

    def test_sets_secure_permissions(self, tmp_path):
        env_file = tmp_path / ".env"
        write_env(env_file, {"SECRET": "s3cret"})
        mode = oct(os.stat(env_file).st_mode & 0o777)
        assert mode == "0o600"

    def test_no_op_with_empty_updates(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEEP=this\n")
        write_env(env_file, {})
        assert env_file.read_text() == "KEEP=this\n"

    def test_idempotent_double_write(self, tmp_path):
        env_file = tmp_path / ".env"
        write_env(env_file, {"A": "1", "B": "2"})
        first = env_file.read_text()
        write_env(env_file, {"A": "1", "B": "2"})
        assert env_file.read_text() == first

    def test_atomic_no_partial_writes(self, tmp_path):
        """If rename fails, original file should be untouched."""
        env_file = tmp_path / ".env"
        env_file.write_text("ORIGINAL=yes\n")
        tmp_file = env_file.with_suffix(".env.tmp")

        with patch("cortex.cli.env_writer.os.chmod"):
            with patch.object(Path, "rename", side_effect=OSError("disk full")):
                with pytest.raises(OSError):
                    write_env(env_file, {"NEW": "val"})
        # Original untouched
        assert env_file.read_text() == "ORIGINAL=yes\n"


# ---------------------------------------------------------------------------
# setup_wizard — step functions
# ---------------------------------------------------------------------------


class TestStepDataDir:
    def test_creates_directory(self, tmp_path):
        from cortex.cli.setup_wizard import _step_data_dir
        from cortex.core.config import CortexConfig

        data_dir = tmp_path / "cortex_test"
        config = CortexConfig(data_dir=data_dir)
        _step_data_dir(config)
        assert data_dir.exists()

    def test_existing_directory_ok(self, tmp_path):
        from cortex.cli.setup_wizard import _step_data_dir
        from cortex.core.config import CortexConfig

        data_dir = tmp_path / "cortex_test"
        data_dir.mkdir()
        config = CortexConfig(data_dir=data_dir)
        _step_data_dir(config)  # should not raise


class TestStepLlm:
    def test_auto_mode_preserves_existing(self, tmp_path):
        from cortex.cli.setup_wizard import _step_llm
        from cortex.core.config import CortexConfig

        config = CortexConfig(
            data_dir=tmp_path,
            llm_model="claude-sonnet-4-20250514",
            llm_api_key="sk-test",
            llm_provider="anthropic",
        )

        with patch("cortex.cli.setup_wizard._test_llm", return_value=(True, "connected")):
            result = _step_llm(config, auto=True)

        assert result["CORTEX_LLM_MODEL"] == "claude-sonnet-4-20250514"
        assert result["CORTEX_LLM_API_KEY"] == "sk-test"
        assert result["CORTEX_LLM_PROVIDER"] == "anthropic"

    def test_auto_mode_empty_if_not_configured(self, tmp_path):
        from cortex.cli.setup_wizard import _step_llm
        from cortex.core.config import CortexConfig

        config = CortexConfig(data_dir=tmp_path)
        result = _step_llm(config, auto=True)
        assert result == {}


class TestStepEmbeddings:
    def test_no_provider_auto_mode_skips(self, tmp_path):
        from cortex.cli.setup_wizard import _step_embeddings
        from cortex.core.config import CortexConfig

        config = CortexConfig(data_dir=tmp_path)
        with patch("cortex.services.embeddings.create_embedding_provider", return_value=None):
            _step_embeddings(config, auto=True)

    def test_no_provider_interactive_decline(self, tmp_path):
        from cortex.cli.setup_wizard import _step_embeddings
        from cortex.core.config import CortexConfig

        config = CortexConfig(data_dir=tmp_path)
        with (
            patch("cortex.services.embeddings.create_embedding_provider", return_value=None),
            patch("cortex.cli.setup_wizard.typer.confirm", return_value=False),
        ):
            _step_embeddings(config, auto=False)  # should not raise

    def test_auto_install_called_on_accept(self, tmp_path):
        from cortex.cli.setup_wizard import _step_embeddings
        from cortex.core.config import CortexConfig

        config = CortexConfig(data_dir=tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: some pip error\n"

        with (
            patch("cortex.services.embeddings.create_embedding_provider", return_value=None),
            patch("cortex.cli.setup_wizard.typer.confirm", return_value=True),
            patch("subprocess.run", return_value=mock_result) as mock_run,
        ):
            _step_embeddings(config, auto=False)
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[1:3] == ["-m", "pip"]
            assert "sentence-transformers>=3.4" in args


class TestStepVerify:
    def test_handles_no_services_running(self, tmp_path):
        from cortex.cli.setup_wizard import _step_verify
        from cortex.core.config import CortexConfig

        config = CortexConfig(data_dir=tmp_path, host="127.0.0.1", port=19999)
        with patch("cortex.cli.setup_wizard.time.sleep"):
            with patch("cortex.cli.setup_wizard._probe_http", return_value=False):
                _step_verify(config)  # should not raise


# ---------------------------------------------------------------------------
# install.py — plist templates include CORTEX_DATA_DIR
# ---------------------------------------------------------------------------


class TestPlistDataDir:
    def test_mcp_plist_includes_data_dir(self):
        import plistlib

        from cortex.cli.install import render_mcp_plist
        from cortex.core.config import CortexConfig

        config = CortexConfig(data_dir=Path("/custom/data"))
        with patch("cortex.cli.install.detect_cortex_binary", return_value="/usr/bin/cortex"):
            content = render_mcp_plist(config, "/usr/bin/cortex")
        plist = plistlib.loads(content.encode())
        assert plist["EnvironmentVariables"]["CORTEX_DATA_DIR"] == "/custom/data"

    def test_dashboard_plist_includes_data_dir(self):
        import plistlib

        from cortex.cli.install import render_dashboard_plist
        from cortex.core.config import CortexConfig

        config = CortexConfig(data_dir=Path("/custom/data"))
        with patch("cortex.cli.install.detect_cortex_binary", return_value="/usr/bin/cortex"):
            content = render_dashboard_plist(config, "/usr/bin/cortex")
        plist = plistlib.loads(content.encode())
        assert plist["EnvironmentVariables"]["CORTEX_DATA_DIR"] == "/custom/data"
