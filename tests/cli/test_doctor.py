"""Tests for the ``cortex doctor`` sub-app (Bundle 8 / B1).

``cortex doctor unlock`` is the guided escape hatch for recovering from a
stale graph.db lock. The bulk of the recovery logic lives in the shared
``_auto_recover_stale_lock`` helper in ``cortex.db.graph_store`` (tested by
``TestAutoRecoverStaleLock`` in ``tests/db/test_graph_store_locking.py``);
these tests focus on the CLI layer's behavior: dry-run, force, refusing a
live holder, idempotence, and no-marker handling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cortex.cli.main as cli_mod
from cortex.cli.main import app
from cortex.db.graph_store import _marker_path_for

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch):
    """Point the CLI at an isolated data dir and reset cached singletons."""
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None
    cli_mod._mcp_client = None
    cli_mod._mcp_probe_done = False
    monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path))
    yield
    cli_mod._store = None
    cli_mod._pipeline = None
    cli_mod._learner = None
    cli_mod._mcp_client = None
    cli_mod._mcp_probe_done = False


def _write_stale_marker(tmp_path: Path, pid: int = 2_000_000) -> tuple[Path, Path]:
    """Create the on-disk remnants of a crashed GraphStore holder.

    Returns ``(marker_path, rocksdb_lock_path)``.
    """
    db = tmp_path / "graph.db"
    db.mkdir(parents=True, exist_ok=True)
    marker = _marker_path_for(db)
    marker.write_text(
        json.dumps(
            {
                "pid": pid,
                "cmdline": "cortex serve --transport mcp-http (crashed)",
                "acquired_at": "2020-01-01T00:00:00+00:00",
            }
        )
    )
    rocksdb_lock = db / "LOCK"
    rocksdb_lock.write_text("")
    return marker, rocksdb_lock


class TestDoctorUnlock:
    def test_unlock_with_no_marker_reports_and_exits_0(self, tmp_path: Path):
        """No marker and no LOCK file → clean no-op."""
        result = runner.invoke(app, ["doctor", "unlock"])
        assert result.exit_code == 0
        combined = result.output + (result.stderr or "")
        assert "nothing to unlock" in combined.lower()

    def test_unlock_with_stale_marker_removes_marker_and_rocksdb_lock(
        self, tmp_path: Path
    ):
        """Dead holder PID → both files removed, exit 0."""
        marker, rocksdb_lock = _write_stale_marker(tmp_path)
        assert marker.exists()
        assert rocksdb_lock.exists()

        result = runner.invoke(app, ["doctor", "unlock"])
        assert result.exit_code == 0, (
            f"unexpected exit: {result.output} {result.stderr}"
        )
        assert not marker.exists(), "marker should be removed"
        assert not rocksdb_lock.exists(), "rocksdb LOCK should be removed"
        combined = result.output + (result.stderr or "")
        assert "Unlocked" in combined

    def test_unlock_refuses_to_unlock_living_holder(
        self, tmp_path: Path
    ):
        """Live holder (this process's own PID) → refuse, exit 1."""
        # Write a marker claiming THIS process is the holder with a matching
        # cmdline so the PID-reuse detection also matches.
        from cortex.db.graph_store import _current_cmdline

        db = tmp_path / "graph.db"
        db.mkdir(parents=True, exist_ok=True)
        marker = _marker_path_for(db)
        marker.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "cmdline": _current_cmdline(),
                    "acquired_at": "2020-01-01T00:00:00+00:00",
                }
            )
        )

        result = runner.invoke(app, ["doctor", "unlock"])
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "still running" in combined or "refusing" in combined
        # Marker must NOT have been removed
        assert marker.exists()

    def test_unlock_dry_run_does_not_remove_files(self, tmp_path: Path):
        marker, rocksdb_lock = _write_stale_marker(tmp_path)

        result = runner.invoke(app, ["doctor", "unlock", "--dry-run"])
        assert result.exit_code == 0
        # Both files untouched
        assert marker.exists()
        assert rocksdb_lock.exists()
        combined = result.output + (result.stderr or "")
        assert "Dry run" in combined
        assert "2000000" in combined or "dead" in combined

    def test_unlock_force_bypasses_pid_check(self, tmp_path: Path):
        """--force skips liveness checks and removes files unconditionally.

        We use this process's own PID in the marker so the normal path
        would refuse; --force should overrule that.
        """
        db = tmp_path / "graph.db"
        db.mkdir(parents=True, exist_ok=True)
        marker = _marker_path_for(db)
        marker.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "cmdline": "fake cmdline that matches neither",
                    "acquired_at": "2020-01-01T00:00:00+00:00",
                }
            )
        )
        rocksdb_lock = db / "LOCK"
        rocksdb_lock.write_text("")

        result = runner.invoke(app, ["doctor", "unlock", "--force"])
        assert result.exit_code == 0, (
            f"unexpected exit: {result.output} {result.stderr}"
        )
        assert not marker.exists()
        assert not rocksdb_lock.exists()
        combined = result.output + (result.stderr or "")
        assert "Force-unlocked" in combined

    def test_unlock_unreadable_marker_reports_clearly(self, tmp_path: Path):
        """Marker file with chmod 000 → clear error, exit 1."""
        db = tmp_path / "graph.db"
        db.mkdir(parents=True, exist_ok=True)
        marker = _marker_path_for(db)
        marker.write_text('{"pid": 12345, "cmdline": "x"}')
        original_mode = marker.stat().st_mode
        marker.chmod(0o000)
        try:
            result = runner.invoke(app, ["doctor", "unlock"])
            assert result.exit_code == 1
            combined = result.output + (result.stderr or "")
            assert "unreadable" in combined.lower()
        finally:
            marker.chmod(original_mode)

    def test_unlock_idempotent(self, tmp_path: Path):
        """Running unlock twice in a row is safe — the second call is
        a no-op with a clean exit.
        """
        _write_stale_marker(tmp_path)

        first = runner.invoke(app, ["doctor", "unlock"])
        assert first.exit_code == 0
        second = runner.invoke(app, ["doctor", "unlock"])
        assert second.exit_code == 0
        combined = second.output + (second.stderr or "")
        assert "nothing to unlock" in combined.lower()

    def test_unlock_pid_reuse_refuses_without_force(self, tmp_path: Path):
        """Marker PID is alive (this process) but cmdline differs → PID
        reuse detection refuses, suggests --force.
        """
        db = tmp_path / "graph.db"
        db.mkdir(parents=True, exist_ok=True)
        marker = _marker_path_for(db)
        marker.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "cmdline": "cortex --impostor",
                    "acquired_at": "2020-01-01T00:00:00+00:00",
                }
            )
        )
        result = runner.invoke(app, ["doctor", "unlock"])
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "does NOT match" in combined or "reuse" in combined.lower()
        assert "--force" in combined
