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


class TestDoctorLogs:
    """Bundle 10.7 / F.4: ``cortex doctor logs`` inspects and rotates the
    LaunchAgent log files under ``~/.cortex/`` (or ``CORTEX_DATA_DIR``).
    """

    def _write_log(self, tmp_path: Path, name: str, content: str) -> Path:
        path = tmp_path / name
        path.write_text(content)
        return path

    def test_logs_summary_reports_sizes_and_counts(self, tmp_path: Path):
        """Default view shows size, line count, mtime, and status badge
        for each existing log file, and reports missing ones.
        """
        self._write_log(
            tmp_path, "mcp-http.log", "line1\nline2\nline3\n"
        )
        self._write_log(tmp_path, "mcp-http.err", "error1\nerror2\n")
        # dashboard.log and dashboard.err intentionally missing

        result = runner.invoke(app, ["doctor", "logs"])
        assert result.exit_code == 0, (
            f"unexpected exit: {result.output} {result.stderr}"
        )
        out = result.output
        # Both present files are listed with a status and line count
        assert "mcp-http.log" in out
        assert "mcp-http.err" in out
        assert "3 lines" in out
        assert "2 lines" in out
        assert "GREEN" in out  # small files → green badge
        # Missing files reported as not present
        assert "dashboard.log" in out
        assert "dashboard.err" in out
        assert "(not present)" in out

    def test_logs_tail_shows_last_n_lines(self, tmp_path: Path):
        """``--tail N`` shows the last N lines of each existing file,
        skipping missing files without error.
        """
        content = "".join(f"line{i}\n" for i in range(1, 11))  # 10 lines
        self._write_log(tmp_path, "mcp-http.log", content)

        result = runner.invoke(app, ["doctor", "logs", "--tail", "3"])
        assert result.exit_code == 0, (
            f"unexpected exit: {result.output} {result.stderr}"
        )
        out = result.output
        # Header with the tailed path
        assert "mcp-http.log" in out
        # Last 3 lines present
        assert "line8" in out
        assert "line9" in out
        assert "line10" in out
        # Earlier lines absent
        assert "line1\n" not in out
        assert "line7" not in out

    def test_logs_rotate_creates_old_and_truncates(self, tmp_path: Path):
        """``--rotate`` copies each non-empty log to ``<file>.old`` and
        truncates the live file to zero length.
        """
        original = "original content\nwith two lines\n"
        log = self._write_log(tmp_path, "mcp-http.log", original)
        assert log.stat().st_size > 0

        result = runner.invoke(app, ["doctor", "logs", "--rotate"])
        assert result.exit_code == 0, (
            f"unexpected exit: {result.output} {result.stderr}"
        )
        # Live file is empty
        assert log.exists()
        assert log.stat().st_size == 0
        # Backup exists and contains the original content
        backup = tmp_path / "mcp-http.log.old"
        assert backup.exists()
        assert backup.read_text() == original
        # Output mentions rotation
        assert "Rotated" in result.output

    def test_logs_rotate_overwrites_previous_old(self, tmp_path: Path):
        """Running ``--rotate`` twice overwrites the previous ``.old``
        with the most recent rotation; no stacking of ``.old.old``.
        """
        log = self._write_log(tmp_path, "mcp-http.log", "first\n")
        # First rotation
        r1 = runner.invoke(app, ["doctor", "logs", "--rotate"])
        assert r1.exit_code == 0
        assert (tmp_path / "mcp-http.log.old").read_text() == "first\n"
        assert log.stat().st_size == 0

        # Write new content and rotate again
        log.write_text("second\n")
        r2 = runner.invoke(app, ["doctor", "logs", "--rotate"])
        assert r2.exit_code == 0
        assert (tmp_path / "mcp-http.log.old").read_text() == "second\n"
        assert log.stat().st_size == 0
        # No .old.old stacking
        assert not (tmp_path / "mcp-http.log.old.old").exists()

    def test_logs_missing_files_graceful(self, tmp_path: Path):
        """With no log files present, all three views exit cleanly
        without raising.
        """
        result = runner.invoke(app, ["doctor", "logs"])
        assert result.exit_code == 0, (
            f"summary failed: {result.output} {result.stderr}"
        )
        assert "(not present)" in result.output

        result = runner.invoke(app, ["doctor", "logs", "--tail", "10"])
        assert result.exit_code == 0, (
            f"tail failed: {result.output} {result.stderr}"
        )

        result = runner.invoke(app, ["doctor", "logs", "--rotate"])
        assert result.exit_code == 0, (
            f"rotate failed: {result.output} {result.stderr}"
        )
        # Rotate with nothing to do should report so
        assert "not present" in result.output or "Skipped" in result.output

    def test_logs_rotate_skips_empty_files(self, tmp_path: Path):
        """An empty live log should not produce a ``.old`` backup."""
        log = self._write_log(tmp_path, "mcp-http.log", "")
        assert log.stat().st_size == 0

        result = runner.invoke(app, ["doctor", "logs", "--rotate"])
        assert result.exit_code == 0
        assert not (tmp_path / "mcp-http.log.old").exists()
        assert "already empty" in result.output or "Skipped" in result.output
