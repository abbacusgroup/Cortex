"""Tests for the GraphStore single-writer lock + PID marker mechanism.

These tests verify Phase 1 of the Oxigraph lock fix:
- A PID marker file is written next to graph.db when the store opens
- The marker is removed on clean close()
- Concurrent open attempts raise StoreLockedError naming the holder
- Stale markers (PID no longer alive) are reported with is_stale=True
- Corrupted, unreadable, or missing markers are handled gracefully
- In-memory mode (path=None) writes no marker
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cortex.core.errors import StoreError, StoreLockedError
from cortex.db.graph_store import (
    GraphStore,
    _marker_path_for,
    _pid_alive,
    _read_marker,
)

# ─── Helpers ──────────────────────────────────────────────────────────────


def _spawn_holder_subprocess(db_path: Path, sentinel: Path) -> subprocess.Popen:
    """Spawn a subprocess that opens the GraphStore and sleeps until killed.

    Touches `sentinel` once the store is open so the parent test can wait for it.
    """
    code = (
        "import sys, time; sys.path.insert(0, 'src');"
        "from pathlib import Path;"
        "from cortex.db.graph_store import GraphStore;"
        f"s = GraphStore(Path({str(db_path)!r}));"
        f"Path({str(sentinel)!r}).write_text('ready');"
        "time.sleep(60)"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
    )


def _wait_for_sentinel(sentinel: Path, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if sentinel.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"Sentinel {sentinel} never appeared")


# ─── 1.B — PID marker write/cleanup ───────────────────────────────────────


class TestPidMarkerLifecycle:
    def test_writes_marker_on_open(self, tmp_path: Path):
        db = tmp_path / "g.db"
        store = GraphStore(db)
        marker = _marker_path_for(db)
        try:
            assert marker.exists(), "Marker file should be created on successful open"
            data = json.loads(marker.read_text())
            assert data["pid"] == os.getpid()
            assert isinstance(data["cmdline"], str) and len(data["cmdline"]) > 0
            assert "acquired_at" in data
        finally:
            store.close()

    def test_removes_marker_on_close(self, tmp_path: Path):
        db = tmp_path / "g.db"
        store = GraphStore(db)
        marker = _marker_path_for(db)
        assert marker.exists()
        store.close()
        assert not marker.exists(), "Marker should be removed on close()"

    def test_close_is_idempotent(self, tmp_path: Path):
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.close()
        store.close()  # second call should not raise

    def test_context_manager_cleans_marker(self, tmp_path: Path):
        db = tmp_path / "g.db"
        marker = _marker_path_for(db)
        with GraphStore(db) as store:
            assert marker.exists()
            assert store is not None
        assert not marker.exists()

    def test_in_memory_mode_writes_no_marker(self, tmp_path: Path):
        # path=None — no on-disk store, no marker should appear anywhere
        store = GraphStore(path=None)
        try:
            assert not (tmp_path / "g.db.lock").exists()
            # Sanity-check the temp directory is empty of any lock files
            lock_files = list(tmp_path.rglob("*.lock"))
            assert lock_files == []
        finally:
            store.close()

    def test_marker_path_is_sibling_not_inside_db_dir(self, tmp_path: Path):
        db = tmp_path / "g.db"
        store = GraphStore(db)
        try:
            marker = _marker_path_for(db)
            assert marker.parent == db.parent
            assert marker.name == "g.db.lock"
            # Marker is NOT inside the RocksDB directory
            assert not (db / "g.db.lock").exists()
        finally:
            store.close()


# ─── 1.C — Lock failure raises StoreLockedError with holder info ─────────


class TestLockFailureRaisesStoreLockedError:
    def test_concurrent_open_in_subprocess_reports_subprocess_pid(
        self, tmp_path: Path
    ):
        db = tmp_path / "g.db"
        sentinel = tmp_path / "ready"
        proc = _spawn_holder_subprocess(db, sentinel)
        try:
            _wait_for_sentinel(sentinel)
            with pytest.raises(StoreLockedError) as exc_info:
                GraphStore(db)
            err = exc_info.value
            assert err.holder_pid == proc.pid
            assert err.holder_cmdline is not None
            assert "python" in err.holder_cmdline.lower() or "cortex" in (
                err.holder_cmdline or ""
            )
            assert err.is_stale is False
            # The error message should be actionable
            s = str(err)
            assert str(proc.pid) in s
            assert "kill" in s.lower() or "stop" in s.lower()
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_locked_error_is_subclass_of_store_error(self, tmp_path: Path):
        """Code that catches StoreError must catch StoreLockedError too."""
        db = tmp_path / "g.db"
        sentinel = tmp_path / "ready"
        proc = _spawn_holder_subprocess(db, sentinel)
        try:
            _wait_for_sentinel(sentinel)
            caught = False
            try:
                GraphStore(db)
            except StoreError:
                caught = True
            assert caught
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_does_NOT_silently_fall_back_to_in_memory(self, tmp_path: Path):  # noqa: N802
        """Regression: the old code silently created an in-memory store on lock fail.

        This test asserts the new code raises instead.
        """
        db = tmp_path / "g.db"
        sentinel = tmp_path / "ready"
        proc = _spawn_holder_subprocess(db, sentinel)
        try:
            _wait_for_sentinel(sentinel)
            with pytest.raises(StoreLockedError):
                GraphStore(db)
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_stale_marker_with_dead_pid_reports_is_stale(self, tmp_path: Path):
        """Write a marker by hand for a non-existent PID, simulate a stuck lock.

        We can't actually create a stuck Oxigraph lock from Python, so we
        simulate by combining (a) a stale marker on disk with (b) the lock
        being held by a real subprocess. The marker contents we read should
        report `is_stale=True` only when the recorded PID is dead.

        For this test we directly call _read_marker + _pid_alive logic to
        verify the staleness detection without needing a real OSError.
        """
        marker = tmp_path / "g.db.lock"
        marker.write_text(
            json.dumps({"pid": 999999, "cmdline": "ghost", "acquired_at": "2026-01-01"})
        )
        data = _read_marker(marker)
        assert data is not None
        assert data["pid"] == 999999
        assert _pid_alive(999999) is False  # PID 999999 should never exist on macOS

    def test_corrupted_marker_handled_gracefully(self, tmp_path: Path):
        """Malformed JSON in the marker file does not cause a Python traceback."""
        marker = tmp_path / "g.db.lock"
        marker.write_text("not-valid-json{{{")
        data = _read_marker(marker)
        assert data is not None
        assert data["pid"] is None
        assert "malformed" in (data["cmdline"] or "").lower()

    def test_missing_marker_returns_none(self, tmp_path: Path):
        marker = tmp_path / "g.db.lock"
        assert not marker.exists()
        assert _read_marker(marker) is None

    def test_marker_with_non_dict_json(self, tmp_path: Path):
        marker = tmp_path / "g.db.lock"
        marker.write_text(json.dumps([1, 2, 3]))
        data = _read_marker(marker)
        assert data is not None
        assert data["pid"] is None
        assert "malformed" in (data["cmdline"] or "").lower()

    def test_locked_error_with_no_marker_file(self, tmp_path: Path):
        """If lock fails AND no marker exists, error reports holder_unknown."""
        db = tmp_path / "g.db"
        sentinel = tmp_path / "ready"
        proc = _spawn_holder_subprocess(db, sentinel)
        try:
            _wait_for_sentinel(sentinel)
            # Delete the marker that the subprocess wrote, simulating a lock
            # held by something that didn't write a marker.
            marker = _marker_path_for(db)
            if marker.exists():
                marker.unlink()
            with pytest.raises(StoreLockedError) as exc_info:
                GraphStore(db)
            err = exc_info.value
            assert err.holder_pid is None
            assert "no marker file found" in str(err) or "unknown" in str(err).lower()
        finally:
            proc.terminate()
            proc.wait(timeout=5)


# ─── Cross-process: marker survives subprocess crash ───────────────────────


class TestCrashRecovery:
    def test_marker_persists_after_sigkill(self, tmp_path: Path):
        """SIGKILL the holder, verify the marker is left behind (atexit didn't run)."""
        db = tmp_path / "g.db"
        sentinel = tmp_path / "ready"
        proc = _spawn_holder_subprocess(db, sentinel)
        try:
            _wait_for_sentinel(sentinel)
            marker = _marker_path_for(db)
            assert marker.exists()
            proc.kill()
            proc.wait(timeout=5)
            # Marker should still be there — atexit doesn't run on SIGKILL
            assert marker.exists()
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)

    def test_marker_does_not_block_open_when_pid_is_dead(self, tmp_path: Path):
        """A stale marker alone (without the actual Oxigraph lock) should NOT
        prevent opening the store. The marker is informational; the actual
        gate is RocksDB's file lock.
        """
        db = tmp_path / "g.db"
        # Write a stale marker for a PID that doesn't exist
        marker = _marker_path_for(db)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"pid": 999998, "cmdline": "ghost", "acquired_at": "2026-01-01"})
        )
        # Open should succeed and the marker should be overwritten by current PID
        store = GraphStore(db)
        try:
            data = json.loads(marker.read_text())
            assert data["pid"] == os.getpid()
        finally:
            store.close()


# ─── Test fixture pollution guard ─────────────────────────────────────────


class TestNoLockFilesLeakBetweenTests:
    """Regression guard: existing tests use tmp_path. Verify each test cleans up."""

    def test_close_removes_marker(self, tmp_path: Path):
        db = tmp_path / "g.db"
        store = GraphStore(db)
        store.close()
        # No .lock files anywhere in tmp_path
        assert list(tmp_path.rglob("*.lock")) == []

    def test_context_manager_removes_marker(self, tmp_path: Path):
        db = tmp_path / "g.db"
        with GraphStore(db):
            pass
        assert list(tmp_path.rglob("*.lock")) == []


# ─── Bundle 1.1 — PID reuse race protection ────────────────────────────────


class TestPidReuseRaceProtection:
    """When the marker's PID is alive but the live process has a DIFFERENT
    cmdline than the marker recorded, the lock holder is ambiguous: either the
    OS reused the PID for an unrelated process, OR the original process did
    something weird like exec into a different binary. Either way, we should
    NOT report it as a normal lock — we should warn that the marker is stale.
    """

    def test_pid_reuse_detected_when_cmdline_mismatch(self, tmp_path: Path):
        """Simulate PID reuse: a real subprocess holds the lock, but we
        manually overwrite the marker with a fake cmdline. The next open
        should detect that the live cmdline differs from the marker's recorded
        cmdline and set ``is_pid_reuse=True``.
        """
        db = tmp_path / "g.db"
        sentinel = tmp_path / "ready"
        proc = _spawn_holder_subprocess(db, sentinel)
        try:
            _wait_for_sentinel(sentinel)
            # Overwrite the marker with the SAME PID but a fabricated cmdline
            marker = _marker_path_for(db)
            marker.write_text(
                json.dumps(
                    {
                        "pid": proc.pid,
                        "cmdline": "totally different fake cmdline that no live process has",
                        "acquired_at": "2026-01-01T00:00:00Z",
                    }
                )
            )

            with pytest.raises(StoreLockedError) as exc_info:
                GraphStore(db)
            err = exc_info.value
            assert err.holder_pid == proc.pid
            assert err.is_pid_reuse is True, (
                "expected is_pid_reuse=True when marker cmdline differs from live cmdline"
            )
            assert err.is_stale is False
            s = str(err)
            assert "reused the PID" in s or "does NOT match" in s
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_no_pid_reuse_flag_when_cmdline_matches(self, tmp_path: Path):
        """Sanity: when the marker's cmdline DOES match the live process,
        is_pid_reuse must be False (this is the normal lock-conflict case).
        """
        db = tmp_path / "g.db"
        sentinel = tmp_path / "ready"
        proc = _spawn_holder_subprocess(db, sentinel)
        try:
            _wait_for_sentinel(sentinel)
            # Don't touch the marker — the subprocess wrote it correctly
            with pytest.raises(StoreLockedError) as exc_info:
                GraphStore(db)
            err = exc_info.value
            assert err.is_pid_reuse is False
            assert err.is_stale is False
            assert err.cmdline_unknown is False
            # Normal "stop the conflicting process" message
            assert "Stop the conflicting process" in str(err)
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_cmdline_unknown_flag_set_when_process_cmdline_returns_none(
        self, tmp_path: Path, monkeypatch
    ):
        """Bundle 10.7 / B.2: when ``_process_cmdline`` returns None for a
        live PID (ps timeout, /proc race, missing permissions), we cannot
        verify the PID-match. Flag ``cmdline_unknown=True``, keep
        ``is_stale=False`` and ``is_pid_reuse=False`` so auto-recovery
        stays disabled, and the error message must say so.
        """
        db = tmp_path / "g.db"
        sentinel = tmp_path / "ready"
        proc = _spawn_holder_subprocess(db, sentinel)
        try:
            _wait_for_sentinel(sentinel)
            # Force _process_cmdline to return None for the live PID
            import cortex.db.graph_store as gs

            monkeypatch.setattr(gs, "_process_cmdline", lambda pid: None)

            with pytest.raises(StoreLockedError) as exc_info:
                GraphStore(db)
            err = exc_info.value
            assert err.holder_pid == proc.pid
            assert err.is_stale is False, (
                "PID is alive, so is_stale must be False"
            )
            assert err.is_pid_reuse is False, (
                "cmdline is unknown, so we cannot claim reuse"
            )
            assert err.cmdline_unknown is True, (
                "expected cmdline_unknown=True when _process_cmdline returns None"
            )
            s = str(err)
            assert "could NOT be read" in s or "cmdline" in s.lower()
            assert "--force" in s
            # context propagation
            assert err.context.get("cmdline_unknown") is True
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_cmdline_unknown_does_not_trigger_auto_recovery(
        self, tmp_path: Path, monkeypatch
    ):
        """Bundle 10.7 / B.2: the auto-recovery path in ``GraphStore.__init__``
        gates on ``is_stale`` — with ``cmdline_unknown=True`` and PID alive,
        ``is_stale`` must remain False so auto-recovery does NOT fire. The
        marker file must still exist after the failed open.
        """
        db = tmp_path / "g.db"
        sentinel = tmp_path / "ready"
        proc = _spawn_holder_subprocess(db, sentinel)
        try:
            _wait_for_sentinel(sentinel)
            import cortex.db.graph_store as gs

            monkeypatch.setattr(gs, "_process_cmdline", lambda pid: None)

            marker = _marker_path_for(db)
            assert marker.exists()  # sanity — subprocess wrote it

            with pytest.raises(StoreLockedError):
                GraphStore(db)

            # Marker must still be present — auto-recovery must NOT have
            # fired. If it had, the marker would be gone.
            assert marker.exists(), (
                "marker was removed — auto-recovery fired despite "
                "cmdline_unknown (regression)"
            )
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_cmdline_unknown_false_when_marker_has_no_cmdline(
        self, tmp_path: Path, monkeypatch
    ):
        """When the marker itself records no cmdline, there's nothing to
        verify against — ``cmdline_unknown`` must stay False even if
        ``_process_cmdline`` also returns None. Otherwise every legacy
        marker without a cmdline field would be flagged as uncertain.
        """
        db = tmp_path / "g.db"
        db.mkdir(parents=True, exist_ok=True)
        marker = _marker_path_for(db)
        marker.write_text(
            json.dumps({"pid": os.getpid(), "acquired_at": "2026-01-01"})
        )
        # Create a dummy LOCK file so the OSError path fires
        (db / "LOCK").write_text("")

        import cortex.db.graph_store as gs

        monkeypatch.setattr(gs, "_process_cmdline", lambda pid: None)
        # Force the open to fail with a lock-style OSError
        real_store = gs.ox.Store

        def fake_store(path):
            raise OSError(f"While lock file: {path}/LOCK: Resource temporarily unavailable")

        monkeypatch.setattr(gs.ox, "Store", fake_store)
        try:
            with pytest.raises(StoreLockedError) as exc_info:
                GraphStore(db)
            err = exc_info.value
            assert err.cmdline_unknown is False, (
                "cmdline_unknown must stay False when marker has no cmdline "
                "recorded — there's nothing to verify against"
            )
        finally:
            monkeypatch.setattr(gs.ox, "Store", real_store)

    def test_stale_marker_with_dead_pid_takes_precedence_over_reuse(
        self, tmp_path: Path
    ):
        """When the marker's PID is dead, is_stale=True (PID reuse check is
        skipped because there's no live cmdline to compare).
        """
        # Write a marker for a definitely-dead PID
        db = tmp_path / "g.db"
        marker = _marker_path_for(db)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {"pid": 999997, "cmdline": "ghost cmdline", "acquired_at": "2026-01-01"}
            )
        )

        # Open should succeed (no actual lock conflict — we only have a stale
        # marker), and overwrite the marker with the current PID
        store = GraphStore(db)
        try:
            data = json.loads(marker.read_text())
            assert data["pid"] == os.getpid()
        finally:
            store.close()

    def test_stale_error_includes_cleanup_hint(self, tmp_path: Path):
        """When we DO surface a StoreLockedError with is_stale=True, the
        message should include the manual cleanup command for both files.
        This is verified by raising the error directly via the helper.
        """
        from cortex.db.graph_store import _raise_locked_error

        # Write a stale marker (PID 999996 is definitely dead on macOS)
        db = tmp_path / "g.db"
        marker = _marker_path_for(db)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {"pid": 999996, "cmdline": "ghost", "acquired_at": "2026-01-01"}
            )
        )

        with pytest.raises(StoreLockedError) as exc_info:
            _raise_locked_error(db, marker, OSError("lock hold by current process"))
        err = exc_info.value
        assert err.is_stale is True
        s = str(err)
        # Cleanup hint mentions BOTH the marker file AND the RocksDB LOCK
        assert str(marker) in s
        assert f"{db}/LOCK" in s
        assert "rm" in s


# ─── Bundle 1.3 — Atomicity test for mid-capture failure ──────────────────


class TestMarkerEdgeCases:
    """Phase 1.B intended-failure tests that catch silent regressions in the
    PID marker mechanism: bad paths, hostile filesystems, concurrent same-process
    opens, regression guard against accidental os.kill() calls.
    """

    def test_marker_NOT_written_if_open_fails_for_non_lock_reason(  # noqa: N802
        self, tmp_path: Path
    ):
        """If ``ox.Store(path)`` raises for a non-lock reason (e.g. permission
        denied on the parent directory), no marker file should be created
        AND the error must be reported as a generic StoreError (not a
        misleading StoreLockedError).
        """
        from cortex.core.errors import StoreError, StoreLockedError

        # Create an unwritable parent directory so RocksDB's mkdir fails
        bad_parent = tmp_path / "readonly"
        bad_parent.mkdir()
        bad_parent.chmod(0o555)  # r-x, no write
        try:
            db = bad_parent / "g.db"
            with pytest.raises(StoreError) as exc_info:
                GraphStore(db)
            # Critical: this is NOT a StoreLockedError. The user must see
            # the real failure (permission denied), not a confusing
            # "graph DB is locked" message.
            assert not isinstance(exc_info.value, StoreLockedError), (
                "permission errors must NOT be reported as lock errors"
            )
            # The underlying OSError is preserved as the cause
            assert exc_info.value.__cause__ is not None
            assert "permission" in str(exc_info.value).lower()

            # No marker file should exist on disk
            marker = _marker_path_for(db)
            assert not marker.exists(), (
                "marker file should NOT be created when open fails for a "
                "non-lock reason"
            )
        finally:
            bad_parent.chmod(0o755)  # restore so cleanup can remove it

    def test_marker_cleanup_does_not_delete_other_files(self, tmp_path: Path):
        """``close()`` must only remove the specific graph.db.lock marker.
        Unrelated files in the same directory must be untouched.
        """
        # Sprinkle some unrelated files alongside what will become the marker
        (tmp_path / "unrelated_1.txt").write_text("keep me")
        (tmp_path / "unrelated_2.json").write_text('{"keep": true}')
        (tmp_path / "another.lock").write_text("not the cortex lock")

        db = tmp_path / "g.db"
        store = GraphStore(db)
        assert _marker_path_for(db).exists()
        store.close()

        # The graph.db.lock marker is gone
        assert not _marker_path_for(db).exists()
        # All unrelated files are still there
        assert (tmp_path / "unrelated_1.txt").read_text() == "keep me"
        assert (tmp_path / "unrelated_2.json").exists()
        assert (tmp_path / "another.lock").read_text() == "not the cortex lock"

    def test_marker_write_failure_does_not_break_open(self, tmp_path: Path):
        """If the marker directory is read-only at the time of the marker
        write, the GraphStore should still open successfully (the marker is
        best-effort, not load-bearing) and log a warning.
        """
        # Create the DB dir, then make the parent read-only AFTER ox.Store can
        # create the DB but BEFORE the marker write. Easiest: monkeypatch
        # _write_marker to return False, simulating a write failure.
        import cortex.db.graph_store as gs_mod

        def fake_write_marker(marker_path: Path) -> bool:
            return False  # Simulate marker write failure

        original = gs_mod._write_marker
        gs_mod._write_marker = fake_write_marker  # type: ignore[assignment]
        try:
            db = tmp_path / "g.db"
            # Should still succeed despite marker write failure
            store = GraphStore(db)
            try:
                # The store object is functional
                assert store.triple_count >= 0
                # The marker file does NOT exist (write was simulated as failed)
                assert not _marker_path_for(db).exists()
            finally:
                store.close()
        finally:
            gs_mod._write_marker = original  # type: ignore[assignment]

    def test_pid_marker_cleaned_up_on_subprocess_clean_exit(
        self, tmp_path: Path
    ):
        """Subprocess opens GraphStore and exits cleanly via sys.exit (which
        triggers atexit cleanup). Parent verifies marker is removed.
        """
        db = tmp_path / "g.db"
        code = (
            "import sys; sys.path.insert(0, 'src');"
            "from pathlib import Path;"
            "from cortex.db.graph_store import GraphStore;"
            f"s = GraphStore(Path({str(db)!r}));"
            "sys.exit(0)"  # clean exit triggers atexit
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, (
            f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        # After clean exit, the marker should be gone (atexit ran)
        marker = _marker_path_for(db)
        assert not marker.exists(), (
            f"marker should be removed after subprocess clean exit, found: {marker}"
        )

    def test_concurrent_open_in_same_process_raises_locked_error(
        self, tmp_path: Path
    ):
        """Opening the same GraphStore path twice in the same Python process
        without closing the first instance should raise StoreLockedError.

        Note: GraphStore.close() drops the pyoxigraph reference and forces gc,
        which releases the RocksDB lock. So this test must hold a strong
        reference to the first instance for the duration of the second open.
        """
        db = tmp_path / "g.db"
        first = GraphStore(db)
        try:
            with pytest.raises(StoreLockedError) as exc_info:
                GraphStore(db)
            err = exc_info.value
            assert err.holder_pid == os.getpid()
            assert err.holder_cmdline is not None
        finally:
            first.close()

    def test_unreadable_marker_reports_specific_error(self, tmp_path: Path):
        """When the marker file exists but is unreadable (chmod 000), the
        error message must distinguish this from the 'no marker file' case
        so the user knows to check file permissions.
        """
        from cortex.db.graph_store import _raise_locked_error

        db = tmp_path / "g.db"
        marker = _marker_path_for(db)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"pid": 12345, "cmdline": "x"}))
        original_mode = marker.stat().st_mode
        marker.chmod(0o000)
        try:
            with pytest.raises(StoreLockedError) as exc_info:
                _raise_locked_error(db, marker, OSError("lock hold"))
            err = exc_info.value
            assert "unreadable" in str(err).lower()
            assert "permissions" in str(err).lower()
        finally:
            marker.chmod(original_mode)

    def test_lock_detection_never_calls_os_kill(self):
        """Regression guard: the lock-detection code path must NEVER call
        ``os.kill(pid, ...)`` to verify a holder, even when checking PID
        liveness. We use ``os.kill(pid, 0)`` which is the *signal-zero*
        liveness probe (sends NO signal), but never any non-zero signal.

        Verified by grepping the source for any non-zero kill() calls.
        """
        from pathlib import Path

        gs_source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "cortex"
            / "db"
            / "graph_store.py"
        ).read_text()
        # The only allowed os.kill call is the signal-0 liveness probe
        # inside _pid_alive. Find every os.kill( occurrence and verify it
        # uses signal 0.
        import re

        # Match os.kill(<anything>, <signal>)
        kill_calls = re.findall(r"os\.kill\([^,]+,\s*([^)]+)\)", gs_source)
        for sig in kill_calls:
            assert sig.strip() == "0", (
                f"graph_store.py contains os.kill(..., {sig.strip()}) — only "
                f"os.kill(pid, 0) is allowed for liveness checks. Lock "
                f"detection must NEVER send signals to other processes."
            )


class TestStoreCreateAtomicity:
    """``Store.create()`` does a dual-write to graph + SQLite. If the SQLite
    write fails AFTER the graph write succeeds, the graph write must be
    rolled back so we don't leak orphan triples.
    """

    def test_sqlite_failure_rolls_back_graph(self, tmp_path: Path):
        from cortex.core.config import CortexConfig
        from cortex.db.store import Store

        config = CortexConfig(data_dir=tmp_path)
        store = Store(config)
        store.initialize()
        try:
            # Get the graph triple count before
            triples_before = store.graph.triple_count
            objects_before = len(store.list_objects(limit=1000))

            # Monkey-patch ContentStore.insert to raise mid-write
            original_insert = store.content.insert

            def failing_insert(*args, **kwargs):
                raise RuntimeError("simulated SQLite failure")

            store.content.insert = failing_insert  # type: ignore[method-assign]

            # Attempt to create — should fail with SyncError
            from cortex.core.errors import SyncError

            with pytest.raises(SyncError):
                store.create(
                    obj_type="fix",
                    title="test fix that should not leak",
                    content="this should be rolled back",
                )

            # Restore the original insert
            store.content.insert = original_insert  # type: ignore[method-assign]

            # Verify NO partial state leaked into either store
            triples_after = store.graph.triple_count
            objects_after = len(store.list_objects(limit=1000))

            assert triples_after == triples_before, (
                f"graph triples leaked: before={triples_before}, after={triples_after}"
            )
            assert objects_after == objects_before, (
                f"SQLite objects leaked: before={objects_before}, after={objects_after}"
            )
        finally:
            store.close()


class TestMarkerNotRequiredForLockAcquisition:
    """Bundle 5 / A20: Phase 1.B weak-point coverage.

    Race condition: a process opens the graph store successfully (acquiring
    the RocksDB lock) but is killed BEFORE it writes the PID marker. A
    subsequent process opens the same path — it should succeed because the
    lock is free (process A is gone) and the absence of a marker is
    interpreted as "no holder" (not as "must have orphan RocksDB lock").

    We simulate this by monkeypatching ``_write_marker`` to return False on
    the first open (simulating the process dying before the marker write).
    Then we close the store (releasing the lock), restore the real
    ``_write_marker``, and open again. The second open must succeed and
    create its own marker.
    """

    def test_no_marker_plus_no_lock_means_clean_open(
        self, tmp_path: Path
    ):
        import cortex.db.graph_store as gs_mod

        db = tmp_path / "g.db"

        # ─── First open: simulate the marker write being interrupted ──
        original_write = gs_mod._write_marker

        def no_marker(marker_path: Path) -> bool:
            return False  # as if we crashed between ox.Store() and marker write

        gs_mod._write_marker = no_marker  # type: ignore[assignment]
        try:
            store_a = GraphStore(db)
            # Marker does NOT exist (write was aborted)
            assert not _marker_path_for(db).exists()
            # Store is still functional
            assert store_a.triple_count >= 0
            # Close releases the RocksDB lock
            store_a.close()
        finally:
            gs_mod._write_marker = original_write  # type: ignore[assignment]

        # Sanity: no marker, no lock held
        assert not _marker_path_for(db).exists()

        # ─── Second open: should succeed and write its own marker ─────
        store_b = GraphStore(db)
        try:
            assert _marker_path_for(db).exists(), (
                "second open should write its own marker"
            )
            data = json.loads(_marker_path_for(db).read_text())
            assert data["pid"] == os.getpid()
            assert store_b.triple_count >= 0
        finally:
            store_b.close()

    def test_stale_marker_but_no_lock_opens_and_overwrites(
        self, tmp_path: Path
    ):
        """Another marker-race variant: a stale marker file exists (from a
        dead process) but no actual RocksDB lock is held. The new process
        should open successfully and overwrite the marker with its own PID.
        """
        db = tmp_path / "g.db"
        marker = _marker_path_for(db)
        marker.parent.mkdir(parents=True, exist_ok=True)
        # Write a stale marker pointing at a bogus PID + cmdline
        stale_data = {
            "pid": 999_999,  # almost certainly not a live PID
            "cmdline": "cortex serve --transport mcp-http (crashed)",
            "acquired_at": "2020-01-01T00:00:00+00:00",
        }
        marker.write_text(json.dumps(stale_data))

        # Precondition: marker exists, no DB dir yet
        assert marker.exists()
        assert not db.exists()

        # The open should succeed because no one actually holds the RocksDB
        # lock — the stale marker alone doesn't prevent opening.
        store = GraphStore(db)
        try:
            # Marker was overwritten with our PID
            assert marker.exists()
            fresh = json.loads(marker.read_text())
            assert fresh["pid"] == os.getpid(), (
                f"marker should now point at current PID, got {fresh['pid']}"
            )
            assert fresh["cmdline"] != stale_data["cmdline"]
        finally:
            store.close()


class TestAutoRecoverStaleLock:
    """Bundle 8 / B1: ``GraphStore.__init__`` auto-recovers when it hits a
    lock error AND the marker says the holder is a dead PID (high-confidence
    stale case). The recovery removes the marker + RocksDB LOCK file and
    retries the open.
    """

    def _simulate_crashed_holder(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        """Set up the on-disk aftermath of a crashed GraphStore holder.

        Opens a real store briefly to initialize the pyoxigraph directory
        structure, closes it, then:

        - Rewrites the marker with a PID that's guaranteed dead (PID 0 or
          a very large number not in use).
        - Leaves a (new, fabricated) RocksDB LOCK file so the auto-recovery
          path exercises the LOCK-removal step.

        Returns ``(db, marker, rocksdb_lock)`` paths.

        NOTE: this does NOT simulate an actually-locked RocksDB store —
        only the on-disk remnants. That's sufficient for the recovery logic
        because we inject a lock error via monkeypatching ox.Store in the
        test itself when we need to test the recovery-triggered-by-lock flow.
        """
        db = tmp_path / "g.db"
        marker = _marker_path_for(db)
        store = GraphStore(db)
        store.close()
        # Rewrite marker with a dead PID
        dead_pid = 2_000_000  # very unlikely to exist
        marker.write_text(
            json.dumps(
                {
                    "pid": dead_pid,
                    "cmdline": "cortex serve --transport mcp-http (crashed)",
                    "acquired_at": "2020-01-01T00:00:00+00:00",
                }
            )
        )
        # Create a fake RocksDB LOCK file (the real one was removed by close())
        rocksdb_lock = db / "LOCK"
        rocksdb_lock.write_text("")
        return db, marker, rocksdb_lock

    def test_stale_marker_triggers_auto_recovery_on_retry(
        self, tmp_path: Path, monkeypatch
    ):
        """When ox.Store raises a lock error and the marker says the holder
        is dead, GraphStore auto-recovers (removes marker + LOCK) and
        retries the open successfully.
        """
        import pyoxigraph as ox_mod

        import cortex.db.graph_store as gs_mod

        db, marker, _rocksdb_lock = self._simulate_crashed_holder(tmp_path)

        # Make the FIRST ox.Store call raise a lock-shaped OSError, then let
        # the retry go through normally.
        original_store_cls = ox_mod.Store
        call_count = {"n": 0}

        def flaky_store(path_arg: str):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError(
                    "While lock file: /tmp/LOCK: Resource temporarily unavailable"
                )
            return original_store_cls(path_arg)

        monkeypatch.setattr(gs_mod.ox, "Store", flaky_store)

        store = GraphStore(db)
        try:
            # Auto-recovery removed the stale marker + LOCK then retried
            assert call_count["n"] == 2, "should have retried after recovery"
            assert marker.exists(), "new marker should be written by this process"
            fresh = json.loads(marker.read_text())
            assert fresh["pid"] == os.getpid()
        finally:
            store.close()

    def test_auto_recovery_removes_rocksdb_lock_file(
        self, tmp_path: Path, monkeypatch
    ):
        """Explicit assertion that the LOCK file inside graph.db/ is removed
        by the auto-recovery path.
        """
        import pyoxigraph as ox_mod

        import cortex.db.graph_store as gs_mod

        db, _marker, rocksdb_lock = self._simulate_crashed_holder(tmp_path)
        assert rocksdb_lock.exists()

        original_store_cls = ox_mod.Store
        seen_rocksdb_lock_exists: list[bool] = []

        call_count = {"n": 0}

        def flaky_store(path_arg: str):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError(
                    "While lock file: /tmp/LOCK: Resource temporarily unavailable"
                )
            # On the retry, record whether the LOCK file was removed
            seen_rocksdb_lock_exists.append(rocksdb_lock.exists())
            return original_store_cls(path_arg)

        monkeypatch.setattr(gs_mod.ox, "Store", flaky_store)

        store = GraphStore(db)
        try:
            assert seen_rocksdb_lock_exists == [False], (
                "auto-recovery should have removed graph.db/LOCK before retry"
            )
        finally:
            store.close()

    def test_auto_recovery_skipped_for_living_holder(self, tmp_path: Path):
        """When the marker's PID is alive (this process's own PID),
        auto-recovery is not triggered — StoreLockedError is raised.
        """
        db = tmp_path / "g.db"
        first = GraphStore(db)
        try:
            # Opening again in this process triggers the in-process lock
            # error. Because holder_pid == os.getpid() (alive), no recovery.
            with pytest.raises(StoreLockedError) as exc_info:
                GraphStore(db)
            err = exc_info.value
            assert err.holder_pid == os.getpid()
            assert err.is_stale is False
            assert "auto_recovery_attempted" not in err.context
        finally:
            first.close()

    def test_auto_recovery_skipped_for_pid_reuse(
        self, tmp_path: Path, monkeypatch
    ):
        """When the marker's PID is alive but its cmdline differs from the
        marker (PID reuse), auto-recovery is not triggered.
        """
        import pyoxigraph as ox_mod

        import cortex.db.graph_store as gs_mod

        db = tmp_path / "g.db"
        marker = _marker_path_for(db)

        # Write a marker with THIS process's PID but a bogus cmdline
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "cmdline": "some-other-program --impostor",
                    "acquired_at": "2020-01-01T00:00:00+00:00",
                }
            )
        )

        # Inject a one-shot lock error so we enter the error path
        original_store_cls = ox_mod.Store
        call_count = {"n": 0}

        def flaky_store(path_arg: str):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError(
                    "While lock file: /tmp/LOCK: Resource temporarily unavailable"
                )
            return original_store_cls(path_arg)

        monkeypatch.setattr(gs_mod.ox, "Store", flaky_store)

        with pytest.raises(StoreLockedError) as exc_info:
            GraphStore(db)
        err = exc_info.value
        assert err.is_pid_reuse is True
        assert err.is_stale is False
        # Crucially: no retry happened, so the call count is 1
        assert call_count["n"] == 1

    def test_auto_recovery_skipped_for_missing_marker(
        self, tmp_path: Path, monkeypatch
    ):
        """When the lock error happens but there's no marker file at all,
        auto-recovery is not triggered (we don't know enough to act safely).
        """
        import pyoxigraph as ox_mod

        import cortex.db.graph_store as gs_mod

        db = tmp_path / "g.db"
        marker = _marker_path_for(db)
        # Simulate "some files in db dir but no marker" by touching the dir
        db.mkdir(parents=True, exist_ok=True)
        assert not marker.exists()

        original_store_cls = ox_mod.Store
        call_count = {"n": 0}

        def flaky_store(path_arg: str):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError(
                    "While lock file: /tmp/LOCK: Resource temporarily unavailable"
                )
            return original_store_cls(path_arg)

        monkeypatch.setattr(gs_mod.ox, "Store", flaky_store)

        with pytest.raises(StoreLockedError) as exc_info:
            GraphStore(db)
        err = exc_info.value
        assert err.holder_pid is None
        assert err.is_stale is False
        # No retry happened
        assert call_count["n"] == 1

    def test_auto_recovery_skipped_for_unreadable_marker(
        self, tmp_path: Path, monkeypatch
    ):
        """When the marker exists but is unreadable (chmod 000), auto-recovery
        is not triggered — we can't tell who the holder is.
        """
        import pyoxigraph as ox_mod

        import cortex.db.graph_store as gs_mod

        db = tmp_path / "g.db"
        marker = _marker_path_for(db)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text('{"pid": 12345, "cmdline": "x"}')
        original_mode = marker.stat().st_mode
        marker.chmod(0o000)

        original_store_cls = ox_mod.Store
        call_count = {"n": 0}

        def flaky_store(path_arg: str):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError(
                    "While lock file: /tmp/LOCK: Resource temporarily unavailable"
                )
            return original_store_cls(path_arg)

        monkeypatch.setattr(gs_mod.ox, "Store", flaky_store)

        try:
            with pytest.raises(StoreLockedError) as exc_info:
                GraphStore(db)
            err = exc_info.value
            assert err.context.get("marker_unreadable") is True
            assert call_count["n"] == 1
        finally:
            marker.chmod(original_mode)

    def test_auto_recovery_re_check_race(
        self, tmp_path: Path, monkeypatch
    ):
        """Simulate a race: `_pid_alive` returns False in the initial
        staleness check, then True during the re-check inside
        `_auto_recover_stale_lock`. The original StoreLockedError should
        be raised without any file cleanup.
        """

        import cortex.db.graph_store as gs_mod

        db, marker, rocksdb_lock = self._simulate_crashed_holder(tmp_path)
        marker_data_before = marker.read_text()
        lock_data_before = rocksdb_lock.read_text()

        # _pid_alive will be called twice: once in _build_locked_error (→ False
        # so is_stale=True), then again in _auto_recover_stale_lock (→ True
        # to simulate a fresh process grabbing that PID).
        call_count = {"n": 0}

        def racy_pid_alive(pid: int) -> bool:
            call_count["n"] += 1
            # First call: mark stale. Second call: race aborts.
            return call_count["n"] >= 2

        monkeypatch.setattr(gs_mod, "_pid_alive", racy_pid_alive)

        def always_locked(path_arg: str):
            raise OSError(
                "While lock file: /tmp/LOCK: Resource temporarily unavailable"
            )

        monkeypatch.setattr(gs_mod.ox, "Store", always_locked)

        with pytest.raises(StoreLockedError) as exc_info:
            GraphStore(db)
        err = exc_info.value
        assert err.is_stale is True
        # Files were NOT touched because the re-check aborted cleanup
        assert marker.read_text() == marker_data_before
        assert rocksdb_lock.read_text() == lock_data_before

    def test_auto_recovery_logs_info_on_success(
        self, tmp_path: Path, monkeypatch, caplog
    ):
        """The successful auto-recovery path must emit a single INFO line
        naming the dead holder PID.
        """
        import logging

        import pyoxigraph as ox_mod

        import cortex.db.graph_store as gs_mod

        db, _marker, _rocksdb_lock = self._simulate_crashed_holder(tmp_path)

        original_store_cls = ox_mod.Store
        call_count = {"n": 0}

        def flaky_store(path_arg: str):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError(
                    "While lock file: /tmp/LOCK: Resource temporarily unavailable"
                )
            return original_store_cls(path_arg)

        monkeypatch.setattr(gs_mod.ox, "Store", flaky_store)

        with caplog.at_level(logging.INFO, logger="cortex.db.graph"):
            store = GraphStore(db)
            try:
                recovery_logs = [
                    rec
                    for rec in caplog.records
                    if "Auto-recovered stale lock" in rec.getMessage()
                ]
                assert len(recovery_logs) == 1
                assert "2000000" in recovery_logs[0].getMessage()
            finally:
                store.close()

    def test_stale_locked_error_message_mentions_doctor_unlock(
        self, tmp_path: Path, monkeypatch
    ):
        """When auto-recovery does NOT fire (e.g. pid_reuse case), the error
        message must still direct users at ``cortex doctor unlock``.
        """

        import cortex.db.graph_store as gs_mod

        db = tmp_path / "g.db"
        marker = _marker_path_for(db)
        marker.parent.mkdir(parents=True, exist_ok=True)
        # Dead PID case → will trigger auto-recovery. Force non-recoverable
        # by stubbing _auto_recover_stale_lock to return False.
        marker.write_text(
            json.dumps(
                {
                    "pid": 2_000_000,
                    "cmdline": "cortex serve (crashed)",
                    "acquired_at": "2020-01-01T00:00:00+00:00",
                }
            )
        )
        (db).mkdir(parents=True, exist_ok=True)
        (db / "LOCK").write_text("")

        def always_locked(path_arg: str):
            raise OSError(
                "While lock file: /tmp/LOCK: Resource temporarily unavailable"
            )

        # Pretend recovery always fails (returns False as if the race aborted)
        monkeypatch.setattr(gs_mod.ox, "Store", always_locked)
        monkeypatch.setattr(
            gs_mod, "_auto_recover_stale_lock", lambda *a, **kw: False
        )

        with pytest.raises(StoreLockedError) as exc_info:
            GraphStore(db)
        msg = str(exc_info.value)
        assert "cortex doctor unlock" in msg
        assert "Manual cleanup:" in msg  # fallback rm commands still present
