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
import multiprocessing
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

    def test_does_NOT_silently_fall_back_to_in_memory(self, tmp_path: Path):
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
            # Normal "stop the conflicting process" message
            assert "Stop the conflicting process" in str(err)
        finally:
            proc.terminate()
            proc.wait(timeout=5)

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

    def test_marker_NOT_written_if_open_fails_for_non_lock_reason(
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
