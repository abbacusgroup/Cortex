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
