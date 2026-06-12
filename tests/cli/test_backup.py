"""Tests for cortex.cli.backup — backup and restore of Cortex data stores."""

from __future__ import annotations

import io
import json
import sqlite3
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from cortex.cli.backup import (
    RestoreVerificationError,
    _archived_doc_count,
    _check_server_running,
    _checkpoint_sqlite,
    _human_size,
    _quick_triple_count,
    _should_exclude,
    _verify_restored_store,
    create_backup,
    do_backup,
    do_restore,
)
from cortex.core.config import CortexConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Create a realistic ~/.cortex/ layout for testing."""
    d = tmp_path / ".cortex"
    d.mkdir()

    # SQLite database (WAL mode, with a documents table)
    db_path = d / "cortex.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT, created_at TEXT)")
    for i in range(5):
        conn.execute(
            "INSERT INTO documents VALUES (?, ?, ?)",
            (f"doc-{i}", f"Title {i}", "2026-04-09"),
        )
    conn.commit()
    conn.close()

    # graph.db/ directory with dummy RocksDB files
    graph = d / "graph.db"
    graph.mkdir()
    (graph / "CURRENT").write_text("MANIFEST-000001\n")
    (graph / "IDENTITY").write_text("test-identity\n")
    (graph / "MANIFEST-000001").write_bytes(b"\x00" * 64)
    (graph / "000010.sst").write_bytes(b"\x00" * 128)
    (graph / "000011.log").write_bytes(b"\x00" * 32)
    (graph / "LOCK").write_text("")  # RocksDB internal lock
    (graph / "LOG").write_text("RocksDB log\n")
    (graph / "LOG.old.1712000000").write_text("old log\n")
    (graph / "LOG.old.1712100000").write_text("old log\n")
    (graph / "OPTIONS-000001").write_text("options\n")

    # Lock marker
    marker = {
        "pid": 99999,
        "cmdline": "cortex serve --transport mcp-http",
        "acquired_at": "2026-04-09T12:00:00+00:00",
    }
    (d / "graph.db.lock").write_text(json.dumps(marker))

    # Server logs
    (d / "mcp-http.log").write_text("log line\n")
    (d / "mcp-http.err").write_text("err line\n")
    (d / "mcp-http.log.old").write_text("old log\n")
    (d / "dashboard.err").write_text("dashboard err\n")

    # .env
    (d / ".env").write_text("CORTEX_LLM_MODEL=test\n")

    return d


@pytest.fixture()
def config(data_dir: Path) -> CortexConfig:
    """Config pointing at the test data_dir."""
    return CortexConfig(data_dir=data_dir)


@pytest.fixture()
def archive_path(config: CortexConfig, tmp_path: Path) -> Path:
    """Produce a valid backup archive for restore tests."""
    # Server not running for backup (PID 99999 is dead)
    return do_backup(config, output=tmp_path / "backups")


@pytest.fixture()
def real_config(tmp_path: Path) -> CortexConfig:
    """A data_dir with a *real* Oxigraph graph.db and a populated SQLite db.

    Unlike the synthetic ``data_dir`` fixture (which uses dummy RocksDB bytes
    that cannot actually be opened), this builds a genuine graph store so the
    post-restore / backup verification path that opens graph.db is exercised
    end to end.
    """
    from cortex.db.graph_store import GraphStore

    d = tmp_path / ".cortex-real"
    d.mkdir()

    db_path = d / "cortex.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT, created_at TEXT)")
    for i in range(3):
        conn.execute(
            "INSERT INTO documents VALUES (?, ?, ?)",
            (f"doc-{i}", f"Title {i}", "2026-06-11"),
        )
    conn.commit()
    conn.close()

    # Real Oxigraph store with a couple of objects (=> nonzero triple_count).
    graph = d / "graph.db"
    with GraphStore(graph) as gs:
        gs.create_object(obj_type="lesson", title="Hello", content="hello")
        gs.create_object(obj_type="fix", title="World", content="world")
    # The lock marker is removed on close(); ensure no stale marker lingers.

    return CortexConfig(data_dir=d)


# ---------------------------------------------------------------------------
# _should_exclude
# ---------------------------------------------------------------------------


class TestShouldExclude:
    def test_excludes_lock_marker(self):
        assert _should_exclude("graph.db.lock")

    def test_excludes_rocksdb_lock(self):
        assert _should_exclude("graph.db/LOCK")

    def test_excludes_env(self):
        assert _should_exclude(".env")

    def test_excludes_wal(self):
        assert _should_exclude("cortex.db-wal")

    def test_excludes_shm(self):
        assert _should_exclude("cortex.db-shm")

    def test_excludes_server_logs(self):
        assert _should_exclude("mcp-http.log")
        assert _should_exclude("mcp-http.err")
        assert _should_exclude("mcp-http.log.old")
        assert _should_exclude("dashboard.err.old")

    def test_excludes_rocksdb_old_logs(self):
        assert _should_exclude("graph.db/LOG.old.1712000000")

    def test_includes_cortex_db(self):
        assert not _should_exclude("cortex.db")

    def test_includes_sst_files(self):
        assert not _should_exclude("graph.db/000010.sst")

    def test_includes_rocksdb_current_log(self):
        assert not _should_exclude("graph.db/LOG")

    def test_includes_rocksdb_manifest(self):
        assert not _should_exclude("graph.db/MANIFEST-000001")

    def test_includes_rocksdb_wal(self):
        assert not _should_exclude("graph.db/000011.log")


# ---------------------------------------------------------------------------
# _check_server_running
# ---------------------------------------------------------------------------


class TestCheckServerRunning:
    def test_no_marker(self, config: CortexConfig):
        (config.data_dir / "graph.db.lock").unlink()
        running, pid, _cmdline = _check_server_running(config)
        assert not running
        assert pid is None

    def test_dead_pid(self, config: CortexConfig):
        # PID 99999 from fixture should be dead
        running, pid, _cmdline = _check_server_running(config)
        assert not running
        assert pid == 99999

    def test_live_pid(self, config: CortexConfig):
        with patch("cortex.db.graph_store._pid_alive", return_value=True):
            running, pid, cmdline = _check_server_running(config)
        assert running
        assert pid == 99999
        assert "cortex serve" in cmdline


# ---------------------------------------------------------------------------
# _checkpoint_sqlite
# ---------------------------------------------------------------------------


class TestCheckpointSqlite:
    def test_checkpoints_wal(self, config: CortexConfig):
        # Write some data to force WAL activity
        conn = sqlite3.connect(str(config.sqlite_db_path))
        conn.execute("INSERT INTO documents VALUES ('extra', 'Extra', '2026-04-09')")
        conn.commit()
        conn.close()

        # Checkpoint should not raise
        _checkpoint_sqlite(config.sqlite_db_path)


# ---------------------------------------------------------------------------
# _human_size
# ---------------------------------------------------------------------------


class TestHumanSize:
    def test_bytes(self):
        assert _human_size(500) == "500 B"

    def test_kilobytes(self):
        assert "KB" in _human_size(2048)

    def test_megabytes(self):
        assert "MB" in _human_size(5 * 1024 * 1024)


# ---------------------------------------------------------------------------
# do_backup
# ---------------------------------------------------------------------------


class TestDoBackup:
    def test_creates_archive(self, config: CortexConfig, tmp_path: Path):
        result = do_backup(config, output=tmp_path / "out")
        assert result.exists()
        assert result.suffix == ".gz"
        assert "cortex-backup-" in result.name

    def test_archive_contains_db(self, config: CortexConfig, tmp_path: Path):
        result = do_backup(config, output=tmp_path / "out")
        with tarfile.open(result, "r:gz") as tar:
            assert "cortex.db" in tar.getnames()

    def test_archive_contains_graph(self, config: CortexConfig, tmp_path: Path):
        result = do_backup(config, output=tmp_path / "out")
        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
        assert any(n.startswith("graph.db/") for n in names)
        assert "graph.db/CURRENT" in names
        assert "graph.db/000010.sst" in names

    def test_archive_excludes_locks(self, config: CortexConfig, tmp_path: Path):
        result = do_backup(config, output=tmp_path / "out")
        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
        assert "graph.db.lock" not in names
        assert "graph.db/LOCK" not in names

    def test_archive_excludes_logs(self, config: CortexConfig, tmp_path: Path):
        result = do_backup(config, output=tmp_path / "out")
        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
        assert "mcp-http.log" not in names
        assert "mcp-http.err" not in names

    def test_archive_excludes_env(self, config: CortexConfig, tmp_path: Path):
        result = do_backup(config, output=tmp_path / "out")
        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
        assert ".env" not in names

    def test_archive_excludes_rocksdb_old_logs(self, config: CortexConfig, tmp_path: Path):
        result = do_backup(config, output=tmp_path / "out")
        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
        assert not any("LOG.old" in n for n in names)

    def test_archive_includes_rocksdb_log(self, config: CortexConfig, tmp_path: Path):
        result = do_backup(config, output=tmp_path / "out")
        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
        assert "graph.db/LOG" in names

    def test_custom_output_dir(self, config: CortexConfig, tmp_path: Path):
        custom = tmp_path / "custom" / "dir"
        result = do_backup(config, output=custom)
        assert result.parent == custom

    def test_warns_when_server_running(self, config: CortexConfig, tmp_path: Path, capsys):
        with patch("cortex.db.graph_store._pid_alive", return_value=True):
            do_backup(config, output=tmp_path / "out")
        captured = capsys.readouterr()
        assert "running" in captured.out.lower()

    def test_fails_if_not_initialized(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        cfg = CortexConfig(data_dir=empty_dir)
        with pytest.raises(typer.Exit):
            do_backup(cfg, output=tmp_path / "out")

    def test_timestamp_in_filename(self, config: CortexConfig, tmp_path: Path):
        result = do_backup(config, output=tmp_path / "out")
        # Format: cortex-backup-YYYY-MM-DDTHHMMSS.tar.gz
        assert result.name.startswith("cortex-backup-")
        assert result.name.endswith(".tar.gz")
        # Extract timestamp part: 2026-04-09T143025
        stamp = result.name.removeprefix("cortex-backup-").removesuffix(".tar.gz")
        assert "T" in stamp
        assert len(stamp) == 17  # YYYY-MM-DDTHHMMSS


# ---------------------------------------------------------------------------
# do_restore
# ---------------------------------------------------------------------------


class TestDoRestore:
    def test_restores_from_archive(self, config: CortexConfig, archive_path: Path):
        # Delete current stores to simulate fresh restore
        config.sqlite_db_path.unlink()
        import shutil

        shutil.rmtree(config.graph_db_path)

        do_restore(config, archive_path)

        assert config.sqlite_db_path.exists()
        assert config.graph_db_path.is_dir()
        assert (config.graph_db_path / "CURRENT").exists()

    def test_refuses_when_server_running(self, config: CortexConfig, archive_path: Path):
        with (
            patch("cortex.db.graph_store._pid_alive", return_value=True),
            pytest.raises(typer.Exit),
        ):
            do_restore(config, archive_path)

    def test_creates_pre_restore(self, config: CortexConfig, archive_path: Path):
        do_restore(config, archive_path)
        pre = config.data_dir / ".pre-restore"
        assert pre.exists()
        assert (pre / "cortex.db").exists()
        assert (pre / "graph.db").is_dir()

    def test_removes_lock_files(self, config: CortexConfig, archive_path: Path, tmp_path: Path):
        # Create a tainted archive that includes lock files
        tainted = tmp_path / "tainted.tar.gz"
        with tarfile.open(tainted, "w:gz") as tar:
            # Copy contents from the good archive
            with tarfile.open(archive_path, "r:gz") as src:
                for member in src.getmembers():
                    tar.addfile(member, src.extractfile(member))
            # Add a fake LOCK file
            import io

            lock_info = tarfile.TarInfo(name="graph.db/LOCK")
            lock_info.size = 0
            tar.addfile(lock_info, io.BytesIO(b""))

        do_restore(config, tainted)
        assert not (config.graph_db_path / "LOCK").exists()

    def test_invalid_archive(self, config: CortexConfig, tmp_path: Path):
        bad = tmp_path / "bad.tar.gz"
        bad.write_text("not a tar file")
        with pytest.raises(typer.Exit):
            do_restore(config, bad)

    def test_missing_archive(self, config: CortexConfig, tmp_path: Path):
        with pytest.raises(typer.Exit):
            do_restore(config, tmp_path / "nonexistent.tar.gz")

    def test_archive_missing_db(self, config: CortexConfig, tmp_path: Path):
        # Archive with graph.db/ but no cortex.db
        bad = tmp_path / "no-db.tar.gz"
        with tarfile.open(bad, "w:gz") as tar:
            import io

            info = tarfile.TarInfo(name="graph.db/CURRENT")
            info.size = 5
            tar.addfile(info, io.BytesIO(b"test\n"))
        with pytest.raises(typer.Exit):
            do_restore(config, bad)

    def test_archive_missing_graph(self, config: CortexConfig, tmp_path: Path):
        # Archive with cortex.db but no graph.db/
        bad = tmp_path / "no-graph.tar.gz"
        with tarfile.open(bad, "w:gz") as tar:
            import io

            info = tarfile.TarInfo(name="cortex.db")
            info.size = 5
            tar.addfile(info, io.BytesIO(b"test\n"))
        with pytest.raises(typer.Exit):
            do_restore(config, bad)

    def test_path_traversal_rejected(self, config: CortexConfig, tmp_path: Path):
        bad = tmp_path / "traversal.tar.gz"
        with tarfile.open(bad, "w:gz") as tar:
            import io

            info = tarfile.TarInfo(name="cortex.db")
            info.size = 5
            tar.addfile(info, io.BytesIO(b"test\n"))
            info2 = tarfile.TarInfo(name="graph.db/../../etc/passwd")
            info2.size = 5
            tar.addfile(info2, io.BytesIO(b"test\n"))
        with pytest.raises(typer.Exit):
            do_restore(config, bad)

    def test_reports_doc_count(self, config: CortexConfig, archive_path: Path, capsys):
        do_restore(config, archive_path)
        captured = capsys.readouterr()
        assert "Documents: 5" in captured.out


# ---------------------------------------------------------------------------
# Initiative 3: new exclusion rules
# ---------------------------------------------------------------------------


class TestNewExclusions:
    def test_excludes_pre_restore_tree(self):
        assert _should_exclude(".pre-restore/cortex.db")
        assert _should_exclude(".pre-restore/graph.db/CURRENT")

    def test_excludes_prior_backup_archives(self):
        assert _should_exclude("cortex-backup-2026-06-11T120000.tar.gz")

    def test_excludes_ds_store(self):
        assert _should_exclude(".DS_Store")

    def test_still_includes_real_stores(self):
        # The new rules must never drop the actual data.
        assert not _should_exclude("cortex.db")
        assert not _should_exclude("graph.db/CURRENT")
        assert not _should_exclude("graph.db/000010.sst")

    def test_backup_excludes_pre_restore_and_self(self, config: CortexConfig, tmp_path: Path):
        # Simulate post-restore litter: a .pre-restore/ tree and a stray prior
        # archive sitting in the data dir.
        pre = config.data_dir / ".pre-restore"
        (pre / "graph.db").mkdir(parents=True)
        (pre / "cortex.db").write_bytes(b"\x00" * 4096)
        (pre / "graph.db" / "CURRENT").write_text("stale\n")
        (config.data_dir / "cortex-backup-2026-01-01T000000.tar.gz").write_bytes(b"old")
        (config.data_dir / ".DS_Store").write_bytes(b"\x00")

        result = do_backup(config, output=tmp_path / "out")
        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
        assert not any(n.startswith(".pre-restore") for n in names)
        assert not any(n.endswith(".tar.gz") for n in names)
        assert ".DS_Store" not in names
        # Real data still present.
        assert "cortex.db" in names
        assert "graph.db/CURRENT" in names

    def test_backup_into_data_dir_does_not_archive_itself(
        self, config: CortexConfig
    ):
        # -o <data_dir>: the in-progress archive lands in the walked tree.
        result = do_backup(config, output=config.data_dir)
        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
        assert not any(n.endswith(".tar.gz") for n in names)


# ---------------------------------------------------------------------------
# Initiative 1: WAL checkpoint result + WAL fallback
# ---------------------------------------------------------------------------


class TestCheckpointResult:
    def test_returns_result_row(self, config: CortexConfig):
        conn = sqlite3.connect(str(config.sqlite_db_path))
        conn.execute("INSERT INTO documents VALUES ('x', 'X', '2026-06-11')")
        conn.commit()
        conn.close()
        busy, log, checkpointed = _checkpoint_sqlite(config.sqlite_db_path)
        # No concurrent holder -> checkpoint should fully succeed.
        assert busy == 0
        assert log == checkpointed

    def test_blocked_checkpoint_reports_busy(self, config: CortexConfig):
        # Hold a read snapshot open in another connection so TRUNCATE cannot
        # complete. busy_timeout makes this deterministic-ish; the key contract
        # is simply that the result row is returned (not discarded).
        holder = sqlite3.connect(str(config.sqlite_db_path))
        holder.execute("BEGIN")
        holder.execute("SELECT COUNT(*) FROM documents").fetchone()  # acquire read lock

        writer = sqlite3.connect(str(config.sqlite_db_path))
        writer.execute("INSERT INTO documents VALUES ('wal-row', 'W', '2026-06-11')")
        writer.commit()
        writer.close()
        try:
            busy, log, checkpointed = _checkpoint_sqlite(config.sqlite_db_path)
            # A held read snapshot prevents a TRUNCATE from resetting the WAL.
            assert busy != 0 or log != checkpointed
        finally:
            holder.close()


class TestWalFallbackRoundTrip:
    def test_wal_survives_hot_backup(self, real_config: CortexConfig, tmp_path: Path):
        """A committed row stranded in the WAL must survive a hot backup.

        Open a second connection holding a read snapshot so the checkpoint is
        blocked, commit a sentinel row via a third connection (it lands in the
        WAL, invisible to the bare main db file), back up, then restore into a
        fresh dir and assert the sentinel is present.
        """
        db = real_config.sqlite_db_path

        # Snapshot holder blocks the checkpoint TRUNCATE.
        holder = sqlite3.connect(str(db))
        holder.execute("BEGIN")
        holder.execute("SELECT COUNT(*) FROM documents").fetchone()

        # Sentinel committed into the WAL by a separate writer.
        writer = sqlite3.connect(str(db))
        writer.execute("INSERT INTO documents VALUES ('sentinel', 'Sentinel', '2026-06-11')")
        writer.commit()
        writer.close()

        try:
            # Backup while the snapshot is still held -> WAL fallback path.
            archive = create_backup(real_config, output=tmp_path / "bk")
            with tarfile.open(archive, "r:gz") as tar:
                names = tar.getnames()
            # The fallback must have bundled the WAL sidecar.
            assert "cortex.db-wal" in names
        finally:
            holder.close()

        # Restore into a pristine data dir.
        dest = tmp_path / "restored"
        dest.mkdir()
        dest_config = CortexConfig(data_dir=dest)
        do_restore(dest_config, archive)

        # The sentinel row (WAL-only at backup time) must be present.
        conn = sqlite3.connect(f"file:{dest_config.sqlite_db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT title FROM documents WHERE id='sentinel'"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0] == "Sentinel"

    def test_clean_checkpoint_excludes_wal(self, real_config: CortexConfig, tmp_path: Path):
        # No concurrent holder -> checkpoint succeeds -> WAL must NOT be bundled.
        conn = sqlite3.connect(str(real_config.sqlite_db_path))
        conn.execute("INSERT INTO documents VALUES ('y', 'Y', '2026-06-11')")
        conn.commit()
        conn.close()
        archive = create_backup(real_config, output=tmp_path / "bk")
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
        assert "cortex.db-wal" not in names
        assert "cortex.db-shm" not in names


# ---------------------------------------------------------------------------
# Initiative 3: backup self-check
# ---------------------------------------------------------------------------


class TestBackupSelfCheck:
    def test_reports_verified(self, real_config: CortexConfig, tmp_path: Path, capsys):
        do_backup(real_config, output=tmp_path / "out")
        out = capsys.readouterr().out
        assert "Backup verified" in out
        assert "3 documents" in out
        assert "Graph triples" in out

    def test_archived_doc_count_matches(self, config: CortexConfig, tmp_path: Path):
        archive = do_backup(config, output=tmp_path / "out")
        assert _archived_doc_count(archive) == 5

    def test_archived_doc_count_none_for_missing_db(self, tmp_path: Path):
        # An archive without cortex.db -> None.
        bad = tmp_path / "no-db.tar.gz"
        with tarfile.open(bad, "w:gz") as tar:
            info = tarfile.TarInfo(name="graph.db/CURRENT")
            info.size = 4
            tar.addfile(info, io.BytesIO(b"test"))
        assert _archived_doc_count(bad) is None

    def test_warns_on_unverifiable_archive(self, config: CortexConfig, tmp_path: Path, capsys):
        # Force the archive's cortex.db to be unreadable so the self-check warns.
        with patch("cortex.cli.backup._archived_doc_count", return_value=None):
            do_backup(config, output=tmp_path / "out")
        err = capsys.readouterr().err
        assert "could not verify archive" in err.lower()

    def test_warns_on_count_mismatch(self, config: CortexConfig, tmp_path: Path, capsys):
        with patch("cortex.cli.backup._archived_doc_count", return_value=2):
            do_backup(config, output=tmp_path / "out")
        err = capsys.readouterr().err
        assert "may be torn or short" in err.lower()

    def test_self_check_counts_wal_bundled_rows(
        self, real_config: CortexConfig, tmp_path: Path, capsys
    ):
        """When the WAL is bundled, the self-check must count WAL-only rows.

        A blocked checkpoint bundles cortex.db-wal; the archived main db file
        alone is short. _archived_doc_count must replay the WAL so the count
        matches the live store and NO spurious mismatch warning fires.
        """
        db = real_config.sqlite_db_path
        holder = sqlite3.connect(str(db))
        holder.execute("BEGIN")
        holder.execute("SELECT COUNT(*) FROM documents").fetchone()
        writer = sqlite3.connect(str(db))
        writer.execute("INSERT INTO documents VALUES ('wal-only', 'W', '2026-06-11')")
        writer.commit()
        writer.close()
        try:
            do_backup(real_config, output=tmp_path / "out")
        finally:
            holder.close()
        captured = capsys.readouterr()
        # Live store now has 4 docs (3 + wal-only); the archive must agree.
        assert "Backup verified: 4 documents" in captured.out
        assert "torn or short" not in captured.err.lower()


# ---------------------------------------------------------------------------
# Initiative 2: atomic restore — rollback in place + completeness check
# ---------------------------------------------------------------------------


class TestRestoreRollback:
    def test_rollback_restores_graph_in_place(
        self, real_config: CortexConfig, tmp_path: Path
    ):
        """A failing extraction must restore the ORIGINAL graph store in place.

        The historic bug: shutil.move(pre_restore/graph.db -> graph_db_path)
        when the destination already exists from a partial extraction nests the
        source as graph.db/graph.db. Assert no such nesting and that the
        original triples are back.
        """
        archive = create_backup(real_config, output=tmp_path / "bk")
        original_triples = _quick_triple_count(real_config.graph_db_path)
        assert original_triples and original_triples > 0

        # Make extractall fail AFTER partially creating the graph.db/ target.
        def failing_extractall(self, *args, **kwargs):
            # Create a partial target so the naive rollback would nest.
            real_config.graph_db_path.mkdir(parents=True, exist_ok=True)
            (real_config.graph_db_path / "PARTIAL").write_text("garbage\n")
            raise RuntimeError("boom: extraction interrupted")

        with (
            patch.object(tarfile.TarFile, "extractall", failing_extractall),
            pytest.raises(RuntimeError, match="boom"),
        ):
            do_restore(real_config, archive)

        # No nesting: graph.db/graph.db must not exist.
        assert not (real_config.graph_db_path / "graph.db").exists()
        # Partial garbage was removed.
        assert not (real_config.graph_db_path / "PARTIAL").exists()
        # Original store is back and openable with the same triple count.
        assert _quick_triple_count(real_config.graph_db_path) == original_triples
        # SQLite original is back too.
        from cortex.cli.backup import _quick_doc_count

        assert _quick_doc_count(real_config.sqlite_db_path) == 3

    def test_rollback_restores_sqlite_in_place(
        self, real_config: CortexConfig, tmp_path: Path
    ):
        archive = create_backup(real_config, output=tmp_path / "bk")

        def failing_extractall(self, *args, **kwargs):
            # Partial cortex.db on disk before failing.
            real_config.sqlite_db_path.write_bytes(b"partial garbage")
            raise RuntimeError("boom")

        with (
            patch.object(tarfile.TarFile, "extractall", failing_extractall),
            pytest.raises(RuntimeError, match="boom"),
        ):
            do_restore(real_config, archive)

        from cortex.cli.backup import _quick_doc_count

        assert _quick_doc_count(real_config.sqlite_db_path) == 3


class TestRestoreCompletenessCheck:
    def test_truncated_archive_triggers_rollback(
        self, real_config: CortexConfig, tmp_path: Path
    ):
        """A restore whose cortex.db opens but has 0 docs must roll back.

        Build a tampered archive that swaps cortex.db for an empty-but-valid
        SQLite db (0 documents). do_restore must detect the empty store, roll
        back, and leave the original 3-doc store in place.
        """
        good = create_backup(real_config, output=tmp_path / "bk")

        # Build an empty (0-row) but structurally valid cortex.db.
        empty_db = tmp_path / "empty.db"
        c = sqlite3.connect(str(empty_db))
        c.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT, created_at TEXT)")
        c.commit()
        c.close()
        empty_bytes = empty_db.read_bytes()

        tampered = tmp_path / "tampered.tar.gz"
        with tarfile.open(tampered, "w:gz") as tar, tarfile.open(good, "r:gz") as src:
            for member in src.getmembers():
                if member.name == "cortex.db":
                    info = tarfile.TarInfo(name="cortex.db")
                    info.size = len(empty_bytes)
                    tar.addfile(info, io.BytesIO(empty_bytes))
                else:
                    f = src.extractfile(member)
                    tar.addfile(member, f)

        from cortex.cli.backup import _quick_doc_count

        with pytest.raises(RestoreVerificationError):
            do_restore(real_config, tampered)

        # Original 3-doc store restored in place.
        assert _quick_doc_count(real_config.sqlite_db_path) == 3
        assert not (real_config.graph_db_path / "graph.db").exists()

    def test_verify_restored_store_passes_for_good_restore(
        self, real_config: CortexConfig, tmp_path: Path
    ):
        archive = create_backup(real_config, output=tmp_path / "bk")
        dest = tmp_path / "restored"
        dest.mkdir()
        dest_config = CortexConfig(data_dir=dest)
        do_restore(dest_config, archive)
        # Self-check passes silently; restored store opens with docs + triples.
        _verify_restored_store(dest_config)
        assert _quick_triple_count(dest_config.graph_db_path) > 0

    def test_verify_raises_on_missing_db(self, tmp_path: Path):
        cfg = CortexConfig(data_dir=tmp_path / "nope")
        with pytest.raises(RestoreVerificationError):
            _verify_restored_store(cfg)


# ---------------------------------------------------------------------------
# Initiative 2: explicit .pre-restore lifecycle
# ---------------------------------------------------------------------------


class TestPreRestoreLifecycle:
    def test_no_pre_restore_when_target_empty(self, tmp_path: Path, real_config: CortexConfig):
        # Build an archive from a populated store, then restore into an EMPTY
        # data dir — no .pre-restore/ should be created (nothing to protect).
        archive = create_backup(real_config, output=tmp_path / "bk")
        empty = tmp_path / "empty-target"
        empty.mkdir()
        empty_config = CortexConfig(data_dir=empty)
        do_restore(empty_config, archive)
        assert not (empty / ".pre-restore").exists()

    def test_pre_restore_path_reported(self, config: CortexConfig, archive_path: Path, capsys):
        do_restore(config, archive_path)
        out = capsys.readouterr().out
        pre = config.data_dir / ".pre-restore"
        assert str(pre) in out
