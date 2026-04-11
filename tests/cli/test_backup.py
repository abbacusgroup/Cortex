"""Tests for cortex.cli.backup — backup and restore of Cortex data stores."""

from __future__ import annotations

import json
import sqlite3
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from cortex.cli.backup import (
    _check_server_running,
    _checkpoint_sqlite,
    _human_size,
    _should_exclude,
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
