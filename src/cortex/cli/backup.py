"""Backup and restore for Cortex data stores.

Creates timestamped tar.gz archives of cortex.db and graph.db/, and
restores them with safety rollback via a .pre-restore/ staging area.
"""

from __future__ import annotations

import shutil
import sqlite3
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import typer

from cortex.core.config import CortexConfig

# ---------------------------------------------------------------------------
# Exclusion rules
# ---------------------------------------------------------------------------

# Relative paths (from data_dir) to exclude from backup archives.
_EXCLUDE_EXACT = frozenset(
    {
        "graph.db.lock",
        "graph.db/LOCK",
        ".env",
        "cortex.db-wal",
        "cortex.db-shm",
    }
)

_EXCLUDE_SUFFIXES = (".log", ".err", ".log.old", ".err.old")


def _should_exclude(rel: str) -> bool:
    """Return True if *rel* (relative to data_dir) should be skipped."""
    if rel in _EXCLUDE_EXACT:
        return True
    # Top-level server logs
    parts = rel.split("/")
    if len(parts) == 1 and rel.endswith(_EXCLUDE_SUFFIXES):
        return True
    # RocksDB diagnostic log archives inside graph.db/
    if parts[0] == "graph.db" and len(parts) == 2:
        name = parts[1]
        if name.startswith("LOG.old"):
            return True
    return False


# ---------------------------------------------------------------------------
# Server-running check
# ---------------------------------------------------------------------------


def _check_server_running(config: CortexConfig) -> tuple[bool, int | None, str | None]:
    """Check if a Cortex server is holding the graph.db lock.

    Returns (is_running, pid_or_none, cmdline_or_none).
    """
    from cortex.db.graph_store import _marker_path_for, _pid_alive, _read_marker

    marker_path = _marker_path_for(config.graph_db_path)
    if not marker_path.exists():
        return (False, None, None)

    marker = _read_marker(marker_path)
    if marker is None or marker.get("_unreadable"):
        return (False, None, None)

    raw_pid = marker.get("pid")
    if not isinstance(raw_pid, int):
        return (False, None, None)

    cmdline = marker.get("cmdline")
    if _pid_alive(raw_pid):
        return (True, raw_pid, cmdline)
    return (False, raw_pid, cmdline)


# ---------------------------------------------------------------------------
# SQLite WAL checkpoint
# ---------------------------------------------------------------------------


def _checkpoint_sqlite(db_path: Path) -> None:
    """Flush the SQLite WAL into the main database file.

    Uses a short-lived direct connection (not ContentStore) to avoid
    acquiring schema locks or interfering with a running server.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Human-readable file size
# ---------------------------------------------------------------------------


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def do_backup(config: CortexConfig, output: Path | None = None) -> Path:
    """Create a tar.gz backup of cortex.db and graph.db/.

    Returns the path to the created archive.
    """
    # 1. Verify stores exist
    if not config.sqlite_db_path.exists():
        typer.secho(
            f"Cortex not initialized — {config.sqlite_db_path} not found.\n"
            "Run `cortex init` first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    if not config.graph_db_path.exists() or not config.graph_db_path.is_dir():
        typer.secho(
            f"Graph store not found at {config.graph_db_path}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # 2. Warn if server is running (safe to proceed)
    running, pid, _cmdline = _check_server_running(config)
    if running:
        typer.secho(
            f"  Cortex server is running (PID {pid}). "
            "Backup will proceed — stores support concurrent reads.",
            fg=typer.colors.YELLOW,
        )

    # 3. Checkpoint SQLite WAL
    typer.echo("  Checkpointing SQLite WAL...")
    _checkpoint_sqlite(config.sqlite_db_path)

    # 4. Build archive
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")
    filename = f"cortex-backup-{stamp}.tar.gz"
    out_dir = output if output else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / filename

    typer.echo(f"  Archiving to {archive_path}...")
    data_dir = config.data_dir
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(data_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(data_dir))
            if _should_exclude(rel):
                continue
            tar.add(str(path), arcname=rel)

    # 5. Report
    size = archive_path.stat().st_size
    doc_count = _quick_doc_count(config.sqlite_db_path)
    typer.secho(f"\nBackup complete: {archive_path}", fg=typer.colors.GREEN)
    typer.echo(f"  Size: {_human_size(size)}")
    if doc_count is not None:
        typer.echo(f"  Documents: {doc_count}")

    return archive_path


def do_restore(config: CortexConfig, archive: Path) -> None:
    """Restore Cortex data from a tar.gz backup archive."""
    # 1. Verify archive exists
    if not archive.exists():
        typer.secho(f"Archive not found: {archive}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # 2. Validate archive
    try:
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
    except tarfile.TarError as e:
        typer.secho(
            f"Invalid archive: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1) from None

    # 3. Validate contents
    has_db = "cortex.db" in names
    has_graph = any(n.startswith("graph.db/") for n in names)
    if not has_db or not has_graph:
        missing = []
        if not has_db:
            missing.append("cortex.db")
        if not has_graph:
            missing.append("graph.db/")
        typer.secho(
            f"Archive is missing required files: {', '.join(missing)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # 4. Path traversal check
    for name in names:
        if name.startswith("/") or ".." in name.split("/"):
            typer.secho(
                f"Archive contains unsafe path: {name!r}. Refusing to extract.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

    # 5. Refuse if server is running
    running, pid, _cmdline = _check_server_running(config)
    if running:
        typer.secho(
            f"Cortex server is running (PID {pid}). "
            f"Stop it first:\n  cortex uninstall\n  # or: kill {pid}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    data_dir = config.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # 6. Safety: move current stores to .pre-restore/
    pre_restore = data_dir / ".pre-restore"
    if pre_restore.exists():
        shutil.rmtree(pre_restore)
    pre_restore.mkdir()

    moved_db = False
    moved_graph = False
    try:
        if config.sqlite_db_path.exists():
            shutil.move(str(config.sqlite_db_path), str(pre_restore / "cortex.db"))
            moved_db = True
            # Also move WAL/SHM if present
            for suffix in ("-wal", "-shm"):
                wal = config.data_dir / f"cortex.db{suffix}"
                if wal.exists():
                    shutil.move(str(wal), str(pre_restore / f"cortex.db{suffix}"))

        if config.graph_db_path.exists():
            shutil.move(str(config.graph_db_path), str(pre_restore / "graph.db"))
            moved_graph = True

        # 7. Extract
        typer.echo(f"  Extracting {archive.name} to {data_dir}...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=str(data_dir), filter="data")

        # 8. Clean lock files from extraction
        lock_marker = data_dir / "graph.db.lock"
        if lock_marker.exists():
            lock_marker.unlink()
        rocksdb_lock = config.graph_db_path / "LOCK"
        if rocksdb_lock.exists():
            rocksdb_lock.unlink()

    except Exception:
        # Attempt rollback
        typer.secho(
            "\nExtraction failed — rolling back from .pre-restore/",
            fg=typer.colors.YELLOW,
            err=True,
        )
        if moved_db and (pre_restore / "cortex.db").exists():
            shutil.move(str(pre_restore / "cortex.db"), str(config.sqlite_db_path))
            for suffix in ("-wal", "-shm"):
                backed = pre_restore / f"cortex.db{suffix}"
                if backed.exists():
                    shutil.move(str(backed), str(config.data_dir / f"cortex.db{suffix}"))
        if moved_graph and (pre_restore / "graph.db").exists():
            shutil.move(str(pre_restore / "graph.db"), str(config.graph_db_path))
        raise

    # 9. Report
    doc_count = _quick_doc_count(config.sqlite_db_path)
    typer.secho(f"\nRestore complete: {data_dir}", fg=typer.colors.GREEN)
    if doc_count is not None:
        typer.echo(f"  Documents: {doc_count}")
    typer.echo("  Old data saved to .pre-restore/ (safe to delete after verifying).")
    typer.echo("  Run `cortex status` to verify.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quick_doc_count(db_path: Path) -> int | None:
    """Quick readonly document count from SQLite. Returns None on any error."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except Exception:
        return None
