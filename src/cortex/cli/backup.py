"""Backup and restore for Cortex data stores.

Creates timestamped tar.gz archives of cortex.db and graph.db/, and
restores them with safety rollback via a .pre-restore/ staging area.
"""

from __future__ import annotations

import contextlib
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import typer

from cortex.core.config import CortexConfig


class RestoreVerificationError(RuntimeError):
    """Raised when a restored store fails its post-extraction self-check."""


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
    parts = rel.split("/")
    # Never archive the restore safety copy — it doubles archive size after a
    # restore and resurrects stale stores over the fresh .pre-restore copy.
    if parts[0] == ".pre-restore":
        return True
    # Top-level strays: prior backup archives (incl. one being written in place)
    # and macOS Finder cruft / stray leading-dot files.
    if len(parts) == 1:
        if rel.startswith("cortex-backup-") and rel.endswith(".tar.gz"):
            return True
        if rel == ".DS_Store":
            return True
        if rel.endswith(_EXCLUDE_SUFFIXES):
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


def _checkpoint_sqlite(db_path: Path) -> tuple[int, int, int]:
    """Flush the SQLite WAL into the main database file.

    Uses a short-lived direct connection (not ContentStore) to avoid
    acquiring schema locks or interfering with a running server.

    Returns the ``(busy, log, checkpointed)`` result row of
    ``PRAGMA wal_checkpoint(TRUNCATE)``:

    * ``busy`` is ``1`` if the checkpoint could not run to completion because
      another connection held a read or write lock (e.g. a live MCP server
      holding a snapshot) — the WAL was *not* fully folded into the main file.
    * ``log`` is the number of frames in the WAL.
    * ``checkpointed`` is the number of frames moved into the main file.

    When ``busy != 0`` or ``log != checkpointed`` the main database file does
    not yet contain every committed transaction, so the caller must archive
    the WAL/SHM sidecars to avoid silently dropping recent writes.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        # Without a busy_timeout the checkpoint gives up instantly (busy=1) the
        # moment any other connection holds a lock, silently leaving committed
        # rows stranded in the WAL. Give it a chance to acquire the lock.
        conn.execute("PRAGMA busy_timeout=5000")
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        conn.close()
    if not row:
        # Older SQLite or unexpected: treat as a non-conclusive checkpoint so
        # the caller errs on the safe side and includes the WAL.
        return (1, -1, 0)
    return (int(row[0]), int(row[1]), int(row[2]))


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


def create_backup(config: CortexConfig, output: Path | None = None) -> Path:
    """Create a tar.gz backup of cortex.db and graph.db/.

    Pure function — no CLI output (apart from a loud warning emitted to stderr
    if the WAL could not be checkpointed; see below).  Returns the path to the
    created archive.  Raises FileNotFoundError if stores don't exist.

    If the SQLite WAL could not be fully folded into the main database file
    (e.g. a live server holds a read snapshot), the WAL/SHM sidecars are
    included in the archive so no committed transaction is silently dropped —
    SQLite replays them on the next open, and :func:`do_restore` relocates any
    pre-existing WAL/SHM before extraction, so this is restore-safe.
    """
    if not config.sqlite_db_path.exists():
        raise FileNotFoundError(f"SQLite not found: {config.sqlite_db_path}")

    if not config.graph_db_path.exists() or not config.graph_db_path.is_dir():
        raise FileNotFoundError(f"Graph store not found: {config.graph_db_path}")

    # Checkpoint SQLite WAL and inspect the result.
    busy, log, checkpointed = _checkpoint_sqlite(config.sqlite_db_path)
    wal_incomplete = busy != 0 or log != checkpointed
    if wal_incomplete:
        typer.secho(
            "  WARNING: SQLite WAL checkpoint was incomplete "
            f"(busy={busy}, log={log}, checkpointed={checkpointed}) — "
            "a live server likely holds a snapshot. Including cortex.db-wal "
            "and cortex.db-shm in the archive so no committed data is lost.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    # Build archive
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")
    filename = f"cortex-backup-{stamp}.tar.gz"
    out_dir = output if output else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / filename

    # When the checkpoint could not flush the WAL, archive the sidecars for
    # this run only (they are normally excluded). Once the checkpoint succeeds
    # they are absent on disk anyway, so we never double-include them.
    wal_sidecars = {"cortex.db-wal", "cortex.db-shm"} if wal_incomplete else frozenset()

    data_dir = config.data_dir
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(data_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(data_dir))
            if rel in wal_sidecars:
                tar.add(str(path), arcname=rel)
                continue
            if _should_exclude(rel):
                continue
            tar.add(str(path), arcname=rel)

    return archive_path


def do_backup(config: CortexConfig, output: Path | None = None) -> Path:
    """Create a tar.gz backup of cortex.db and graph.db/.

    CLI wrapper around :func:`create_backup` — adds typer output.
    Returns the path to the created archive.
    """
    # 1. Verify stores exist
    if not config.sqlite_db_path.exists():
        typer.secho(
            f"Cortex not initialized — {config.sqlite_db_path} not found.\n"
            "Run `cortex setup` first.",
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

    # 3. Create backup using core logic
    typer.echo("  Checkpointing SQLite WAL...")
    archive_path = create_backup(config, output)

    # 4. Report
    typer.echo(f"  Archiving to {archive_path}...")
    size = archive_path.stat().st_size
    live_count = _quick_doc_count(config.sqlite_db_path)
    typer.secho(f"\nBackup complete: {archive_path}", fg=typer.colors.GREEN)
    typer.echo(f"  Size: {_human_size(size)}")

    # 5. Self-check: open the archive we just wrote and count documents in the
    # archived cortex.db, comparing against the live store. This proves the
    # archive is readable and complete rather than trusting the live count.
    archived_count = _archived_doc_count(archive_path)
    if archived_count is None:
        typer.secho(
            "  WARNING: could not verify archive — the archived cortex.db is "
            "unreadable or missing. Do NOT rely on this backup.",
            fg=typer.colors.RED,
            err=True,
        )
    elif live_count is not None and archived_count != live_count:
        typer.secho(
            f"  WARNING: archive has {archived_count} documents but the live "
            f"store has {live_count}. The archive may be torn or short.",
            fg=typer.colors.RED,
            err=True,
        )
    else:
        typer.secho(
            f"  Backup verified: {archived_count} documents "
            "(archive matches live store).",
            fg=typer.colors.GREEN,
        )

    # Also surface graph store coverage so the user sees both stores accounted
    # for. Best-effort: a count failure here is informational only.
    triples = _quick_triple_count(config.graph_db_path)
    if triples is not None:
        typer.echo(f"  Graph triples: {triples}")

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

    # 6. Safety: move current stores aside to .pre-restore/. Only create the
    # staging dir if there is prior data to protect, so an empty-target restore
    # does not leave a confusing empty .pre-restore/ behind.
    pre_restore = data_dir / ".pre-restore"
    had_prior_data = config.sqlite_db_path.exists() or config.graph_db_path.exists()
    if had_prior_data:
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
        _clean_extracted_locks(config)

        # 9. Completeness self-check: prove the restored store actually opens
        # and is non-empty before declaring success. A torn/short archive (e.g.
        # from a non-crash-consistent hot backup) would otherwise "restore"
        # cleanly and only fail later.
        _verify_restored_store(config)

        # Opening graph.db for verification re-creates the RocksDB LOCK file;
        # clear it again so a subsequent server start isn't blocked.
        _clean_extracted_locks(config)

    except Exception:
        # Attempt rollback. The destination dirs/files may now exist from a
        # partial extraction; shutil.move into an existing directory NESTS the
        # source inside it (graph.db/graph.db), so we must remove the partial
        # target first. This rmtree/unlink is confined to the rollback path and
        # only touches the (re-creatable) extraction target — never the
        # pre-restore copy.
        typer.secho(
            f"\nRestore failed — rolling back from {pre_restore}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        if moved_db and (pre_restore / "cortex.db").exists():
            if config.sqlite_db_path.exists():
                config.sqlite_db_path.unlink()
            for suffix in ("-wal", "-shm"):
                stray = config.data_dir / f"cortex.db{suffix}"
                if stray.exists():
                    stray.unlink()
            shutil.move(str(pre_restore / "cortex.db"), str(config.sqlite_db_path))
            for suffix in ("-wal", "-shm"):
                backed = pre_restore / f"cortex.db{suffix}"
                if backed.exists():
                    shutil.move(str(backed), str(config.data_dir / f"cortex.db{suffix}"))
        if moved_graph and (pre_restore / "graph.db").exists():
            if config.graph_db_path.exists():
                shutil.rmtree(config.graph_db_path)
            shutil.move(str(pre_restore / "graph.db"), str(config.graph_db_path))
        raise

    # 10. Report
    doc_count = _quick_doc_count(config.sqlite_db_path)
    typer.secho(f"\nRestore complete: {data_dir}", fg=typer.colors.GREEN)
    if doc_count is not None:
        typer.echo(f"  Documents: {doc_count}")
    if had_prior_data:
        typer.echo(
            f"  Old data saved to {pre_restore} (safe to delete after verifying)."
        )
    typer.echo("  Run `cortex status` to verify.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_extracted_locks(config: CortexConfig) -> None:
    """Remove the graph.db lock marker and RocksDB LOCK file if present.

    Safe to call repeatedly. Called both after extraction (an archive may have
    captured stale lock files) and after the verification open (which re-creates
    the RocksDB LOCK), so a subsequent server start is never blocked.
    """
    lock_marker = config.data_dir / "graph.db.lock"
    if lock_marker.exists():
        lock_marker.unlink()
    rocksdb_lock = config.graph_db_path / "LOCK"
    if rocksdb_lock.exists():
        rocksdb_lock.unlink()


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


def _quick_triple_count(graph_db_path: Path) -> int | None:
    """Best-effort triple count from a graph store directory.

    Opens the store read/write (Oxigraph has no read-only mode), counts
    triples, and closes immediately to release the RocksDB lock. Returns None
    on any error — this is informational reporting, not a correctness gate.
    """
    try:
        from cortex.db.graph_store import GraphStore

        with GraphStore(graph_db_path) as gs:
            return gs.triple_count
    except Exception:
        return None


def _archived_doc_count(archive_path: Path) -> int | None:
    """Open *archive_path*, extract its cortex.db, and count its documents.

    Returns the archived document count, or None if the archive is unreadable
    or its cortex.db is missing/corrupt. This verifies the bytes that were
    actually written rather than trusting the live store.

    If the archive bundled the WAL/SHM sidecars (the blocked-checkpoint
    fallback path), they are extracted alongside cortex.db so SQLite replays
    them on open — otherwise the count would understate the real, restorable
    document total and produce a spurious "torn/short" warning.
    """
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            try:
                member = tar.getmember("cortex.db")
            except KeyError:
                return None
            with tempfile.TemporaryDirectory() as tmp:
                tar.extract(member, path=tmp, filter="data")
                for sidecar in ("cortex.db-wal", "cortex.db-shm"):
                    with contextlib.suppress(KeyError):
                        tar.extract(tar.getmember(sidecar), path=tmp, filter="data")
                # Use a read/write connection on the disposable temp copy so
                # SQLite replays any bundled WAL (a read-only connection cannot,
                # which would undercount rows committed only in the WAL).
                db_copy = Path(tmp) / "cortex.db"
                conn = sqlite3.connect(str(db_copy))
                try:
                    row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
                    return row[0] if row else None
                finally:
                    conn.close()
    except Exception:
        return None


def _verify_restored_store(config: CortexConfig) -> None:
    """Verify the freshly-restored stores open and are non-empty.

    Raises RestoreVerificationError if the restored SQLite store cannot be
    opened or reports zero documents — that signals a torn/short archive and
    must trigger the rollback. The graph store is opened as a best-effort probe
    (and its lock released immediately); a graph-open failure is surfaced as a
    loud warning rather than discarding an otherwise-good restore, because the
    SQLite count is the authoritative completeness gate.
    """
    doc_count = _quick_doc_count(config.sqlite_db_path)
    if doc_count is None:
        raise RestoreVerificationError(
            "restored cortex.db could not be opened or has no documents table"
        )
    if doc_count == 0:
        raise RestoreVerificationError(
            "restored cortex.db opened but contains 0 documents — "
            "the archive appears truncated"
        )

    triples = _quick_triple_count(config.graph_db_path)
    if triples is None:
        typer.secho(
            "  WARNING: restored graph.db could not be opened for verification. "
            "Documents restored; verify the graph store with `cortex status`.",
            fg=typer.colors.RED,
            err=True,
        )
