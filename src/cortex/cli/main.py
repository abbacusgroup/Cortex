"""Cortex CLI — Typer application.

Commands: init, capture, search, read, list, status, context, dossier, graph, synthesize, entities.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

import typer

from cortex.core.config import CortexConfig, load_config
from cortex.core.constants import KNOWLEDGE_TYPES
from cortex.core.errors import StoreLockedError
from cortex.core.logging import setup_logging
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.pipeline.orchestrator import PipelineOrchestrator
from cortex.retrieval.learner import LearningLoop

app = typer.Typer(
    name="cortex",
    help="Cognitive knowledge system with formal ontology and reasoning.",
    no_args_is_help=True,
)

# Module-level singletons (initialized lazily)
_store: Store | None = None
_pipeline: PipelineOrchestrator | None = None
_learner: LearningLoop | None = None

# Phase 3 — global --direct flag state and MCP client singleton.
# When --direct is set, MCP-routed commands bypass the HTTP client and open
# the store directly (existing behavior). Default is False (MCP routing).
_direct_mode: bool = False
_mcp_client: Any | None = None  # CortexMCPClient — Any-typed to avoid import cycle
_mcp_probe_done: bool = False


@app.callback()
def _global_options(
    direct: bool = typer.Option(
        False,
        "--direct",
        help=(
            "Bypass the running MCP HTTP server and open the graph store directly. "
            "Use this when the MCP server is unreachable, or for offline admin work. "
            "Will fail with a lock error if the MCP server is currently running."
        ),
    ),
) -> None:
    """Global Cortex CLI options. Applies to all subcommands."""
    global _direct_mode, _mcp_client, _mcp_probe_done
    _direct_mode = direct
    # Reset the singleton state per-invocation so CliRunner tests don't leak
    # the previous invocation's state into the next one.
    _mcp_client = None
    _mcp_probe_done = False


def _open_store_or_exit(config: CortexConfig) -> Store:
    """Open a Store, exiting cleanly with a user-friendly message if the graph DB is locked.

    This is the single chokepoint for surfacing StoreLockedError to CLI users.
    """
    try:
        return Store(config)
    except StoreLockedError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e


# ─── Phase 3 helpers — MCP client routing ──────────────────────────────────


def _get_mcp_client() -> Any:
    """Singleton CortexMCPClient bound to ``config.mcp_server_url``.

    Constructed lazily on first use. Returns the same instance for the rest of
    the CLI process. Reset by ``_global_options`` between CliRunner invocations.

    Bundle 9.1: 30s default headroom for slow CI hardware running real capture
    pipelines (embedding model inference + LLM calls can take 15-25s). The
    probe client at 10s (_get_probe_client) still gives fast-fail detection
    for unreachable/hung servers, so users never actually wait 30s just to
    find out the server is down.

    Bundle 10.6: timeout is overridable via the ``CORTEX_MCP_CLIENT_TIMEOUT_SECONDS``
    environment variable. CI sets this to 60s because the GitHub Actions
    macOS runner + 3 concurrent xdist workers racing for CPU can push a
    cold-start ``cortex capture`` past 30s (sentence-transformers embedding
    model load + full pipeline). Local users keep the 30s default for
    fast-fail UX.
    """
    global _mcp_client
    if _mcp_client is not None:
        return _mcp_client
    import os as _os

    from cortex.transport.mcp.client import CortexMCPClient

    config = load_config()
    timeout_str = _os.environ.get("CORTEX_MCP_CLIENT_TIMEOUT_SECONDS", "").strip()
    try:
        timeout = float(timeout_str) if timeout_str else 30.0
    except ValueError:
        timeout = 30.0
    _mcp_client = CortexMCPClient(config.mcp_server_url, timeout_seconds=timeout)
    return _mcp_client


def _get_probe_client() -> Any:
    """Construct a *short-timeout* CortexMCPClient for the lazy probe.

    Bundle 9 / D.2 fix: the probe needs to be cheap and fail fast — every
    extra second of timeout is felt by the user when the server is hung.
    10s is fast enough for a healthy probe (``list_tools`` is one cheap
    RTT) but bounds the user's wait against a SIGSTOP'd /
    network-partitioned server to well under the 30s singleton timeout.

    Bundle 9.4: the probe timeout is overridable via the
    ``CORTEX_PROBE_TIMEOUT_SECONDS`` environment variable. CI sets this
    to 30s because the full ``cortex capture`` subprocess path on slow
    runners spends 5-8s on Python startup + cortex import chain before
    the probe even runs, leaving less than 10s for the actual probe
    RTT. Local users keep the 10s default for fast-fail UX.

    Tests patch this function to inject a fake client (same monkeypatch
    pattern they use for ``_get_mcp_client``).
    """
    import os as _os

    from cortex.transport.mcp.client import CortexMCPClient

    config = load_config()
    timeout_str = _os.environ.get("CORTEX_PROBE_TIMEOUT_SECONDS", "").strip()
    try:
        timeout = float(timeout_str) if timeout_str else 10.0
    except ValueError:
        timeout = 10.0
    return CortexMCPClient(config.mcp_server_url, timeout_seconds=timeout)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync CLI code, handling running-loop edge cases.

    If no event loop is currently running on this thread, ``asyncio.run`` is
    used directly. If a loop IS already running (rare in CLI but possible if
    a future command uses asyncio internally), the coroutine is dispatched on
    a fresh loop in a worker thread to avoid the
    "asyncio.run() cannot be called from a running event loop" RuntimeError.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running on this thread — fast path
        return asyncio.run(coro)

    # A loop is already running — fall back to a thread
    import threading

    box: list[Any] = [None]
    exc_box: list[BaseException | None] = [None]

    def _runner() -> None:
        new_loop = asyncio.new_event_loop()
        try:
            box[0] = new_loop.run_until_complete(coro)
        except BaseException as e:
            exc_box[0] = e
        finally:
            new_loop.close()

    t = threading.Thread(target=_runner)
    t.start()
    t.join()
    if exc_box[0] is not None:
        raise exc_box[0]
    return box[0]


# Tools the MCP HTTP server must expose for Phase 3 routing to work. The probe
# checks all of these are present before any MCP-routed command runs.
_REQUIRED_MCP_ROUTING_TOOLS = frozenset(
    {
        "cortex_search",
        "cortex_capture",
        "cortex_read",
        "cortex_list",
        "cortex_status",
        "cortex_context",
        "cortex_dossier",
        "cortex_graph",
        "cortex_synthesize",
        "cortex_list_entities",
        "cortex_pipeline",
        "cortex_reason",
    }
)


def _probe_mcp_lazy() -> None:
    """Verify the MCP HTTP server is reachable AND has all required tools.

    Runs at most once per CLI process (cached via ``_mcp_probe_done``). Fails
    fast with an actionable error if the server isn't reachable, or if it's
    missing one of the tools the CLI's MCP-routed commands depend on.

    In ``--direct`` mode this function is never called.

    Bundle 9 / D.2 fix: the probe uses a dedicated short-timeout client
    (3s) via ``_get_probe_client`` instead of the singleton (10s). The
    singleton's wider timeout is needed for slow tools like
    ``cortex_search`` and ``cortex_capture``, but a *probe* just needs to
    know whether the server is alive — and against a hung server, every
    extra second of timeout is felt by the user. Before this change,
    ``cortex list`` against a SIGSTOP'd server waited ~10s before
    erroring; now it's ~3s.
    """
    global _mcp_probe_done
    if _mcp_probe_done:
        return
    from cortex.transport.mcp.client import MCPClientError

    probe_client = _get_probe_client()
    try:
        available = set(_run_async(probe_client.list_tools()))
    except MCPClientError as e:
        config = load_config()
        typer.secho(
            f"Cannot reach Cortex MCP server at {config.mcp_server_url}",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(f"  {e}", fg=typer.colors.RED, err=True)
        direct_hint = (
            f"    cortex --direct {' '.join(sys.argv[1:])}"
            if len(sys.argv) > 1
            else "    cortex --direct ..."
        )
        typer.echo(
            "  Either start the server in another terminal:\n"
            "    cortex serve --transport mcp-http --host 127.0.0.1 --port 1314\n"
            "  Or bypass it for this command with --direct:\n"
            f"{direct_hint}",
            err=True,
        )
        raise typer.Exit(1) from e

    missing = _REQUIRED_MCP_ROUTING_TOOLS - available
    if missing:
        typer.secho(
            f"MCP server is missing required tools: {', '.join(sorted(missing))}",
            fg=typer.colors.RED,
            err=True,
        )
        typer.echo(
            "This usually means the MCP server is from an older version of Cortex. "
            "Update or restart it.",
            err=True,
        )
        raise typer.Exit(1)

    _mcp_probe_done = True


def _use_mcp() -> bool:
    """Return True if the current command should route through the MCP HTTP server.

    Centralizes the routing decision: when `--direct` is set we always go to
    the local store. Otherwise we run the lazy probe (which fails fast if the
    server isn't reachable) and route through MCP.

    Bootstrap commands (init, setup, import-v1, import-vault) call
    `_get_store()` directly and never go through this helper, so they always
    use the direct path regardless of `--direct`.
    """
    if _direct_mode:
        return False
    _probe_mcp_lazy()
    return True


def _mcp_call_or_exit(coro_factory: Any) -> Any:
    """Run an MCP client call and convert any error into a clean typer.Exit.

    Each MCP-routed CLI command wraps its tool call in this helper:

        result = _mcp_call_or_exit(lambda: _get_mcp_client().search("foo"))

    The factory pattern (lambda) defers coroutine creation so we can catch
    construction errors too. On success returns the unwrapped result; on any
    MCP error prints a red message to stderr and raises ``typer.Exit(1)``.
    """
    from cortex.transport.mcp.client import (
        MCPConnectionError,
        MCPServerError,
        MCPTimeoutError,
        MCPToolError,
    )

    try:
        return _run_async(coro_factory())
    except MCPConnectionError as e:
        typer.secho(f"MCP connection error: {e}", fg=typer.colors.RED, err=True)
        typer.echo(
            "  Either start the server in another terminal:\n"
            "    cortex serve --transport mcp-http --host 127.0.0.1 --port 1314\n"
            "  Or bypass it with --direct.",
            err=True,
        )
        raise typer.Exit(1) from e
    except MCPTimeoutError as e:
        typer.secho(f"MCP server timed out: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    except MCPServerError as e:
        typer.secho(f"MCP server error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    except MCPToolError as e:
        typer.secho(f"MCP tool error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e


def _get_store(*, must_init: bool = True) -> Store:
    """Get or create the unified store."""
    global _store
    if _store is not None:
        return _store

    config = load_config()
    setup_logging(level=config.log_level, json_output=False)
    store = _open_store_or_exit(config)

    # Auto-initialize if data dir has stores
    try:
        ontology_path = find_ontology()
        store.initialize(ontology_path)
    except FileNotFoundError:
        if must_init:
            typer.echo("Error: Cortex not initialized. Run `cortex init` first.", err=True)
            raise typer.Exit(1) from None

    _store = store
    return store


def _get_pipeline() -> PipelineOrchestrator:
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    store = _get_store()
    _pipeline = PipelineOrchestrator(store, store.config)
    return _pipeline


def _get_learner() -> LearningLoop:
    global _learner
    if _learner is not None:
        return _learner
    _learner = LearningLoop(_get_store())
    return _learner


def _warmup_embeddings(config: CortexConfig) -> None:
    """Download and cache the embedding model if sentence-transformers is installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        typer.echo("  Embeddings: not installed (install cortex[embeddings] for semantic search)")
        return

    model_name = config.embedding_model
    typer.echo(f"  Embeddings: loading {model_name}...")
    try:
        SentenceTransformer(model_name)
        typer.echo(f"  Embeddings: {model_name} ready")
    except Exception as e:
        typer.echo(f"  Embeddings: warm-up failed ({e}) — will retry on first search")


@app.command()
def init(
    data_dir: str | None = typer.Option(None, "--data-dir", "-d", help="Data directory path"),
) -> None:
    """Initialize Cortex — create data directory, load ontology, set up stores."""
    config = load_config(data_dir=Path(data_dir) if data_dir else None)
    setup_logging(level=config.log_level, json_output=False)

    store = _open_store_or_exit(config)

    try:
        ontology_path = find_ontology()
    except FileNotFoundError:
        typer.echo("Error: Ontology file not found", err=True)
        raise typer.Exit(1) from None

    triples = store.graph.load_ontology(ontology_path)

    typer.echo(f"Cortex initialized at {config.data_dir}")
    typer.echo(f"  Ontology: {triples} triples loaded")
    typer.echo(f"  Graph DB: {config.graph_db_path}")
    typer.echo(f"  SQLite:   {config.sqlite_db_path}")
    _warmup_embeddings(config)

    global _store
    _store = store


@app.command()
def capture(
    title: str = typer.Argument(..., help="Title for the knowledge object"),
    obj_type: str = typer.Option("capture", "--type", "-t", help="Knowledge type"),
    content: str | None = typer.Option(None, "--content", "-c", help="Content text"),
    project: str = typer.Option("", "--project", "-p", help="Project name"),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags"),
) -> None:
    """Capture a knowledge object."""
    # Normalize type
    obj_type = obj_type.lower()
    if obj_type == "capture":
        obj_type = "idea"  # Default to idea for untyped captures
    if obj_type not in KNOWLEDGE_TYPES:
        typer.echo(
            f"Error: Invalid type '{obj_type}'. Valid types: {', '.join(sorted(KNOWLEDGE_TYPES))}",
            err=True,
        )
        raise typer.Exit(1)

    # Read content from stdin if not provided
    body = content or ""
    if not body and not sys.stdin.isatty():
        body = sys.stdin.read()

    if not body:
        typer.echo("Error: No content provided. Use --content or pipe via stdin.", err=True)
        raise typer.Exit(1)

    if _use_mcp():
        result = _mcp_call_or_exit(
            lambda: _get_mcp_client().capture(
                title=title,
                content=body,
                obj_type=obj_type,
                project=project,
                tags=tags,
            )
        )
    else:
        result = _get_pipeline().capture(
            title=title,
            content=body,
            obj_type=obj_type,
            project=project,
            tags=tags,
            captured_by="cli",
            run_pipeline=True,
        )
    obj_id = result["id"]
    typer.echo(f"Captured {obj_type}: {obj_id}")
    typer.echo(f"  Title: {title}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    obj_type: str | None = typer.Option(None, "--type", "-t", help="Filter by type"),
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by project"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
) -> None:
    """Search knowledge objects with full-text search."""
    if _use_mcp():
        results = _mcp_call_or_exit(
            lambda: _get_mcp_client().search(
                query=query,
                doc_type=obj_type or "",
                project=project or "",
                limit=limit,
            )
        )
    else:
        store = _get_store()
        results = store.search(query, doc_type=obj_type, project=project, limit=limit)

    if not results:
        typer.echo("No results found.")
        return

    typer.echo(f"Found {len(results)} result(s):\n")
    for doc in results:
        _print_summary(doc)


@app.command()
def read(
    obj_id: str = typer.Argument(..., help="Object ID to read"),
) -> None:
    """Read a knowledge object in full."""
    if _use_mcp():
        doc = _mcp_call_or_exit(lambda: _get_mcp_client().read(obj_id))
        # cortex_read returns a string "Not found: {id}" when missing
        if isinstance(doc, str) or doc is None:
            typer.echo(f"Not found: {obj_id}", err=True)
            raise typer.Exit(1)
        # Access tracking is done server-side inside cortex_read
    else:
        store = _get_store()
        doc = store.read(obj_id)
        if doc is None:
            typer.echo(f"Not found: {obj_id}", err=True)
            raise typer.Exit(1)
        _get_learner().record_access(obj_id)

    typer.echo(f"{'=' * 60}")
    typer.echo(f"ID:      {doc['id']}")
    typer.echo(f"Type:    {doc.get('type', 'unknown')}")
    typer.echo(f"Title:   {doc.get('title', '')}")
    typer.echo(f"Project: {doc.get('project', '')}")
    typer.echo(f"Tags:    {doc.get('tags', '')}")
    typer.echo(f"Tier:    {doc.get('tier', '')}")
    typer.echo(f"Created: {doc.get('created_at', '')}")
    typer.echo(f"{'=' * 60}")

    content = doc.get("content", "")
    if content:
        typer.echo(f"\n{content}\n")

    rels = doc.get("relationships", [])
    if rels:
        typer.echo(f"{'─' * 60}")
        typer.echo("Relationships:")
        for r in rels:
            direction = "→" if r["direction"] == "outgoing" else "←"
            typer.echo(f"  {direction} {r['rel_type']} {r['other_id']}")


@app.command(name="list")
def list_objects(
    obj_type: str | None = typer.Option(None, "--type", "-t", help="Filter by type"),
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by project"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
) -> None:
    """List knowledge objects."""
    if _use_mcp():
        results = _mcp_call_or_exit(
            lambda: _get_mcp_client().list_objects(
                doc_type=obj_type or "",
                project=project or "",
                limit=limit,
            )
        )
    else:
        store = _get_store()
        results = store.list_objects(obj_type=obj_type, project=project, limit=limit)

    if not results:
        typer.echo("No objects found.")
        return

    typer.echo(f"{len(results)} object(s):\n")
    for doc in results:
        _print_summary(doc)


@app.command()
def status() -> None:
    """Show Cortex status and health."""
    if _use_mcp():
        stats = _mcp_call_or_exit(lambda: _get_mcp_client().status())
    else:
        store = _get_store(must_init=False)
        stats = store.status()

    typer.echo("Cortex v0.1.0")
    typer.echo(f"  Initialized: {stats['initialized']}")
    typer.echo(f"  Documents:   {stats['sqlite_total']}")
    typer.echo(f"  Triples:     {stats['graph_triples']}")
    typer.echo(f"  Entities:    {stats['entities']}")

    counts = stats.get("counts_by_type", {})
    if counts:
        typer.echo("\n  By type:")
        for t, c in sorted(counts.items()):
            typer.echo(f"    {t:12s} {c}")


@app.command()
def context(
    topic: str = typer.Argument(..., help="Topic to get context for"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
) -> None:
    """Get a briefing (summaries only) for a topic."""
    if _use_mcp():
        briefs = _mcp_call_or_exit(
            lambda: _get_mcp_client().context(topic=topic, limit=limit)
        )
    else:
        store = _get_store()
        from cortex.retrieval.engine import RetrievalEngine
        from cortex.retrieval.presenters import BriefingPresenter

        engine = RetrievalEngine(store)
        results = engine.search(topic, limit=limit)
        briefs = BriefingPresenter().render(results) if results else []

    if not briefs:
        typer.echo("No context found.")
        return

    typer.echo(f"Context for '{topic}' ({len(briefs)} results):\n")
    for b in briefs:
        score = f" ({b['score']:.3f})" if b.get("score") is not None else ""
        typer.echo(f"  [{b['type']}] {b['title']}{score}")
        if b.get("summary") and b["summary"] != b["title"]:
            typer.echo(f"    {b['summary'][:120]}")


@app.command()
def dossier(
    topic: str = typer.Argument(..., help="Entity or topic for dossier"),
) -> None:
    """Build an intelligence dossier around an entity or topic."""
    if _use_mcp():
        result = _mcp_call_or_exit(
            lambda: _get_mcp_client().dossier(topic=topic)
        )
    else:
        store = _get_store()
        from cortex.retrieval.presenters import DossierPresenter

        presenter = DossierPresenter(store)
        result = presenter.render(topic)

    if result.get("status") == "no_knowledge_found":
        typer.echo(f"No knowledge found for '{topic}'.")
        return

    typer.echo(f"{'=' * 60}")
    typer.echo(f"Dossier: {topic}")
    if result.get("entity"):
        typer.echo(f"Entity:  {result['entity']['name']} ({result['entity']['type']})")
    typer.echo(f"Objects: {result.get('object_count', 0)}")
    typer.echo(f"{'=' * 60}")

    for obj in result.get("objects", []):
        typer.echo(f"\n  [{obj['type']}] {obj['title']}")
        if obj.get("summary") and obj["summary"] != obj["title"]:
            typer.echo(f"    {obj['summary'][:120]}")

    contradictions = result.get("contradictions", [])
    if contradictions:
        typer.echo(f"\n{'─' * 60}")
        typer.echo(f"Contradictions ({len(contradictions)}):")
        for c in contradictions:
            typer.echo(f"  ! {c.get('title_a', c['object_a'][:8])}")

    related = result.get("related_entities", [])
    if related:
        typer.echo(f"\n{'─' * 60}")
        typer.echo("Related entities:")
        for e in related:
            typer.echo(f"  - {e['name']} ({e['type']})")


@app.command()
def graph(
    obj_id: str = typer.Argument(..., help="Object ID to show graph for"),
) -> None:
    """Show an object's relationships and graph neighborhood."""
    if _use_mcp():
        result = _mcp_call_or_exit(
            lambda: _get_mcp_client().graph(obj_id=obj_id)
        )
        chain = result.get("causal_chain", [])
        timeline = result.get("evolution", [])
        rels = result.get("relationships", [])
    else:
        store = _get_store()
        from cortex.retrieval.graph import GraphQueries

        gq = GraphQueries(store)
        chain = gq.causal_chain(obj_id)
        timeline = gq.evolution_timeline(obj_id)
        rels = store.get_relationships(obj_id)

    # Show causal chain if applicable
    if len(chain) > 1:
        typer.echo("Causal chain:")
        for i, node in enumerate(chain):
            prefix = "  → " if i > 0 else "  "
            typer.echo(f"{prefix}[{node['type']}] {node['title']}")
        typer.echo()

    # Show evolution timeline
    if len(timeline) > 1:
        typer.echo("Evolution timeline:")
        for node in timeline:
            typer.echo(f"  {node.get('created_at', '?')[:10]}  {node['title']}")
        typer.echo()

    # Show direct relationships
    if rels:
        typer.echo("Relationships:")
        for r in rels:
            direction = "→" if r["direction"] == "outgoing" else "←"
            typer.echo(f"  {direction} {r['rel_type']} {r['other_id'][:8]}")
    else:
        typer.echo("No relationships found.")


@app.command()
def synthesize(
    period: int = typer.Option(7, "--period", "-d", help="Period in days"),
    project: str | None = typer.Option(None, "--project", "-p", help="Project"),
) -> None:
    """Generate a synthesis of recent knowledge."""
    if _use_mcp():
        result = _mcp_call_or_exit(
            lambda: _get_mcp_client().synthesize(
                period_days=period, project=project or ""
            )
        )
    else:
        store = _get_store()
        from cortex.retrieval.presenters import SynthesisPresenter

        presenter = SynthesisPresenter(store)
        result = presenter.render(period_days=period, project=project)

    if result.get("status") == "nothing_to_synthesize":
        typer.echo(f"Nothing to synthesize in the last {period} days.")
        return

    typer.echo(f"{'=' * 60}")
    typer.echo(f"Synthesis — last {period} days")
    if project:
        typer.echo(f"Project: {project}")
    typer.echo(f"Objects: {result.get('object_count', 0)}")
    typer.echo(f"{'=' * 60}\n")

    themes = result.get("themes", [])
    if themes:
        typer.echo("Themes:")
        for t in themes:
            typer.echo(f"  - {t['name']}: {t['count']}")
        typer.echo()

    narrative = result.get("narrative", "")
    if narrative:
        typer.echo(narrative)


@app.command()
def entities(
    entity_type: str | None = typer.Option(None, "--type", "-t", help="Filter by type"),
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by project"),
) -> None:
    """List resolved entities in the knowledge graph."""
    if _use_mcp() and not project:
        # MCP path: simple type-filtered listing.
        ents = _mcp_call_or_exit(
            lambda: _get_mcp_client().list_entities(entity_type=entity_type or "")
        )
    else:
        # Direct path used either when --direct OR when --project is set
        # (project_overview is direct-only — no MCP tool yet for that path).
        store = _get_store()
        if project:
            from cortex.retrieval.graph import GraphQueries

            gq = GraphQueries(store)
            overview = gq.project_overview(project)
            ents = overview.get("entities", [])
        else:
            ents = store.list_entities(entity_type=entity_type)

    if not ents:
        typer.echo("No entities found.")
        return

    typer.echo(f"{len(ents)} entity(ies):\n")
    for e in ents:
        typer.echo(f"  {e['id'][:8]}  {e['type']:12s} {e['name']}")


@app.command()
def register(
    legacy_stdio: bool = typer.Option(
        False,
        "--legacy-stdio",
        help="Register the old stdio transport instead of HTTP. Use only "
        "if you don't run a long-lived MCP HTTP server.",
    ),
) -> None:
    """Register Cortex MCP server with Claude Code.

    By default, registers the new HTTP transport pointing at
    ``http://127.0.0.1:1314/mcp``. The MCP HTTP server (started via
    ``cortex serve --transport mcp-http`` or via a launchd/systemd unit)
    must be running for Claude Code to connect.

    Pass ``--legacy-stdio`` to register the old stdio transport instead;
    this is the pre-Phase-2 behavior, where Claude Code spawns its own
    Cortex stdio child process per session.
    """
    import json

    settings_path = Path.home() / ".claude" / "settings.json"
    settings: dict[str, Any] = {}

    if settings_path.exists():
        settings = json.loads(settings_path.read_text())

    mcp_servers = settings.setdefault("mcpServers", {})

    if legacy_stdio:
        import shutil

        cortex_bin = shutil.which("cortex")
        if cortex_bin:
            cmd = cortex_bin
            args = ["serve", "--transport", "stdio"]
        else:
            cmd = sys.executable
            args = ["-m", "cortex.transport.mcp"]

        mcp_servers["cortex"] = {
            "command": cmd,
            "args": args,
            "env": {},
        }
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")

        typer.echo(f"Registered Cortex MCP (stdio) at {settings_path}")
        typer.echo(f"  Command: {cmd} {' '.join(args)}")
    else:
        config = load_config()
        mcp_servers["cortex"] = {
            "type": "http",
            "url": config.mcp_server_url,
        }
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")

        typer.echo(f"Registered Cortex MCP (http) at {settings_path}")
        typer.echo(f"  URL: {config.mcp_server_url}")
        typer.echo(
            "  Make sure the MCP HTTP server is running:\n"
            "    cortex serve --transport mcp-http --host 127.0.0.1 --port 1314"
        )

    typer.echo("  Restart Claude Code to activate.")


@app.command()
def install(
    service: str = typer.Option("all", "--service", help="mcp, dashboard, or all"),
) -> None:
    """Install Cortex as a background service (auto-start on login)."""
    from cortex.cli.install import do_install

    config = load_config()
    do_install(config, service=service)


@app.command()
def uninstall(
    service: str = typer.Option("all", "--service", help="mcp, dashboard, or all"),
) -> None:
    """Remove Cortex background services."""
    from cortex.cli.install import do_uninstall

    config = load_config()
    do_uninstall(config, service=service)


@app.command()
def backup(
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output directory for the backup archive"
    ),
) -> None:
    """Create a backup of Cortex data (cortex.db + graph.db)."""
    from cortex.cli.backup import do_backup

    config = load_config()
    do_backup(config, output=Path(output) if output else None)


@app.command()
def restore(
    archive: str = typer.Argument(..., help="Path to backup archive (.tar.gz)"),
) -> None:
    """Restore Cortex data from a backup archive."""
    from cortex.cli.backup import do_restore

    config = load_config()
    do_restore(config, archive=Path(archive))


def _start_parent_watchdog() -> None:
    """Spawn a daemon thread that exits the process when the parent dies.

    Bundle 9 / A.3: When ``cortex dashboard --spawn-mcp`` launches an MCP
    HTTP child, an ``atexit`` handler in the parent terminates the child
    on graceful exit. ``atexit`` does NOT run on ``SIGKILL`` / hard crash,
    so the child would otherwise outlive the parent and keep the lock on
    ``graph.db``. This watchdog closes that gap by relying on the OS:
    when the parent dies, its end of the stdin pipe is closed, our
    blocking ``sys.stdin.read()`` returns, and we ``os._exit(0)``.

    Why ``os._exit`` and not ``sys.exit``: FastMCP's transport runs an
    asyncio event loop with its own signal handlers; raising SystemExit
    from a background thread does not necessarily reach those handlers
    cleanly. ``os._exit`` terminates the process immediately so the lock
    on ``graph.db`` is released and the next caller can auto-recover.
    """
    import os as _os
    import sys as _sys
    import threading

    def _watch() -> None:
        with contextlib.suppress(Exception):
            # Blocks until the pipe is closed (parent died) or EOF.
            _sys.stdin.read()
        # Parent is gone — exit hard so the lock is released immediately.
        _os._exit(0)

    t = threading.Thread(target=_watch, name="cortex-parent-watchdog", daemon=True)
    t.start()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(1314, "--port", help="Bind port"),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        help=(
            "Transport: stdio (default, for Claude Code), "
            "mcp-http (HTTP MCP server), or http (REST API)"
        ),
    ),
    parent_watchdog: bool = typer.Option(
        False,
        "--parent-watchdog",
        hidden=True,
        help=(
            "(internal) Exit when stdin closes. Set automatically by "
            "`cortex dashboard --spawn-mcp` so the spawned MCP child cannot "
            "outlive the dashboard even if the dashboard is SIGKILL'd."
        ),
    ),
) -> None:
    """Start the Cortex server.

    Transports:
        stdio    — MCP over stdio (default; for Claude Code, Cursor, etc.)
        mcp-http — MCP over streamable-http (for browser dashboard + multi-client setups)
        http     — REST API with API-key auth (for remote agents)
    """
    if transport == "stdio":
        from cortex.transport.mcp.server import run_stdio
        try:
            run_stdio()
        except StoreLockedError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from e
    elif transport == "mcp-http":
        from cortex.transport.mcp.server import run_http
        typer.echo(f"Cortex MCP (streamable-http) at http://{host}:{port}/mcp")
        if parent_watchdog:
            _start_parent_watchdog()
        try:
            run_http(host=host, port=port)
        except StoreLockedError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from e
        except PermissionError:
            typer.secho(
                f"Permission denied binding to port {port}. Use a port >= 1024 "
                f"or run with elevated privileges.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1) from None
        except OSError as e:
            # Catches "address already in use", DNS errors, other network
            # binding failures. Produces a clean message instead of a traceback.
            typer.secho(
                f"Cannot start MCP HTTP server on {host}:{port}: {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1) from e
    elif transport == "http":
        import uvicorn

        from cortex.transport.api.server import create_api
        from cortex.transport.mcp.client import MCPClientError

        cfg = load_config()
        # Phase 4: the REST API is now a thin MCP HTTP client. Probe the
        # MCP server at startup and fail fast with an actionable error if
        # it's unreachable or missing required tools. Mirrors the
        # dashboard startup probe.
        try:
            available = _probe_mcp_server(cfg.mcp_server_url)
        except MCPClientError as e:
            typer.secho(
                f"Cannot reach Cortex MCP server at {cfg.mcp_server_url}",
                fg=typer.colors.RED,
                err=True,
            )
            typer.secho(f"  {e}", fg=typer.colors.RED, err=True)
            typer.echo(
                "  Start it in another terminal:\n"
                "    cortex serve --transport mcp-http --host 127.0.0.1 --port 1314",
                err=True,
            )
            raise typer.Exit(1) from e
        missing = _REQUIRED_MCP_TOOLS - available
        if missing:
            typer.secho(
                f"MCP server at {cfg.mcp_server_url} is missing required "
                f"tools: {', '.join(sorted(missing))}",
                fg=typer.colors.RED,
                err=True,
            )
            typer.echo(
                "  The MCP server may be from a different Cortex version. "
                "Restart it from the current working directory.",
                err=True,
            )
            raise typer.Exit(1)

        api = create_api(cfg)
        uvicorn.run(api, host=host, port=port)
    else:
        typer.echo(
            f"Unknown transport: {transport}. Valid: stdio, mcp-http, http",
            err=True,
        )
        raise typer.Exit(1)


@app.command()
def setup(
    auto: bool = typer.Option(False, "--auto", help="Non-interactive with defaults"),
) -> None:
    """Set up Cortex — interactive wizard or auto mode."""
    import bcrypt

    config = load_config()
    setup_logging(level=config.log_level, json_output=False)

    typer.echo("Cortex Setup\n")

    # 1. Data directory
    typer.echo(f"  Data directory: {config.data_dir}")
    if config.data_dir.exists():
        typer.echo("  (already exists)")
    else:
        config.data_dir.mkdir(parents=True, exist_ok=True)
        typer.echo("  (created)")

    # 2. Initialize stores
    store = _open_store_or_exit(config)
    try:
        ontology_path = find_ontology()
        store.initialize(ontology_path)
        typer.echo("  Ontology loaded")
    except FileNotFoundError:
        typer.echo("  Ontology not found — skipping")

    _warmup_embeddings(config)

    # 3. Dashboard password
    if not auto:
        set_pw = typer.confirm("  Set a dashboard password?", default=False)
        if set_pw:
            pw = typer.prompt("  Password", hide_input=True)
            pw_hash = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
            store.content.set_config("dashboard_password_hash", pw_hash)
            typer.echo("  Password set")
    else:
        typer.echo("  Dashboard: open access (no password)")

    # 4. LLM test
    if config.llm_model and config.llm_api_key:
        typer.echo(f"  LLM: {config.llm_model}")
        try:
            from cortex.services.llm import LLMClient
            llm = LLMClient(config)
            llm.complete("Say 'connected' in one word.")
            typer.echo("  LLM: Connected")
        except Exception as e:
            typer.echo(f"  LLM: Failed — {e}")
    else:
        typer.echo("  LLM: not configured (set CORTEX_LLM_MODEL and CORTEX_LLM_API_KEY)")

    typer.echo(f"\nCortex ready at {config.data_dir}")
    typer.echo("  Run `cortex status` to verify.")
    typer.echo("  Run `cortex serve` to start the server.")

    global _store
    _store = store


@app.command(name="import-v1")
def import_v1(
    db_path: str = typer.Argument(..., help="Path to Cortex v1 SQLite database"),
) -> None:
    """Import from a Cortex v1 database."""
    store = _get_store()
    from cortex.pipeline.importer import CortexV1Importer

    importer = CortexV1Importer(store)
    result = importer.run(Path(db_path))

    if result.get("status") == "error":
        typer.echo(f"Error: {result.get('message')}", err=True)
        raise typer.Exit(1)

    typer.echo("Import complete:")
    typer.echo(f"  Imported: {result['imported']}")
    typer.echo(f"  Skipped:  {result['skipped']} (duplicates)")
    typer.echo(f"  Failed:   {result['failed']}")


@app.command(name="import-vault")
def import_vault(
    vault_path: str = typer.Argument(..., help="Path to Obsidian vault"),
    skip_pipeline: bool = typer.Option(
        False, "--skip-pipeline", help="Fast import without running pipeline stages"
    ),
) -> None:
    """Import from an Obsidian vault."""
    store = _get_store()
    pipeline = _get_pipeline() if not skip_pipeline else None
    from cortex.pipeline.importer import ObsidianImporter

    importer = ObsidianImporter(store, pipeline=pipeline)
    result = importer.run(Path(vault_path))

    if result.get("status") == "error":
        typer.echo(f"Error: {result.get('message')}", err=True)
        raise typer.Exit(1)

    typer.echo("Import complete:")
    typer.echo(f"  Imported: {result['imported']}")
    typer.echo(f"  Skipped:  {result['skipped']} (duplicates/filtered)")
    typer.echo(f"  Failed:   {result['failed']}")
    wiki_links = result.get("wiki_links_created", 0)
    if wiki_links:
        typer.echo(f"  Wiki-link relationships: {wiki_links}")


_REQUIRED_MCP_TOOLS = frozenset(
    {
        "cortex_search",
        "cortex_list",
        "cortex_read",
        "cortex_capture",
        "cortex_dossier",
        "cortex_status",
        "cortex_query_trail",
        "cortex_graph_data",
        "cortex_list_entities",
    }
)


def _probe_mcp_server(url: str, *, retries: int = 3, retry_delay: float = 1.0) -> set[str]:
    """Verify the MCP HTTP server is reachable AND has the expected tools.

    Returns the set of tool names exposed. Raises ``MCPConnectionError`` /
    ``MCPTimeoutError`` after exhausting retries. Used by ``cortex dashboard``
    to fail fast with a clear error if the MCP server isn't running.

    Bundle 10.9: the per-attempt timeout is now overridable via the
    ``CORTEX_PROBE_TIMEOUT_SECONDS`` env var (same knob that
    ``_get_probe_client`` honours), defaulting to 3s for fast local
    fail-detection. CI and integration tests set this higher to
    accommodate slow cold starts on macOS runners.
    """
    import asyncio
    import os
    import time

    from cortex.transport.mcp.client import (
        CortexMCPClient,
        MCPClientError,
    )

    timeout_str = os.environ.get("CORTEX_PROBE_TIMEOUT_SECONDS", "").strip()
    try:
        timeout = float(timeout_str) if timeout_str else 3.0
    except ValueError:
        timeout = 3.0
    client = CortexMCPClient(url, timeout_seconds=timeout)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return set(asyncio.run(client.list_tools()))
        except MCPClientError as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(retry_delay)
    assert last_error is not None
    raise last_error


def _spawn_mcp_subprocess(url: str, data_dir: Path):
    """Spawn ``cortex serve --transport mcp-http`` and wait for it ready.

    Bundle 8 / B2: called by ``cortex dashboard --spawn-mcp`` when the
    initial probe fails. Launches the MCP server as a subprocess on the
    host/port derived from ``url``, polls for readiness via
    ``_probe_mcp_server``, and returns the Popen handle so the caller can
    register cleanup on exit.

    Raises:
        RuntimeError: If the subprocess failed to become ready within the
            timeout. The subprocess has already been terminated in that
            case.

    Log files:
        stdout → ``<data_dir>/mcp-http.log``
        stderr → ``<data_dir>/mcp-http.err``
        (Same paths the LaunchAgent uses — consistent for users.)
    """
    import os as _os
    import subprocess
    import time
    from urllib.parse import urlparse

    from cortex.transport.mcp.client import MCPClientError

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 1314

    data_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = data_dir / "mcp-http.log"
    stderr_path = data_dir / "mcp-http.err"
    stdout_f = stdout_path.open("a")
    stderr_f = stderr_path.open("a")

    env = _os.environ.copy()
    env.setdefault("CORTEX_DATA_DIR", str(data_dir))

    # Bundle 9 / A.3: pass stdin=PIPE and --parent-watchdog so the spawned
    # child exits if our process dies (even by SIGKILL, which atexit can't
    # handle). The watchdog thread in the child blocks on sys.stdin.read();
    # when our process exits the OS closes the pipe, the read returns, and
    # the child os._exit(0)'s, releasing the lock on graph.db.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "cortex.cli.main",
            "serve",
            "--transport",
            "mcp-http",
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watchdog",
        ],
        env=env,
        stdin=subprocess.PIPE,
        stdout=stdout_f,
        stderr=stderr_f,
    )

    # Poll readiness for up to ~15s total (5 retry cycles x 3s timeout each).
    deadline = time.time() + 15.0
    last_error: Exception | None = None
    while time.time() < deadline:
        # Bail early if the subprocess died — no point polling a zombie.
        if proc.poll() is not None:
            stdout_f.close()
            stderr_f.close()
            log_tail = ""
            with contextlib.suppress(OSError):
                log_tail = stderr_path.read_text()[-500:]
            raise RuntimeError(
                f"Spawned MCP subprocess exited with code {proc.returncode} "
                f"before becoming ready. See {stderr_path} for details.\n"
                f"--- tail of stderr ---\n{log_tail}"
            )
        try:
            _probe_mcp_server(url, retries=1, retry_delay=0.0)
            return proc
        except MCPClientError as e:
            last_error = e
            time.sleep(0.5)

    # Timeout — kill the subprocess so we don't leave orphans behind.
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    finally:
        stdout_f.close()
        stderr_f.close()

    raise RuntimeError(
        f"Spawned MCP subprocess did not become ready within 15s at {url}. "
        f"Last probe error: {last_error}"
    )


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(1315, "--port", help="Bind port"),
    spawn_mcp: bool = typer.Option(
        False,
        "--spawn-mcp",
        help=(
            "If the MCP HTTP server is unreachable, spawn it as a subprocess "
            "and wait for it to be ready. The subprocess is terminated when "
            "the dashboard exits. Skipped silently if the MCP server is "
            "already running."
        ),
    ),
) -> None:
    """Start the web dashboard.

    The dashboard is a thin client of the MCP HTTP server. It probes the
    configured ``mcp_server_url`` at startup and refuses to start if the
    server isn't reachable or is missing required tools.

    With ``--spawn-mcp`` (Bundle 8 / B2), an unreachable MCP server is
    launched as a subprocess on the configured host/port and waited for.
    The subprocess is terminated when the dashboard process exits.
    """
    import atexit

    import uvicorn

    from cortex.dashboard.server import create_dashboard
    from cortex.transport.mcp.client import MCPClientError

    config = load_config()

    # Probe the MCP server before starting uvicorn so the user gets an
    # actionable error instead of a half-broken dashboard.
    spawned_proc = None
    try:
        available = _probe_mcp_server(config.mcp_server_url)
    except MCPClientError as probe_err:
        if not spawn_mcp:
            typer.secho(
                f"Cannot reach Cortex MCP server at {config.mcp_server_url}",
                fg=typer.colors.RED,
                err=True,
            )
            typer.secho(f"  {probe_err}", fg=typer.colors.RED, err=True)
            typer.echo(
                "  Start it in another terminal:\n"
                "    cortex serve --transport mcp-http --host 127.0.0.1 --port 1314\n"
                "  Or rerun with --spawn-mcp to have the dashboard launch it for you.",
                err=True,
            )
            raise typer.Exit(1) from probe_err

        # --spawn-mcp: launch the MCP server as a subprocess and wait.
        typer.secho(
            f"MCP server at {config.mcp_server_url} unreachable — "
            f"spawning one via --spawn-mcp…",
            fg=typer.colors.YELLOW,
            err=True,
        )
        try:
            spawned_proc = _spawn_mcp_subprocess(
                config.mcp_server_url, config.data_dir
            )
        except RuntimeError as spawn_err:
            typer.secho(
                f"Failed to spawn MCP subprocess: {spawn_err}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1) from spawn_err

        # Register cleanup so the child doesn't outlive the dashboard.
        def _terminate_spawned():
            if spawned_proc is None or spawned_proc.poll() is not None:
                return
            try:
                spawned_proc.terminate()
                spawned_proc.wait(timeout=10)
            except Exception:
                try:
                    spawned_proc.kill()
                    spawned_proc.wait()
                except Exception:
                    pass

        atexit.register(_terminate_spawned)

        typer.secho(
            f"Spawned MCP server (PID {spawned_proc.pid}) — "
            f"logs in {config.data_dir}/mcp-http.{{log,err}}",
            fg=typer.colors.GREEN,
            err=True,
        )

        # Re-probe now that the subprocess is up.
        try:
            available = _probe_mcp_server(config.mcp_server_url)
        except MCPClientError as e:
            typer.secho(
                f"Spawned MCP server is not responding: {e}",
                fg=typer.colors.RED,
                err=True,
            )
            _terminate_spawned()
            raise typer.Exit(1) from e

    missing = _REQUIRED_MCP_TOOLS - available
    if missing:
        typer.secho(
            f"MCP server at {config.mcp_server_url} is missing required tools: "
            f"{', '.join(sorted(missing))}",
            fg=typer.colors.RED,
            err=True,
        )
        typer.echo(
            "This usually means the MCP server is from an older version of Cortex. "
            "Update or restart it.",
            err=True,
        )
        raise typer.Exit(1)

    dash = create_dashboard(config)
    typer.echo(f"Dashboard at http://{host}:{port}")
    uvicorn.run(dash, host=host, port=port)


@app.command(name="pipeline")
def run_pipeline_cmd(
    obj_id: str = typer.Argument(None, help="Object ID to run pipeline on"),
    batch: bool = typer.Option(False, "--batch", help="Process all un-pipelined docs"),
) -> None:
    """Re-run the intelligence pipeline on one or all objects."""
    if batch:
        # Batch mode reaches into store.content._db for the SQL query — keep
        # it on the direct path. The MCP server has no batch tool today.
        if not _direct_mode:
            typer.secho(
                "--batch requires --direct (no MCP tool for batch pipeline runs).",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        store = _get_store()

        docs = store.content._db.execute(
            "SELECT id, title FROM documents"
            " WHERE pipeline_stage = 'ingest'"
            " ORDER BY created_at ASC"
        ).fetchall()

        total = len(docs)
        if total == 0:
            typer.echo("No documents pending pipeline processing.")
            return

        typer.echo(f"Processing {total} documents through pipeline...\n")

        pipe = _get_pipeline()
        succeeded = 0
        failed = 0

        for i, doc in enumerate(docs, 1):
            doc_id = doc["id"]
            title = doc["title"][:50]
            try:
                typer.echo(f"  [{i}/{total}] {title}...", nl=False)
                result = pipe.run_pipeline(doc_id)
                status = result.get("status", "?")
                typer.echo(f" {status}")
                succeeded += 1
            except Exception as e:
                typer.echo(f" FAILED: {e}")
                failed += 1

        typer.echo(f"\nBatch complete: {succeeded} succeeded, {failed} failed out of {total}")
        return

    if obj_id is None:
        typer.echo("Error: provide OBJ_ID or use --batch", err=True)
        raise typer.Exit(1)

    if _use_mcp():
        result = _mcp_call_or_exit(
            lambda: _get_mcp_client().pipeline(obj_id=obj_id)
        )
        if "error" in result:
            typer.echo(result["error"], err=True)
            raise typer.Exit(1)
    else:
        store = _get_store()
        doc = store.read(obj_id)
        if doc is None:
            typer.echo(f"Not found: {obj_id}", err=True)
            raise typer.Exit(1)

        pipe = _get_pipeline()
        result = pipe.run_pipeline(obj_id)
    typer.echo(f"Pipeline {result.get('status', 'unknown')} for {obj_id[:12]}")
    for stage, data in result.get("pipeline_stages", {}).items():
        typer.echo(f"  {stage}: {data.get('status', '?')}")


@app.command()
def reason() -> None:
    """Run advanced reasoning (contradictions, patterns, gaps, staleness)."""
    if _use_mcp():
        results = _mcp_call_or_exit(lambda: _get_mcp_client().reason())
    else:
        store = _get_store()
        from cortex.pipeline.advanced_reason import AdvancedReasoner

        reasoner = AdvancedReasoner(store)
        results = reasoner.run_all()

    total = sum(len(v) for v in results.values() if isinstance(v, list))
    typer.echo(f"Advanced reasoning: {total} finding(s)")
    for category, findings in results.items():
        if findings:
            typer.echo(f"\n  {category} ({len(findings)}):")
            for f in findings:
                typer.echo(f"    [{f.get('severity', '?')}] {f.get('message', '')}")


def _print_summary(doc: dict) -> None:
    """Print a single-line summary of a document."""
    doc_id = doc.get("id", "?")
    doc_type = doc.get("type", "?")
    title = doc.get("title", "Untitled")
    project = doc.get("project", "")
    proj_str = f" [{project}]" if project else ""
    typer.echo(f"  {doc_id[:8]}  {doc_type:12s} {title}{proj_str}")


# ─── `cortex doctor` sub-app (Bundle 8 / B1) ─────────────────────────────

doctor_app = typer.Typer(
    name="doctor",
    help="Diagnostic and recovery commands for a Cortex install.",
    no_args_is_help=True,
)
app.add_typer(doctor_app, name="doctor")


@doctor_app.command("unlock")
def doctor_unlock(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report what would be removed without touching anything.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Skip the PID-alive check and remove the marker + RocksDB LOCK "
            "file unconditionally. Use when a cross-user PermissionError "
            "prevents the normal liveness check from working."
        ),
    ),
) -> None:
    """Clean up a stale graph.db lock from a crashed Cortex process.

    Reads ``~/.cortex/graph.db.lock``, verifies the recorded holder PID is
    really dead, then removes both the marker file and the RocksDB ``LOCK``
    file inside the graph DB directory. Safe to run when nothing is locked
    (no-op). Refuses to unlock a living holder unless ``--force`` is given.
    """
    from cortex.db.graph_store import (
        _auto_recover_stale_lock,
        _marker_path_for,
        _pid_alive,
        _process_cmdline,
        _read_marker,
    )

    config = load_config()
    db_path = config.data_dir / "graph.db"
    marker_path = _marker_path_for(db_path)
    rocksdb_lock = db_path / "LOCK"

    if not marker_path.exists() and not rocksdb_lock.exists():
        typer.secho(
            "No marker file or RocksDB LOCK found — nothing to unlock.",
            fg=typer.colors.GREEN,
        )
        raise typer.Exit(0)

    marker = _read_marker(marker_path) if marker_path.exists() else None

    if marker is not None and marker.get("_unreadable"):
        typer.secho(
            f"Marker file {marker_path} exists but is unreadable. "
            f"Check its permissions (chmod/chown) and retry.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    holder_pid: int | None = None
    holder_cmdline: str | None = None
    if marker is not None:
        raw_pid = marker.get("pid")
        if isinstance(raw_pid, int):
            holder_pid = raw_pid
        raw_cmdline = marker.get("cmdline")
        if isinstance(raw_cmdline, str):
            holder_cmdline = raw_cmdline

    if dry_run:
        typer.echo("Dry run — no files will be removed.")
        typer.echo(f"  marker: {marker_path} (exists={marker_path.exists()})")
        typer.echo(
            f"  rocksdb LOCK: {rocksdb_lock} (exists={rocksdb_lock.exists()})"
        )
        if holder_pid is not None:
            alive = _pid_alive(holder_pid)
            typer.echo(
                f"  holder PID: {holder_pid} "
                f"({'alive' if alive else 'dead'})"
            )
            if holder_cmdline:
                typer.echo(f"  holder cmdline: {holder_cmdline}")
        raise typer.Exit(0)

    if force:
        removed: list[str] = []
        if marker_path.exists():
            try:
                marker_path.unlink(missing_ok=True)
                removed.append(str(marker_path))
            except OSError as e:
                typer.secho(
                    f"Could not remove marker {marker_path}: {e}",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(1) from e
        if rocksdb_lock.exists():
            try:
                rocksdb_lock.unlink(missing_ok=True)
                removed.append(str(rocksdb_lock))
            except OSError as e:
                typer.secho(
                    f"Could not remove RocksDB LOCK {rocksdb_lock}: {e}",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(1) from e
        typer.secho(
            f"Force-unlocked. Removed: {', '.join(removed) or 'nothing'}",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(0)

    # Normal path: require that we can prove the holder is gone.
    if holder_pid is None:
        # No marker (or malformed). Just remove whatever stale file is left.
        removed = []
        if marker_path.exists():
            marker_path.unlink(missing_ok=True)
            removed.append(str(marker_path))
        if rocksdb_lock.exists():
            rocksdb_lock.unlink(missing_ok=True)
            removed.append(str(rocksdb_lock))
        typer.secho(
            f"Unlocked. No holder PID known; removed: "
            f"{', '.join(removed) or 'nothing'}",
            fg=typer.colors.GREEN,
        )
        raise typer.Exit(0)

    if _pid_alive(holder_pid):
        # Before refusing, check for PID reuse (same PID, different cmdline).
        live_cmdline = _process_cmdline(holder_pid)
        is_reuse = (
            live_cmdline is not None
            and holder_cmdline is not None
            and live_cmdline != holder_cmdline
        )
        cmdline_unknown = (
            live_cmdline is None and holder_cmdline is not None
        )
        if is_reuse:
            typer.secho(
                f"PID {holder_pid} is alive but its cmdline does NOT match "
                f"the marker. The OS has reused the PID for an unrelated "
                f"process. Pass --force to unlock anyway.",
                fg=typer.colors.RED,
                err=True,
            )
        elif cmdline_unknown:
            # Bundle 10.7 / B.2: ps returned nothing for the live PID, so
            # we can't confirm this is the same process. Stay conservative
            # and refuse without --force, but tell the user exactly why.
            typer.secho(
                f"PID {holder_pid} is alive, but its cmdline could NOT be "
                f"read (ps/procfs returned nothing). We cannot verify this "
                f"is the same process that holds the lock. Refusing to "
                f"unlock without --force.",
                fg=typer.colors.RED,
                err=True,
            )
            typer.echo(
                "  If you are sure the marker is stale, run: "
                "cortex doctor unlock --force",
                err=True,
            )
        else:
            typer.secho(
                f"PID {holder_pid} is still running — refusing to unlock "
                f"a live holder.",
                fg=typer.colors.RED,
                err=True,
            )
            typer.echo(
                f"  Stop it first: kill {holder_pid}\n"
                f"  Or override with: cortex doctor unlock --force",
                err=True,
            )
        raise typer.Exit(1)

    # Holder is dead — safe to auto-recover via the shared helper.
    cleaned = _auto_recover_stale_lock(db_path, marker_path, holder_pid)
    if not cleaned:
        # Race: PID came back alive during the re-check.
        typer.secho(
            f"PID {holder_pid} became alive during the re-check; aborted.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    typer.secho(
        f"Unlocked. Holder PID {holder_pid} was dead; removed "
        f"{marker_path} and {rocksdb_lock}.",
        fg=typer.colors.GREEN,
    )


# --------------------------------------------------------------------------- #
# Bundle 10.7 / F.4 — ``cortex doctor logs``                                  #
# --------------------------------------------------------------------------- #

# Log files written by the LaunchAgents (see deploy/*.plist) and by
# ``cortex dashboard --spawn-mcp`` via ``_spawn_mcp_subprocess``.
_LAUNCHAGENT_LOG_FILENAMES = (
    "mcp-http.log",
    "mcp-http.err",
    "dashboard.log",
    "dashboard.err",
)

# Size thresholds for the summary status badge.
_LOG_SIZE_GREEN_MAX = 10 * 1024 * 1024   # 10 MB
_LOG_SIZE_YELLOW_MAX = 100 * 1024 * 1024  # 100 MB


def _format_bytes(n: int) -> str:
    """Return a compact human-readable byte string (e.g. ``17.4 MB``)."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _log_status_color(size_bytes: int) -> str:
    if size_bytes < _LOG_SIZE_GREEN_MAX:
        return typer.colors.GREEN
    if size_bytes < _LOG_SIZE_YELLOW_MAX:
        return typer.colors.YELLOW
    return typer.colors.RED


def _log_status_label(size_bytes: int) -> str:
    if size_bytes < _LOG_SIZE_GREEN_MAX:
        return "GREEN"
    if size_bytes < _LOG_SIZE_YELLOW_MAX:
        return "YELLOW"
    return "RED"


def _count_lines(path: Path) -> int:
    """Count newlines in a file without loading it into memory."""
    count = 0
    with path.open("rb") as f:
        while chunk := f.read(65536):
            count += chunk.count(b"\n")
    return count


def _tail_lines(path: Path, n: int) -> list[str]:
    """Return the last ``n`` lines of ``path`` using a bounded deque."""
    from collections import deque

    with path.open("r", errors="replace") as f:
        return list(deque(f, maxlen=n))


def _summarize_logs(log_paths: list[Path]) -> None:
    """Default view for ``cortex doctor logs`` — show per-file size,
    line count, mtime, and a status badge.
    """
    import datetime

    any_present = False
    for path in log_paths:
        if not path.exists():
            typer.echo(f"  {path.name:<20}  (not present)")
            continue
        any_present = True
        stat = path.stat()
        size_bytes = stat.st_size
        lines = _count_lines(path)
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        color = _log_status_color(size_bytes)
        label = _log_status_label(size_bytes)
        typer.echo(
            f"  {path.name:<20}  {_format_bytes(size_bytes):>10}  "
            f"{lines:>10} lines  last modified {mtime}  ",
            nl=False,
        )
        typer.secho(f"[{label}]", fg=color)
    if not any_present:
        typer.secho(
            "  No log files found under the current data dir.",
            fg=typer.colors.GREEN,
        )


def _tail_logs(log_paths: list[Path], n: int) -> None:
    """``--tail N`` view: show the last ``n`` lines of each existing
    log file.
    """
    for path in log_paths:
        if not path.exists():
            continue
        typer.secho(f"==> {path} (last {n} lines) <==", fg=typer.colors.CYAN)
        lines = _tail_lines(path, n)
        if not lines:
            typer.echo("  (empty)")
            continue
        for line in lines:
            # Preserve existing newlines, strip only the trailing one.
            typer.echo(line.rstrip("\n"))
        typer.echo("")


def _rotate_logs(log_paths: list[Path]) -> None:
    """``--rotate``: back up each non-empty log to ``<file>.old`` and
    truncate the live file to zero length.

    Safe while ``launchd`` is running the LaunchAgent because
    ``StandardOutPath`` / ``StandardErrorPath`` are opened with
    ``O_APPEND``: every subsequent ``write()`` in the server process
    atomically seeks to EOF first, so after ``os.truncate(path, 0)`` the
    next write lands at offset 0. No restart needed.
    """
    import os as _os
    import shutil

    total_freed = 0
    rotated: list[tuple[str, int]] = []
    skipped: list[str] = []

    for path in log_paths:
        if not path.exists():
            skipped.append(f"{path.name} (not present)")
            continue
        size_before = path.stat().st_size
        if size_before == 0:
            skipped.append(f"{path.name} (already empty)")
            continue

        backup = path.with_name(path.name + ".old")
        try:
            if backup.exists():
                backup.unlink()
            shutil.copyfile(path, backup)
            _os.truncate(path, 0)
        except OSError as e:
            typer.secho(
                f"  Failed to rotate {path}: {e}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1) from e

        rotated.append((path.name, size_before))
        total_freed += size_before

    if rotated:
        typer.secho("Rotated:", fg=typer.colors.GREEN)
        for name, size in rotated:
            typer.echo(f"  {name:<20}  {_format_bytes(size):>10}  → {name}.old")
        typer.secho(
            f"Total freed: {_format_bytes(total_freed)}",
            fg=typer.colors.GREEN,
        )
    if skipped:
        typer.echo("Skipped:")
        for label in skipped:
            typer.echo(f"  {label}")


@doctor_app.command("logs")
def doctor_logs(
    tail: int = typer.Option(
        0,
        "--tail",
        "-n",
        help="Show the last N lines of each log file (0 = summary view).",
    ),
    rotate: bool = typer.Option(
        False,
        "--rotate",
        help=(
            "Back up current log files to <file>.old and truncate them to "
            "zero length. Safe while the LaunchAgent is running — launchd "
            "opens stdout/stderr with O_APPEND, so the server's next "
            "write lands at offset 0 without needing a restart."
        ),
    ),
) -> None:
    """Inspect or rotate the Cortex LaunchAgent log files.

    Default: show size, line count, last-modified time, and a traffic-light
    status badge for each of ``mcp-http.log``, ``mcp-http.err``,
    ``dashboard.log``, ``dashboard.err`` under the current data dir.
    """
    config = load_config()
    log_paths = [config.data_dir / name for name in _LAUNCHAGENT_LOG_FILENAMES]

    if rotate:
        _rotate_logs(log_paths)
        raise typer.Exit(0)

    if tail > 0:
        _tail_logs(log_paths, tail)
        raise typer.Exit(0)

    typer.secho(f"Log files under {config.data_dir}:", fg=typer.colors.CYAN)
    _summarize_logs(log_paths)


# --------------------------------------------------------------------------- #
# S4 + S5 — ``cortex doctor check`` / ``cortex doctor repair``                #
# --------------------------------------------------------------------------- #


@doctor_app.command("check")
def doctor_check() -> None:
    """Run health checks on the Cortex data stores.

    Validates store presence, FTS5 index consistency, reasoner fixpoint,
    and server connectivity. Fully read-only — safe to run at any time.
    """
    config = load_config()
    all_ok = True

    typer.secho("Cortex health check:", fg=typer.colors.CYAN)

    # 1. Stores exist
    db_ok = config.sqlite_db_path.exists()
    graph_ok = config.graph_db_path.exists() and config.graph_db_path.is_dir()
    if db_ok and graph_ok:
        typer.echo("  Stores:     OK (cortex.db + graph.db present)")
    else:
        missing = []
        if not db_ok:
            missing.append("cortex.db")
        if not graph_ok:
            missing.append("graph.db")
        typer.secho(
            f"  Stores:     FAIL — missing: {', '.join(missing)}",
            fg=typer.colors.RED,
        )
        typer.echo("  Run `cortex init` to create data stores.")
        raise typer.Exit(1)

    # 2. FTS5 consistency
    from cortex.db.content_store import ContentStore

    content = ContentStore(path=config.sqlite_db_path)
    try:
        fts = content.fts_integrity_check()
        if fts["ok"]:
            typer.echo(
                f"  FTS5 index: OK ({fts['documents_count']} documents indexed)"
            )
        else:
            all_ok = False
            typer.secho(
                f"  FTS5 index: WARN — index inconsistent: {fts['error']}",
                fg=typer.colors.YELLOW,
            )
            typer.echo("              Run `cortex doctor repair` to rebuild.")
    finally:
        content.close()

    # 3. Reasoner fixpoint
    try:
        from cortex.db.graph_store import GraphStore
        from cortex.pipeline.reason import ReasonStage

        graph = GraphStore(path=config.graph_db_path)
        try:
            reasoner = ReasonStage(graph)
            fixpoint = reasoner.check_fixpoint()
            if fixpoint["ok"]:
                typer.echo("  Reasoner:   OK (fixpoint reached, 0 pending triples)")
            else:
                all_ok = False
                typer.secho(
                    f"  Reasoner:   WARN — {fixpoint['total_pending']} "
                    "triples pending inference",
                    fg=typer.colors.YELLOW,
                )
                typer.echo("              Run `cortex doctor repair` to reach fixpoint.")
        finally:
            graph.close()
    except StoreLockedError:
        typer.secho(
            "  Reasoner:   SKIP (graph.db locked — server is running)",
            fg=typer.colors.YELLOW,
        )
        typer.echo("              Stop server first, or check results via MCP.")

    # 4. Server connectivity
    from cortex.cli.backup import _check_server_running

    running, pid, _cmdline = _check_server_running(config)
    if running:
        typer.echo(f"  Server:     OK (PID {pid}, port {config.port})")
    else:
        typer.echo("  Server:     not running")

    if all_ok:
        typer.secho("\nAll checks passed.", fg=typer.colors.GREEN)
    else:
        typer.secho(
            "\nSome checks reported warnings. Run `cortex doctor repair` to fix.",
            fg=typer.colors.YELLOW,
        )


@doctor_app.command("repair")
def doctor_repair() -> None:
    """Repair Cortex data stores.

    Rebuilds the FTS5 search index, runs the reasoner to fixpoint, and
    checkpoints the SQLite WAL. Requires the server to be stopped.
    """
    config = load_config()

    # Refuse if server is running
    from cortex.cli.backup import _check_server_running, _checkpoint_sqlite

    running, pid, _cmdline = _check_server_running(config)
    if running:
        typer.secho(
            f"Cortex server is running (PID {pid}). "
            f"Stop it first:\n  cortex uninstall\n  # or: kill {pid}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # Verify stores exist
    if not config.sqlite_db_path.exists():
        typer.secho(
            "cortex.db not found. Run `cortex init` first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    typer.secho("Cortex repair:", fg=typer.colors.CYAN)

    # 1. FTS5 rebuild
    from cortex.db.content_store import ContentStore

    content = ContentStore(path=config.sqlite_db_path)
    try:
        result = content.fts_rebuild()
        typer.echo(
            f"  FTS5 index: rebuilt ({result['documents_count']} documents reindexed)"
        )
    finally:
        content.close()

    # 2. Reasoner fixpoint
    if config.graph_db_path.exists():
        from cortex.db.graph_store import GraphStore
        from cortex.pipeline.reason import ReasonStage

        graph = GraphStore(path=config.graph_db_path)
        try:
            reasoner = ReasonStage(graph)
            result = reasoner.run()
            typer.echo(
                f"  Reasoner:   {result['total_inferred']} triples inferred "
                f"(fixpoint in {result['iterations']} iteration(s))"
            )
        finally:
            graph.close()
    else:
        typer.echo("  Reasoner:   skipped (graph.db not found)")

    # 3. WAL checkpoint
    _checkpoint_sqlite(config.sqlite_db_path)
    typer.echo("  WAL:        checkpointed")

    typer.secho("\nRepair complete.", fg=typer.colors.GREEN)


# Allow `python -m cortex.cli.main ...` invocation alongside the `cortex`
# console_scripts entry point. Used by the integration test suite.
if __name__ == "__main__":
    app()
