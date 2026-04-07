"""Cortex CLI — Typer application.

Commands: init, capture, search, read, list, status, context, dossier, graph, synthesize, entities.
"""

from __future__ import annotations

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


def _open_store_or_exit(config: CortexConfig) -> Store:
    """Open a Store, exiting cleanly with a user-friendly message if the graph DB is locked.

    This is the single chokepoint for surfacing StoreLockedError to CLI users.
    """
    try:
        return Store(config)
    except StoreLockedError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


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
            raise typer.Exit(1)

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
        raise typer.Exit(1)

    triples = store.graph.load_ontology(ontology_path)

    typer.echo(f"Cortex initialized at {config.data_dir}")
    typer.echo(f"  Ontology: {triples} triples loaded")
    typer.echo(f"  Graph DB: {config.graph_db_path}")
    typer.echo(f"  SQLite:   {config.sqlite_db_path}")

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
    store = _get_store()
    from cortex.retrieval.engine import RetrievalEngine
    from cortex.retrieval.presenters import BriefingPresenter

    engine = RetrievalEngine(store)
    results = engine.search(topic, limit=limit)

    if not results:
        typer.echo("No context found.")
        return

    presenter = BriefingPresenter()
    briefs = presenter.render(results)

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
    store = _get_store()
    from cortex.retrieval.graph import GraphQueries

    gq = GraphQueries(store)

    # Show causal chain if applicable
    chain = gq.causal_chain(obj_id)
    if len(chain) > 1:
        typer.echo("Causal chain:")
        for i, node in enumerate(chain):
            prefix = "  → " if i > 0 else "  "
            typer.echo(f"{prefix}[{node['type']}] {node['title']}")
        typer.echo()

    # Show evolution timeline
    timeline = gq.evolution_timeline(obj_id)
    if len(timeline) > 1:
        typer.echo("Evolution timeline:")
        for node in timeline:
            typer.echo(f"  {node.get('created_at', '?')[:10]}  {node['title']}")
        typer.echo()

    # Show direct relationships
    rels = store.get_relationships(obj_id)
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
def register() -> None:
    """Register Cortex MCP server with Claude Code."""
    import json

    settings_path = Path.home() / ".claude" / "settings.json"
    settings: dict[str, Any] = {}

    if settings_path.exists():
        settings = json.loads(settings_path.read_text())

    mcp_servers = settings.setdefault("mcpServers", {})

    # Find the cortex package entry point
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

    typer.echo(f"Registered Cortex MCP at {settings_path}")
    typer.echo(f"  Command: {cmd} {' '.join(args)}")
    typer.echo("  Restart Claude Code to activate.")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(1314, "--port", help="Bind port"),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        help="Transport: stdio (default, for Claude Code), mcp-http (HTTP MCP server), or http (REST API)",
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
            raise typer.Exit(1)
    elif transport == "mcp-http":
        from cortex.transport.mcp.server import run_http
        typer.echo(f"Cortex MCP (streamable-http) at http://{host}:{port}/mcp")
        try:
            run_http(host=host, port=port)
        except StoreLockedError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
    elif transport == "http":
        import uvicorn

        from cortex.transport.api.server import create_api
        try:
            api = create_api()
        except StoreLockedError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
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
    """
    import asyncio
    import time

    from cortex.dashboard.mcp_client import (
        CortexMCPClient,
        MCPClientError,
    )

    client = CortexMCPClient(url, timeout_seconds=3.0)
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


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(1315, "--port", help="Bind port"),
) -> None:
    """Start the web dashboard.

    The dashboard is a thin client of the MCP HTTP server. It probes the
    configured ``mcp_server_url`` at startup and refuses to start if the
    server isn't reachable or is missing required tools.
    """
    import uvicorn

    from cortex.dashboard.mcp_client import MCPClientError
    from cortex.dashboard.server import create_dashboard

    config = load_config()

    # Probe the MCP server before starting uvicorn so the user gets an
    # actionable error instead of a half-broken dashboard.
    try:
        available = _probe_mcp_server(config.mcp_server_url)
    except MCPClientError as e:
        typer.secho(
            f"Cannot reach Cortex MCP server at {config.mcp_server_url}",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(f"  {e}", fg=typer.colors.RED, err=True)
        typer.echo(
            "  Start it in another terminal:\n"
            "    cortex serve --transport mcp-http --host 127.0.0.1 --port 1314",
            err=True,
        )
        raise typer.Exit(1)

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
    store = _get_store()

    if batch:
        # Query for docs at 'ingest' stage
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


# Allow `python -m cortex.cli.main ...` invocation alongside the `cortex`
# console_scripts entry point. Used by the integration test suite.
if __name__ == "__main__":
    app()
