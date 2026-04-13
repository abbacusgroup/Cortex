"""Cortex Web Dashboard — FastAPI + Jinja2 + HTMX.

The dashboard is a thin client of the Cortex MCP HTTP server. It does NOT
open ``graph.db`` or ``cortex.db`` directly — every read and write goes
through ``CortexMCPClient`` to a running ``cortex serve --transport mcp-http``
process. This avoids the Oxigraph single-writer lock and lets the dashboard
coexist with Claude Code's MCP usage.

Authentication is bcrypt-hash-based via the ``dashboard_password`` config
field (set via ``CORTEX_DASHBOARD_PASSWORD`` env var). Sessions are kept in
process memory.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

import bcrypt
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cortex.core.config import CortexConfig, load_config
from cortex.core.logging import get_logger, setup_logging
from cortex.transport.mcp.client import (
    CortexMCPClient,
    MCPClientError,
    MCPConnectionError,
    MCPTimeoutError,
    MCPToolError,
)

logger = get_logger("dashboard")

DASHBOARD_DIR = Path(__file__).parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"

# In-memory session store (simple for single-user)
_sessions: dict[str, dict[str, Any]] = {}
SESSION_COOKIE = "cortex_session"


def create_dashboard(
    config: CortexConfig | None = None,
    *,
    mcp_client: CortexMCPClient | None = None,
) -> FastAPI:
    """Create the dashboard FastAPI application.

    Args:
        config: Cortex configuration. If None, loads from env.
        mcp_client: MCP client instance. If None, one is created from
            ``config.mcp_server_url``. Tests inject a fake here.
    """
    if config is None:
        config = load_config()

    setup_logging(level=config.log_level, json_output=False)

    if mcp_client is None:
        mcp_client = CortexMCPClient(config.mcp_server_url, timeout_seconds=10.0)

    app = FastAPI(title="Cortex Dashboard", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Stash on app state for tests + auth helpers
    app.state.config = config
    app.state.mcp_client = mcp_client

    # ─── Auth Helpers ──────────────────────────────────────────────

    def _get_password_hash() -> str | None:
        """Get the dashboard password hash from config (env var)."""
        return config.dashboard_password or None

    def _verify_password(password: str) -> bool:
        stored = _get_password_hash()
        if not stored:
            return True  # No password = open access
        try:
            return bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
        except Exception:
            return False

    def _needs_auth() -> bool:
        return _get_password_hash() is not None

    def _get_session(request: Request) -> dict[str, Any] | None:
        session_id = request.cookies.get(SESSION_COOKIE)
        if session_id and session_id in _sessions:
            return _sessions[session_id]
        return None

    def _require_auth(request: Request) -> dict[str, Any] | None:
        if not _needs_auth():
            return {"user": "anonymous"}
        return _get_session(request)

    def _ctx(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    # ─── MCP error handlers ────────────────────────────────────────

    def _mcp_error_html(request: Request, exc: MCPClientError, status: int):
        return templates.TemplateResponse(
            request,
            "error.html",
            _ctx(
                error_message=str(exc),
                error_code=exc.code,
                status_code=status,
            ),
            status_code=status,
        )

    def _mcp_error_json(exc: MCPClientError, status: int):
        return JSONResponse(
            {"error": str(exc), "code": exc.code},
            status_code=status,
        )

    @app.exception_handler(MCPConnectionError)
    async def _connection_handler(request: Request, exc: MCPConnectionError):
        if request.url.path.startswith("/api/"):
            return _mcp_error_json(exc, 503)
        return _mcp_error_html(request, exc, 503)

    @app.exception_handler(MCPTimeoutError)
    async def _timeout_handler(request: Request, exc: MCPTimeoutError):
        if request.url.path.startswith("/api/"):
            return _mcp_error_json(exc, 504)
        return _mcp_error_html(request, exc, 504)

    @app.exception_handler(MCPToolError)
    async def _tool_handler(request: Request, exc: MCPToolError):
        if request.url.path.startswith("/api/"):
            return _mcp_error_json(exc, 502)
        return _mcp_error_html(request, exc, 502)

    # ─── Auth Routes ───────────────────────────────────────────────

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        if not _needs_auth():
            return RedirectResponse("/", status_code=302)
        return templates.TemplateResponse(request, "login.html")

    @app.post("/login")
    async def login_submit(request: Request, password: str = Form(...)):
        if _verify_password(password):
            session_id = secrets.token_hex(32)
            _sessions[session_id] = {"user": "admin"}
            response = RedirectResponse("/", status_code=302)
            response.set_cookie(SESSION_COOKIE, session_id, httponly=True)
            return response
        return templates.TemplateResponse(
            request,
            "login.html",
            _ctx(error="Invalid password"),
            status_code=401,
        )

    @app.get("/logout")
    async def logout(request: Request):
        session_id = request.cookies.get(SESSION_COOKIE)
        if session_id and session_id in _sessions:
            del _sessions[session_id]
        response = RedirectResponse("/login", status_code=302)
        response.delete_cookie(SESSION_COOKIE)
        return response

    # ─── Page Routes ───────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        stats = await mcp_client.status()
        alerts = stats.get("alerts", []) if isinstance(stats, dict) else []

        # Derive projects from document project fields.
        all_docs = await mcp_client.list_objects(limit=500)

        doc_counts: dict[str, int] = {}
        last_activity: dict[str, str] = {}
        for doc in all_docs:
            proj = doc.get("project", "")
            if proj:
                doc_counts[proj] = doc_counts.get(proj, 0) + 1
                created = doc.get("created_at", "")
                if created > last_activity.get(proj, ""):
                    last_activity[proj] = created

        # Build project list sorted by doc count descending.
        projects = [
            {
                "name": name,
                "doc_count": count,
                "last_activity": last_activity.get(name, "")[:10],
            }
            for name, count in sorted(doc_counts.items(), key=lambda x: -x[1])
        ]

        recent = all_docs[:10]

        return templates.TemplateResponse(
            request,
            "home.html",
            _ctx(stats=stats, alerts=alerts, projects=projects, recent=recent),
        )

    @app.get("/project/{project_name}", response_class=HTMLResponse)
    async def project_detail(request: Request, project_name: str):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        docs = await mcp_client.list_objects(project=project_name, limit=50)

        return templates.TemplateResponse(
            request,
            "project.html",
            _ctx(project_name=project_name, documents=docs),
        )

    @app.get("/documents", response_class=HTMLResponse)
    async def documents_page(
        request: Request,
        doc_type: str = "",
        project: str = "",
        q: str = "",
    ):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        if q:
            docs = await mcp_client.search(
                q,
                doc_type=doc_type,
                project=project,
                limit=50,
            )
        else:
            docs = await mcp_client.list_objects(
                doc_type=doc_type,
                project=project,
                limit=50,
            )

        # Get distinct project names for the dropdown.
        all_docs = await mcp_client.list_objects(limit=500)
        project_names = sorted({d.get("project", "") for d in all_docs} - {""})

        return templates.TemplateResponse(
            request,
            "documents.html",
            _ctx(
                documents=docs,
                query=q,
                doc_type=doc_type,
                project=project,
                project_names=project_names,
            ),
        )

    @app.get("/documents/{obj_id}", response_class=HTMLResponse)
    async def document_detail(request: Request, obj_id: str):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        doc = await mcp_client.read(obj_id)
        # cortex_read returns a string "Not found: {id}" when missing
        if isinstance(doc, str) or doc is None:
            raise HTTPException(status_code=404, detail="Not found")
        # Access tracking happens server-side inside cortex_read.
        return templates.TemplateResponse(request, "detail.html", _ctx(doc=doc))

    @app.get("/explore", response_class=HTMLResponse)
    async def explore_page(
        request: Request,
        topic: str = "",
        entity_type: str = "",
    ):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        dossier = None
        connections: list[dict[str, Any]] = []

        if topic:
            # Fetch dossier for the topic.
            try:
                dossier = await mcp_client.dossier(topic)
            except MCPClientError:
                dossier = None

            # If dossier found objects, resolve relationships for the
            # first object to show how it connects.
            if dossier and dossier.get("objects"):
                first_id = dossier["objects"][0]["id"]
                try:
                    graph_info = await mcp_client.graph(obj_id=first_id)
                    rels = graph_info.get("relationships", [])
                    obj_lookup = {
                        o["id"]: o for o in dossier.get("objects", [])
                    }
                    for rel in rels:
                        other = obj_lookup.get(rel["other_id"])
                        if other:
                            connections.append({
                                "direction": rel["direction"],
                                "rel_type": rel["rel_type"],
                                "other_id": rel["other_id"],
                                "other_title": other.get("title", rel["other_id"][:8]),
                                "other_type": other.get("type", ""),
                            })
                except MCPClientError:
                    pass

        # Always fetch entities for the browse section.
        entities = await mcp_client.list_entities(entity_type=entity_type)

        # Compute connection counts per entity from graph data.
        graph_data = await mcp_client.graph_data(limit=500)
        edge_counts: dict[str, int] = {}
        node_lookup: dict[str, str] = {}  # id -> label
        for node in graph_data.get("nodes", []):
            d = node.get("data", {})
            node_lookup[d.get("id", "")] = d.get("label", "")
        for edge in graph_data.get("edges", []):
            d = edge.get("data", {})
            src = node_lookup.get(d.get("source", ""), "")
            tgt = node_lookup.get(d.get("target", ""), "")
            if src:
                edge_counts[src] = edge_counts.get(src, 0) + 1
            if tgt:
                edge_counts[tgt] = edge_counts.get(tgt, 0) + 1

        # Attach connection count and size tier to each entity.
        max_count = max((edge_counts.get(e.get("name", ""), 0) for e in entities), default=1) or 1
        for ent in entities:
            count = edge_counts.get(ent.get("name", ""), 0)
            ent["connection_count"] = count
            # Size tiers: sm (0-20%), md (20-50%), lg (50-80%), xl (80%+)
            ratio = count / max_count if max_count else 0
            if ratio >= 0.8:
                ent["size"] = "xl"
            elif ratio >= 0.5:
                ent["size"] = "lg"
            elif ratio >= 0.2:
                ent["size"] = "md"
            else:
                ent["size"] = "sm"

        # Sort entities by connection count descending.
        entities.sort(key=lambda e: e.get("connection_count", 0), reverse=True)

        return templates.TemplateResponse(
            request,
            "explore.html",
            _ctx(
                topic=topic,
                dossier=dossier,
                connections=connections,
                entities=entities,
                entity_type=entity_type,
            ),
        )

    @app.get("/graph", response_class=HTMLResponse)
    async def graph_page(request: Request):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        all_docs = await mcp_client.list_objects(limit=500)
        project_names = sorted({d.get("project", "") for d in all_docs} - {""})

        return templates.TemplateResponse(
            request, "graph.html", _ctx(project_names=project_names)
        )

    @app.get("/entities", response_class=HTMLResponse)
    async def entities_page(request: Request, entity_type: str = ""):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        ents = await mcp_client.list_entities(entity_type=entity_type)
        return templates.TemplateResponse(
            request,
            "entities.html",
            _ctx(entities=ents, entity_type=entity_type),
        )

    @app.get("/create", response_class=HTMLResponse)
    async def create_page(request: Request):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)
        return templates.TemplateResponse(request, "create.html")

    @app.post("/create")
    async def create_submit(
        request: Request,
        title: str = Form(...),
        content: str = Form(""),
        obj_type: str = Form("idea"),
        project: str = Form(""),
        tags: str = Form(""),
    ):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        result = await mcp_client.capture(
            title=title,
            content=content,
            obj_type=obj_type,
            project=project,
            tags=tags,
        )
        obj_id = result["id"]
        return RedirectResponse(f"/documents/{obj_id}", status_code=302)

    @app.get("/trail", response_class=HTMLResponse)
    async def query_trail(request: Request):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        logs = await mcp_client.query_trail(limit=50)
        return templates.TemplateResponse(request, "trail.html", _ctx(logs=logs))

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request, msg: str = "", msg_type: str = "info"):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        return templates.TemplateResponse(
            request,
            "settings.html",
            _ctx(config=config, weights={}, msg=msg, msg_type=msg_type),
        )

    @app.post("/settings/import")
    async def settings_import(request: Request, vault_path: str = Form(...)):
        """Import an Obsidian vault into Cortex."""
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        from pathlib import Path as _Path

        vault = _Path(vault_path).expanduser().resolve()
        if not vault.is_dir():
            return RedirectResponse(
                f"/settings?msg=Directory not found: {vault_path}&msg_type=danger",
                status_code=302,
            )

        try:
            from cortex.db.store import Store
            from cortex.pipeline.importer import ObsidianImporter

            store = Store(config.data_dir)
            importer = ObsidianImporter(store, pipeline=None)
            result = importer.run(vault)
            store.close()

            imported = result.get("imported", 0)
            skipped = result.get("skipped", 0)
            failed = result.get("failed", 0)
            wiki = result.get("wiki_links_created", 0)
            msg = (
                f"Import complete: {imported} imported, {skipped} skipped, "
                f"{failed} failed, {wiki} wiki-links"
            )
            return RedirectResponse(
                f"/settings?msg={msg}&msg_type=info", status_code=302,
            )
        except Exception as e:
            return RedirectResponse(
                f"/settings?msg=Import error: {e}&msg_type=danger",
                status_code=302,
            )

    @app.post("/settings/export")
    async def settings_export(request: Request, export_path: str = Form(...)):
        """Export all Cortex documents as an Obsidian-compatible vault."""
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        from pathlib import Path as _Path

        target = _Path(export_path).expanduser().resolve()
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return RedirectResponse(
                f"/settings?msg=Cannot create directory: {e}&msg_type=danger",
                status_code=302,
            )

        try:
            all_docs = await mcp_client.list_objects(limit=5000)
            exported = 0
            for doc in all_docs:
                obj_id = doc.get("id", "")
                full = await mcp_client.read(obj_id)
                if isinstance(full, str) or full is None:
                    continue

                title = full.get("title", obj_id[:8])
                # Sanitize filename.
                safe_name = "".join(
                    c if c.isalnum() or c in " -_" else "_" for c in title
                ).strip()[:100]
                if not safe_name:
                    safe_name = obj_id[:8]

                # Build markdown with YAML frontmatter.
                md = "---\n"
                md += f"id: {obj_id}\n"
                md += f"type: {full.get('type', '')}\n"
                md += f"project: {full.get('project', '')}\n"
                md += f"tags: {full.get('tags', '')}\n"
                md += f"created: {full.get('created_at', '')}\n"
                md += "---\n\n"
                md += f"# {title}\n\n"
                md += full.get("content", "")

                filepath = target / f"{safe_name}.md"
                filepath.write_text(md, encoding="utf-8")
                exported += 1

            msg = f"Exported {exported} documents to {target}"
            return RedirectResponse(
                f"/settings?msg={msg}&msg_type=info", status_code=302,
            )
        except Exception as e:
            return RedirectResponse(
                f"/settings?msg=Export error: {e}&msg_type=danger",
                status_code=302,
            )

    # ─── API Endpoints (for HTMX/Cytoscape) ───────────────────────

    @app.get("/api/graph-data")
    async def graph_data(
        request: Request,
        project: str = "",
        doc_type: str = "",
        limit: int = 500,
        offset: int = 0,
    ):
        """Return graph data in Cytoscape.js format."""
        session = _require_auth(request)
        if session is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        return await mcp_client.graph_data(
            project=project,
            doc_type=doc_type,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/dossier/{topic}")
    async def api_dossier(request: Request, topic: str):
        session = _require_auth(request)
        if session is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await mcp_client.dossier(topic)

    return app
