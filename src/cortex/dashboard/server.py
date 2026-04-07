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
        recent = await mcp_client.list_objects(limit=10)
        alerts = stats.get("alerts", []) if isinstance(stats, dict) else []

        return templates.TemplateResponse(
            request,
            "home.html",
            _ctx(stats=stats, alerts=alerts, recent=recent),
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

        return templates.TemplateResponse(
            request,
            "documents.html",
            _ctx(
                documents=docs,
                query=q,
                doc_type=doc_type,
                project=project,
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

    @app.get("/graph", response_class=HTMLResponse)
    async def graph_page(request: Request):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)
        return templates.TemplateResponse(request, "graph.html")

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
    async def settings_page(request: Request):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        return templates.TemplateResponse(
            request,
            "settings.html",
            _ctx(config=config, weights={}),
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
