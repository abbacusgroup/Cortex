"""Cortex Web Dashboard — FastAPI + Jinja2 + HTMX.

Visual exploration of the knowledge graph with Abbacus brand styling.
Authentication via bcrypt + session cookies.
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
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.pipeline.orchestrator import PipelineOrchestrator
from cortex.retrieval.engine import RetrievalEngine
from cortex.retrieval.learner import LearningLoop
from cortex.retrieval.presenters import AlertPresenter, DossierPresenter
from cortex.services.llm import LLMClient

logger = get_logger("dashboard")

DASHBOARD_DIR = Path(__file__).parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"

# In-memory session store (simple for single-user)
_sessions: dict[str, dict[str, Any]] = {}
SESSION_COOKIE = "cortex_session"


def create_dashboard(config: CortexConfig | None = None) -> FastAPI:
    """Create the dashboard FastAPI application."""
    if config is None:
        config = load_config()

    setup_logging(level=config.log_level, json_output=False)

    store = Store(config)
    try:
        ontology_path = find_ontology()
        store.initialize(ontology_path)
    except FileNotFoundError:
        pass

    llm = LLMClient(config)
    pipeline = PipelineOrchestrator(store, config)
    engine = RetrievalEngine(store)
    learner = LearningLoop(store)

    app = FastAPI(title="Cortex Dashboard", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Store on app state for testing
    app.state.config = config
    app.state.store = store

    # ─── Auth Helpers ──────────────────────────────────────────────

    def _get_password_hash() -> str | None:
        """Get stored password hash, or None if no password set."""
        if config.dashboard_password:
            return config.dashboard_password
        return store.content.get_config("dashboard_password_hash", "") or None

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
        """Build template context."""
        return kwargs

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

        stats = store.status()
        alerts = AlertPresenter(store).render()
        recent = store.list_objects(limit=10)

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
            docs = engine.search(
                q,
                doc_type=doc_type or None,
                project=project or None,
                limit=50,
            )
        else:
            docs = store.list_objects(
                obj_type=doc_type or None,
                project=project or None,
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

        doc = store.read(obj_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Not found")

        learner.record_access(obj_id)
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

        ents = store.list_entities(entity_type=entity_type or None)
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

        result = pipeline.capture(
            title=title,
            content=content,
            obj_type=obj_type,
            project=project,
            tags=tags,
            captured_by="dashboard",
            run_pipeline=True,
        )
        obj_id = result["id"]
        return RedirectResponse(f"/documents/{obj_id}", status_code=302)

    @app.get("/trail", response_class=HTMLResponse)
    async def query_trail(request: Request):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        logs = store.content.get_query_log(limit=50)
        return templates.TemplateResponse(request, "trail.html", _ctx(logs=logs))

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        return templates.TemplateResponse(
            request,
            "settings.html",
            _ctx(
                config=config,
                weights=learner.get_weights(),
            ),
        )

    # ─── API Endpoints (for HTMX/Cytoscape) ───────────────────────

    @app.get("/api/graph-data")
    async def graph_data(
        request: Request,
        project: str = "",
        doc_type: str = "",
    ):
        """Return graph data in Cytoscape.js format."""
        session = _require_auth(request)
        if session is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        objects = store.list_objects(
            obj_type=doc_type or None,
            project=project or None,
            limit=200,
        )

        nodes = []
        edges = []
        seen_edges: set[str] = set()

        for obj in objects:
            obj_id = obj.get("id", "")
            nodes.append(
                {
                    "data": {
                        "id": obj_id,
                        "label": obj.get("title", "")[:40],
                        "type": obj.get("type", ""),
                        "project": obj.get("project", ""),
                    },
                }
            )

            rels = store.get_relationships(obj_id)
            for rel in rels:
                if rel["direction"] == "outgoing":
                    edge_key = f"{obj_id}-{rel['rel_type']}-{rel['other_id']}"
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(
                            {
                                "data": {
                                    "source": obj_id,
                                    "target": rel["other_id"],
                                    "rel_type": rel["rel_type"],
                                },
                            }
                        )

        # Add entity nodes and mention edges
        for entity in store.list_entities():
            eid = f"entity:{entity['id']}"
            nodes.append({
                "data": {
                    "id": eid,
                    "label": entity["name"],
                    "type": f"entity:{entity['type']}",
                    "project": "",
                },
            })
            for mid in store.graph.get_entity_mentions(entity["id"]):
                edge_key = f"{mid}-mentions-{eid}"
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "data": {
                            "source": mid,
                            "target": eid,
                            "rel_type": "mentions",
                        },
                    })

        return {"nodes": nodes, "edges": edges}

    @app.get("/api/dossier/{topic}")
    async def api_dossier(request: Request, topic: str):
        session = _require_auth(request)
        if session is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return DossierPresenter(store, llm).render(topic)

    return app
