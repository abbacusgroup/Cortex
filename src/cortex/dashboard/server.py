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

import html as html_mod
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
        msg: str = "",
        msg_type: str = "",
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
                msg=msg,
                msg_type=msg_type,
            ),
        )

    @app.get("/documents/{obj_id}", response_class=HTMLResponse)
    async def document_detail(
        request: Request,
        obj_id: str,
        msg: str = "",
        msg_type: str = "",
    ):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        doc = await mcp_client.read(obj_id)
        # cortex_read returns a string "Not found: {id}" when missing
        if isinstance(doc, str) or doc is None:
            raise HTTPException(status_code=404, detail="Not found")
        # Access tracking happens server-side inside cortex_read.
        return templates.TemplateResponse(
            request, "detail.html", _ctx(doc=doc, msg=msg, msg_type=msg_type),
        )

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
            result = await mcp_client.import_obsidian(str(vault))
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

    def _sanitize_filename(title: str, fallback: str) -> str:
        """Sanitize a title into an Obsidian-safe filename (no extension)."""
        safe = "".join(
            c if c.isalnum() or c in " -_" else "_" for c in title
        ).strip()[:100]
        return safe if safe else fallback

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
            import re as _re

            all_docs = await mcp_client.list_objects(limit=5000)

            # Pass 1: collect all docs and build id -> filename lookup.
            # Filenames include relative path for folder structure.
            id_to_filename: dict[str, str] = {}
            doc_cache: list[tuple[str, str, dict]] = []

            for doc in all_docs:
                obj_id = doc.get("id", "")
                full = await mcp_client.read(obj_id)
                if isinstance(full, str) or full is None:
                    continue
                title = full.get("title", obj_id[:8])
                safe_name = _sanitize_filename(title, obj_id[:8])

                # Build relative path: project/type/filename
                project = full.get("project", "") or "_unscoped"
                doc_type = full.get("type", "") or "other"
                safe_project = _sanitize_filename(project, "unknown")
                rel_path = f"{safe_project}/{doc_type}/{safe_name}"

                id_to_filename[obj_id] = rel_path
                doc_cache.append((obj_id, rel_path, full))

            # Pass 2: render markdown with relationships and write files.
            exported = 0
            for obj_id, rel_path, full in doc_cache:
                title = full.get("title", obj_id[:8])

                # Build markdown with YAML frontmatter.
                md = "---\n"
                md += f"id: {obj_id}\n"
                md += f"type: {full.get('type', '')}\n"
                md += f"project: {full.get('project', '')}\n"
                raw_tags = full.get("tags", "")
                tag_list = [
                    t.strip() for t in raw_tags.split(",") if t.strip()
                ] if raw_tags else []
                md += f"tags: {tag_list}\n"
                md += f"created: {full.get('created_at', '')}\n"
                summary = full.get("summary", "")
                if summary:
                    # Escape multiline summaries for YAML
                    md += f"summary: \"{summary}\"\n"
                md += "---\n\n"

                # Strip old ## Related sections from content (legacy imports)
                # to avoid duplicate sections and broken wiki-links.
                content = full.get("content", "")
                content = _re.split(r"\n## Related\b", content)[0].rstrip()

                # Avoid duplicate title heading
                if not content.lstrip().startswith("# "):
                    md += f"# {title}\n\n"
                md += content

                # Append relationships as Obsidian wiki-links.
                relationships = full.get("relationships", [])
                entities = full.get("entities", [])

                if relationships or entities:
                    md += "\n\n## Related\n\n"

                if relationships:
                    for rel in relationships:
                        other_id = rel.get("other_id", "")
                        other_name = id_to_filename.get(other_id)
                        if other_name is None:
                            continue
                        rel_type = rel.get("rel_type", "related")
                        direction = rel.get("direction", "outgoing")
                        if direction == "outgoing":
                            md += f"- {rel_type} [[{other_name}]]\n"
                        else:
                            md += f"- {rel_type} (from) [[{other_name}]]\n"

                if entities:
                    entity_tags = ", ".join(
                        f"#{e.get('type', 'concept')}/{e['name'].replace(' ', '_')}"
                        for e in entities
                        if e.get("name")
                    )
                    if entity_tags:
                        md += f"\n**Entities:** {entity_tags}\n"

                filepath = target / f"{rel_path}.md"
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(md, encoding="utf-8")
                exported += 1

            # Pass 3: generate project index/hub files.
            projects: dict[str, list[tuple[str, str, str]]] = {}
            for _obj_id, rel_path, full in doc_cache:
                proj = full.get("project", "")
                if proj:
                    projects.setdefault(proj, []).append(
                        (full.get("title", ""), rel_path, full.get("type", ""))
                    )

            for proj_name, docs in projects.items():
                by_type: dict[str, list[tuple[str, str]]] = {}
                for doc_title, doc_path, dtype in docs:
                    by_type.setdefault(dtype or "other", []).append(
                        (doc_title, doc_path)
                    )

                hub_md = "---\n"
                hub_md += "type: source\n"
                hub_md += f"project: {proj_name}\n"
                hub_md += "tags: [project-index]\n"
                hub_md += "---\n\n"
                hub_md += f"# {proj_name}\n\n"
                hub_md += f"Project index — {len(docs)} documents\n\n"
                for dtype, items in sorted(by_type.items()):
                    hub_md += f"## {dtype.capitalize()}s ({len(items)})\n\n"
                    for doc_title, doc_path in sorted(items):
                        hub_md += f"- [[{doc_path}|{doc_title}]]\n"
                    hub_md += "\n"

                safe_proj = _sanitize_filename(proj_name, proj_name)
                hub_path = target / safe_proj / f"{safe_proj}.md"
                hub_path.parent.mkdir(parents=True, exist_ok=True)
                hub_path.write_text(hub_md, encoding="utf-8")
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

    # ─── Backup & Single-Object Export ────────────────────────────

    @app.post("/settings/backup")
    async def settings_backup(request: Request):
        """Download a backup archive of all Cortex data."""
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        import tempfile

        from cortex.cli.backup import create_backup

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                from pathlib import Path as _Path

                archive_path = create_backup(config, output=_Path(tmp_dir))
                content = archive_path.read_bytes()
                filename = archive_path.name

            from fastapi.responses import Response

            return Response(
                content=content,
                media_type="application/gzip",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                },
            )
        except FileNotFoundError as e:
            return RedirectResponse(
                f"/settings?msg=Backup failed: {e}&msg_type=danger",
                status_code=302,
            )
        except Exception as e:
            return RedirectResponse(
                f"/settings?msg=Backup error: {e}&msg_type=danger",
                status_code=302,
            )

    @app.get("/documents/{obj_id}/export")
    async def document_export(request: Request, obj_id: str):
        """Export a single knowledge object as a markdown file."""
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        result = await mcp_client.export_object(obj_id)
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        from fastapi.responses import Response

        return Response(
            content=content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="{obj_id[:12]}.md"',
            },
        )

    # ─── Document Actions ──────────────────────────────────────────

    @app.post("/documents/{obj_id}/delete")
    async def document_delete(request: Request, obj_id: str):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        result = await mcp_client.delete(obj_id)
        status = result.get("status", "error") if isinstance(result, dict) else "error"
        if status == "deleted":
            return RedirectResponse(
                "/documents?msg=Document deleted&msg_type=info",
                status_code=302,
            )
        return RedirectResponse(
            f"/documents/{obj_id}?msg=Delete failed: {status}&msg_type=danger",
            status_code=302,
        )

    @app.post("/documents/{obj_id}/edit")
    async def document_edit(
        request: Request,
        obj_id: str,
        title: str = Form(""),
        content: str = Form(""),
        tags: str = Form(""),
        project: str = Form(""),
    ):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        await mcp_client.update(
            obj_id, title=title, content=content, tags=tags, project=project,
        )
        return RedirectResponse(f"/documents/{obj_id}", status_code=302)

    @app.post("/documents/{obj_id}/classify")
    async def document_classify(
        request: Request,
        obj_id: str,
        obj_type: str = Form(""),
        summary: str = Form(""),
    ):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        await mcp_client.classify(
            obj_id, obj_type=obj_type, summary=summary,
        )
        return RedirectResponse(f"/documents/{obj_id}", status_code=302)

    @app.post("/documents/{obj_id}/pipeline")
    async def document_pipeline(request: Request, obj_id: str):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        await mcp_client.pipeline(obj_id)
        return RedirectResponse(f"/documents/{obj_id}", status_code=302)

    @app.post("/documents/{obj_id}/link")
    async def document_link(
        request: Request,
        obj_id: str,
        rel_type: str = Form("relatedTo"),
        direction: str = Form("outgoing"),
        target_id: str = Form(""),
    ):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        if not target_id:
            return RedirectResponse(
                f"/documents/{obj_id}?msg=Target ID required&msg_type=danger",
                status_code=302,
            )

        if direction == "outgoing":
            await mcp_client.link(obj_id, rel_type, target_id)
        else:
            await mcp_client.link(target_id, rel_type, obj_id)
        return RedirectResponse(f"/documents/{obj_id}", status_code=302)

    @app.post("/documents/{obj_id}/unlink")
    async def document_unlink(
        request: Request,
        obj_id: str,
        from_id: str = Form(""),
        rel_type: str = Form(""),
        to_id: str = Form(""),
    ):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        await mcp_client.unlink(from_id, rel_type, to_id)
        return RedirectResponse(f"/documents/{obj_id}", status_code=302)

    # ─── Synthesis & Insights ────────────────────────────────────

    @app.get("/synthesis", response_class=HTMLResponse)
    async def synthesis_page(request: Request):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        return templates.TemplateResponse(
            request, "synthesis.html", _ctx(synthesis=None),
        )

    @app.post("/synthesis", response_class=HTMLResponse)
    async def synthesis_generate(
        request: Request,
        period_days: int = Form(7),
        project: str = Form(""),
    ):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        result = await mcp_client.synthesize(period_days=period_days, project=project)

        return templates.TemplateResponse(
            request,
            "synthesis.html",
            _ctx(synthesis=result, period_days=period_days, project=project),
        )

    @app.get("/insights", response_class=HTMLResponse)
    async def insights_page(request: Request):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        return templates.TemplateResponse(
            request, "insights.html", _ctx(analysis=None),
        )

    @app.post("/insights", response_class=HTMLResponse)
    async def insights_generate(request: Request):
        session = _require_auth(request)
        if session is None:
            return RedirectResponse("/login", status_code=302)

        result = await mcp_client.reason()

        return templates.TemplateResponse(
            request, "insights.html", _ctx(analysis=result),
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

    @app.get("/api/search")
    async def api_search(request: Request, q: str = ""):
        session = _require_auth(request)
        if session is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if not q or len(q) < 2:
            return HTMLResponse("")
        results = await mcp_client.search(q, limit=8)
        # Return HTML fragment for HTMX
        html_out = ""
        for doc in results:
            obj_id = html_mod.escape(doc.get("id", ""), quote=True)
            title = html_mod.escape(doc.get("title", obj_id[:12]), quote=True)
            doc_type = html_mod.escape(doc.get("type", ""), quote=True)
            html_out += (
                f'<div class="search-item" '
                f'data-id="{obj_id}" '
                f"onclick=\"document.getElementById('target_id').value=this.dataset.id; "
                f"document.getElementById('search-results').innerHTML='';\">"
                f'<span class="badge badge-{doc_type}" style="font-size:0.7rem;">{doc_type}</span> '
                f"{title[:60]}"
                f'<small style="color:var(--text-muted);"> {obj_id[:8]}</small>'
                f"</div>"
            )
        return HTMLResponse(html_out)

    @app.post("/api/feedback")
    async def api_feedback(request: Request):
        session = _require_auth(request)
        if session is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        body = await request.json()
        obj_id = body.get("obj_id", "")
        relevant = body.get("relevant", True)
        result = await mcp_client.feedback(obj_id, relevant)
        return JSONResponse(result)

    return app
