"""Cortex REST API — FastAPI server backed by the MCP HTTP client.

Post-Phase-4 (2026-04-08): the REST API is now a thin client of the MCP
HTTP server. It does NOT open ``graph.db`` directly. This was the third
and final entry point (after the dashboard and CLI) to be migrated off
the direct ``Store(config)`` pattern so multiple entry points can
coexist under a single-writer MCP server without lock conflicts.

Authentication keys are read from the ``CORTEX_API_KEYS`` environment
variable (comma-separated) at startup. If the env var is empty the API
runs in dev mode and accepts any non-empty key (matching legacy
behavior). The previous SQLite-backed key storage has been removed —
there is no API-level admin UI for managing keys, and the env-var
approach is both simpler and keeps the REST API out of the store
entirely.
"""

from __future__ import annotations

import hmac
import os
import time
from collections import defaultdict
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from cortex.core.config import CortexConfig, load_config
from cortex.core.logging import get_logger, setup_logging
from cortex.transport.mcp.client import (
    CortexMCPClient,
    MCPConnectionError,
    MCPServerError,
    MCPTimeoutError,
    MCPToolError,
)

logger = get_logger("transport.api")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Simple in-memory rate limiter
_rate_limits: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 100  # requests per window


def _load_api_keys_from_env() -> set[str]:
    """Parse ``CORTEX_API_KEYS`` (comma-separated) into a set.

    Empty or unset returns an empty set (dev mode — any non-empty key
    is accepted).
    """
    raw = os.environ.get("CORTEX_API_KEYS", "").strip()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def create_api(
    config: CortexConfig | None = None,
    *,
    mcp_client: Any | None = None,
) -> FastAPI:
    """Create the Cortex REST API.

    Args:
        config: Cortex configuration. If None, loads from env.
        mcp_client: MCP client instance. If None, a fresh
            :class:`CortexMCPClient` is constructed from
            ``config.mcp_server_url``. Tests inject
            :class:`tests.conftest.FakeMCPClient` here.

    Returns:
        Configured FastAPI app.
    """
    if config is None:
        config = load_config()

    setup_logging(level=config.log_level, json_output=config.log_json)

    if mcp_client is None:
        mcp_client = CortexMCPClient(
            config.mcp_server_url, timeout_seconds=10.0
        )

    app = FastAPI(
        title="Cortex API",
        description=(
            "Cognitive knowledge system — REST surface over the canonical "
            "MCP HTTP server."
        ),
        version="0.2.0",
    )

    app.state.config = config
    app.state.mcp_client = mcp_client

    # ─── MCP error handlers — translate typed client errors to HTTP ──

    @app.exception_handler(MCPConnectionError)
    async def _mcp_conn_handler(
        request: Request, exc: MCPConnectionError
    ):
        return _error_response(503, f"MCP server unreachable: {exc}")

    @app.exception_handler(MCPTimeoutError)
    async def _mcp_timeout_handler(
        request: Request, exc: MCPTimeoutError
    ):
        return _error_response(504, f"MCP server timed out: {exc}")

    @app.exception_handler(MCPServerError)
    async def _mcp_server_handler(
        request: Request, exc: MCPServerError
    ):
        return _error_response(502, f"MCP server error: {exc}")

    @app.exception_handler(MCPToolError)
    async def _mcp_tool_handler(request: Request, exc: MCPToolError):
        return _error_response(502, f"MCP tool error: {exc}")

    # ─── Auth ──────────────────────────────────────────────────────

    def _get_api_keys() -> set[str]:
        """Return valid API keys from the CORTEX_API_KEYS env var."""
        return _load_api_keys_from_env()

    async def verify_api_key(
        api_key: str | None = Security(api_key_header),
    ) -> str:
        """Verify the API key from the X-API-Key header.

        Dev mode: if no keys are configured via ``CORTEX_API_KEYS``, any
        non-empty key is accepted. This matches pre-Phase-4 behavior and
        keeps existing tests working without extra env setup.

        Bundle 9 / Group 3 #1 fix: the API key is ``.strip()``'d before
        comparison so trailing/leading whitespace pasted from a config
        file no longer causes asymmetric 401 responses (h11 strips
        leading OWS but preserves trailing — the bug hunt observed 200
        for ``"  good"`` and 401 for ``"good "``). Comparison uses
        :func:`hmac.compare_digest` for constant-time equality so the
        auth path is not vulnerable to timing-based key recovery.
        """
        if api_key is None:
            raise HTTPException(status_code=401, detail="Missing API key")

        # Normalize: strip OWS so leading/trailing whitespace can't fool
        # the comparison either way.
        api_key = api_key.strip()
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API key")

        valid_keys = _get_api_keys()
        if not valid_keys:
            # Dev mode — any non-empty key passes
            return api_key

        # Constant-time comparison against each valid key. We use
        # hmac.compare_digest so an attacker who can measure response
        # time cannot incrementally guess characters of a valid key.
        for candidate in valid_keys:
            if hmac.compare_digest(api_key, candidate):
                return api_key
        raise HTTPException(status_code=401, detail="Invalid API key")

    # ─── Rate Limiting ─────────────────────────────────────────────

    async def rate_limit(request: Request) -> None:
        """Simple rate limiter per client IP."""
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()

        _rate_limits[client] = [
            t for t in _rate_limits[client] if now - t < RATE_LIMIT_WINDOW
        ]

        if len(_rate_limits[client]) >= RATE_LIMIT_MAX:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
            )

        _rate_limits[client].append(now)

    # ─── Health ────────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    # ─── Public Endpoints ──────────────────────────────────────────

    @app.post("/search", dependencies=[Depends(rate_limit)])
    async def search(
        query: str,
        doc_type: str = "",
        project: str = "",
        limit: int = 20,
        _key: str = Depends(verify_api_key),
    ) -> list[dict[str, Any]]:
        """Hybrid search for knowledge objects."""
        return await mcp_client.search(
            query, doc_type=doc_type, project=project, limit=limit
        )

    @app.post("/context", dependencies=[Depends(rate_limit)])
    async def context(
        topic: str,
        limit: int = 10,
        _key: str = Depends(verify_api_key),
    ) -> list[dict[str, Any]]:
        """Get briefing (summaries) for a topic."""
        return await mcp_client.context(topic, limit=limit)

    @app.post("/dossier", dependencies=[Depends(rate_limit)])
    async def dossier(
        topic: str,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Build entity/topic dossier."""
        return await mcp_client.dossier(topic)

    @app.get("/read/{obj_id}", dependencies=[Depends(rate_limit)])
    async def read(
        obj_id: str,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Read a knowledge object in full."""
        result = await mcp_client.read(obj_id)
        # cortex_read returns the string ``f"Not found: {obj_id}"`` for
        # missing objects, or a dict for hits. Normalize to 404 here.
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"Not found: {obj_id}"
            )
        if isinstance(result, str):
            raise HTTPException(status_code=404, detail=result)
        if isinstance(result, dict) and result.get("error"):
            raise HTTPException(
                status_code=404, detail=str(result.get("error"))
            )
        return result

    @app.post("/capture", dependencies=[Depends(rate_limit)])
    async def capture(
        title: str,
        content: str = "",
        obj_type: str = "idea",
        project: str = "",
        tags: str = "",
        template: str = "",
        run_pipeline: bool = True,
        summary: str = "",
        entities: str = "",
        properties: str = "",
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Capture a knowledge object with optional pre-classification."""
        return await mcp_client.capture(
            title=title,
            content=content,
            obj_type=obj_type,
            project=project,
            tags=tags,
            template=template,
            run_pipeline=run_pipeline,
            summary=summary,
            entities=entities,
            properties=properties,
        )

    @app.post("/link", dependencies=[Depends(rate_limit)])
    async def link(
        from_id: str,
        rel_type: str,
        to_id: str,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Create a relationship."""
        result = await mcp_client.link(
            from_id=from_id, rel_type=rel_type, to_id=to_id
        )
        if isinstance(result, dict) and result.get("status") == "error":
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "link failed"),
            )
        return result

    @app.post("/feedback", dependencies=[Depends(rate_limit)])
    async def feedback(
        obj_id: str,
        relevant: bool = True,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Provide relevance feedback."""
        return await mcp_client.feedback(obj_id, relevant=relevant)

    @app.post("/classify/{obj_id}", dependencies=[Depends(rate_limit)])
    async def classify(
        obj_id: str,
        summary: str = "",
        obj_type: str = "",
        tags: str = "",
        project: str = "",
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Classify or reclassify an existing knowledge object."""
        result = await mcp_client.classify(
            obj_id=obj_id,
            summary=summary,
            obj_type=obj_type,
            tags=tags,
            project=project,
        )
        if isinstance(result, dict) and result.get("status") == "error":
            msg = result.get("message", "")
            if "Not found" in msg or "not found" in msg:
                raise HTTPException(status_code=404, detail=msg)
            raise HTTPException(status_code=400, detail=msg)
        return result

    @app.get("/graph/{obj_id}", dependencies=[Depends(rate_limit)])
    async def graph_obj(
        obj_id: str,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Get graph around an object."""
        return await mcp_client.graph(obj_id=obj_id)

    @app.get(
        "/graph/entity/{entity_name}", dependencies=[Depends(rate_limit)]
    )
    async def graph_entity(
        entity_name: str,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Get graph around an entity."""
        return await mcp_client.graph(entity=entity_name)

    @app.get("/list", dependencies=[Depends(rate_limit)])
    async def list_objects(
        doc_type: str = "",
        project: str = "",
        limit: int = 50,
        _key: str = Depends(verify_api_key),
    ) -> list[dict[str, Any]]:
        """List knowledge objects."""
        return await mcp_client.list_objects(
            doc_type=doc_type, project=project, limit=limit
        )

    # ─── Pipeline & Reasoning ─────────────────────────────────────

    @app.post("/pipeline/{obj_id}", dependencies=[Depends(rate_limit)])
    async def run_pipeline_endpoint(
        obj_id: str,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Re-run the intelligence pipeline on an existing object."""
        result = await mcp_client.pipeline(obj_id)
        if isinstance(result, dict) and result.get("error"):
            raise HTTPException(
                status_code=404, detail=result.get("error")
            )
        return result

    @app.post("/reason", dependencies=[Depends(rate_limit)])
    async def run_reason(
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Run advanced reasoning checks."""
        return await mcp_client.reason()

    # ─── Admin Endpoints ───────────────────────────────────────────

    @app.get("/status", dependencies=[Depends(rate_limit)])
    async def status(
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Get Cortex status and health."""
        return await mcp_client.status()

    @app.post("/synthesize", dependencies=[Depends(rate_limit)])
    async def synthesize(
        period_days: int = 7,
        project: str = "",
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Generate synthesis."""
        return await mcp_client.synthesize(
            period_days=period_days, project=project
        )

    @app.delete("/delete/{obj_id}", dependencies=[Depends(rate_limit)])
    async def delete(
        obj_id: str,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Delete a knowledge object."""
        result = await mcp_client.delete(obj_id)
        if isinstance(result, dict):
            status_val = result.get("status")
            if status_val == "not_found" or (
                status_val == "error"
                and "not found" in result.get("message", "").lower()
            ):
                raise HTTPException(
                    status_code=404, detail=f"Not found: {obj_id}"
                )
        return result

    return app


def _error_response(status_code: int, detail: str):
    """Build a plain JSON response for the MCP error handlers."""
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"detail": detail})
