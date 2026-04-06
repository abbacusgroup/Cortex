"""Cortex REST API — FastAPI server mirroring MCP tools.

Provides HTTP endpoints with API key authentication, rate limiting,
and auto-generated OpenAPI docs.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from cortex.core.config import CortexConfig, load_config
from cortex.core.logging import get_logger, setup_logging
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.pipeline.orchestrator import PipelineOrchestrator
from cortex.retrieval.engine import RetrievalEngine
from cortex.retrieval.graph import GraphQueries
from cortex.retrieval.learner import LearningLoop
from cortex.retrieval.presenters import (
    AlertPresenter,
    BriefingPresenter,
    DocumentPresenter,
    DossierPresenter,
    SynthesisPresenter,
)
from cortex.services.llm import LLMClient

logger = get_logger("transport.api")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Simple in-memory rate limiter
_rate_limits: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 100  # requests per window


def create_api(config: CortexConfig | None = None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        config: Cortex configuration. If None, loads from env.

    Returns:
        Configured FastAPI app.
    """
    if config is None:
        config = load_config()

    setup_logging(level=config.log_level, json_output=config.log_json)

    store = Store(config)
    try:
        ontology_path = find_ontology()
        store.initialize(ontology_path)
    except FileNotFoundError:
        pass

    llm = LLMClient(config)
    pipeline = PipelineOrchestrator(store, config)
    engine = RetrievalEngine(store)
    graph_queries = GraphQueries(store)
    learner = LearningLoop(store)

    app = FastAPI(
        title="Cortex API",
        description="Cognitive knowledge system with formal ontology and reasoning.",
        version="0.1.0",
    )

    # Store config on app state for dependency injection
    app.state.config = config
    app.state.store = store

    # ─── Auth ──────────────────────────────────────────────────────

    def _get_api_keys() -> set[str]:
        """Get valid API keys from config."""
        stored = store.content.get_config("api_keys", "")
        keys = set()
        if stored:
            keys.update(k.strip() for k in stored.split(",") if k.strip())
        # Always allow a configured key
        master = store.content.get_config("master_api_key", "")
        if master:
            keys.add(master)
        return keys

    async def verify_api_key(
        api_key: str | None = Security(api_key_header),
    ) -> str:
        """Verify the API key from the X-API-Key header."""
        if api_key is None:
            raise HTTPException(status_code=401, detail="Missing API key")

        valid_keys = _get_api_keys()
        if not valid_keys:
            # No keys configured — allow all (dev mode)
            return api_key

        if api_key not in valid_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return api_key

    # ─── Rate Limiting ─────────────────────────────────────────────

    async def rate_limit(request: Request) -> None:
        """Simple rate limiter per client IP."""
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()

        # Clean old entries
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
        return engine.search(
            query,
            doc_type=doc_type or None,
            project=project or None,
            limit=limit,
        )

    @app.post("/context", dependencies=[Depends(rate_limit)])
    async def context(
        topic: str,
        limit: int = 10,
        _key: str = Depends(verify_api_key),
    ) -> list[dict[str, Any]]:
        """Get briefing (summaries) for a topic."""
        results = engine.search(topic, limit=limit)
        return BriefingPresenter().render(results)

    @app.post("/dossier", dependencies=[Depends(rate_limit)])
    async def dossier(
        topic: str,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Build entity/topic dossier."""
        return DossierPresenter(store, llm).render(topic)

    @app.get("/read/{obj_id}", dependencies=[Depends(rate_limit)])
    async def read(
        obj_id: str,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Read a knowledge object in full."""
        result = DocumentPresenter(store).render(obj_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Not found: {obj_id}")
        learner.record_access(obj_id)
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
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Capture a knowledge object."""
        return pipeline.capture(
            title=title,
            content=content,
            obj_type=obj_type,
            project=project,
            tags=tags,
            template=template or None,
            captured_by="api",
            run_pipeline=run_pipeline,
        )

    @app.post("/link", dependencies=[Depends(rate_limit)])
    async def link(
        from_id: str,
        rel_type: str,
        to_id: str,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Create a relationship."""
        try:
            store.create_relationship(
                from_id=from_id, rel_type=rel_type, to_id=to_id
            )
            return {"status": "created"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/feedback", dependencies=[Depends(rate_limit)])
    async def feedback(
        obj_id: str,
        relevant: bool = True,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Provide relevance feedback."""
        if relevant:
            learner.record_access(obj_id)
        return {"status": "recorded", "obj_id": obj_id}

    @app.get("/graph/{obj_id}", dependencies=[Depends(rate_limit)])
    async def graph_obj(
        obj_id: str,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Get graph around an object."""
        return {
            "causal_chain": graph_queries.causal_chain(obj_id),
            "evolution": graph_queries.evolution_timeline(obj_id),
            "relationships": store.get_relationships(obj_id),
        }

    @app.get("/graph/entity/{entity_name}", dependencies=[Depends(rate_limit)])
    async def graph_entity(
        entity_name: str,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Get graph around an entity."""
        return graph_queries.entity_neighborhood(entity_name)

    @app.get("/list", dependencies=[Depends(rate_limit)])
    async def list_objects(
        doc_type: str = "",
        project: str = "",
        limit: int = 50,
        _key: str = Depends(verify_api_key),
    ) -> list[dict[str, Any]]:
        """List knowledge objects."""
        return store.list_objects(
            obj_type=doc_type or None,
            project=project or None,
            limit=limit,
        )

    # ─── Pipeline & Reasoning ─────────────────────────────────────

    @app.post("/pipeline/{obj_id}", dependencies=[Depends(rate_limit)])
    async def run_pipeline(
        obj_id: str,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Re-run the intelligence pipeline on an existing object."""
        doc = store.read(obj_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Not found: {obj_id}")
        return pipeline.run_pipeline(obj_id)

    @app.post("/reason", dependencies=[Depends(rate_limit)])
    async def run_reason(
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Run advanced reasoning checks."""
        from cortex.pipeline.advanced_reason import AdvancedReasoner
        reasoner = AdvancedReasoner(store, llm)
        return reasoner.run_all()

    # ─── Admin Endpoints ───────────────────────────────────────────

    @app.get("/status", dependencies=[Depends(rate_limit)])
    async def status(
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Get Cortex status and health."""
        stats = store.status()
        stats["alerts"] = AlertPresenter(store).render()
        return stats

    @app.post("/synthesize", dependencies=[Depends(rate_limit)])
    async def synthesize(
        period_days: int = 7,
        project: str = "",
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Generate synthesis."""
        return SynthesisPresenter(store, llm).render(
            period_days=period_days, project=project or None
        )

    @app.delete("/delete/{obj_id}", dependencies=[Depends(rate_limit)])
    async def delete(
        obj_id: str,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """Delete a knowledge object."""
        deleted = store.delete(obj_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Not found: {obj_id}")
        return {"status": "deleted", "obj_id": obj_id}

    return app
