"""Cortex MCP server — 17 tools for AI agent integration.

Provides both stdio and StreamableHTTP transports.
Admin tools (status, synthesize, delete, export, safety_check, reason) are
only exposed on stdio, not HTTP.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from cortex.core.config import CortexConfig, load_config
from cortex.core.logging import get_logger, setup_logging
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.pipeline.link import LinkStage
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

logger = get_logger("transport.mcp")

# Admin tool names — gated by include_admin (true for stdio + localhost mcp-http).
ADMIN_TOOLS = frozenset({
    "cortex_status",
    "cortex_synthesize",
    "cortex_delete",
    "cortex_export",
    "cortex_safety_check",
    "cortex_reason",
    "cortex_query_trail",
    "cortex_graph_data",
    "cortex_list_entities",
})

# Maximum result size caps for the new dashboard-aggregation tools.
QUERY_TRAIL_MAX_LIMIT = 1000
GRAPH_DATA_MAX_OBJECTS = 1000


def create_mcp_server(
    config: CortexConfig | None = None,
    *,
    include_admin: bool = True,
) -> FastMCP:
    """Create and configure the Cortex MCP server.

    Args:
        config: Cortex configuration. If None, loads from env.
        include_admin: If True, register admin tools (stdio mode).

    Returns:
        Configured FastMCP server instance.
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

    mcp = FastMCP("cortex")

    # ─── Public Tools ──────────────────────────────────────────────

    @mcp.tool()
    def cortex_search(
        query: str,
        doc_type: str = "",
        project: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search knowledge objects with hybrid keyword + semantic + graph ranking.

        Args:
            query: Search query text.
            doc_type: Filter by type (decision, lesson, fix, session, etc.)
            project: Filter by project name.
            limit: Maximum results (default 20).
        """
        return engine.search(
            query,
            doc_type=doc_type or None,
            project=project or None,
            limit=limit,
        )

    @mcp.tool()
    def cortex_context(
        topic: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get a briefing (summaries only) for a topic. Token-efficient.

        Args:
            topic: Topic to get context for.
            limit: Maximum results (default 10).
        """
        results = engine.search(topic, limit=limit)
        presenter = BriefingPresenter()
        return presenter.render(results)

    @mcp.tool()
    def cortex_dossier(
        topic: str,
    ) -> dict[str, Any]:
        """Build an entity/topic-centric intelligence brief.

        Includes related objects, contradictions, timeline, and related entities.

        Args:
            topic: Entity name or topic to build dossier for.
        """
        presenter = DossierPresenter(store, llm)
        return presenter.render(topic)

    @mcp.tool()
    def cortex_read(
        obj_id: str,
    ) -> dict[str, Any] | str:
        """Read a knowledge object in full detail.

        Args:
            obj_id: Object ID to read.
        """
        presenter = DocumentPresenter(store)
        result = presenter.render(obj_id)
        if result is None:
            return f"Not found: {obj_id}"
        learner.record_access(obj_id)
        return result

    @mcp.tool()
    def cortex_capture(
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
    ) -> dict[str, Any]:
        """Capture a knowledge object. When calling via MCP, YOU (Claude) are the classifier.

        Analyze the content and provide classification data directly:
        - summary: 1-2 sentence summary of the content
        - entities: JSON array of entities mentioned,
          e.g. [{"name": "Python", "type": "technology"}]
          Entity types: technology, project, pattern, concept
        - obj_type: Best-fit type from: decision, lesson, fix,
          session, research, source, synthesis, idea

        Optionally provide type-specific properties as JSON:
          decision: {"rationale": "...", "chosen": "..."}
          fix: {"symptom": "...", "rootCause": "...", "resolution": "..."}
          session: {"goal": "...", "worked": "...", "failed": "...", "nextSteps": "..."}
          lesson: {"cause": "...", "impact": "...", "prevention": "..."}

        Args:
            title: Title for the knowledge object.
            content: Full text content.
            obj_type: Type classification.
            project: Project name.
            tags: Comma-separated tags.
            template: Template name.
            run_pipeline: Run intelligence pipeline (default True).
            summary: Your 1-2 sentence summary.
            entities: JSON array of entities: [{"name": "...", "type": "..."}].
            properties: JSON of type-specific properties.
        """
        parsed_entities = None
        if entities:
            try:
                parsed_entities = json.loads(entities)
            except json.JSONDecodeError:
                parsed_entities = None

        parsed_properties = None
        if properties:
            try:
                parsed_properties = json.loads(properties)
            except json.JSONDecodeError:
                parsed_properties = None

        return pipeline.capture(
            title=title,
            content=content,
            obj_type=obj_type,
            project=project,
            tags=tags,
            template=template or None,
            captured_by="mcp",
            run_pipeline=run_pipeline,
            summary=summary,
            entities=parsed_entities,
            extra_properties=parsed_properties,
            confidence=0.9 if summary else 0.0,
        )

    @mcp.tool()
    def cortex_link(
        from_id: str,
        rel_type: str,
        to_id: str,
    ) -> dict[str, Any]:
        """Create a relationship between two knowledge objects.

        Args:
            from_id: Source object ID.
            rel_type: Relationship type (causedBy, contradicts, supports, etc.).
            to_id: Target object ID.
        """
        try:
            store.create_relationship(from_id=from_id, rel_type=rel_type, to_id=to_id)
            return {"status": "created", "from": from_id, "rel_type": rel_type, "to": to_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @mcp.tool()
    def cortex_feedback(
        obj_id: str,
        relevant: bool = True,
    ) -> dict[str, Any]:
        """Provide explicit relevance feedback for an object.

        Args:
            obj_id: Object ID to provide feedback for.
            relevant: True if the object was relevant/useful.
        """
        if relevant:
            learner.record_access(obj_id)
        return {
            "status": "recorded",
            "obj_id": obj_id,
            "relevant": relevant,
            "access_count": learner.get_access_count(obj_id),
        }

    @mcp.tool()
    def cortex_graph(
        obj_id: str = "",
        entity: str = "",
    ) -> dict[str, Any]:
        """Get the knowledge graph around an object or entity.

        Args:
            obj_id: Object ID to get graph for.
            entity: Entity name to get neighborhood for (alternative to obj_id).
        """
        if entity:
            return graph_queries.entity_neighborhood(entity)
        if obj_id:
            return {
                "causal_chain": graph_queries.causal_chain(obj_id),
                "evolution": graph_queries.evolution_timeline(obj_id),
                "relationships": store.get_relationships(obj_id),
            }
        return {"error": "Provide either obj_id or entity"}

    @mcp.tool()
    def cortex_list(
        doc_type: str = "",
        project: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List knowledge objects with optional filters.

        Args:
            doc_type: Filter by type.
            project: Filter by project.
            limit: Maximum results (default 50).
        """
        return store.list_objects(
            obj_type=doc_type or None,
            project=project or None,
            limit=limit,
        )

    @mcp.tool()
    def cortex_pipeline(
        obj_id: str,
    ) -> dict[str, Any]:
        """Re-run the intelligence pipeline on an existing object.

        Args:
            obj_id: Object ID to re-process.
        """
        doc = store.read(obj_id)
        if doc is None:
            return {"error": f"Not found: {obj_id}"}
        return pipeline.run_pipeline(obj_id)

    @mcp.tool()
    def cortex_classify(
        obj_id: str,
        summary: str = "",
        obj_type: str = "",
        entities: str = "",
        properties: str = "",
        tags: str = "",
        project: str = "",
    ) -> dict[str, Any]:
        """Classify or reclassify an existing knowledge object.

        Use cortex_read first to examine the object, then call this to enrich it.

        Provide:
        - summary: 1-2 sentence summary
        - obj_type: Corrected type (decision, lesson, fix,
          session, research, source, synthesis, idea)
        - entities: JSON array of [{"name": "...", "type": "technology|project|pattern|concept"}]

        Args:
            obj_id: ID of the object to classify.
            summary: Your summary of the content.
            obj_type: Corrected type classification.
            entities: JSON array of entities.
            properties: JSON of type-specific properties.
            tags: Updated comma-separated tags.
            project: Updated project name.
        """
        doc = store.read(obj_id)
        if doc is None:
            return {"status": "error", "message": f"Not found: {obj_id}"}

        updates: dict[str, Any] = {"confidence": 0.9}
        if summary:
            updates["summary"] = summary
        if obj_type:
            updates["type"] = obj_type
        if tags:
            updates["tags"] = tags
        if project:
            updates["project"] = project

        try:
            store.content.update(obj_id, **updates)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        # Update graph with properties
        graph_updates: dict[str, str] = {}
        if summary:
            graph_updates["summary"] = summary
        if properties:
            try:
                parsed = json.loads(properties)
                graph_updates.update({k: v for k, v in parsed.items() if isinstance(v, str)})
            except json.JSONDecodeError:
                pass
        if graph_updates:
            try:
                store.graph.update_object(obj_id, **graph_updates)
            except Exception:
                pass

        # Resolve entities
        resolved = []
        if entities:
            try:
                parsed_entities = json.loads(entities)
                linker = LinkStage(store, llm)
                resolved = linker._resolve_entities(obj_id, parsed_entities)
            except Exception as e:
                logger.warning("Entity resolution failed: %s", e)

        return {
            "status": "classified",
            "obj_id": obj_id,
            "updates": updates,
            "entities_resolved": len(resolved),
        }

    # ─── Admin Tools ───────────────────────────────────────────────

    if include_admin:

        @mcp.tool()
        def cortex_status() -> dict[str, Any]:
            """Get Cortex status, health metrics, and object counts."""
            stats = store.status()
            alerts = AlertPresenter(store).render()
            stats["alerts"] = alerts
            stats["alert_count"] = len(alerts)
            return stats

        @mcp.tool()
        def cortex_synthesize(
            period_days: int = 7,
            project: str = "",
        ) -> dict[str, Any]:
            """Generate a synthesis of recent knowledge.

            Args:
                period_days: Number of days to cover (default 7).
                project: Scope to a specific project.
            """
            presenter = SynthesisPresenter(store, llm)
            return presenter.render(
                period_days=period_days,
                project=project or None,
            )

        @mcp.tool()
        def cortex_delete(
            obj_id: str,
        ) -> dict[str, Any]:
            """Delete a knowledge object.

            Args:
                obj_id: Object ID to delete.
            """
            deleted = store.delete(obj_id)
            return {
                "status": "deleted" if deleted else "not_found",
                "obj_id": obj_id,
            }

        @mcp.tool()
        def cortex_export(
            obj_id: str,
            format: str = "markdown",
        ) -> dict[str, Any]:
            """Export a knowledge object.

            Args:
                obj_id: Object ID to export.
                format: Export format (markdown).
            """
            doc = store.read(obj_id)
            if doc is None:
                return {"status": "not_found", "obj_id": obj_id}

            if format == "markdown":
                md = f"# {doc.get('title', '')}\n\n"
                md += f"**Type:** {doc.get('type', '')}\n"
                md += f"**Project:** {doc.get('project', '')}\n"
                md += f"**Tags:** {doc.get('tags', '')}\n"
                md += f"**Created:** {doc.get('created_at', '')}\n\n"
                md += doc.get("content", "")
                return {"status": "exported", "format": format, "content": md}

            return {"status": "error", "message": f"Unknown format: {format}"}

        @mcp.tool()
        def cortex_safety_check(
            action: str,
            target: str,
        ) -> dict[str, Any]:
            """Review a potentially destructive action before executing.

            Args:
                action: The action to review (delete, bulk_delete, etc.)
                target: What will be affected.
            """
            return {
                "action": action,
                "target": target,
                "warning": f"This will {action} '{target}'. Confirm before proceeding.",
                "confirmed": False,
            }

        @mcp.tool()
        def cortex_reason() -> dict[str, Any]:
            """Run advanced reasoning: contradictions, patterns, gaps, staleness."""
            from cortex.pipeline.advanced_reason import AdvancedReasoner
            reasoner = AdvancedReasoner(store, llm)
            return reasoner.run_all()

        @mcp.tool()
        def cortex_list_entities(entity_type: str = "") -> list[dict[str, Any]]:
            """List entities in the knowledge graph, optionally filtered by type.

            Args:
                entity_type: Optional filter (technology, project, pattern, concept).
            """
            return store.list_entities(entity_type=entity_type or None)

        @mcp.tool()
        def cortex_query_trail(limit: int = 50) -> list[dict[str, Any]]:
            """Return the most recent search query log entries.

            Used by the dashboard's /trail page. Capped at 1000 entries.

            Args:
                limit: Number of entries to return (default 50, max 1000).
            """
            if limit <= 0:
                return []
            capped = min(limit, QUERY_TRAIL_MAX_LIMIT)
            return store.content.get_query_log(limit=capped)

        @mcp.tool()
        def cortex_graph_data(
            project: str = "",
            doc_type: str = "",
            limit: int = 500,
            offset: int = 0,
        ) -> dict[str, Any]:
            """Return graph data in Cytoscape.js format for dashboard visualization.

            Aggregates objects, relationships, and entities into a single payload.

            Args:
                project: Filter by project name.
                doc_type: Filter by knowledge type.
                limit: Maximum number of objects (default 500, max 1000).
                offset: Pagination offset.
            """
            capped_limit = min(max(limit, 0), GRAPH_DATA_MAX_OBJECTS)
            objects = store.list_objects(
                obj_type=doc_type or None,
                project=project or None,
                limit=capped_limit,
                offset=max(offset, 0),
            )

            nodes: list[dict[str, Any]] = []
            edges: list[dict[str, Any]] = []
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
                for rel in store.get_relationships(obj_id):
                    if rel["direction"] != "outgoing":
                        continue
                    edge_key = f"{obj_id}-{rel['rel_type']}-{rel['other_id']}"
                    if edge_key in seen_edges:
                        continue
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

            for entity in store.list_entities():
                eid = f"entity:{entity['id']}"
                nodes.append(
                    {
                        "data": {
                            "id": eid,
                            "label": entity["name"],
                            "type": f"entity:{entity['type']}",
                            "project": "",
                        },
                    }
                )
                for mid in store.graph.get_entity_mentions(entity["id"]):
                    edge_key = f"{mid}-mentions-{eid}"
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    edges.append(
                        {
                            "data": {
                                "source": mid,
                                "target": eid,
                                "rel_type": "mentions",
                            },
                        }
                    )

            return {
                "nodes": nodes,
                "edges": edges,
                "total": store.content.total_count(),
                "limit": capped_limit,
                "offset": max(offset, 0),
            }

    return mcp


# Localhost-equivalent host strings — we trust these and expose admin tools.
_LOCALHOST_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def run_stdio() -> None:
    """Run the MCP server on stdio (for Claude Code, Cursor, etc.)"""
    mcp = create_mcp_server(include_admin=True)
    mcp.run(transport="stdio")


def run_http(host: str = "127.0.0.1", port: int = 1314) -> None:
    """Run the MCP server on StreamableHTTP.

    When bound to a loopback host (127.0.0.1, localhost, ::1) we trust the
    caller and expose admin tools — this is the mode the local dashboard uses.
    For any other host (e.g. 0.0.0.0 or a public IP) admin tools are excluded
    so untrusted remote agents cannot trigger destructive operations.
    """
    is_local = host in _LOCALHOST_HOSTS
    mcp = create_mcp_server(include_admin=is_local)
    if not is_local:
        logger.warning(
            "Binding MCP HTTP server to non-localhost host %s — admin tools disabled",
            host,
        )
    # FastMCP reads host/port from its settings object — they're not accepted
    # as ``run()`` kwargs in this version of the SDK.
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="streamable-http")
