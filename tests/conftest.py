"""Shared pytest fixtures and helpers.

The star of the show here is :class:`FakeMCPClient`, an in-process drop-in
for :class:`cortex.transport.mcp.client.CortexMCPClient`. It holds a real
``Store`` behind a real MCP server and dispatches calls to the tool
functions directly instead of going out over HTTP. Both the dashboard and
REST API test suites inject this client so they can exercise their real
code paths (endpoint routing, template rendering, auth) against a live
store without spinning up subprocesses.
"""

from __future__ import annotations

from typing import Any

from cortex.db.store import Store


class FakeMCPClient:
    """In-process drop-in for ``CortexMCPClient``.

    Wraps a real :class:`cortex.transport.mcp.server.create_mcp_server`
    instance and routes each async method to the corresponding registered
    tool function. Async signatures match the real client so callers'
    ``await`` lines keep working transparently.

    This is the shared test helper used by both ``tests/dashboard`` and
    ``tests/transport/test_api.py`` — keeping the two in sync.
    """

    def __init__(self, mcp):
        self._mcp = mcp
        self._tools = mcp._tool_manager._tools

    # ─── internal helpers ──────────────────────────────────────────────

    def _call(self, name: str, **kwargs) -> Any:
        return self._tools[name].fn(**kwargs)

    @property
    def store(self) -> Store:
        """Reach into the closure of the MCP server to expose the Store.

        Tests that want to seed data directly (bypassing the pipeline)
        use this. The closure layout is stable because ``create_mcp_server``
        captures ``store`` in every tool's closure.
        """
        return self._tools["cortex_list"].fn.__closure__[0].cell_contents

    # ─── public async methods (match CortexMCPClient signature) ────────

    async def search(
        self, query: str, doc_type: str = "", project: str = "", limit: int = 20
    ) -> list[dict[str, Any]]:
        return self._call(
            "cortex_search",
            query=query,
            doc_type=doc_type,
            project=project,
            limit=limit,
        )

    async def context(
        self, topic: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        return self._call("cortex_context", topic=topic, limit=limit)

    async def dossier(self, topic: str) -> dict[str, Any]:
        return self._call("cortex_dossier", topic=topic)

    async def read(self, obj_id: str) -> dict[str, Any] | str:
        return self._call("cortex_read", obj_id=obj_id)

    async def capture(
        self,
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
        return self._call(
            "cortex_capture",
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

    async def list_objects(
        self, doc_type: str = "", project: str = "", limit: int = 50
    ) -> list[dict[str, Any]]:
        return self._call(
            "cortex_list", doc_type=doc_type, project=project, limit=limit
        )

    async def graph(
        self, obj_id: str = "", entity: str = ""
    ) -> dict[str, Any]:
        return self._call("cortex_graph", obj_id=obj_id, entity=entity)

    async def status(self) -> dict[str, Any]:
        return self._call("cortex_status")

    async def query_trail(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._call("cortex_query_trail", limit=limit)

    async def list_entities(
        self, entity_type: str = ""
    ) -> list[dict[str, Any]]:
        return self._call("cortex_list_entities", entity_type=entity_type)

    async def graph_data(
        self,
        project: str = "",
        doc_type: str = "",
        limit: int = 500,
        offset: int = 0,
    ) -> dict[str, Any]:
        return self._call(
            "cortex_graph_data",
            project=project,
            doc_type=doc_type,
            limit=limit,
            offset=offset,
        )

    # ─── Phase 3 methods (CLI routing) ─────────────────────────────────

    async def pipeline(self, obj_id: str) -> dict[str, Any]:
        return self._call("cortex_pipeline", obj_id=obj_id)

    async def synthesize(
        self, period_days: int = 7, project: str = ""
    ) -> dict[str, Any]:
        return self._call(
            "cortex_synthesize", period_days=period_days, project=project
        )

    async def reason(self) -> dict[str, Any]:
        return self._call("cortex_reason")

    # ─── Bundle 7 / Phase 4 methods (REST API routing) ─────────────────

    async def link(
        self, from_id: str, rel_type: str, to_id: str
    ) -> dict[str, Any]:
        return self._call(
            "cortex_link", from_id=from_id, rel_type=rel_type, to_id=to_id
        )

    async def feedback(
        self, obj_id: str, relevant: bool = True
    ) -> dict[str, Any]:
        return self._call(
            "cortex_feedback", obj_id=obj_id, relevant=relevant
        )

    async def classify(
        self,
        obj_id: str,
        summary: str = "",
        obj_type: str = "",
        entities: str = "",
        properties: str = "",
        tags: str = "",
        project: str = "",
    ) -> dict[str, Any]:
        return self._call(
            "cortex_classify",
            obj_id=obj_id,
            summary=summary,
            obj_type=obj_type,
            entities=entities,
            properties=properties,
            tags=tags,
            project=project,
        )

    async def delete(self, obj_id: str) -> dict[str, Any]:
        return self._call("cortex_delete", obj_id=obj_id)

    # ─── transport layer ───────────────────────────────────────────────

    async def list_tools(self) -> list[str]:
        return list(self._tools.keys())
