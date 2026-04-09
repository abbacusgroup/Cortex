"""Async MCP HTTP client wrapper used by the dashboard.

The dashboard never opens ``graph.db`` directly. Instead it talks to a running
``cortex serve --transport mcp-http`` process via this wrapper, which uses the
official ``mcp`` SDK's streamable-http client.

Each public method maps to a single MCP tool call and returns a JSON-shaped
``dict`` or ``list`` ready for template rendering. Connection errors, timeouts,
HTTP failures, and tool errors are surfaced as typed exceptions so dashboard
endpoints can map them to the right HTTP status code.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from cortex.core.errors import CortexError
from cortex.core.logging import get_logger

logger = get_logger("dashboard.mcp_client")


# ─── Typed exceptions ──────────────────────────────────────────────────────


class MCPClientError(CortexError):
    """Base error for the dashboard MCP client."""

    code = "CORTEX_MCP_CLIENT_ERROR"


class MCPConnectionError(MCPClientError):
    """The MCP server is unreachable (connection refused, DNS error, etc.)."""

    code = "CORTEX_MCP_CONNECTION_ERROR"


class MCPTimeoutError(MCPClientError):
    """A request to the MCP server timed out."""

    code = "CORTEX_MCP_TIMEOUT"


class MCPServerError(MCPClientError):
    """The MCP server responded with a 5xx or other server-side failure."""

    code = "CORTEX_MCP_SERVER_ERROR"


class MCPToolError(MCPClientError):
    """The remote MCP tool returned an error result."""

    code = "CORTEX_MCP_TOOL_ERROR"


# ─── Helpers ───────────────────────────────────────────────────────────────


# Sentinel for _call() — see comment in _call for why this exists.
_UNSET: Any = object()


@asynccontextmanager
async def _http_client_session(url: str, timeout_seconds: float):
    """Bridge old (timeout=N) → new (http_client=...) API.

    The deprecated ``streamablehttp_client`` took ``timeout=N`` directly.
    The canonical ``streamable_http_client`` requires an ``httpx.AsyncClient``
    with the timeout configured on it. This helper hides the boilerplate so
    call sites can stay as ``async with _http_client_session(url, t) as (r, w, sid):``.
    """
    async with (
        httpx.AsyncClient(timeout=timeout_seconds) as http_client,
        streamable_http_client(url, http_client=http_client) as result,
    ):
        yield result


# ─── Bundle 10.8: BaseExceptionGroup unwrap helpers ───────────────────────


def _flatten_exception_group(eg: BaseExceptionGroup) -> list[BaseException]:
    """Recursively flatten an exception group to its leaf (non-group) exceptions."""
    leaves: list[BaseException] = []

    def _collect(e: BaseException) -> None:
        if isinstance(e, BaseExceptionGroup):
            for sub in e.exceptions:
                _collect(sub)
        else:
            leaves.append(e)

    _collect(eg)
    return leaves


def _pick_significant_leaf(leaves: list[BaseException]) -> BaseException:
    """Pick the most significant leaf from a flattened exception group.

    Priority: already-classified ``MCPClientError`` > timeout > HTTP status >
    connection error > first leaf as fallback.
    """
    if not leaves:
        return RuntimeError("empty exception group")
    for leaf in leaves:
        if isinstance(leaf, MCPClientError):
            return leaf
    for leaf in leaves:
        if isinstance(leaf, (httpx.TimeoutException, TimeoutError)):
            return leaf
    for leaf in leaves:
        if isinstance(leaf, httpx.HTTPStatusError):
            return leaf
    for leaf in leaves:
        if isinstance(leaf, (httpx.ConnectError, ConnectionError)):
            return leaf
    return leaves[0]


def _classify_transport_exception(
    exc: BaseException,
    *,
    url: str,
    timeout: float,
) -> MCPClientError:
    """Map a single transport-layer exception to a typed MCP client error.

    Used for both bare exceptions (from the normal ``except Exception`` path)
    and for the representative leaf of a ``BaseExceptionGroup``. Cancellation
    is NOT handled here — callers must check for ``asyncio.CancelledError``
    before calling this helper.
    """
    if isinstance(exc, MCPClientError):
        return exc
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return MCPTimeoutError(
            f"MCP server at {url} timed out after {timeout}s",
            context={"url": url, "timeout": timeout},
            cause=exc,
        )
    if isinstance(exc, httpx.HTTPStatusError):
        return MCPServerError(
            f"MCP server at {url} returned {exc.response.status_code}",
            context={"url": url, "status": exc.response.status_code},
            cause=exc,
        )
    if isinstance(exc, (httpx.ConnectError, ConnectionError)):
        return MCPConnectionError(
            f"Cannot reach MCP server at {url}",
            context={"url": url},
            cause=exc,
        )
    return MCPConnectionError(
        f"MCP transport error talking to {url}: {exc}",
        context={"url": url},
        cause=exc,
    )


def _unwrap_call_tool_result(name: str, result: Any) -> Any:
    """Convert ``CallToolResult`` into the underlying JSON dict/list/str.

    FastMCP wraps tool returns into a list of ``TextContent`` blocks containing
    JSON-encoded strings. We undo that wrapping so dashboard endpoints get
    plain Python objects.
    """
    if getattr(result, "isError", False):
        # MCP returned an error result
        message = ""
        try:
            message = result.content[0].text  # type: ignore[union-attr]
        except (AttributeError, IndexError):
            message = str(result)
        raise MCPToolError(
            f"Tool {name!r} returned error: {message}",
            context={"tool": name, "raw": message},
        )

    # Prefer the structured result when present (MCP SDK 1.6+)
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        # Some tool definitions wrap their return value in {"result": ...}
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured

    # Fall back to parsing the first content block
    content = getattr(result, "content", None) or []
    if not content:
        return None
    first = content[0]
    text = getattr(first, "text", None)
    if text is None:
        return None
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text


# ─── Client ────────────────────────────────────────────────────────────────


class CortexMCPClient:
    """Thin async client for Cortex's MCP HTTP server.

    Each public method opens a fresh ``streamablehttp_client`` session, calls
    one tool, and returns the unwrapped result. Sessions are NOT pooled —
    they're cheap enough for a local dashboard and pooling would complicate
    cancellation and reconnection semantics.
    """

    def __init__(self, url: str, *, timeout_seconds: float = 10.0):
        self.url = url
        self.timeout = timedelta(seconds=timeout_seconds)
        self._timeout_seconds = timeout_seconds

    async def _call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a single MCP tool and return its unwrapped result."""
        # Defensive sentinel: under rare concurrency conditions a task
        # group inside ClientSession.__aexit__ may suppress an inner error
        # and exit the ``async with`` without raising. Without this
        # sentinel, control flow would reach the ``return`` below with
        # ``result`` unbound and crash with ``UnboundLocalError``. We
        # surface a clean MCPConnectionError instead.
        result: Any = _UNSET
        try:
            async with (
                _http_client_session(self.url, self._timeout_seconds) as (
                    read_stream,
                    write_stream,
                    _get_session_id,
                ),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                result = await session.call_tool(
                    name,
                    arguments=arguments or {},
                    read_timeout_seconds=self.timeout,
                )
        except MCPClientError:
            raise
        except BaseExceptionGroup as eg:
            # Bundle 10.8: anyio wraps transport errors in
            # BaseExceptionGroup when they fire inside ClientSession's
            # internal TaskGroup. Without this handler, timeouts surface
            # as "unhandled errors in a TaskGroup (1 sub-exception)".
            leaves = _flatten_exception_group(eg)
            cancelled = next(
                (e for e in leaves if isinstance(e, asyncio.CancelledError)),
                None,
            )
            if cancelled is not None:
                raise cancelled from None
            raise _classify_transport_exception(
                _pick_significant_leaf(leaves),
                url=self.url,
                timeout=self._timeout_seconds,
            ) from eg
        except Exception as e:
            raise _classify_transport_exception(
                e, url=self.url, timeout=self._timeout_seconds
            ) from e
        if result is _UNSET:
            raise MCPConnectionError(
                f"MCP call to {name} completed without a result "
                f"(likely transport cancellation or task group teardown)",
                context={"url": self.url, "tool": name},
            )
        return _unwrap_call_tool_result(name, result)

    async def list_tools(self) -> list[str]:
        """Return the names of tools the MCP server exposes.

        Used at startup to verify the server is the right version.
        """
        result: Any = _UNSET
        try:
            async with (
                _http_client_session(self.url, self._timeout_seconds) as (
                    read_stream,
                    write_stream,
                    _,
                ),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                result = await session.list_tools()
        except MCPClientError:
            raise
        except BaseExceptionGroup as eg:
            leaves = _flatten_exception_group(eg)
            cancelled = next(
                (e for e in leaves if isinstance(e, asyncio.CancelledError)),
                None,
            )
            if cancelled is not None:
                raise cancelled from None
            raise _classify_transport_exception(
                _pick_significant_leaf(leaves),
                url=self.url,
                timeout=self._timeout_seconds,
            ) from eg
        except Exception as e:
            raise _classify_transport_exception(
                e, url=self.url, timeout=self._timeout_seconds
            ) from e
        if result is _UNSET:
            raise MCPConnectionError(
                "MCP list_tools call completed without a result "
                "(likely transport cancellation or task group teardown)",
                context={"url": self.url},
            )
        return [t.name for t in result.tools]

    # ─── Tool methods ──────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        doc_type: str = "",
        project: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return await self._call(
            "cortex_search",
            {"query": query, "doc_type": doc_type, "project": project, "limit": limit},
        )

    async def context(self, topic: str, limit: int = 10) -> list[dict[str, Any]]:
        return await self._call("cortex_context", {"topic": topic, "limit": limit})

    async def dossier(self, topic: str) -> dict[str, Any]:
        return await self._call("cortex_dossier", {"topic": topic})

    async def read(self, obj_id: str) -> dict[str, Any] | str:
        return await self._call("cortex_read", {"obj_id": obj_id})

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
        """Capture a knowledge object.

        The extra kwargs (``template``, ``run_pipeline``, ``summary``,
        ``entities``, ``properties``) mirror the ``cortex_capture`` MCP
        tool and the REST API ``/capture`` endpoint so every caller can
        ship the same shape of request.
        """
        return await self._call(
            "cortex_capture",
            {
                "title": title,
                "content": content,
                "obj_type": obj_type,
                "project": project,
                "tags": tags,
                "template": template,
                "run_pipeline": run_pipeline,
                "summary": summary,
                "entities": entities,
                "properties": properties,
            },
        )

    async def link(
        self, from_id: str, rel_type: str, to_id: str
    ) -> dict[str, Any]:
        """Create a typed relationship between two knowledge objects."""
        return await self._call(
            "cortex_link",
            {"from_id": from_id, "rel_type": rel_type, "to_id": to_id},
        )

    async def feedback(
        self, obj_id: str, relevant: bool = True
    ) -> dict[str, Any]:
        """Record explicit relevance feedback for a knowledge object."""
        return await self._call(
            "cortex_feedback", {"obj_id": obj_id, "relevant": relevant}
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
        """Classify or reclassify an existing knowledge object."""
        return await self._call(
            "cortex_classify",
            {
                "obj_id": obj_id,
                "summary": summary,
                "obj_type": obj_type,
                "entities": entities,
                "properties": properties,
                "tags": tags,
                "project": project,
            },
        )

    async def delete(self, obj_id: str) -> dict[str, Any]:
        """Delete a knowledge object (admin tool)."""
        return await self._call("cortex_delete", {"obj_id": obj_id})

    async def list_objects(
        self, doc_type: str = "", project: str = "", limit: int = 50
    ) -> list[dict[str, Any]]:
        return await self._call(
            "cortex_list",
            {"doc_type": doc_type, "project": project, "limit": limit},
        )

    async def graph(self, obj_id: str = "", entity: str = "") -> dict[str, Any]:
        return await self._call(
            "cortex_graph", {"obj_id": obj_id, "entity": entity}
        )

    async def status(self) -> dict[str, Any]:
        return await self._call("cortex_status")

    async def query_trail(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._call("cortex_query_trail", {"limit": limit})

    async def list_entities(self, entity_type: str = "") -> list[dict[str, Any]]:
        return await self._call(
            "cortex_list_entities", {"entity_type": entity_type}
        )

    async def graph_data(
        self,
        project: str = "",
        doc_type: str = "",
        limit: int = 500,
        offset: int = 0,
    ) -> dict[str, Any]:
        return await self._call(
            "cortex_graph_data",
            {
                "project": project,
                "doc_type": doc_type,
                "limit": limit,
                "offset": offset,
            },
        )

    # ─── Phase 3: methods used by CLI commands ────────────────────────

    async def pipeline(self, obj_id: str) -> dict[str, Any]:
        return await self._call("cortex_pipeline", {"obj_id": obj_id})

    async def synthesize(
        self, period_days: int = 7, project: str = ""
    ) -> dict[str, Any]:
        return await self._call(
            "cortex_synthesize",
            {"period_days": period_days, "project": project},
        )

    async def reason(self) -> dict[str, Any]:
        return await self._call("cortex_reason")
