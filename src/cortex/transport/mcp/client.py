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


@asynccontextmanager
async def _http_client_session(url: str, timeout_seconds: float):
    """Bridge old (timeout=N) → new (http_client=...) API.

    The deprecated ``streamablehttp_client`` took ``timeout=N`` directly.
    The canonical ``streamable_http_client`` requires an ``httpx.AsyncClient``
    with the timeout configured on it. This helper hides the boilerplate so
    call sites can stay as ``async with _http_client_session(url, t) as (r, w, sid):``.
    """
    async with httpx.AsyncClient(timeout=timeout_seconds) as http_client:
        async with streamable_http_client(url, http_client=http_client) as result:
            yield result


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

    def __init__(self, url: str, *, timeout_seconds: float = 5.0):
        self.url = url
        self.timeout = timedelta(seconds=timeout_seconds)
        self._timeout_seconds = timeout_seconds

    async def _call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a single MCP tool and return its unwrapped result."""
        try:
            async with _http_client_session(
                self.url, self._timeout_seconds
            ) as (read_stream, write_stream, _get_session_id):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        name,
                        arguments=arguments or {},
                        read_timeout_seconds=self.timeout,
                    )
        except httpx.ConnectError as e:
            raise MCPConnectionError(
                f"Cannot reach MCP server at {self.url}",
                context={"url": self.url},
                cause=e,
            )
        except httpx.TimeoutException as e:
            raise MCPTimeoutError(
                f"MCP server at {self.url} timed out after {self._timeout_seconds}s",
                context={"url": self.url, "timeout": self._timeout_seconds},
                cause=e,
            )
        except httpx.HTTPStatusError as e:
            raise MCPServerError(
                f"MCP server at {self.url} returned {e.response.status_code}",
                context={"url": self.url, "status": e.response.status_code},
                cause=e,
            )
        except (MCPClientError,):
            raise
        except Exception as e:
            # Wrap any other transport-level error so callers don't have to
            # know about anyio/httpx internals.
            raise MCPConnectionError(
                f"MCP transport error talking to {self.url}: {e}",
                context={"url": self.url},
                cause=e,
            )
        return _unwrap_call_tool_result(name, result)

    async def list_tools(self) -> list[str]:
        """Return the names of tools the MCP server exposes.

        Used at startup to verify the server is the right version.
        """
        try:
            async with _http_client_session(
                self.url, self._timeout_seconds
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
        except httpx.ConnectError as e:
            raise MCPConnectionError(
                f"Cannot reach MCP server at {self.url}",
                context={"url": self.url},
                cause=e,
            )
        except httpx.TimeoutException as e:
            raise MCPTimeoutError(
                f"MCP server at {self.url} timed out",
                context={"url": self.url},
                cause=e,
            )
        except (MCPClientError,):
            raise
        except Exception as e:
            raise MCPConnectionError(
                f"MCP transport error: {e}",
                context={"url": self.url},
                cause=e,
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
    ) -> dict[str, Any]:
        return await self._call(
            "cortex_capture",
            {
                "title": title,
                "content": content,
                "obj_type": obj_type,
                "project": project,
                "tags": tags,
            },
        )

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
