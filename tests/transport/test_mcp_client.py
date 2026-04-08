"""Tests for the Cortex MCP HTTP client wrapper.

The wrapper is mocked at the ``_http_client_session`` + ``ClientSession``
boundary so tests don't need a running server. Used by both the dashboard
and the CLI (Phase 3).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from cortex.transport.mcp.client import (
    CortexMCPClient,
    MCPClientError,
    MCPConnectionError,
    MCPServerError,
    MCPTimeoutError,
    MCPToolError,
    _http_client_session,
    _unwrap_call_tool_result,
)


# ─── _unwrap_call_tool_result helpers ─────────────────────────────────────


class _FakeTextContent:
    def __init__(self, text: str):
        self.text = text


class _FakeCallToolResult:
    def __init__(
        self,
        *,
        text: str | None = None,
        structured: object | None = None,
        is_error: bool = False,
    ):
        if text is not None:
            self.content = [_FakeTextContent(text)]
        else:
            self.content = []
        self.structuredContent = structured
        self.isError = is_error


class TestUnwrapResult:
    def test_unwraps_json_text_content(self):
        result = _FakeCallToolResult(text=json.dumps({"hello": "world"}))
        out = _unwrap_call_tool_result("test_tool", result)
        assert out == {"hello": "world"}

    def test_unwraps_json_array(self):
        result = _FakeCallToolResult(text=json.dumps([1, 2, 3]))
        out = _unwrap_call_tool_result("test_tool", result)
        assert out == [1, 2, 3]

    def test_returns_plain_string_when_not_json(self):
        result = _FakeCallToolResult(text="plain text response")
        out = _unwrap_call_tool_result("test_tool", result)
        assert out == "plain text response"

    def test_prefers_structured_content_when_present(self):
        result = _FakeCallToolResult(
            text=json.dumps({"old": "ignore me"}),
            structured={"hello": "world"},
        )
        out = _unwrap_call_tool_result("test_tool", result)
        assert out == {"hello": "world"}

    def test_unwraps_result_wrapper_in_structured(self):
        """FastMCP often wraps non-dict tool returns as ``{'result': ...}``."""
        result = _FakeCallToolResult(structured={"result": [1, 2, 3]})
        out = _unwrap_call_tool_result("test_tool", result)
        assert out == [1, 2, 3]

    def test_returns_none_when_no_content(self):
        result = _FakeCallToolResult()
        out = _unwrap_call_tool_result("test_tool", result)
        assert out is None

    def test_raises_tool_error_when_result_is_error(self):
        result = _FakeCallToolResult(text="something failed", is_error=True)
        with pytest.raises(MCPToolError) as exc_info:
            _unwrap_call_tool_result("foo", result)
        assert "foo" in str(exc_info.value)
        assert "something failed" in str(exc_info.value)


# ─── CortexMCPClient with mocked transport ────────────────────────────────


@pytest.fixture
def fake_client_session():
    """Fixture that patches both _http_client_session and ClientSession.

    Returns a tuple ``(session_mock, set_call_tool_return)`` where the second
    element is a helper to set what ``call_tool`` returns.
    """
    session_mock = MagicMock()
    session_mock.initialize = AsyncMock()
    session_mock.call_tool = AsyncMock()
    session_mock.list_tools = AsyncMock()

    # _http_client_session returns an async context manager yielding
    # (read_stream, write_stream, get_session_id_callback)
    transport_cm = MagicMock()
    transport_cm.__aenter__ = AsyncMock(
        return_value=(MagicMock(), MagicMock(), MagicMock())
    )
    transport_cm.__aexit__ = AsyncMock(return_value=None)

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session_mock)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "cortex.transport.mcp.client._http_client_session",
        return_value=transport_cm,
    ), patch(
        "cortex.transport.mcp.client.ClientSession",
        return_value=session_cm,
    ):
        yield session_mock


class TestCortexMCPClientHappyPath:
    @pytest.mark.asyncio
    async def test_search_calls_cortex_search_tool(self, fake_client_session):
        fake_client_session.call_tool.return_value = _FakeCallToolResult(
            text=json.dumps([{"id": "1", "title": "Result"}])
        )
        client = CortexMCPClient("http://localhost:1314/mcp")
        result = await client.search("hello", limit=5)
        assert result == [{"id": "1", "title": "Result"}]
        fake_client_session.call_tool.assert_called_once()
        args = fake_client_session.call_tool.call_args
        assert args.args[0] == "cortex_search"
        assert args.kwargs["arguments"]["query"] == "hello"
        assert args.kwargs["arguments"]["limit"] == 5

    @pytest.mark.asyncio
    async def test_capture_passes_all_kwargs(self, fake_client_session):
        fake_client_session.call_tool.return_value = _FakeCallToolResult(
            text=json.dumps({"id": "abc", "status": "complete"})
        )
        client = CortexMCPClient("http://localhost:1314/mcp")
        result = await client.capture(
            title="T",
            content="C",
            obj_type="fix",
            project="proj",
            tags="a,b",
        )
        assert result == {"id": "abc", "status": "complete"}
        args = fake_client_session.call_tool.call_args
        assert args.args[0] == "cortex_capture"
        kwargs = args.kwargs["arguments"]
        assert kwargs["title"] == "T"
        assert kwargs["content"] == "C"
        assert kwargs["obj_type"] == "fix"
        assert kwargs["project"] == "proj"
        assert kwargs["tags"] == "a,b"

    @pytest.mark.asyncio
    async def test_read_returns_dict(self, fake_client_session):
        fake_client_session.call_tool.return_value = _FakeCallToolResult(
            text=json.dumps({"id": "x", "title": "Y"})
        )
        client = CortexMCPClient("http://localhost:1314/mcp")
        result = await client.read("x")
        assert result == {"id": "x", "title": "Y"}

    @pytest.mark.asyncio
    async def test_list_objects_returns_list(self, fake_client_session):
        fake_client_session.call_tool.return_value = _FakeCallToolResult(
            text=json.dumps([{"id": "1"}, {"id": "2"}])
        )
        client = CortexMCPClient("http://localhost:1314/mcp")
        result = await client.list_objects(limit=10)
        assert result == [{"id": "1"}, {"id": "2"}]
        args = fake_client_session.call_tool.call_args
        assert args.args[0] == "cortex_list"
        assert args.kwargs["arguments"]["limit"] == 10

    @pytest.mark.asyncio
    async def test_dossier_returns_dict(self, fake_client_session):
        fake_client_session.call_tool.return_value = _FakeCallToolResult(
            text=json.dumps(
                {"topic": "SQLite", "objects": [], "related_entities": []}
            )
        )
        client = CortexMCPClient("http://localhost:1314/mcp")
        result = await client.dossier("SQLite")
        assert result["topic"] == "SQLite"

    @pytest.mark.asyncio
    async def test_graph_data_returns_cytoscape_shape(self, fake_client_session):
        fake_client_session.call_tool.return_value = _FakeCallToolResult(
            text=json.dumps({"nodes": [], "edges": [], "total": 0})
        )
        client = CortexMCPClient("http://localhost:1314/mcp")
        result = await client.graph_data()
        assert "nodes" in result
        assert "edges" in result

    @pytest.mark.asyncio
    async def test_query_trail(self, fake_client_session):
        fake_client_session.call_tool.return_value = _FakeCallToolResult(
            text=json.dumps([{"tool": "cortex_search"}])
        )
        client = CortexMCPClient("http://localhost:1314/mcp")
        result = await client.query_trail(limit=20)
        assert result == [{"tool": "cortex_search"}]
        args = fake_client_session.call_tool.call_args
        assert args.kwargs["arguments"]["limit"] == 20

    @pytest.mark.asyncio
    async def test_status(self, fake_client_session):
        fake_client_session.call_tool.return_value = _FakeCallToolResult(
            text=json.dumps({"sqlite_total": 100, "graph_triples": 1000})
        )
        client = CortexMCPClient("http://localhost:1314/mcp")
        result = await client.status()
        assert result["sqlite_total"] == 100

    @pytest.mark.asyncio
    async def test_list_tools_returns_names(self, fake_client_session):
        fake_tool = MagicMock()
        fake_tool.name = "cortex_search"
        fake_tool2 = MagicMock()
        fake_tool2.name = "cortex_capture"
        fake_result = MagicMock()
        fake_result.tools = [fake_tool, fake_tool2]
        fake_client_session.list_tools.return_value = fake_result

        client = CortexMCPClient("http://localhost:1314/mcp")
        result = await client.list_tools()
        assert set(result) == {"cortex_search", "cortex_capture"}


class TestCortexMCPClientErrors:
    @pytest.mark.asyncio
    async def test_connection_refused_raises_mcp_connection_error(self):
        # Patch the transport to raise httpx.ConnectError on enter
        bad_cm = MagicMock()
        bad_cm.__aenter__ = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        bad_cm.__aexit__ = AsyncMock(return_value=None)
        with patch(
            "cortex.transport.mcp.client._http_client_session",
            return_value=bad_cm,
        ):
            client = CortexMCPClient("http://127.0.0.1:1/mcp")
            with pytest.raises(MCPConnectionError) as exc_info:
                await client.search("anything")
            assert "127.0.0.1:1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_raises_mcp_timeout_error(self):
        slow_cm = MagicMock()
        slow_cm.__aenter__ = AsyncMock(
            side_effect=httpx.ReadTimeout("read timed out")
        )
        slow_cm.__aexit__ = AsyncMock(return_value=None)
        with patch(
            "cortex.transport.mcp.client._http_client_session",
            return_value=slow_cm,
        ):
            client = CortexMCPClient("http://localhost:1314/mcp", timeout_seconds=0.5)
            with pytest.raises(MCPTimeoutError):
                await client.search("anything")

    @pytest.mark.asyncio
    async def test_unknown_transport_error_wrapped(self):
        bad_cm = MagicMock()
        bad_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("weird error"))
        bad_cm.__aexit__ = AsyncMock(return_value=None)
        with patch(
            "cortex.transport.mcp.client._http_client_session",
            return_value=bad_cm,
        ):
            client = CortexMCPClient("http://localhost:1314/mcp")
            with pytest.raises(MCPConnectionError) as exc_info:
                await client.search("anything")
            assert "weird error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tool_error_propagates_as_tool_error(self, fake_client_session):
        fake_client_session.call_tool.return_value = _FakeCallToolResult(
            text="tool failed: missing param", is_error=True
        )
        client = CortexMCPClient("http://localhost:1314/mcp")
        with pytest.raises(MCPToolError) as exc_info:
            await client.search("anything")
        assert "missing param" in str(exc_info.value)

    def test_all_error_classes_subclass_mcp_client_error(self):
        assert issubclass(MCPConnectionError, MCPClientError)
        assert issubclass(MCPTimeoutError, MCPClientError)
        assert issubclass(MCPServerError, MCPClientError)
        assert issubclass(MCPToolError, MCPClientError)

    @pytest.mark.asyncio
    async def test_invalid_tool_name_surfaces_as_tool_error(
        self, fake_client_session
    ):
        """Phase 2.C: when an MCP method calls a tool that doesn't exist on
        the server (renamed, removed, version mismatch), the server returns
        an error result and the client surfaces it as MCPToolError, not as
        a generic exception or silent None.
        """
        fake_client_session.call_tool.return_value = _FakeCallToolResult(
            text="Unknown tool: cortex_nonexistent",
            is_error=True,
        )
        client = CortexMCPClient("http://localhost:1314/mcp")
        with pytest.raises(MCPToolError) as exc_info:
            # Call any method — the underlying call_tool returns an error
            # result which _unwrap_call_tool_result surfaces as MCPToolError
            await client.search("anything")
        assert "Unknown tool" in str(exc_info.value)


# ─── Bundle 4.1: _http_client_session helper + deprecation guard ──────────


class TestHttpClientSession:
    """Bundle 4 Step 4.1: the _http_client_session helper bridges the
    deprecated ``streamablehttp_client(url, timeout=N)`` API to the canonical
    ``streamable_http_client(url, http_client=httpx.AsyncClient(...))`` API.
    """

    @pytest.mark.asyncio
    async def test_helper_yields_three_tuple_via_streamable_http_client(self):
        """Helper wraps streamable_http_client and yields its (r, w, sid) tuple."""
        fake_tuple = (MagicMock(), MagicMock(), MagicMock())
        inner_cm = MagicMock()
        inner_cm.__aenter__ = AsyncMock(return_value=fake_tuple)
        inner_cm.__aexit__ = AsyncMock(return_value=None)
        with patch(
            "cortex.transport.mcp.client.streamable_http_client",
            return_value=inner_cm,
        ) as mock_streamable:
            async with _http_client_session("http://test/mcp", 5.0) as yielded:
                assert yielded == fake_tuple
            # Verify streamable_http_client got called with http_client=<AsyncClient>
            assert mock_streamable.called
            _args, kwargs = mock_streamable.call_args
            assert "http_client" in kwargs
            assert isinstance(kwargs["http_client"], httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_helper_closes_httpx_client_on_exit(self):
        """The httpx.AsyncClient created by the helper must be closed when
        the context manager exits (no resource leaks).
        """
        inner_cm = MagicMock()
        inner_cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
        inner_cm.__aexit__ = AsyncMock(return_value=None)

        captured_client: list[httpx.AsyncClient] = []

        def capture(url, *, http_client, **kw):
            captured_client.append(http_client)
            return inner_cm

        with patch(
            "cortex.transport.mcp.client.streamable_http_client",
            side_effect=capture,
        ):
            async with _http_client_session("http://test/mcp", 2.0):
                pass
        assert len(captured_client) == 1
        # After exit, the client should be closed
        assert captured_client[0].is_closed

    @pytest.mark.asyncio
    async def test_helper_propagates_connection_error(self):
        """When the inner transport raises ConnectError, it bubbles up."""
        inner_cm = MagicMock()
        inner_cm.__aenter__ = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        inner_cm.__aexit__ = AsyncMock(return_value=None)
        with patch(
            "cortex.transport.mcp.client.streamable_http_client",
            return_value=inner_cm,
        ):
            with pytest.raises(httpx.ConnectError):
                async with _http_client_session("http://test/mcp", 5.0):
                    pass

    @pytest.mark.asyncio
    async def test_50_concurrent_helper_invocations_no_leak(self):
        """Stress: 50 helpers run concurrently without resource leaks."""
        import asyncio

        captured_clients: list[httpx.AsyncClient] = []

        def make_cm(url, *, http_client, **kw):
            captured_clients.append(http_client)
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with patch(
            "cortex.transport.mcp.client.streamable_http_client",
            side_effect=make_cm,
        ):
            async def one():
                async with _http_client_session("http://test/mcp", 5.0):
                    await asyncio.sleep(0)

            await asyncio.gather(*[one() for _ in range(50)])

        assert len(captured_clients) == 50
        # All httpx clients closed after their contexts exited
        assert all(c.is_closed for c in captured_clients)


class TestNoDeprecationWarnings:
    """Bundle 4.1 regression guard: the mcp.client module should no longer
    emit DeprecationWarning on import or use of CortexMCPClient.
    """

    def test_import_client_does_not_warn(self):
        import importlib
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import cortex.transport.mcp.client as _c
            importlib.reload(_c)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            # Filter to just warnings from our module
            ours = [
                x for x in dep_warnings
                if "streamablehttp_client" in str(x.message)
                or "streamable_http_client" in str(x.message)
            ]
            assert len(ours) == 0, (
                f"unexpected deprecation warnings from cortex.transport.mcp.client: {ours}"
            )

    def test_old_name_not_imported(self):
        """The deprecated streamablehttp_client name must not appear in our
        module globals — we use streamable_http_client exclusively.
        """
        import cortex.transport.mcp.client as mod
        assert not hasattr(mod, "streamablehttp_client"), (
            "cortex.transport.mcp.client should NOT re-export the deprecated name"
        )
        assert hasattr(mod, "streamable_http_client"), (
            "cortex.transport.mcp.client must import streamable_http_client"
        )
