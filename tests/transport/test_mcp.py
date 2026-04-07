"""Tests for cortex.transport.mcp.server — tool discovery and registration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cortex.core.config import CortexConfig
from cortex.transport.mcp.server import (
    ADMIN_TOOLS,
    _LOCALHOST_HOSTS,
    create_mcp_server,
    run_http,
    run_stdio,
)

EXPECTED_PUBLIC_TOOLS = frozenset(
    {
        "cortex_search",
        "cortex_context",
        "cortex_dossier",
        "cortex_read",
        "cortex_capture",
        "cortex_link",
        "cortex_feedback",
        "cortex_graph",
        "cortex_list",
        "cortex_pipeline",
        "cortex_classify",
    }
)

EXPECTED_ADMIN_TOOLS = frozenset(
    {
        "cortex_status",
        "cortex_synthesize",
        "cortex_delete",
        "cortex_export",
        "cortex_safety_check",
        "cortex_reason",
        "cortex_query_trail",
        "cortex_graph_data",
        "cortex_list_entities",
    }
)

ALL_EXPECTED_TOOLS = EXPECTED_PUBLIC_TOOLS | EXPECTED_ADMIN_TOOLS


def _tool_names(mcp) -> set[str]:
    """Extract registered tool names from a FastMCP instance."""
    return set(mcp._tool_manager._tools.keys())


@pytest.fixture()
def config(tmp_path: Path) -> CortexConfig:
    return CortexConfig(data_dir=tmp_path)


# -- Tool counts -------------------------------------------------------------


class TestToolCounts:
    """Verify the correct number of tools are registered."""

    def test_all_tools_count(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=True)
        assert len(_tool_names(mcp)) == 20

    def test_public_tools_count(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=False)
        assert len(_tool_names(mcp)) == 11

    def test_admin_exclusion_removes_exactly_nine(self, config: CortexConfig):
        all_mcp = create_mcp_server(config, include_admin=True)
        pub_mcp = create_mcp_server(
            CortexConfig(data_dir=config.data_dir / "pub"),
            include_admin=False,
        )
        diff = _tool_names(all_mcp) - _tool_names(pub_mcp)
        assert len(diff) == 9
        assert diff == EXPECTED_ADMIN_TOOLS


# -- Tool naming convention ---------------------------------------------------


class TestToolNaming:
    """All tool names must follow the cortex_ prefix convention."""

    def test_all_tools_have_cortex_prefix(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=True)
        for name in _tool_names(mcp):
            assert name.startswith("cortex_"), f"Tool {name!r} missing cortex_ prefix"

    def test_public_tools_have_cortex_prefix(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=False)
        for name in _tool_names(mcp):
            assert name.startswith("cortex_"), f"Tool {name!r} missing cortex_ prefix"


# -- Public tool registration -------------------------------------------------


class TestPublicToolRegistration:
    """Public tools must always be present regardless of admin mode."""

    def test_public_tools_present_in_admin_mode(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=True)
        names = _tool_names(mcp)
        assert EXPECTED_PUBLIC_TOOLS.issubset(names)

    def test_public_tools_present_in_public_mode(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=False)
        names = _tool_names(mcp)
        assert EXPECTED_PUBLIC_TOOLS.issubset(names)

    @pytest.mark.parametrize("tool_name", sorted(EXPECTED_PUBLIC_TOOLS))
    def test_each_public_tool_registered(self, config: CortexConfig, tool_name: str):
        mcp = create_mcp_server(config, include_admin=False)
        assert tool_name in _tool_names(mcp)


# -- Admin tool registration --------------------------------------------------


class TestAdminToolRegistration:
    """Admin tools should only appear when include_admin=True."""

    def test_admin_tools_present_in_admin_mode(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=True)
        names = _tool_names(mcp)
        assert EXPECTED_ADMIN_TOOLS.issubset(names)

    def test_admin_tools_absent_in_public_mode(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=False)
        names = _tool_names(mcp)
        assert names.isdisjoint(EXPECTED_ADMIN_TOOLS)

    @pytest.mark.parametrize("tool_name", sorted(EXPECTED_ADMIN_TOOLS))
    def test_each_admin_tool_excluded_in_public_mode(self, config: CortexConfig, tool_name: str):
        mcp = create_mcp_server(config, include_admin=False)
        assert tool_name not in _tool_names(mcp)

    @pytest.mark.parametrize("tool_name", sorted(EXPECTED_ADMIN_TOOLS))
    def test_each_admin_tool_included_in_admin_mode(self, config: CortexConfig, tool_name: str):
        mcp = create_mcp_server(config, include_admin=True)
        assert tool_name in _tool_names(mcp)


# -- ADMIN_TOOLS constant consistency ----------------------------------------


class TestAdminToolsConstant:
    """The ADMIN_TOOLS constant in server.py must match expectations."""

    def test_admin_tools_matches_expected(self):
        assert ADMIN_TOOLS == EXPECTED_ADMIN_TOOLS

    def test_admin_tools_is_frozenset(self):
        assert isinstance(ADMIN_TOOLS, frozenset)

    def test_no_overlap_between_public_and_admin(self):
        assert EXPECTED_PUBLIC_TOOLS.isdisjoint(EXPECTED_ADMIN_TOOLS)


# -- Complete tool set --------------------------------------------------------


class TestCompleteToolSet:
    """The full set of registered tools matches the expected roster."""

    def test_no_unexpected_tools_in_admin_mode(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=True)
        names = _tool_names(mcp)
        unexpected = names - ALL_EXPECTED_TOOLS
        assert unexpected == set(), f"Unexpected tools registered: {unexpected}"

    def test_no_unexpected_tools_in_public_mode(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=False)
        names = _tool_names(mcp)
        unexpected = names - EXPECTED_PUBLIC_TOOLS
        assert unexpected == set(), f"Unexpected tools registered: {unexpected}"

    def test_all_tools_exactly_match(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=True)
        assert _tool_names(mcp) == ALL_EXPECTED_TOOLS


# -- Server metadata ---------------------------------------------------------


class TestServerMetadata:
    """The FastMCP instance is configured with correct metadata."""

    def test_server_name(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=True)
        assert mcp.name == "cortex"


# -- run_http() host-aware admin gating (Phase 2.B) --------------------------


class TestRunHttpAdminGating:
    """run_http() includes admin tools only when bound to a loopback address.

    The actual ``mcp.run`` call is mocked because it would otherwise block
    indefinitely waiting for HTTP connections.
    """

    @pytest.mark.parametrize("local_host", ["127.0.0.1", "localhost", "::1"])
    def test_localhost_includes_admin_tools(self, local_host, monkeypatch):
        captured: dict[str, object] = {}

        def fake_create_mcp_server(config=None, *, include_admin: bool = True):
            captured["include_admin"] = include_admin
            mock = MagicMock()
            mock.run = MagicMock()
            return mock

        monkeypatch.setattr(
            "cortex.transport.mcp.server.create_mcp_server",
            fake_create_mcp_server,
        )
        run_http(host=local_host, port=1234)
        assert captured["include_admin"] is True

    @pytest.mark.parametrize("non_local_host", ["0.0.0.0", "192.168.1.5", "10.0.0.1"])
    def test_non_localhost_excludes_admin_tools(self, non_local_host, monkeypatch):
        captured: dict[str, object] = {}

        def fake_create_mcp_server(config=None, *, include_admin: bool = True):
            captured["include_admin"] = include_admin
            mock = MagicMock()
            mock.run = MagicMock()
            return mock

        monkeypatch.setattr(
            "cortex.transport.mcp.server.create_mcp_server",
            fake_create_mcp_server,
        )
        run_http(host=non_local_host, port=1234)
        assert captured["include_admin"] is False

    def test_run_http_passes_streamable_http_transport(self, monkeypatch):
        mock_mcp = MagicMock()
        mock_mcp.settings = MagicMock()

        def fake_create_mcp_server(config=None, *, include_admin: bool = True):
            return mock_mcp

        monkeypatch.setattr(
            "cortex.transport.mcp.server.create_mcp_server",
            fake_create_mcp_server,
        )
        run_http(host="127.0.0.1", port=4242)
        mock_mcp.run.assert_called_once_with(transport="streamable-http")
        # host/port should be set on settings, not passed to run()
        assert mock_mcp.settings.host == "127.0.0.1"
        assert mock_mcp.settings.port == 4242

    def test_run_stdio_unchanged_includes_admin(self, monkeypatch):
        """Regression: stdio mode still includes admin tools."""
        captured: dict[str, object] = {}

        def fake_create_mcp_server(config=None, *, include_admin: bool = True):
            captured["include_admin"] = include_admin
            mock = MagicMock()
            return mock

        monkeypatch.setattr(
            "cortex.transport.mcp.server.create_mcp_server",
            fake_create_mcp_server,
        )
        run_stdio()
        assert captured["include_admin"] is True

    def test_localhost_set_contents(self):
        """Belt-and-suspenders: the localhost set has exactly the expected entries."""
        assert _LOCALHOST_HOSTS == frozenset({"127.0.0.1", "localhost", "::1"})


# -- New aggregation tools (Phase 2.E) ----------------------------------------


def _call_tool(mcp, name: str, **kwargs):
    """Invoke a registered tool by name and return its result.

    FastMCP wraps tools in a manager; the underlying Python function lives at
    ``mcp._tool_manager._tools[name].fn``.
    """
    tool = mcp._tool_manager._tools[name]
    return tool.fn(**kwargs)


class TestCortexQueryTrail:
    def test_returns_empty_list_when_no_logs(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=True)
        result = _call_tool(mcp, "cortex_query_trail")
        assert result == []

    def test_returns_logged_queries(self, config: CortexConfig):
        from cortex.db.store import Store

        # Populate via a local store, then close it BEFORE creating the MCP
        # server (which would otherwise hit the lock — Phase 1's honest mode).
        store = Store(config)
        store.content.log_query(
            tool="cortex_search",
            params={"query": "test query"},
            result_ids=["1", "2"],
            duration_ms=10,
        )
        store.close()

        mcp = create_mcp_server(config, include_admin=True)
        result = _call_tool(mcp, "cortex_query_trail", limit=10)
        assert isinstance(result, list)
        assert any(r.get("tool") == "cortex_search" for r in result)

    def test_negative_limit_returns_empty(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=True)
        assert _call_tool(mcp, "cortex_query_trail", limit=-1) == []
        assert _call_tool(mcp, "cortex_query_trail", limit=0) == []

    def test_huge_limit_is_capped(self, config: CortexConfig, monkeypatch):
        """Limits over QUERY_TRAIL_MAX_LIMIT (1000) are capped, not passed through."""
        import cortex.transport.mcp.server as mcp_mod

        captured_limits: list[int] = []

        def fake_get_query_log(self, *, limit: int):
            captured_limits.append(limit)
            return []

        monkeypatch.setattr(
            "cortex.db.content_store.ContentStore.get_query_log",
            fake_get_query_log,
        )
        mcp = create_mcp_server(config, include_admin=True)
        _call_tool(mcp, "cortex_query_trail", limit=999_999)
        assert captured_limits, "get_query_log should have been called"
        assert max(captured_limits) <= mcp_mod.QUERY_TRAIL_MAX_LIMIT

    def test_query_trail_only_in_admin_mode(self, config: CortexConfig):
        pub_mcp = create_mcp_server(config, include_admin=False)
        assert "cortex_query_trail" not in _tool_names(pub_mcp)


class TestCortexGraphData:
    def test_empty_store_returns_empty_shape(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=True)
        result = _call_tool(mcp, "cortex_graph_data")
        assert result["nodes"] == []
        assert result["edges"] == []
        assert "total" in result

    def test_returns_cytoscape_shape_with_objects(self, config: CortexConfig):
        from cortex.db.store import Store

        # Populate via local store, then close it before creating MCP server.
        store = Store(config)
        store.initialize()
        obj_id = store.create(
            obj_type="fix",
            title="Test fix",
            content="content",
            project="testproj",
        )
        store.close()

        mcp = create_mcp_server(config, include_admin=True)
        result = _call_tool(mcp, "cortex_graph_data")
        assert isinstance(result["nodes"], list)
        assert isinstance(result["edges"], list)
        node_ids = [n["data"]["id"] for n in result["nodes"]]
        assert obj_id in node_ids
        for n in result["nodes"]:
            assert "id" in n["data"]
            assert "label" in n["data"]
            assert "type" in n["data"]

    def test_huge_limit_is_capped(self, config: CortexConfig):
        import cortex.transport.mcp.server as mcp_mod

        mcp = create_mcp_server(config, include_admin=True)
        result = _call_tool(mcp, "cortex_graph_data", limit=999_999)
        assert result["limit"] <= mcp_mod.GRAPH_DATA_MAX_OBJECTS

    def test_negative_limit_handled(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=True)
        result = _call_tool(mcp, "cortex_graph_data", limit=-5)
        # No crash, no negative limit
        assert result["limit"] == 0
        assert result["nodes"] == []

    def test_negative_offset_handled(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=True)
        result = _call_tool(mcp, "cortex_graph_data", offset=-100)
        assert result["offset"] == 0

    def test_graph_data_only_in_admin_mode(self, config: CortexConfig):
        pub_mcp = create_mcp_server(config, include_admin=False)
        assert "cortex_graph_data" not in _tool_names(pub_mcp)

    def test_filter_by_project(self, config: CortexConfig):
        from cortex.db.store import Store

        store = Store(config)
        store.initialize()
        id_a = store.create(obj_type="fix", title="A", project="alpha")
        id_b = store.create(obj_type="fix", title="B", project="beta")
        store.close()

        mcp = create_mcp_server(config, include_admin=True)
        result = _call_tool(mcp, "cortex_graph_data", project="alpha")
        obj_node_ids = {
            n["data"]["id"]
            for n in result["nodes"]
            if not n["data"]["id"].startswith("entity:")
        }
        assert id_a in obj_node_ids
        assert id_b not in obj_node_ids
