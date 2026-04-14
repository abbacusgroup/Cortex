"""Tests for cortex.transport.mcp.server — tool discovery and registration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cortex.core.config import CortexConfig
from cortex.transport.mcp.server import (
    _LOCALHOST_HOSTS,
    ADMIN_TOOLS,
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
        "cortex_update",
        "cortex_unlink",
        "cortex_export",
        "cortex_safety_check",
        "cortex_reason",
        "cortex_query_trail",
        "cortex_graph_data",
        "cortex_list_entities",
        "cortex_debug_sessions",
        "cortex_debug_memory",
        "cortex_import",
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
        assert len(_tool_names(mcp)) == 25

    def test_public_tools_count(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=False)
        assert len(_tool_names(mcp)) == 11

    def test_admin_exclusion_removes_exactly_thirteen(self, config: CortexConfig):
        all_mcp = create_mcp_server(config, include_admin=True)
        pub_mcp = create_mcp_server(
            CortexConfig(data_dir=config.data_dir / "pub"),
            include_admin=False,
        )
        diff = _tool_names(all_mcp) - _tool_names(pub_mcp)
        assert len(diff) == 14
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
        assert frozenset({"127.0.0.1", "localhost", "::1"}) == _LOCALHOST_HOSTS


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

    def test_default_limit_is_50(self, config: CortexConfig):
        """Bundle 4 / A10: calling cortex_query_trail() with no arguments must
        use the default limit of 50. Seed 75 entries to distinguish "default"
        from "return everything".
        """
        from cortex.db.store import Store

        store = Store(config)
        for i in range(75):
            store.content.log_query(
                tool="cortex_search",
                params={"query": f"q{i}"},
                result_ids=[],
                duration_ms=1,
            )
        store.close()

        mcp = create_mcp_server(config, include_admin=True)
        result = _call_tool(mcp, "cortex_query_trail")  # no args
        assert len(result) == 50, f"default limit should be 50, got {len(result)} entries"

    def test_default_returns_all_when_fewer_than_50_logged(self, config: CortexConfig):
        """When < 50 entries exist, default should return all of them, not 0
        and not exactly 50.
        """
        from cortex.db.store import Store

        store = Store(config)
        for i in range(7):
            store.content.log_query(
                tool="cortex_search",
                params={"query": f"q{i}"},
                result_ids=[],
                duration_ms=1,
            )
        store.close()

        mcp = create_mcp_server(config, include_admin=True)
        result = _call_tool(mcp, "cortex_query_trail")
        assert len(result) == 7


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
            n["data"]["id"] for n in result["nodes"] if not n["data"]["id"].startswith("entity:")
        }
        assert id_a in obj_node_ids
        assert id_b not in obj_node_ids

    def test_invalid_doc_type_falls_back_to_knowledge_object(self, config: CortexConfig):
        """Bundle 4 / A11: unknown ``doc_type`` must not crash. The current
        code's else-branch (``graph_store.list_objects`` ~line 573) falls back
        to querying ``cortex:KnowledgeObject`` when the type isn't in
        ``CLASS_MAP``. This test pins that behavior.
        """
        mcp = create_mcp_server(config, include_admin=True)
        result = _call_tool(mcp, "cortex_graph_data", doc_type="not_a_real_type")
        # Does not crash, returns the cytoscape shape (possibly empty).
        assert "nodes" in result
        assert "edges" in result
        assert isinstance(result["nodes"], list)
        assert isinstance(result["edges"], list)

    def test_safe_project_names_do_not_crash(self, config: CortexConfig):
        """A11: baseline safety — simple project names containing only
        alphanumerics / dashes / underscores must not crash the tool.
        """
        mcp = create_mcp_server(config, include_admin=True)
        for safe in ["alpha", "project-1", "proj_2", "team.alpha", ""]:
            result = _call_tool(mcp, "cortex_graph_data", project=safe)
            assert "nodes" in result

    def test_hostile_project_names_do_not_crash(self, config: CortexConfig):
        """A11: hostile project names must NOT raise an unhandled exception
        out of the MCP tool.

        NOTE: ``graph_store.list_objects()`` currently interpolates the
        ``project`` parameter directly into a SPARQL string literal via
        ``f'?s cortex:project "{project}" .'`` (graph_store.py:577). In
        principle this is a SPARQL injection surface. In practice, pyoxigraph's
        SPARQL parser handles these malformed literals by returning zero
        matches (no crash, no data leakage). This test pins that observed
        behavior: hostile inputs MUST NOT crash the tool and MUST return an
        empty-or-unfiltered valid response.

        The interpolation should still be escaped properly in a future
        hardening commit; this test would catch a regression to a crashy
        parser variant.
        """
        mcp = create_mcp_server(config, include_admin=True)
        hostile_inputs = [
            'with"quote',  # SPARQL string terminator
            "with'apostrophe",  # alternate delimiter
            "with\\backslash",  # escape character
            "with\nnewline",  # whitespace
            '"; DROP ALL; "',  # injection attempt
        ]
        for hostile in hostile_inputs:
            result = _call_tool(mcp, "cortex_graph_data", project=hostile)
            assert "nodes" in result
            assert "edges" in result
            assert isinstance(result["nodes"], list)
            assert isinstance(result["edges"], list)


# -- A.2 diagnostic tools ---------------------------------------------------


# -- E.3 security probe: cortex_read with hostile obj_ids -------------------


class TestCortexReadPathTraversal:
    """E.3: cortex_read must safely handle hostile object IDs.

    The obj_id flows through parameterized SQLite queries and RDF API calls
    (not string interpolation), so injection is not possible at the data
    layer. These tests prove that contract and catch regressions if the
    implementation ever changes.
    """

    @pytest.mark.parametrize(
        "hostile_id",
        [
            "../../../etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "/etc/shadow",
            "....//....//etc/passwd",
            "obj_id\x00.txt",
            "'; DROP TABLE documents; --",
            '" OR 1=1 --',
            "SELECT * FROM documents",
            "<script>alert(1)</script>",
            "${7*7}",
            "{{7*7}}",
            "",
            " ",
            "a" * 10_000,
        ],
    )
    def test_hostile_obj_id_returns_not_found(self, config: CortexConfig, hostile_id: str):
        mcp = create_mcp_server(config, include_admin=True)
        result = _call_tool(mcp, "cortex_read", obj_id=hostile_id)
        # Should return "Not found: ..." string, never raise or leak data
        assert isinstance(result, str)
        assert "Not found" in result


class TestCortexDebugSessions:
    def test_returns_session_diagnostics(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=True)
        result = _call_tool(mcp, "cortex_debug_sessions")
        assert "session_count" in result
        assert "terminated_count" in result
        assert "active_count" in result
        assert "rss_mb" in result
        assert "pid" in result
        assert isinstance(result["session_count"], int)
        assert result["session_count"] >= 0

    def test_not_exposed_without_admin(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=False)
        assert "cortex_debug_sessions" not in mcp._tool_manager._tools


class TestCortexDebugMemory:
    def test_snapshot_without_start_returns_error(self, config: CortexConfig):
        import tracemalloc

        was_tracing = tracemalloc.is_tracing()
        if was_tracing:
            tracemalloc.stop()
        try:
            mcp = create_mcp_server(config, include_admin=True)
            result = _call_tool(mcp, "cortex_debug_memory", action="snapshot")
            assert "error" in result
        finally:
            if was_tracing:
                tracemalloc.start()

    def test_start_snapshot_stop_lifecycle(self, config: CortexConfig):
        import tracemalloc

        was_tracing = tracemalloc.is_tracing()
        if was_tracing:
            tracemalloc.stop()
        try:
            mcp = create_mcp_server(config, include_admin=True)

            # Start tracing
            result = _call_tool(mcp, "cortex_debug_memory", action="start")
            assert result["status"] == "tracing_started"

            # Baseline snapshot
            result = _call_tool(mcp, "cortex_debug_memory", action="snapshot")
            assert result["status"] == "snapshot_taken"
            assert result["mode"] == "baseline"
            assert "top_allocations" in result
            assert "rss_mb" in result

            # Diff snapshot
            result = _call_tool(mcp, "cortex_debug_memory", action="snapshot")
            assert result["status"] == "snapshot_taken"
            assert result["mode"] == "diff"

            # Stop tracing
            result = _call_tool(mcp, "cortex_debug_memory", action="stop")
            assert result["status"] == "tracing_stopped"
        finally:
            if was_tracing:
                tracemalloc.start()

    def test_unknown_action_returns_error(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=True)
        result = _call_tool(mcp, "cortex_debug_memory", action="bogus")
        assert "error" in result

    def test_not_exposed_without_admin(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=False)
        assert "cortex_debug_memory" not in mcp._tool_manager._tools
