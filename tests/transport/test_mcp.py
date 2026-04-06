"""Tests for cortex.transport.mcp.server — tool discovery and registration."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.core.config import CortexConfig
from cortex.transport.mcp.server import ADMIN_TOOLS, create_mcp_server

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
        assert len(_tool_names(mcp)) == 17

    def test_public_tools_count(self, config: CortexConfig):
        mcp = create_mcp_server(config, include_admin=False)
        assert len(_tool_names(mcp)) == 11

    def test_admin_exclusion_removes_exactly_six(self, config: CortexConfig):
        all_mcp = create_mcp_server(config, include_admin=True)
        pub_mcp = create_mcp_server(
            CortexConfig(data_dir=config.data_dir / "pub"),
            include_admin=False,
        )
        diff = _tool_names(all_mcp) - _tool_names(pub_mcp)
        assert len(diff) == 6
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
