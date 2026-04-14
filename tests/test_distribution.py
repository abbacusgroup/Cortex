"""Distribution validation tests — verify packaging artifacts exist and are correct.

No Docker builds; these are file-content checks only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


class TestDockerfile:
    def test_exists(self):
        assert (PROJECT_ROOT / "Dockerfile").exists()

    def test_base_image(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "python:3.13" in content

    def test_healthcheck(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "HEALTHCHECK" in content

    def test_exposes_port(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "EXPOSE 1314" in content


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------


class TestDockerCompose:
    def test_exists(self):
        assert (PROJECT_ROOT / "docker-compose.yml").exists()

    def test_cortex_service(self):
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "cortex" in content

    def test_volume_mapping(self):
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        assert "volumes:" in content


# ---------------------------------------------------------------------------
# pyproject.toml
# ---------------------------------------------------------------------------


class TestPyprojectToml:
    def test_cortex_entrypoint(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert 'cortex = "cortex.cli.main:app"' in content

    def test_required_dependencies(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text()
        for dep in ("typer", "fastapi", "pyoxigraph", "litellm"):
            assert dep in content, f"Missing dependency: {dep}"

    def test_embeddings_are_optional(self):
        content = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert 'embeddings = ["sentence-transformers' in content

    def test_version_consistency(self):
        """pyproject.toml version must match cortex.__version__."""
        import cortex

        toml_content = (PROJECT_ROOT / "pyproject.toml").read_text()
        # Extract version = "X.Y.Z" from pyproject.toml
        for line in toml_content.splitlines():
            if line.startswith("version = "):
                toml_version = line.split('"')[1]
                break
        else:
            pytest.fail("No version found in pyproject.toml")
        assert toml_version == cortex.__version__, (
            f"pyproject.toml has {toml_version!r} but cortex.__version__ is {cortex.__version__!r}"
        )


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


class TestReadme:
    def test_exists(self):
        assert (PROJECT_ROOT / "README.md").exists()

    def test_quick_start(self):
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "Quick Start" in content


# ---------------------------------------------------------------------------
# llms.txt
# ---------------------------------------------------------------------------


class TestLlmsTxt:
    def test_exists(self):
        assert (PROJECT_ROOT / "llms.txt").exists()

    def test_lists_all_tools(self):
        content = (PROJECT_ROOT / "llms.txt").read_text()
        tools = [
            "cortex_search",
            "cortex_context",
            "cortex_dossier",
            "cortex_read",
            "cortex_capture",
            "cortex_classify",
            "cortex_pipeline",
            "cortex_link",
            "cortex_feedback",
            "cortex_graph",
            "cortex_list",
            "cortex_status",
            "cortex_synthesize",
            "cortex_delete",
            "cortex_export",
            "cortex_safety_check",
            "cortex_reason",
            "cortex_list_entities",
            "cortex_query_trail",
            "cortex_graph_data",
            "cortex_debug_sessions",
            "cortex_debug_memory",
        ]
        for tool in tools:
            assert tool in content, f"Missing tool: {tool}"
        assert len(tools) == 22

    def test_knowledge_types(self):
        content = (PROJECT_ROOT / "llms.txt").read_text()
        for ktype in (
            "decision",
            "lesson",
            "fix",
            "session",
            "research",
            "source",
            "synthesis",
            "idea",
        ):
            assert ktype in content, f"Missing knowledge type: {ktype}"
