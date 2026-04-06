"""Distribution validation tests — verify packaging artifacts exist and are correct.

No Docker builds; these are file-content checks only.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


class TestDockerfile:
    def test_exists(self):
        assert (PROJECT_ROOT / "Dockerfile").exists()

    def test_base_image(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "python:3.12" in content

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
        ]
        for tool in tools:
            assert tool in content, f"Missing tool: {tool}"
        assert len(tools) == 17

    def test_knowledge_types(self):
        content = (PROJECT_ROOT / "llms.txt").read_text()
        for ktype in (
            "decision", "lesson", "fix", "session",
            "research", "source", "synthesis", "idea",
        ):
            assert ktype in content, f"Missing knowledge type: {ktype}"
