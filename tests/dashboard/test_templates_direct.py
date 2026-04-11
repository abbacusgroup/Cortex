"""Direct Jinja2 render tests for dashboard templates (Bundle 10.2).

These tests close Weak Point #6 from the original A+D plan: every
production data path through the dashboard passes **dicts** (returned
by ``CortexMCPClient`` tool calls). Jinja2's ``{{ obj.foo }}`` notation
tries attribute access first and falls back to ``__getitem__``, so
dicts usually work transparently — but a dict key that shadows a
``dict`` method (``items``, ``keys``, ``values``) would silently
return the method object instead of the data, and a typo in a field
name in the data or template would only be caught when the affected
page is actually rendered under specific conditions.

``tests/dashboard/test_dashboard.py`` exercises templates indirectly
via ``TestClient`` + ``FakeMCPClient``, but only asserts status 200.
These tests instead render each template directly against a
hand-built dict fixture that mirrors the MCP client return shape and
assert that specific field values appear in the output. That catches
both the method-shadowing trap and any silent field-name drift.

``settings.html`` is deliberately rendered with a real
:class:`cortex.core.config.CortexConfig` to keep the attribute-access
path covered too — the dashboard passes the config object straight
through (see ``settings_page`` in ``cortex/dashboard/server.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape

from cortex.core.config import CortexConfig
from cortex.dashboard.server import DASHBOARD_DIR

TEMPLATES_DIR: Path = DASHBOARD_DIR / "templates"


@pytest.fixture(scope="module")
def env() -> Environment:
    """A bare Jinja2 ``Environment`` pointed at the dashboard templates.

    We deliberately do NOT use FastAPI's ``Jinja2Templates`` wrapper
    because it demands a ``Request`` object. Rendering templates in
    isolation is the whole point — this is the shape the production
    code path reaches when a page actually renders.
    """
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )


# ==========================================================================
# home.html — stats + alerts + recent dicts
# ==========================================================================


class TestHomeTemplate:
    def test_renders_with_dict_stats_and_recent(self, env: Environment) -> None:
        html = env.get_template("home.html").render(
            stats={
                "sqlite_total": 172,
                "graph_triples": 10572,
                "entities": 636,
            },
            alerts=[
                {
                    "severity": "high",
                    "type": "contradiction",
                    "message": "Two fixes disagree",
                },
            ],
            recent=[
                {
                    "id": "obj-1",
                    "type": "decision",
                    "title": "Use Oxigraph",
                    "project": "cortex",
                    "created_at": "2026-04-08T12:34:56+00:00",
                },
            ],
        )
        assert "172" in html
        assert "10572" in html
        assert "636" in html
        assert "Contradiction" in html
        assert "Two fixes disagree" in html
        assert "Use Oxigraph" in html
        assert "/documents/obj-1" in html
        # Created-at slice: first 10 chars of ISO datetime
        assert "2026-04-08" in html

    def test_renders_with_empty_alerts_and_recent(self, env: Environment) -> None:
        html = env.get_template("home.html").render(
            stats={"sqlite_total": 0, "graph_triples": 0, "entities": 0},
            alerts=[],
            recent=[],
        )
        # No "Active Alerts" heading when alerts is empty
        assert "Active Alerts" not in html
        # Recent table still renders (just empty rows)
        assert "Recent Activity" in html

    def test_handles_missing_created_at(self, env: Environment) -> None:
        html = env.get_template("home.html").render(
            stats={"sqlite_total": 1, "graph_triples": 1, "entities": 1},
            alerts=[],
            recent=[
                {
                    "id": "obj-x",
                    "type": "idea",
                    "title": "No timestamp",
                    "project": "",
                    "created_at": None,
                },
            ],
        )
        assert "No timestamp" in html  # must not crash on None slice


# ==========================================================================
# documents.html — list of dicts + form state
# ==========================================================================


class TestDocumentsTemplate:
    def test_renders_with_dict_documents(self, env: Environment) -> None:
        docs = [
            {
                "id": "abc-123",
                "type": "lesson",
                "title": "Prefer dicts at boundaries",
                "project": "cortex",
                "tags": "api,design",
            },
            {
                "id": "def-456",
                "type": "fix",
                "title": "Escape SPARQL literals",
                "project": "cortex",
                "tags": "security",
            },
        ]
        html = env.get_template("documents.html").render(
            documents=docs, query="", doc_type="", project=""
        )
        assert "Prefer dicts at boundaries" in html
        assert "Escape SPARQL literals" in html
        assert "/documents/abc-123" in html
        assert "/documents/def-456" in html
        assert "2 result(s)" in html

    def test_renders_form_state(self, env: Environment) -> None:
        html = env.get_template("documents.html").render(
            documents=[],
            query="hello",
            doc_type="fix",
            project="cortex",
        )
        # Query input value reflects current search
        assert 'value="hello"' in html
        # Project input value reflects filter
        assert 'value="cortex"' in html
        # selected option reflects doc_type
        assert 'value="fix" selected' in html
        assert "0 result(s)" in html

    def test_renders_empty_state(self, env: Environment) -> None:
        html = env.get_template("documents.html").render(
            documents=[], query="", doc_type="", project=""
        )
        assert "No documents found." in html


# ==========================================================================
# detail.html — single doc dict + relationships
# ==========================================================================


class TestDetailTemplate:
    def test_renders_with_dict_doc(self, env: Environment) -> None:
        doc = {
            "id": "obj-1",
            "type": "decision",
            "title": "Adopt HTTP MCP transport",
            "project": "cortex",
            "tags": "mcp,transport",
            "tier": "canonical",
            "confidence": 0.95,
            "created_at": "2026-04-08T12:34:56+00:00",
            "content": "We chose streamable-http over stdio because...",
            "relationships": [
                {
                    "direction": "outgoing",
                    "rel_type": "supersedes",
                    "other_id": "obj-0000-prev",
                },
            ],
        }
        html = env.get_template("detail.html").render(doc=doc)
        assert "Adopt HTTP MCP transport" in html
        assert "badge-decision" in html
        assert "obj-1" in html
        assert "cortex" in html
        assert "mcp,transport" in html
        assert "canonical" in html
        assert "0.95" in html
        # created_at sliced to the first 19 chars
        assert "2026-04-08T12:34:56" in html
        assert "streamable-http over stdio" in html
        assert "supersedes" in html
        assert "/documents/obj-0000-prev" in html

    def test_renders_with_no_content_or_relationships(self, env: Environment) -> None:
        doc = {
            "id": "obj-2",
            "type": "idea",
            "title": "Half-empty doc",
            "project": None,
            "tags": None,
            "tier": None,
            "confidence": 0.1,
            "created_at": None,
            "content": "",
            "relationships": [],
        }
        html = env.get_template("detail.html").render(doc=doc)
        assert "Half-empty doc" in html
        # em-dash fallback for None fields
        assert "—" in html
        # No Content card
        assert "Content</div>" not in html
        # No Relationships card
        assert "Relationships</div>" not in html


# ==========================================================================
# entities.html — list of dicts
# ==========================================================================


class TestEntitiesTemplate:
    def test_renders_with_dict_entities(self, env: Environment) -> None:
        ents = [
            {"id": "ent-1234567890abcdef", "name": "Oxigraph", "type": "technology"},
            {"id": "ent-fedcba0987654321", "name": "Cortex v2", "type": "project"},
        ]
        html = env.get_template("entities.html").render(entities=ents, entity_type="")
        assert "Oxigraph" in html
        assert "Cortex v2" in html
        assert "technology" in html
        # ID is sliced to [:12]
        assert "ent-12345678" in html
        assert "2 entity(ies)" in html

    def test_renders_empty_state(self, env: Environment) -> None:
        html = env.get_template("entities.html").render(entities=[], entity_type="")
        assert "No entities found." in html


# ==========================================================================
# trail.html — query log list of dicts
# ==========================================================================


class TestTrailTemplate:
    def test_renders_with_dict_logs(self, env: Environment) -> None:
        logs = [
            {
                "timestamp": "2026-04-08T12:00:00+00:00",
                "tool": "cortex_search",
                "params": '{"query":"oxigraph"}',
                "result_count": 7,
                "duration_ms": 123.4,
            },
        ]
        html = env.get_template("trail.html").render(logs=logs)
        # Slice [:19] on ISO timestamp
        assert "2026-04-08T12:00:00" in html
        assert "cortex_search" in html
        assert "7" in html
        assert "123.4ms" in html
        assert "Recent queries (1)" in html

    def test_renders_empty_state(self, env: Environment) -> None:
        html = env.get_template("trail.html").render(logs=[])
        assert "No queries yet." in html


# ==========================================================================
# settings.html — real CortexConfig object (attribute access path)
# ==========================================================================


class TestSettingsTemplate:
    def test_renders_with_real_config_object(self, env: Environment, tmp_path: Path) -> None:
        config = CortexConfig(
            data_dir=tmp_path,
            host="127.0.0.1",
            port=1314,
            llm_provider="anthropic",
            llm_model="claude-opus-4-6",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            log_level="INFO",
        )
        html = env.get_template("settings.html").render(
            config=config, weights={"recency": 0.3, "similarity": 0.7}
        )
        assert "127.0.0.1" in html
        assert "1314" in html
        assert "anthropic" in html
        assert "claude-opus-4-6" in html
        assert "all-MiniLM-L6-v2" in html
        assert "INFO" in html
        assert "Recency" in html
        assert "0.3" in html
        assert "Similarity" in html
        assert "0.7" in html

    def test_renders_with_empty_weights(self, env: Environment, tmp_path: Path) -> None:
        config = CortexConfig(data_dir=tmp_path)
        html = env.get_template("settings.html").render(config=config, weights={})
        # Must not crash when weights is empty
        assert "Retrieval Weights" in html
        assert "Configuration" in html


# ==========================================================================
# error.html — exception-handler context
# ==========================================================================


class TestErrorTemplate:
    def test_renders_connection_error_context(self, env: Environment) -> None:
        html = env.get_template("error.html").render(
            status_code=503,
            error_code="mcp_connection_error",
            error_message="Cannot reach http://127.0.0.1:1314/mcp",
        )
        assert "503" in html
        assert "mcp_connection_error" in html
        assert "Cannot reach http://127.0.0.1:1314/mcp" in html
        assert "cortex serve --transport mcp-http" in html


# ==========================================================================
# login.html — form + optional error banner
# ==========================================================================


class TestLoginTemplate:
    def test_renders_without_error(self, env: Environment) -> None:
        html = env.get_template("login.html").render(error=None)
        assert "Sign In" in html
        assert "alert-danger" not in html

    def test_renders_with_error(self, env: Environment) -> None:
        html = env.get_template("login.html").render(error="Invalid password")
        assert "Invalid password" in html
        assert "alert-danger" in html


# ==========================================================================
# create.html + graph.html — static templates, sanity check they compile
# ==========================================================================


class TestStaticTemplates:
    def test_create_template_compiles(self, env: Environment) -> None:
        html = env.get_template("create.html").render()
        assert "Capture" in html
        assert 'name="title"' in html
        assert 'name="content"' in html
        assert 'name="project"' in html

    def test_graph_template_compiles(self, env: Environment) -> None:
        html = env.get_template("graph.html").render()
        assert "Knowledge Graph" in html
        # Cytoscape mount point
        assert 'id="cy"' in html
