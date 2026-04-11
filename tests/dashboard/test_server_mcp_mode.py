"""Tests for the dashboard's MCP-client refactor (Phase 2.D).

These tests verify the contracts that distinguish the new MCP-client dashboard
from the old direct-store dashboard:

1. The dashboard process never opens ``graph.db`` directly.
2. Every endpoint flows through the injected MCP client.
3. MCP errors map to friendly HTTP status codes (503/504/502).
4. Auth still wraps every authenticated endpoint.
"""

from __future__ import annotations

import bcrypt
import pytest
from starlette.testclient import TestClient

from cortex.core.config import CortexConfig
from cortex.dashboard.server import _sessions, create_dashboard
from cortex.transport.mcp.client import (
    MCPConnectionError,
    MCPTimeoutError,
    MCPToolError,
)


class CallRecorder:
    """A fake MCP client that just records the methods invoked on it.

    Returns minimal-but-valid shapes for each method so templates render.
    """

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def _record(self, name: str, **kwargs):
        self.calls.append((name, kwargs))

    async def search(self, query, doc_type="", project="", limit=20):
        self._record("search", query=query, doc_type=doc_type, project=project, limit=limit)
        return []

    async def list_objects(self, doc_type="", project="", limit=50):
        self._record("list_objects", doc_type=doc_type, project=project, limit=limit)
        return []

    async def list_entities(self, entity_type=""):
        self._record("list_entities", entity_type=entity_type)
        return []

    async def read(self, obj_id):
        self._record("read", obj_id=obj_id)
        return {"id": obj_id, "title": "X", "type": "fix", "content": ""}

    async def capture(self, title, content="", obj_type="idea", project="", tags=""):
        self._record(
            "capture",
            title=title,
            content=content,
            obj_type=obj_type,
            project=project,
            tags=tags,
        )
        return {"id": "fake-id"}

    async def status(self):
        self._record("status")
        return {
            "sqlite_total": 0,
            "graph_triples": 0,
            "counts_by_type": {},
            "graph_counts_by_type": {},
            "entities": 0,
            "alerts": [],
            "alert_count": 0,
        }

    async def query_trail(self, limit=50):
        self._record("query_trail", limit=limit)
        return []

    async def graph_data(self, project="", doc_type="", limit=500, offset=0):
        self._record("graph_data", project=project, doc_type=doc_type, limit=limit, offset=offset)
        return {"nodes": [], "edges": [], "total": 0}

    async def dossier(self, topic):
        self._record("dossier", topic=topic)
        return {"topic": topic, "objects": [], "related_entities": []}

    async def context(self, topic, limit=10):
        self._record("context", topic=topic, limit=limit)
        return []

    async def graph(self, obj_id="", entity=""):
        self._record("graph", obj_id=obj_id, entity=entity)
        return {}

    async def list_tools(self):
        return []


@pytest.fixture
def call_recorder() -> CallRecorder:
    return CallRecorder()


@pytest.fixture
def app_with_recorder(tmp_path, call_recorder):
    _sessions.clear()
    config = CortexConfig(data_dir=tmp_path)
    return create_dashboard(config, mcp_client=call_recorder)


class TestEndpointsRouteThroughMcpClient:
    """Each dashboard endpoint must call the corresponding MCP method."""

    def test_documents_calls_list_objects(self, app_with_recorder, call_recorder):
        client = TestClient(app_with_recorder)
        client.get("/documents")
        names = [c[0] for c in call_recorder.calls]
        assert "list_objects" in names

    def test_documents_with_query_calls_search(self, app_with_recorder, call_recorder):
        client = TestClient(app_with_recorder)
        client.get("/documents?q=needle")
        names = [c[0] for c in call_recorder.calls]
        assert "search" in names
        # And search received the query
        search_call = next(c for c in call_recorder.calls if c[0] == "search")
        assert search_call[1]["query"] == "needle"

    def test_document_detail_calls_read(self, app_with_recorder, call_recorder):
        client = TestClient(app_with_recorder)
        client.get("/documents/abc123")
        names = [c[0] for c in call_recorder.calls]
        assert "read" in names
        read_call = next(c for c in call_recorder.calls if c[0] == "read")
        assert read_call[1]["obj_id"] == "abc123"

    def test_post_create_calls_capture(self, app_with_recorder, call_recorder):
        client = TestClient(app_with_recorder)
        client.post(
            "/create",
            data={
                "title": "T",
                "content": "C",
                "obj_type": "fix",
                "project": "p",
                "tags": "a,b",
            },
            follow_redirects=False,
        )
        names = [c[0] for c in call_recorder.calls]
        assert "capture" in names
        capture_call = next(c for c in call_recorder.calls if c[0] == "capture")
        assert capture_call[1]["title"] == "T"
        assert capture_call[1]["obj_type"] == "fix"
        assert capture_call[1]["project"] == "p"
        assert capture_call[1]["tags"] == "a,b"

    def test_entities_page_calls_list_entities(self, app_with_recorder, call_recorder):
        client = TestClient(app_with_recorder)
        client.get("/entities")
        names = [c[0] for c in call_recorder.calls]
        assert "list_entities" in names

    def test_trail_calls_query_trail(self, app_with_recorder, call_recorder):
        client = TestClient(app_with_recorder)
        client.get("/trail")
        names = [c[0] for c in call_recorder.calls]
        assert "query_trail" in names

    def test_api_graph_data_calls_graph_data(self, app_with_recorder, call_recorder):
        client = TestClient(app_with_recorder)
        client.get("/api/graph-data")
        names = [c[0] for c in call_recorder.calls]
        assert "graph_data" in names

    def test_api_dossier_calls_dossier(self, app_with_recorder, call_recorder):
        client = TestClient(app_with_recorder)
        client.get("/api/dossier/SQLite")
        names = [c[0] for c in call_recorder.calls]
        assert "dossier" in names
        call = next(c for c in call_recorder.calls if c[0] == "dossier")
        assert call[1]["topic"] == "SQLite"

    def test_home_calls_status_and_list(self, app_with_recorder, call_recorder):
        client = TestClient(app_with_recorder)
        client.get("/")
        names = [c[0] for c in call_recorder.calls]
        assert "status" in names
        assert "list_objects" in names


class TestDashboardDoesNotImportStore:
    """Strong regression guard: dashboard/server.py must not import the Store layer."""

    def test_server_module_does_not_import_store(self):
        # Inspect the module's source for forbidden imports
        import inspect

        import cortex.dashboard.server as server_mod

        src = inspect.getsource(server_mod)
        # The dashboard should NOT import any of these
        forbidden = [
            "from cortex.db.store import Store",
            "from cortex.pipeline.orchestrator import PipelineOrchestrator",
            "from cortex.retrieval.engine import RetrievalEngine",
            "from cortex.retrieval.learner import LearningLoop",
            "from cortex.retrieval.presenters import",
            "from cortex.services.llm import LLMClient",
        ]
        for line in forbidden:
            assert line not in src, f"Dashboard must not import {line!r} after Phase 2.D"


class TestMcpErrorHandlers:
    """MCP errors map to user-friendly HTTP responses."""

    def _make_failing_client(self, exc_type, exc_args=None):
        """Build a fake client where every method raises the given error."""
        from cortex.transport.mcp.client import MCPClientError  # noqa: F401

        class FailingClient:
            async def search(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def list_objects(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def list_entities(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def read(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def capture(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def status(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def query_trail(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def graph_data(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def dossier(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def context(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def graph(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def list_tools(self):
                raise exc_type(*(exc_args or ()))

        return FailingClient()

    def test_connection_error_returns_503_html(self, tmp_path):
        _sessions.clear()
        client = self._make_failing_client(MCPConnectionError, ("MCP unreachable",))
        config = CortexConfig(data_dir=tmp_path)
        app = create_dashboard(config, mcp_client=client)
        tc = TestClient(app)
        resp = tc.get("/documents")
        assert resp.status_code == 503
        assert "MCP" in resp.text
        # Friendly error page, not a Python traceback
        assert "Traceback" not in resp.text

    def test_connection_error_returns_503_json_for_api(self, tmp_path):
        _sessions.clear()
        client = self._make_failing_client(MCPConnectionError, ("MCP unreachable",))
        config = CortexConfig(data_dir=tmp_path)
        app = create_dashboard(config, mcp_client=client)
        tc = TestClient(app)
        resp = tc.get("/api/graph-data")
        assert resp.status_code == 503
        body = resp.json()
        assert "error" in body
        assert body["code"] == "CORTEX_MCP_CONNECTION_ERROR"

    def test_timeout_error_returns_504(self, tmp_path):
        _sessions.clear()
        client = self._make_failing_client(MCPTimeoutError, ("timed out",))
        config = CortexConfig(data_dir=tmp_path)
        app = create_dashboard(config, mcp_client=client)
        tc = TestClient(app)
        resp = tc.get("/documents")
        assert resp.status_code == 504
        assert "Traceback" not in resp.text

    def test_tool_error_returns_502(self, tmp_path):
        _sessions.clear()
        client = self._make_failing_client(MCPToolError, ("tool blew up",))
        config = CortexConfig(data_dir=tmp_path)
        app = create_dashboard(config, mcp_client=client)
        tc = TestClient(app)
        resp = tc.get("/documents")
        assert resp.status_code == 502
        assert "Traceback" not in resp.text


class TestAuthStillEnforced:
    """Auth wrappers must survive the refactor."""

    def test_documents_requires_auth_when_password_set(self, tmp_path, call_recorder):
        _sessions.clear()
        pw_hash = bcrypt.hashpw(b"x", bcrypt.gensalt()).decode()
        config = CortexConfig(data_dir=tmp_path, dashboard_password=pw_hash)
        app = create_dashboard(config, mcp_client=call_recorder)
        tc = TestClient(app, follow_redirects=False)
        resp = tc.get("/documents")
        assert resp.status_code == 302
        assert "/login" in resp.headers["location"]
        # And no MCP call was made
        assert not call_recorder.calls

    def test_create_post_requires_auth(self, tmp_path, call_recorder):
        _sessions.clear()
        pw_hash = bcrypt.hashpw(b"x", bcrypt.gensalt()).decode()
        config = CortexConfig(data_dir=tmp_path, dashboard_password=pw_hash)
        app = create_dashboard(config, mcp_client=call_recorder)
        tc = TestClient(app, follow_redirects=False)
        resp = tc.post(
            "/create",
            data={"title": "X"},
        )
        assert resp.status_code == 302
        # No capture call attempted
        assert not any(c[0] == "capture" for c in call_recorder.calls)

    def test_api_graph_data_requires_auth(self, tmp_path, call_recorder):
        _sessions.clear()
        pw_hash = bcrypt.hashpw(b"x", bcrypt.gensalt()).decode()
        config = CortexConfig(data_dir=tmp_path, dashboard_password=pw_hash)
        app = create_dashboard(config, mcp_client=call_recorder)
        tc = TestClient(app, follow_redirects=False)
        resp = tc.get("/api/graph-data")
        # API endpoints return 401 JSON instead of redirecting
        assert resp.status_code == 401


class TestSettingsPageRendersWithoutStore:
    """Bundle 5 / A8: Phase 2.D — the ``/settings`` page must render without
    invoking any store-touching MCP call.

    The general :class:`TestDashboardDoesNotImportStore` covers the
    import-level contract. This test exercises the /settings endpoint
    specifically to prove it doesn't indirectly hit a code path that
    would open the store, even after future refactors.
    """

    def test_get_settings_renders_without_store_calls(self, app_with_recorder, call_recorder):
        client = TestClient(app_with_recorder)
        resp = client.get("/settings")
        assert resp.status_code == 200
        # Any MCP method that would cause the server to touch the store.
        store_touching = {
            "search",
            "read",
            "list_objects",
            "list_entities",
            "capture",
            "dossier",
            "context",
            "graph",
            "graph_data",
            "status",
            "query_trail",
        }
        called = {c[0] for c in call_recorder.calls}
        overlap = called & store_touching
        assert not overlap, (
            f"/settings should not invoke any store-touching MCP method, "
            f"but called: {sorted(overlap)}"
        )

    def test_get_settings_does_not_crash_with_failing_store_client(self, tmp_path):
        """Even if every MCP method would raise, /settings still renders —
        because it doesn't call any of them. Proves the endpoint has zero
        dependency on the MCP client's live connection.
        """
        _sessions.clear()

        class FailingForEverything:
            async def search(self, *a, **kw):
                raise MCPConnectionError("unreachable")

            async def list_objects(self, *a, **kw):
                raise MCPConnectionError("unreachable")

            async def list_entities(self, *a, **kw):
                raise MCPConnectionError("unreachable")

            async def read(self, *a, **kw):
                raise MCPConnectionError("unreachable")

            async def capture(self, *a, **kw):
                raise MCPConnectionError("unreachable")

            async def status(self, *a, **kw):
                raise MCPConnectionError("unreachable")

            async def query_trail(self, *a, **kw):
                raise MCPConnectionError("unreachable")

            async def graph_data(self, *a, **kw):
                raise MCPConnectionError("unreachable")

            async def dossier(self, *a, **kw):
                raise MCPConnectionError("unreachable")

            async def context(self, *a, **kw):
                raise MCPConnectionError("unreachable")

            async def graph(self, *a, **kw):
                raise MCPConnectionError("unreachable")

            async def list_tools(self):
                raise MCPConnectionError("unreachable")

        config = CortexConfig(data_dir=tmp_path)
        app = create_dashboard(config, mcp_client=FailingForEverything())
        client = TestClient(app)
        resp = client.get("/settings")
        # Renders successfully — /settings has no MCP dependency
        assert resp.status_code == 200


class TestDashboardDoesNotDoubleCountAccess:
    """Phase 2.F regression guard: viewing a document via the dashboard must
    record exactly ONE access (the server-side cortex_read tool does it).
    The dashboard's old direct learner.record_access call was removed in
    Phase 2.D — this test ensures we don't accidentally re-add it.
    """

    def test_get_document_calls_read_exactly_once(self, app_with_recorder, call_recorder):
        client = TestClient(app_with_recorder)
        client.get("/documents/some-obj-id")
        # Filter to just the 'read' calls
        read_calls = [c for c in call_recorder.calls if c[0] == "read"]
        assert len(read_calls) == 1, (
            f"expected exactly one read call, got {len(read_calls)}: {read_calls}"
        )

    def test_three_views_produce_three_read_calls(self, app_with_recorder, call_recorder):
        """Sanity: three GET /documents/{id} requests → three read calls.
        Catches a future regression where someone batches/dedupes accesses.
        """
        client = TestClient(app_with_recorder)
        client.get("/documents/id-1")
        client.get("/documents/id-2")
        client.get("/documents/id-3")
        read_calls = [c for c in call_recorder.calls if c[0] == "read"]
        assert len(read_calls) == 3
        ids_seen = [c[1]["obj_id"] for c in read_calls]
        assert ids_seen == ["id-1", "id-2", "id-3"]


class TestPostCreateInputValidation:
    """Phase 2.D: POST /create with missing required fields should reject the
    request without making any MCP call.
    """

    def test_post_create_missing_title_returns_422(self, app_with_recorder, call_recorder):
        """``title`` is a required form field; omitting it must return 422
        from FastAPI's form validation, NOT call MCP, NOT crash.
        """
        client = TestClient(app_with_recorder)
        # POST /create without the required `title` field
        resp = client.post("/create", data={"content": "no title"})
        assert resp.status_code == 422
        # No MCP capture call should have been made
        assert not any(c[0] == "capture" for c in call_recorder.calls)

    def test_post_create_with_only_title_succeeds(self, app_with_recorder, call_recorder):
        """Sanity: title alone is enough; the other Form fields default."""
        client = TestClient(app_with_recorder)
        resp = client.post("/create", data={"title": "T"}, follow_redirects=False)
        assert resp.status_code == 302  # redirect to /documents/<id>
        # MCP capture WAS called
        assert any(c[0] == "capture" for c in call_recorder.calls)
