"""Tests for cortex.transport.api.server (REST API).

Post-Phase-4 (Bundle 7): the REST API is now a thin MCP HTTP client. The
tests inject :class:`tests.conftest.FakeMCPClient` — an in-process
MCP-server proxy wired to a real ``Store`` — so the existing behavioral
assertions (capture-then-read, list filters, etc.) continue to work
without spinning up a real MCP HTTP server.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from cortex.core.config import CortexConfig
from cortex.ontology.resolver import find_ontology
from cortex.transport.api.server import create_api
from cortex.transport.mcp.server import create_mcp_server
from tests.conftest import FakeMCPClient

ONTOLOGY_PATH = find_ontology()

API_KEY = "test-key"
AUTH = {"X-API-Key": API_KEY}


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    """TestClient wired to a fresh Cortex API backed by tmp_path.

    Uses ``FakeMCPClient`` to route every endpoint through the same MCP
    tool functions the real deployment would hit, without spinning up a
    real HTTP MCP server.
    """
    config = CortexConfig(data_dir=tmp_path)
    mcp = create_mcp_server(config, include_admin=True)
    fake_client = FakeMCPClient(mcp)
    app = create_api(config, mcp_client=fake_client)
    return TestClient(app)


def _capture(
    client: TestClient,
    *,
    title: str = "Sample",
    content: str = "body",
    obj_type: str = "fix",
) -> dict:
    """Helper: capture an object and return the JSON response."""
    resp = client.post(
        "/capture",
        params={
            "title": title,
            "content": content,
            "obj_type": obj_type,
            "run_pipeline": "false",
        },
        headers=AUTH,
    )
    assert resp.status_code == 200
    return resp.json()


# -- Health ---------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_health_requires_no_auth(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200


# -- Auth (dev mode — no keys configured) --------------------------------


class TestAuth:
    def test_request_with_api_key_is_allowed(self, client: TestClient):
        resp = client.post("/search", params={"query": "test"}, headers=AUTH)
        assert resp.status_code == 200

    def test_request_without_api_key_returns_401(self, client: TestClient):
        resp = client.post("/search", params={"query": "test"})
        assert resp.status_code == 401

    def test_any_key_value_works_in_dev_mode(self, client: TestClient):
        resp = client.post(
            "/search",
            params={"query": "test"},
            headers={"X-API-Key": "literally-anything"},
        )
        assert resp.status_code == 200


# -- Search ---------------------------------------------------------------


class TestSearch:
    def test_search_returns_list(self, client: TestClient):
        resp = client.post("/search", params={"query": "test"}, headers=AUTH)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_search_finds_captured_object(self, client: TestClient):
        _capture(
            client,
            title="Unique quantum discovery",
            content="quantum entanglement breakthrough",
            obj_type="research",
        )
        resp = client.post("/search", params={"query": "quantum"}, headers=AUTH)
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) >= 1
        assert any("quantum" in r.get("title", "").lower() for r in results)

    def test_search_empty_query_returns_empty(self, client: TestClient):
        resp = client.post("/search", params={"query": ""}, headers=AUTH)
        assert resp.status_code == 200
        assert resp.json() == []


# -- Capture --------------------------------------------------------------


class TestCapture:
    def test_capture_returns_id_and_status(self, client: TestClient):
        data = _capture(client, title="Test capture")
        assert "id" in data
        assert data["id"] != ""
        assert data["type"] == "fix"

    def test_capture_with_different_type(self, client: TestClient):
        data = _capture(
            client,
            title="A lesson",
            content="learned something",
            obj_type="lesson",
        )
        assert data["type"] == "lesson"

    def test_capture_returns_ingested_status(self, client: TestClient):
        data = _capture(client)
        assert data["status"] == "ingested"


# -- Read -----------------------------------------------------------------


class TestRead:
    def test_read_existing_object(self, client: TestClient):
        captured = _capture(client, title="Readable doc")
        obj_id = captured["id"]

        resp = client.get(f"/read/{obj_id}", headers=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == obj_id
        assert body["title"] == "Readable doc"

    def test_read_nonexistent_returns_404(self, client: TestClient):
        resp = client.get("/read/nonexistent-id-xyz", headers=AUTH)
        assert resp.status_code == 404


# -- List -----------------------------------------------------------------


class TestList:
    def test_list_returns_list(self, client: TestClient):
        resp = client.get("/list", headers=AUTH)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_includes_captured_objects(self, client: TestClient):
        _capture(client, title="Listed item")
        resp = client.get("/list", headers=AUTH)
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) >= 1
        assert any(i.get("title") == "Listed item" for i in items)

    def test_list_filter_by_type(self, client: TestClient):
        _capture(client, title="Fix A", obj_type="fix")
        _capture(client, title="Lesson B", obj_type="lesson")

        resp = client.get(
            "/list",
            params={"doc_type": "fix"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        items = resp.json()
        assert all(i.get("type") == "fix" for i in items)


# -- Status ---------------------------------------------------------------


class TestStatus:
    def test_status_returns_counts(self, client: TestClient):
        _capture(client, title="Status item")

        resp = client.get("/status", headers=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert "sqlite_total" in body
        assert body["sqlite_total"] >= 1
        assert "counts_by_type" in body
        assert "alerts" in body

    def test_status_reflects_multiple_types(self, client: TestClient):
        _capture(client, title="Fix", obj_type="fix")
        _capture(client, title="Lesson", obj_type="lesson")

        resp = client.get("/status", headers=AUTH)
        assert resp.status_code == 200
        counts = resp.json()["counts_by_type"]
        assert "fix" in counts
        assert "lesson" in counts


# -- Delete ---------------------------------------------------------------


class TestDelete:
    def test_delete_existing_object(self, client: TestClient):
        captured = _capture(client, title="To delete")
        obj_id = captured["id"]

        resp = client.delete(f"/delete/{obj_id}", headers=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "deleted"
        assert body["obj_id"] == obj_id

        # Confirm it is gone
        resp = client.get(f"/read/{obj_id}", headers=AUTH)
        assert resp.status_code == 404

    def test_delete_nonexistent_returns_404(self, client: TestClient):
        resp = client.delete("/delete/nonexistent-id-xyz", headers=AUTH)
        assert resp.status_code == 404


# -- Context --------------------------------------------------------------


class TestContext:
    def test_context_returns_list(self, client: TestClient):
        resp = client.post(
            "/context",
            params={"topic": "test"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_context_returns_briefing_fields(self, client: TestClient):
        _capture(
            client,
            title="Context target",
            content="context topic material",
        )
        resp = client.post(
            "/context",
            params={"topic": "context"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        results = resp.json()
        if results:
            item = results[0]
            assert "id" in item
            assert "title" in item
            assert "summary" in item


# -- Dossier --------------------------------------------------------------


class TestDossier:
    def test_dossier_returns_dict(self, client: TestClient):
        resp = client.post(
            "/dossier",
            params={"topic": "test"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)
        assert "topic" in body
        assert body["topic"] == "test"

    def test_dossier_unknown_topic_shape(self, client: TestClient):
        resp = client.post(
            "/dossier",
            params={"topic": "ZzzzNonexistent"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "no_knowledge_found")
        assert "objects" in body


# -- Pipeline -------------------------------------------------------------


class TestPipeline:
    def test_pipeline_on_existing_object(self, client: TestClient):
        captured = _capture(client, title="Pipeline target")
        obj_id = captured["id"]

        resp = client.post(f"/pipeline/{obj_id}", headers=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert "pipeline_stages" in body

    def test_pipeline_nonexistent_returns_404(self, client: TestClient):
        resp = client.post("/pipeline/nonexistent-id-xyz", headers=AUTH)
        assert resp.status_code == 404


# -- Reason ---------------------------------------------------------------


class TestReason:
    def test_reason_returns_expected_keys(self, client: TestClient):
        resp = client.post("/reason", headers=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        for key in ("contradictions", "patterns", "gaps", "staleness"):
            assert key in body


# -- Capture with pre-classification -------------------------------------


class TestCaptureWithPreClassification:
    def test_capture_with_summary(self, client: TestClient):
        resp = client.post(
            "/capture",
            params={
                "title": "Pre-classified",
                "content": "body",
                "obj_type": "fix",
                "summary": "A pre-classified fix",
                "run_pipeline": "false",
            },
            headers=AUTH,
        )
        assert resp.status_code == 200
        obj_id = resp.json()["id"]

        read_resp = client.get(f"/read/{obj_id}", headers=AUTH)
        assert read_resp.json()["summary"] == "A pre-classified fix"


# -- Classify endpoint ----------------------------------------------------


class TestClassifyEndpoint:
    def test_classify_existing(self, client: TestClient):
        cap = client.post(
            "/capture",
            params={
                "title": "To classify",
                "content": "x",
                "run_pipeline": "false",
            },
            headers=AUTH,
        )
        obj_id = cap.json()["id"]

        resp = client.post(
            f"/classify/{obj_id}",
            params={"summary": "Classified!", "obj_type": "lesson"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "classified"

    def test_classify_nonexistent(self, client: TestClient):
        resp = client.post(
            "/classify/fake-id",
            params={"summary": "x"},
            headers=AUTH,
        )
        assert resp.status_code == 404


# -- Phase 4 / Bundle 7 contracts -----------------------------------------


class TestRestApiDoesNotImportStore:
    """Bundle 7 / Phase 4 regression guard: ``transport/api/server.py``
    must not import the Store layer after Phase 4.

    The REST API is now a thin MCP HTTP client. Re-introducing any
    direct-store dependency would resurrect the lock-fight problem that
    Phase 4 just fixed.
    """

    def test_server_module_does_not_import_store(self):
        import inspect

        import cortex.transport.api.server as api_mod

        src = inspect.getsource(api_mod)
        forbidden = [
            "from cortex.db.store import Store",
            "from cortex.pipeline.orchestrator import PipelineOrchestrator",
            "from cortex.retrieval.engine import RetrievalEngine",
            "from cortex.retrieval.graph import GraphQueries",
            "from cortex.retrieval.learner import LearningLoop",
            "from cortex.retrieval.presenters import",
            "from cortex.services.llm import LLMClient",
            "from cortex.ontology.resolver import find_ontology",
        ]
        for line in forbidden:
            assert line not in src, f"REST API must not import {line!r} after Phase 4"

    def test_server_module_imports_mcp_client(self):
        import inspect

        import cortex.transport.api.server as api_mod

        src = inspect.getsource(api_mod)
        assert "from cortex.transport.mcp.client import" in src
        # The exception classes we need for HTTP error mapping
        assert "MCPConnectionError" in src
        assert "MCPTimeoutError" in src
        assert "MCPServerError" in src
        assert "MCPToolError" in src


class TestRestApiMcpErrorMapping:
    """MCP client errors map to the expected HTTP status codes."""

    def _failing_client(self, exc_type, exc_args=None):
        """Build a minimal fake client where every tool call raises."""

        class FailingClient:
            async def search(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def list_objects(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def read(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def capture(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def status(self, *a, **kw):
                raise exc_type(*(exc_args or ()))

            async def list_tools(self):
                raise exc_type(*(exc_args or ()))

        return FailingClient()

    def _make_app(self, tmp_path: Path, failing_client):
        config = CortexConfig(data_dir=tmp_path)
        return create_api(config, mcp_client=failing_client)

    def test_connection_error_maps_to_503(self, tmp_path: Path):
        from cortex.transport.mcp.client import MCPConnectionError

        client = self._failing_client(MCPConnectionError, ("server down",))
        app = self._make_app(tmp_path, client)
        tc = TestClient(app)
        resp = tc.post("/search", params={"query": "x"}, headers=AUTH)
        assert resp.status_code == 503
        assert "unreachable" in resp.json()["detail"].lower()

    def test_timeout_error_maps_to_504(self, tmp_path: Path):
        from cortex.transport.mcp.client import MCPTimeoutError

        client = self._failing_client(MCPTimeoutError, ("timed out",))
        app = self._make_app(tmp_path, client)
        tc = TestClient(app)
        resp = tc.post("/search", params={"query": "x"}, headers=AUTH)
        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"].lower()

    def test_server_error_maps_to_502(self, tmp_path: Path):
        from cortex.transport.mcp.client import MCPServerError

        client = self._failing_client(MCPServerError, ("500 from upstream",))
        app = self._make_app(tmp_path, client)
        tc = TestClient(app)
        resp = tc.post("/search", params={"query": "x"}, headers=AUTH)
        assert resp.status_code == 502
        assert "server error" in resp.json()["detail"].lower()

    def test_tool_error_maps_to_502(self, tmp_path: Path):
        from cortex.transport.mcp.client import MCPToolError

        client = self._failing_client(MCPToolError, ("boom",))
        app = self._make_app(tmp_path, client)
        tc = TestClient(app)
        resp = tc.post("/search", params={"query": "x"}, headers=AUTH)
        assert resp.status_code == 502
        assert "tool error" in resp.json()["detail"].lower()


class TestRestApiKeyFromEnvVar:
    """Bundle 7: auth keys now come from ``CORTEX_API_KEYS`` env var.
    Legacy SQLite-backed key storage was removed.
    """

    def test_no_env_var_is_dev_mode(self, client: TestClient, monkeypatch):
        monkeypatch.delenv("CORTEX_API_KEYS", raising=False)
        resp = client.post(
            "/search",
            params={"query": "x"},
            headers={"X-API-Key": "anything-goes"},
        )
        assert resp.status_code == 200

    def test_configured_keys_accept_matching(self, client: TestClient, monkeypatch):
        monkeypatch.setenv("CORTEX_API_KEYS", "good-key,another")
        resp = client.post(
            "/search",
            params={"query": "x"},
            headers={"X-API-Key": "good-key"},
        )
        assert resp.status_code == 200

    def test_configured_keys_reject_nonmatching(self, client: TestClient, monkeypatch):
        monkeypatch.setenv("CORTEX_API_KEYS", "good-key")
        resp = client.post(
            "/search",
            params={"query": "x"},
            headers={"X-API-Key": "bad-key"},
        )
        assert resp.status_code == 401

    def test_configured_keys_accept_trailing_whitespace(self, client: TestClient, monkeypatch):
        """Bundle 9 / Group 3 #1: trailing whitespace in the X-API-Key
        header used to return 401 because h11 preserved trailing OWS
        while stripping leading OWS, producing an asymmetric and
        confusing comparison. The auth path now ``.strip()``'s the key
        before comparison so both sides round-trip correctly.
        """
        monkeypatch.setenv("CORTEX_API_KEYS", "good-key")
        resp = client.post(
            "/search",
            params={"query": "x"},
            headers={"X-API-Key": "good-key "},  # trailing space
        )
        assert resp.status_code == 200

    def test_configured_keys_accept_leading_whitespace(self, client: TestClient, monkeypatch):
        """Bundle 9 / Group 3 #1: leading whitespace was already stripped
        by h11 so this case used to ALSO return 200, but for the wrong
        reason (HTTP-parser-level normalization, not auth-level
        normalization). After the fix both directions are handled at
        auth-level for symmetry.
        """
        monkeypatch.setenv("CORTEX_API_KEYS", "good-key")
        resp = client.post(
            "/search",
            params={"query": "x"},
            headers={"X-API-Key": "  good-key"},  # leading space
        )
        assert resp.status_code == 200

    def test_whitespace_only_key_is_rejected(self, client: TestClient, monkeypatch):
        """Bundle 9 / Group 3 #1: stripping must not turn a whitespace
        key into the empty string and silently let it through. The auth
        path explicitly re-checks for emptiness after stripping.
        """
        monkeypatch.setenv("CORTEX_API_KEYS", "good-key")
        resp = client.post(
            "/search",
            params={"query": "x"},
            headers={"X-API-Key": "   "},
        )
        assert resp.status_code == 401


# -- E.2 security probe: API key bypass edge cases --------------------------


class TestApiKeyBypassEdgeCases:
    """E.2: additional edge cases beyond the Bundle 9 whitespace tests.

    The auth path uses hmac.compare_digest (constant-time) and strips
    whitespace. These tests exercise further bypass vectors.
    """

    def test_empty_string_key_is_rejected(self, client: TestClient, monkeypatch):
        monkeypatch.setenv("CORTEX_API_KEYS", "good-key")
        resp = client.post(
            "/search",
            params={"query": "x"},
            headers={"X-API-Key": ""},
        )
        assert resp.status_code == 401

    def test_case_sensitivity(self, client: TestClient, monkeypatch):
        """Keys must be case-sensitive — 'Good-Key' != 'good-key'."""
        monkeypatch.setenv("CORTEX_API_KEYS", "good-key")
        resp = client.post(
            "/search",
            params={"query": "x"},
            headers={"X-API-Key": "Good-Key"},
        )
        assert resp.status_code == 401

    def test_null_byte_in_key_is_rejected(self, client: TestClient, monkeypatch):
        monkeypatch.setenv("CORTEX_API_KEYS", "good-key")
        resp = client.post(
            "/search",
            params={"query": "x"},
            headers={"X-API-Key": "good-key\x00extra"},
        )
        assert resp.status_code == 401

    def test_comma_separated_empty_entries_ignored(self, client: TestClient, monkeypatch):
        """'key1,,key2' should not create an empty-string key."""
        monkeypatch.setenv("CORTEX_API_KEYS", "key1,,key2")
        # Empty key should still be rejected
        resp = client.post(
            "/search",
            params={"query": "x"},
            headers={"X-API-Key": ""},
        )
        assert resp.status_code == 401
        # Valid key should work
        resp = client.post(
            "/search",
            params={"query": "x"},
            headers={"X-API-Key": "key2"},
        )
        assert resp.status_code == 200

    def test_very_long_key_is_rejected(self, client: TestClient, monkeypatch):
        monkeypatch.setenv("CORTEX_API_KEYS", "good-key")
        resp = client.post(
            "/search",
            params={"query": "x"},
            headers={"X-API-Key": "x" * 100_000},
        )
        assert resp.status_code == 401

    def test_key_with_special_characters(self, client: TestClient, monkeypatch):
        """Keys containing special chars should match exactly."""
        special_key = "k3y-w1th_$pecial.chars!@#"
        monkeypatch.setenv("CORTEX_API_KEYS", special_key)
        resp = client.post(
            "/search",
            params={"query": "x"},
            headers={"X-API-Key": special_key},
        )
        assert resp.status_code == 200
