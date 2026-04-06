"""Tests for cortex.transport.api.server (REST API)."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from cortex.core.config import CortexConfig
from cortex.ontology.resolver import find_ontology
from cortex.transport.api.server import create_api

ONTOLOGY_PATH = find_ontology()

API_KEY = "test-key"
AUTH = {"X-API-Key": API_KEY}


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    """TestClient wired to a fresh Cortex API backed by tmp_path."""
    config = CortexConfig(data_dir=tmp_path)
    app = create_api(config)
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
    def test_request_with_api_key_is_allowed(
        self, client: TestClient
    ):
        resp = client.post(
            "/search", params={"query": "test"}, headers=AUTH
        )
        assert resp.status_code == 200

    def test_request_without_api_key_returns_401(
        self, client: TestClient
    ):
        resp = client.post("/search", params={"query": "test"})
        assert resp.status_code == 401

    def test_any_key_value_works_in_dev_mode(
        self, client: TestClient
    ):
        resp = client.post(
            "/search",
            params={"query": "test"},
            headers={"X-API-Key": "literally-anything"},
        )
        assert resp.status_code == 200


# -- Search ---------------------------------------------------------------


class TestSearch:
    def test_search_returns_list(self, client: TestClient):
        resp = client.post(
            "/search", params={"query": "test"}, headers=AUTH
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_search_finds_captured_object(
        self, client: TestClient
    ):
        _capture(
            client,
            title="Unique quantum discovery",
            content="quantum entanglement breakthrough",
            obj_type="research",
        )
        resp = client.post(
            "/search", params={"query": "quantum"}, headers=AUTH
        )
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) >= 1
        assert any(
            "quantum" in r.get("title", "").lower()
            for r in results
        )

    def test_search_empty_query_returns_empty(
        self, client: TestClient
    ):
        resp = client.post(
            "/search", params={"query": ""}, headers=AUTH
        )
        assert resp.status_code == 200
        assert resp.json() == []


# -- Capture --------------------------------------------------------------


class TestCapture:
    def test_capture_returns_id_and_status(
        self, client: TestClient
    ):
        data = _capture(client, title="Test capture")
        assert "id" in data
        assert data["id"] != ""
        assert data["type"] == "fix"

    def test_capture_with_different_type(
        self, client: TestClient
    ):
        data = _capture(
            client,
            title="A lesson",
            content="learned something",
            obj_type="lesson",
        )
        assert data["type"] == "lesson"

    def test_capture_returns_ingested_status(
        self, client: TestClient
    ):
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

    def test_read_nonexistent_returns_404(
        self, client: TestClient
    ):
        resp = client.get(
            "/read/nonexistent-id-xyz", headers=AUTH
        )
        assert resp.status_code == 404


# -- List -----------------------------------------------------------------


class TestList:
    def test_list_returns_list(self, client: TestClient):
        resp = client.get("/list", headers=AUTH)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_includes_captured_objects(
        self, client: TestClient
    ):
        _capture(client, title="Listed item")
        resp = client.get("/list", headers=AUTH)
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) >= 1
        assert any(
            i.get("title") == "Listed item" for i in items
        )

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
        assert all(
            i.get("type") == "fix" for i in items
        )


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

    def test_status_reflects_multiple_types(
        self, client: TestClient
    ):
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

        resp = client.delete(
            f"/delete/{obj_id}", headers=AUTH
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "deleted"
        assert body["obj_id"] == obj_id

        # Confirm it is gone
        resp = client.get(f"/read/{obj_id}", headers=AUTH)
        assert resp.status_code == 404

    def test_delete_nonexistent_returns_404(
        self, client: TestClient
    ):
        resp = client.delete(
            "/delete/nonexistent-id-xyz", headers=AUTH
        )
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

    def test_context_returns_briefing_fields(
        self, client: TestClient
    ):
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

    def test_dossier_unknown_topic_shape(
        self, client: TestClient
    ):
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

    def test_pipeline_nonexistent_returns_404(
        self, client: TestClient
    ):
        resp = client.post(
            "/pipeline/nonexistent-id-xyz", headers=AUTH
        )
        assert resp.status_code == 404


# -- Reason ---------------------------------------------------------------


class TestReason:
    def test_reason_returns_expected_keys(
        self, client: TestClient
    ):
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
