"""Tests for cortex.dashboard.server (web dashboard).

The dashboard is a thin MCP HTTP client. To test it without spinning up a real
MCP HTTP server, we inject an in-process ``FakeMCPClient`` (from
``tests/conftest.py``) that wraps a real ``Store`` and dispatches the same
MCP tool calls the dashboard would make. This lets the existing assertion
patterns ("create from dashboard, see in list") work unchanged while keeping
tests fast.
"""

from __future__ import annotations

from pathlib import Path

import bcrypt
import pytest
from starlette.testclient import TestClient

from cortex.core.config import CortexConfig
from cortex.dashboard.server import _sessions, create_dashboard
from cortex.transport.mcp.server import create_mcp_server
from tests.conftest import FakeMCPClient


def _make_client(tmp_path: Path, *, password_hash: str | None = None) -> TestClient:
    _sessions.clear()
    config = CortexConfig(
        data_dir=tmp_path,
        dashboard_password=password_hash or "",
    )
    mcp = create_mcp_server(config, include_admin=True)
    fake_client = FakeMCPClient(mcp)
    app = create_dashboard(config, mcp_client=fake_client)
    follow = password_hash is None
    return TestClient(app, follow_redirects=follow)


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    """TestClient wired to a fresh dashboard with no password (open access)."""
    return _make_client(tmp_path)


@pytest.fixture()
def auth_client(tmp_path: Path) -> TestClient:
    """TestClient wired to a dashboard with password auth enabled."""
    pw_hash = bcrypt.hashpw(b"testpass", bcrypt.gensalt()).decode()
    return _make_client(tmp_path, password_hash=pw_hash)


# -- Pages Render (open access) ------------------------------------------


class TestPagesRender:
    """All pages should return 200 and contain key text in open mode."""

    def test_home_page(self, client: TestClient):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Dashboard" in resp.text

    def test_documents_page(self, client: TestClient):
        resp = client.get("/documents")
        assert resp.status_code == 200
        assert "Documents" in resp.text

    def test_graph_page(self, client: TestClient):
        resp = client.get("/graph")
        assert resp.status_code == 200
        assert "Knowledge Graph" in resp.text

    def test_entities_page(self, client: TestClient):
        resp = client.get("/entities")
        assert resp.status_code == 200
        assert "Entities" in resp.text

    def test_create_page(self, client: TestClient):
        resp = client.get("/create")
        assert resp.status_code == 200
        assert "Create" in resp.text

    def test_trail_page(self, client: TestClient):
        resp = client.get("/trail")
        assert resp.status_code == 200
        assert "Query Trail" in resp.text

    def test_settings_page(self, client: TestClient):
        resp = client.get("/settings")
        assert resp.status_code == 200
        assert "Settings" in resp.text


# -- Document Detail ------------------------------------------------------


class TestDocumentDetail:
    def test_detail_for_existing_object(self, client: TestClient):
        store = client.app.state.mcp_client.store
        obj_id = store.create(
            obj_type="idea",
            title="Test Idea",
            content="Some content",
        )
        resp = client.get(f"/documents/{obj_id}")
        assert resp.status_code == 200
        assert "Test Idea" in resp.text

    def test_detail_nonexistent_returns_404(self, client: TestClient):
        resp = client.get("/documents/nonexistent-id-xyz")
        assert resp.status_code == 404


# -- Create Form ----------------------------------------------------------


class TestCreateForm:
    def test_post_create_redirects_to_detail(self, client: TestClient):
        resp = client.post(
            "/create",
            data={
                "title": "New Dashboard Object",
                "content": "Created via form",
                "obj_type": "idea",
                "project": "",
                "tags": "",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 302
        location = resp.headers["location"]
        assert location.startswith("/documents/")

    def test_created_object_appears_in_documents(self, client: TestClient):
        client.post(
            "/create",
            data={
                "title": "Visible Object",
                "content": "Should appear in list",
                "obj_type": "fix",
                "project": "",
                "tags": "",
            },
        )
        resp = client.get("/documents")
        assert resp.status_code == 200
        assert "Visible Object" in resp.text


# -- Search ---------------------------------------------------------------


class TestSearch:
    def test_search_with_query_returns_200(self, client: TestClient):
        resp = client.get("/documents", params={"q": "test"})
        assert resp.status_code == 200

    def test_search_finds_matching_document(self, client: TestClient):
        store = client.app.state.mcp_client.store
        store.create(
            obj_type="research",
            title="Quantum Entanglement",
            content="A study on quantum entanglement",
        )
        resp = client.get("/documents", params={"q": "quantum"})
        assert resp.status_code == 200
        assert "Quantum" in resp.text


# -- Graph Data API -------------------------------------------------------


class TestGraphDataAPI:
    def test_graph_data_returns_json(self, client: TestClient):
        resp = client.get("/api/graph-data")
        assert resp.status_code == 200
        body = resp.json()
        assert "nodes" in body
        assert "edges" in body

    def test_graph_data_empty_by_default(self, client: TestClient):
        resp = client.get("/api/graph-data")
        body = resp.json()
        assert body["nodes"] == []
        assert body["edges"] == []

    def test_graph_data_includes_seeded_objects(self, client: TestClient):
        store = client.app.state.mcp_client.store
        store.create(
            obj_type="idea",
            title="Graph Node A",
            content="first node",
        )
        store.create(
            obj_type="fix",
            title="Graph Node B",
            content="second node",
        )
        resp = client.get("/api/graph-data")
        body = resp.json()
        labels = [n["data"]["label"] for n in body["nodes"]]
        assert "Graph Node A" in labels
        assert "Graph Node B" in labels


# -- Auth (password protected) --------------------------------------------


class TestAuth:
    def test_home_redirects_to_login(self, auth_client: TestClient):
        resp = auth_client.get("/")
        assert resp.status_code == 302
        assert "/login" in resp.headers["location"]

    def test_login_correct_password_redirects(self, auth_client: TestClient):
        resp = auth_client.post("/login", data={"password": "testpass"})
        assert resp.status_code == 302
        assert resp.headers["location"] == "/"
        assert "cortex_session" in resp.cookies

    def test_login_wrong_password_returns_401(self, auth_client: TestClient):
        resp = auth_client.post("/login", data={"password": "wrongpass"})
        assert resp.status_code == 401

    def test_authenticated_session_accesses_home(self, auth_client: TestClient):
        # Log in first
        login_resp = auth_client.post("/login", data={"password": "testpass"})
        session_cookie = login_resp.cookies.get("cortex_session")
        assert session_cookie is not None

        # Access home with session cookie set on client
        auth_client.cookies.set("cortex_session", session_cookie)
        resp = auth_client.get("/")
        assert resp.status_code == 200

    def test_logout_clears_session(self, auth_client: TestClient):
        # Log in
        login_resp = auth_client.post("/login", data={"password": "testpass"})
        session_cookie = login_resp.cookies.get("cortex_session")

        # Logout with session cookie set on client
        auth_client.cookies.set("cortex_session", session_cookie)
        resp = auth_client.get("/logout")
        assert resp.status_code == 302
        assert "/login" in resp.headers["location"]

        # Session should be invalid now — clear client cookies
        auth_client.cookies.clear()
        auth_client.cookies.set("cortex_session", session_cookie)
        resp = auth_client.get("/")
        assert resp.status_code == 302


# -- Login Page Behavior --------------------------------------------------


class TestLoginPage:
    def test_login_page_redirects_when_no_password(self, client: TestClient):
        resp = client.get("/login", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/"

    def test_login_page_renders_when_password_set(self, auth_client: TestClient):
        resp = auth_client.get("/login")
        assert resp.status_code == 200


# -- CSRF / Origin hardening ----------------------------------------------


class TestCsrfOriginGuard:
    """State-changing routes must reject cross-site POSTs even with no password.

    The default install has no dashboard password, so ``_require_auth`` lets
    every request through as ``anonymous``. The Origin/Host guard is the only
    thing standing between a drive-by ``evil.com`` page and a CSRF-driven
    delete/export against the local KB.
    """

    def test_cross_origin_post_is_rejected(self, client: TestClient):
        # A real object so the route would otherwise succeed.
        store = client.app.state.mcp_client.store
        obj_id = store.create(
            obj_type="idea", title="Victim", content="do not delete me",
        )
        resp = client.post(
            f"/documents/{obj_id}/delete",
            headers={"origin": "http://evil.com"},
            follow_redirects=False,
        )
        assert resp.status_code == 403
        # The guard runs before routing, so the object must still exist.
        assert store.read(obj_id) is not None

    def test_cross_origin_referer_is_rejected(self, client: TestClient):
        # No Origin header, but a foreign Referer must also be rejected.
        resp = client.post(
            "/create",
            data={"title": "X", "content": "", "obj_type": "idea"},
            headers={"referer": "http://evil.com/attack.html"},
            follow_redirects=False,
        )
        assert resp.status_code == 403

    def test_same_origin_post_still_works(self, client: TestClient):
        # Browser same-origin form posts carry an Origin matching the host.
        resp = client.post(
            "/create",
            data={
                "title": "Same Origin Object",
                "content": "from a real form",
                "obj_type": "idea",
                "project": "",
                "tags": "",
            },
            headers={"origin": "http://testserver"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert resp.headers["location"].startswith("/documents/")

    def test_loopback_origin_is_allowed(self, client: TestClient):
        resp = client.post(
            "/create",
            data={"title": "Localhost Object", "content": "", "obj_type": "idea"},
            headers={"origin": "http://localhost:1315"},
            follow_redirects=False,
        )
        assert resp.status_code == 302

    def test_post_without_origin_still_works(self, client: TestClient):
        # The existing CreateForm tests rely on header-less posts continuing
        # to work (curl / TestClient default). Guard must not break those.
        resp = client.post(
            "/create",
            data={"title": "Headerless Object", "content": "", "obj_type": "idea"},
            follow_redirects=False,
        )
        assert resp.status_code == 302

    def test_get_with_foreign_origin_is_unaffected(self, client: TestClient):
        # GET is a safe method; the guard must never touch it.
        resp = client.get("/documents", headers={"origin": "http://evil.com"})
        assert resp.status_code == 200

    def test_cross_origin_export_is_rejected(self, client: TestClient):
        # /settings/export is the worst case: a cross-site directory-create +
        # arbitrary .md write primitive. It must be blocked.
        resp = client.post(
            "/settings/export",
            data={"export_path": "/tmp/cortex-csrf-target"},
            headers={"origin": "http://evil.com"},
            follow_redirects=False,
        )
        assert resp.status_code == 403


# -- Feedback API ----------------------------------------------------------


class TestFeedbackAPI:
    """Regression for the content-type mismatch: the HTMX buttons in
    documents.html post application/x-www-form-urlencoded (HTMX 2.x default,
    no json-enc extension is loaded), but the handler parsed JSON
    unconditionally — every click 500'd and the relevance signal was lost."""

    @staticmethod
    def _seed(client: TestClient) -> str:
        store = client.app.state.mcp_client.store
        return store.create(
            obj_type="idea", title="Feedback target", content="body",
        )

    @staticmethod
    def _access_count(client: TestClient, obj_id: str) -> int:
        """Read the learner's persisted access counter straight from the store."""
        store = client.app.state.mcp_client.store
        return int(store.content.get_config(f"access_count:{obj_id}", "0"))

    def test_form_encoded_relevant_true_is_recorded(self, client: TestClient):
        obj_id = self._seed(client)
        # Exactly what the documents.html HTMX button sends:
        # hx-vals='{"obj_id": "...", "relevant": true}' as form encoding.
        resp = client.post(
            "/api/feedback",
            data={"obj_id": obj_id, "relevant": "true"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"
        assert body["relevant"] is True
        assert body["access_count"] == 1
        # The signal actually persisted for the learner.
        assert self._access_count(client, obj_id) == 1

    def test_form_encoded_relevant_false_is_parsed_as_false(
        self, client: TestClient
    ):
        # bool("false") is truthy — the handler must parse the string.
        obj_id = self._seed(client)
        resp = client.post(
            "/api/feedback",
            data={"obj_id": obj_id, "relevant": "false"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["relevant"] is False
        # Not-relevant feedback must not count as an access.
        assert self._access_count(client, obj_id) == 0

    def test_json_body_still_works(self, client: TestClient):
        obj_id = self._seed(client)
        resp = client.post(
            "/api/feedback", json={"obj_id": obj_id, "relevant": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"
        assert body["relevant"] is True
        assert self._access_count(client, obj_id) == 1

    def test_json_relevant_false(self, client: TestClient):
        obj_id = self._seed(client)
        resp = client.post(
            "/api/feedback", json={"obj_id": obj_id, "relevant": False},
        )
        assert resp.status_code == 200
        assert resp.json()["relevant"] is False
        assert self._access_count(client, obj_id) == 0

    def test_missing_obj_id_returns_400(self, client: TestClient):
        resp = client.post("/api/feedback", data={"relevant": "true"})
        assert resp.status_code == 400

    def test_invalid_json_returns_400(self, client: TestClient):
        resp = client.post(
            "/api/feedback",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_unauthenticated_returns_401(self, auth_client: TestClient):
        resp = auth_client.post(
            "/api/feedback", data={"obj_id": "x", "relevant": "true"},
        )
        assert resp.status_code == 401
