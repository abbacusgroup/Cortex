"""Tests for the dashboard vault export (POST /settings/export).

Covers the P1 export fixes:

- pagination over the full corpus (the old single ``list_objects(limit=5000)``
  call silently truncated larger KBs),
- case-insensitive de-collision of project hub files (empirically, projects
  'Cortex' and 'cortex' overwrote each other's hub on macOS APFS),
- case-insensitive de-collision of document filenames, and
- an honest exported-file count (files actually written, docs and project
  indexes reported separately, resolved collisions surfaced).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from starlette.testclient import TestClient

from cortex.core.config import CortexConfig
from cortex.dashboard.server import _sessions, create_dashboard
from cortex.transport.mcp.server import create_mcp_server
from tests.conftest import FakeMCPClient


class PaginatingFakeMCPClient(FakeMCPClient):
    """FakeMCPClient plus the ``offset`` param the real client now supports.

    Also records every ``list_objects`` call so tests can assert the export
    actually paginates instead of issuing one giant request.
    """

    def __init__(self, mcp):
        super().__init__(mcp)
        self.list_calls: list[dict[str, Any]] = []

    async def list_objects(
        self,
        doc_type: str = "",
        project: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self.list_calls.append({"limit": limit, "offset": offset})
        return self._call(
            "cortex_list",
            doc_type=doc_type,
            project=project,
            limit=limit,
            offset=offset,
        )


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    """Dashboard TestClient (open access) backed by a fresh in-process store."""
    _sessions.clear()
    config = CortexConfig(data_dir=tmp_path / "data", dashboard_password="")
    mcp = create_mcp_server(config, include_admin=True)
    fake_client = PaginatingFakeMCPClient(mcp)
    app = create_dashboard(config, mcp_client=fake_client)
    return TestClient(app, follow_redirects=False)


def _seed(client: TestClient, title: str, project: str, *, seq: int) -> str:
    """Create a doc with a strictly distinct created_at so the export's
    ``ORDER BY created_at DESC`` processing order is deterministic
    (higher ``seq`` = newer = processed first)."""
    store = client.app.state.mcp_client.store
    return store.create(
        obj_type="idea",
        title=title,
        content=f"content of {title}",
        project=project,
        created_at=f"2026-01-01T00:{seq // 60:02d}:{seq % 60:02d}+00:00",
    )


def _export(client: TestClient, target: Path) -> str:
    """POST the export form and return the decoded redirect message."""
    resp = client.post("/settings/export", data={"export_path": str(target)})
    assert resp.status_code == 302
    query = parse_qs(urlsplit(resp.headers["location"]).query)
    msg = query.get("msg", [""])[0]
    assert query.get("msg_type", [""])[0] == "info", msg
    return msg


def _md_files(target: Path) -> list[Path]:
    return sorted(target.rglob("*.md"))


class TestExportPagination:
    def test_export_paginates_past_a_single_page(
        self, client: TestClient, tmp_path: Path, monkeypatch
    ):
        # Shrink the page size so 8 docs need three pages (3 + 3 + 2).
        monkeypatch.setattr("cortex.dashboard.server.EXPORT_PAGE_SIZE", 3)
        for i in range(8):
            _seed(client, f"Doc {i:02d}", "alpha", seq=i)
        target = tmp_path / "vault"

        msg = _export(client, target)

        files = _md_files(target)
        # 8 docs + 1 project hub, none silently dropped by the page cap.
        assert len(files) == 9
        assert "Exported 8 documents and 1 project indexes" in msg
        # The fake recorded the actual paging behavior.
        offsets = [c["offset"] for c in client.app.state.mcp_client.list_calls]
        assert offsets == [0, 3, 6]
        assert all(c["limit"] == 3 for c in client.app.state.mcp_client.list_calls)

    def test_export_smaller_than_one_page_is_single_call(
        self, client: TestClient, tmp_path: Path
    ):
        _seed(client, "Only Doc", "alpha", seq=0)
        target = tmp_path / "vault"

        msg = _export(client, target)

        assert "Exported 1 documents and 1 project indexes" in msg
        assert [c["offset"] for c in client.app.state.mcp_client.list_calls] == [0]


class TestExportHubCollisions:
    def test_case_insensitive_project_hubs_get_distinct_paths(
        self, client: TestClient, tmp_path: Path
    ):
        # Empirical macOS bug: 'Cortex' and 'cortex' hub files resolved to
        # the same path on case-insensitive APFS, one overwrote the other.
        # seq makes 'Cortex' the newest doc, so it claims its folder first.
        _seed(client, "Upper Doc", "Cortex", seq=10)
        _seed(client, "Lower Doc A", "cortex", seq=2)
        _seed(client, "Lower Doc B", "cortex", seq=1)
        target = tmp_path / "vault"

        msg = _export(client, target)

        files = _md_files(target)
        # 3 docs + 2 hubs, all on disk even on a case-insensitive filesystem.
        assert len(files) == 5
        assert "Exported 3 documents and 2 project indexes" in msg
        assert "collisions resolved" in msg

        hub_upper = target / "Cortex" / "Cortex.md"
        hub_lower = target / "cortex-2" / "cortex-2.md"
        assert hub_upper.is_file()
        assert hub_lower.is_file()
        # Each hub kept its own project's links (nothing overwritten/merged).
        assert "project: Cortex" in hub_upper.read_text(encoding="utf-8")
        lower_text = hub_lower.read_text(encoding="utf-8")
        assert "project: cortex" in lower_text
        assert "2 documents" in lower_text
        assert "Lower Doc A" in lower_text
        assert "Lower Doc B" in lower_text

    def test_distinct_projects_unaffected(self, client: TestClient, tmp_path: Path):
        _seed(client, "Doc One", "ProjectA", seq=1)
        _seed(client, "Doc Two", "ProjectB", seq=0)
        target = tmp_path / "vault"

        msg = _export(client, target)

        assert (target / "ProjectA" / "ProjectA.md").is_file()
        assert (target / "ProjectB" / "ProjectB.md").is_file()
        assert "collisions resolved" not in msg


class TestExportDocCollisions:
    def test_same_title_docs_get_id_suffixed_filenames(
        self, client: TestClient, tmp_path: Path
    ):
        id_new = _seed(client, "Same Title", "proj", seq=1)
        id_old = _seed(client, "Same Title", "proj", seq=0)
        target = tmp_path / "vault"

        msg = _export(client, target)

        doc_dir = target / "proj" / "idea"
        # Newest doc is processed first and keeps the plain name; the older
        # one is de-collided with its object-id prefix.
        assert (doc_dir / "Same Title.md").is_file()
        assert (doc_dir / f"Same Title-{id_old[:8]}.md").is_file()
        assert id_new != id_old
        assert "Exported 2 documents and 1 project indexes" in msg
        assert "(1 filename collisions resolved)" in msg

    def test_titles_differing_only_by_case_are_de_collided(
        self, client: TestClient, tmp_path: Path
    ):
        _seed(client, "NOTES", "proj", seq=1)
        id_old = _seed(client, "notes", "proj", seq=0)
        target = tmp_path / "vault"

        msg = _export(client, target)

        doc_dir = target / "proj" / "idea"
        assert (doc_dir / "NOTES.md").is_file()
        assert (doc_dir / f"notes-{id_old[:8]}.md").is_file()
        assert "Exported 2 documents and 1 project indexes" in msg


class TestExportHonestCount:
    def test_message_counts_match_files_on_disk(
        self, client: TestClient, tmp_path: Path
    ):
        # Mixed bag: case-colliding projects AND a case-colliding title.
        _seed(client, "Hub Doc", "Cortex", seq=5)
        _seed(client, "Shared", "cortex", seq=4)
        _seed(client, "shared", "cortex", seq=3)
        _seed(client, "Loose Doc", "", seq=2)  # unscoped: no hub
        target = tmp_path / "vault"

        msg = _export(client, target)

        files = _md_files(target)
        # 4 docs + 2 hubs (no hub for the unscoped doc).
        assert "Exported 4 documents and 2 project indexes" in msg
        assert len(files) == 6
        # The unscoped doc landed under _unscoped/ with no hub file.
        assert (target / "_unscoped" / "idea" / "Loose Doc.md").is_file()
        assert not (target / "_unscoped" / "_unscoped.md").exists()

    def test_empty_store_exports_zero(self, client: TestClient, tmp_path: Path):
        target = tmp_path / "vault"
        msg = _export(client, target)
        assert "Exported 0 documents and 0 project indexes" in msg
        assert _md_files(target) == []
