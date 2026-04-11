"""Tests for TemporalVersioning — version history and state-at-time queries."""

from __future__ import annotations

import time

import pytest

from cortex.db.content_store import ContentStore
from cortex.pipeline.temporal import TemporalVersioning


@pytest.fixture
def content_store():
    return ContentStore(path=None)  # in-memory


@pytest.fixture
def versioning(content_store):
    return TemporalVersioning(content_store)


def _insert_doc(content_store, doc_id="doc-1", title="Original Title"):
    """Insert a document directly into the content store."""
    content_store.insert(
        doc_id=doc_id,
        title=title,
        content="original content",
        doc_type="idea",
    )
    return doc_id


class TestSnapshotBeforeUpdate:
    """Tests for snapshot_before_update."""

    def test_snapshot_creates_version_one(self, content_store, versioning):
        doc_id = _insert_doc(content_store)
        version = versioning.snapshot_before_update(doc_id)

        assert version == 1

    def test_multiple_snapshots_increment(self, content_store, versioning):
        doc_id = _insert_doc(content_store)

        v1 = versioning.snapshot_before_update(doc_id)
        content_store.update(doc_id, title="Updated v1")
        v2 = versioning.snapshot_before_update(doc_id)
        content_store.update(doc_id, title="Updated v2")
        v3 = versioning.snapshot_before_update(doc_id)

        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    def test_nonexistent_doc_returns_none(self, versioning):
        result = versioning.snapshot_before_update("does-not-exist")

        assert result is None

    def test_five_updates_five_versions(self, content_store, versioning):
        doc_id = _insert_doc(content_store)

        for i in range(5):
            versioning.snapshot_before_update(doc_id)
            content_store.update(doc_id, title=f"Revision {i + 1}")

        assert versioning.version_count(doc_id) == 5


class TestGetVersion:
    """Tests for get_version — retrieving specific version data."""

    def test_get_version_returns_correct_data(self, content_store, versioning):
        doc_id = _insert_doc(content_store, title="V1 Title")
        versioning.snapshot_before_update(doc_id)

        content_store.update(doc_id, title="V2 Title")
        versioning.snapshot_before_update(doc_id)

        v1_data = versioning.get_version(doc_id, 1)
        v2_data = versioning.get_version(doc_id, 2)

        assert v1_data["title"] == "V1 Title"
        assert v2_data["title"] == "V2 Title"

    def test_get_version_nonexistent_returns_none(self, versioning):
        result = versioning.get_version("doc-1", 999)

        assert result is None


class TestGetStateAt:
    """Tests for get_state_at — point-in-time queries."""

    def test_future_date_returns_current(self, content_store, versioning):
        doc_id = _insert_doc(content_store)

        state = versioning.get_state_at(doc_id, "2099-01-01T00:00:00+00:00")

        assert state is not None
        assert state["id"] == doc_id

    def test_before_creation_returns_none(self, content_store, versioning):
        _insert_doc(content_store)

        state = versioning.get_state_at("doc-1", "1990-01-01T00:00:00+00:00")

        assert state is None

    def test_state_at_returns_version_valid_at_time(self, content_store, versioning):
        doc_id = _insert_doc(content_store, title="First")
        versioning.snapshot_before_update(doc_id)

        # Small delay so timestamps differ
        time.sleep(0.05)
        content_store.update(doc_id, title="Second")

        # Grab the version metadata to find valid_from/valid_to
        versions = versioning.list_versions(doc_id)
        assert len(versions) == 1

        v1_from = versions[0]["valid_from"]
        v1_to = versions[0]["valid_to"]

        # Query midpoint of v1's validity window
        state = versioning.get_state_at(doc_id, v1_from)
        assert state is not None
        assert state["title"] == "First"

        # After v1's valid_to, should get current version
        state_after = versioning.get_state_at(doc_id, v1_to)
        # valid_to is exclusive in the query (valid_to > at_time),
        # so at_time == valid_to falls outside the version window.
        # It should return the current doc instead.
        assert state_after is not None


class TestListVersions:
    """Tests for list_versions — ordered version metadata."""

    def test_list_versions_ordered_oldest_first(self, content_store, versioning):
        doc_id = _insert_doc(content_store)

        for i in range(3):
            versioning.snapshot_before_update(doc_id)
            content_store.update(doc_id, title=f"Rev {i + 1}")

        versions = versioning.list_versions(doc_id)

        assert len(versions) == 3
        assert versions[0]["version_num"] == 1
        assert versions[1]["version_num"] == 2
        assert versions[2]["version_num"] == 3

    def test_list_versions_empty_for_no_snapshots(self, versioning):
        versions = versioning.list_versions("never-snapped")

        assert versions == []


class TestVersionCount:
    """Tests for version_count."""

    def test_version_count_matches_actual(self, content_store, versioning):
        doc_id = _insert_doc(content_store)

        versioning.snapshot_before_update(doc_id)
        assert versioning.version_count(doc_id) == 1

        content_store.update(doc_id, title="v2")
        versioning.snapshot_before_update(doc_id)
        assert versioning.version_count(doc_id) == 2

    def test_version_count_zero_for_unknown_doc(self, versioning):
        assert versioning.version_count("unknown") == 0
