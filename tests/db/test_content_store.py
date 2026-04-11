"""Tests for cortex.db.content_store."""

from __future__ import annotations

import time

import pytest

from cortex.core.errors import NotFoundError, StoreError
from cortex.db.content_store import ContentStore


@pytest.fixture()
def store() -> ContentStore:
    """In-memory ContentStore, fresh per test."""
    return ContentStore(path=None)


# ── CRUD Happy Path ──────────────────────────────────────────────────


class TestCRUD:
    def test_insert_returns_id_and_get_returns_all_fields(self, store: ContentStore):
        doc_id = store.insert(
            doc_id="doc-1",
            title="My Title",
            content="Body text",
            raw_markdown="# Body text",
            doc_type="decision",
            project="cortex",
            tags="python,testing",
            summary="A summary",
            tier="recall",
            pipeline_stage="enrich",
            confidence=0.95,
            captured_by="test",
        )
        assert doc_id == "doc-1"

        doc = store.get(doc_id)
        assert doc is not None
        assert doc["id"] == "doc-1"
        assert doc["title"] == "My Title"
        assert doc["content"] == "Body text"
        assert doc["raw_markdown"] == "# Body text"
        assert doc["type"] == "decision"
        assert doc["project"] == "cortex"
        assert doc["tags"] == "python,testing"
        assert doc["summary"] == "A summary"
        assert doc["tier"] == "recall"
        assert doc["pipeline_stage"] == "enrich"
        assert doc["confidence"] == pytest.approx(0.95)
        assert doc["captured_by"] == "test"
        assert doc["created_at"] is not None
        assert doc["updated_at"] is not None

    def test_update_changes_fields_and_bumps_updated_at(self, store: ContentStore):
        store.insert(doc_id="u1", title="Old")
        before = store.get("u1")
        assert before is not None
        old_ts = before["updated_at"]

        # Tiny sleep so the timestamp can differ
        time.sleep(0.01)
        store.update("u1", title="New", tier="reflex")

        after = store.get("u1")
        assert after is not None
        assert after["title"] == "New"
        assert after["tier"] == "reflex"
        assert after["updated_at"] > old_ts

    def test_update_nonexistent_raises(self, store: ContentStore):
        with pytest.raises(NotFoundError):
            store.update("ghost", title="nope")

    def test_delete_removes_doc_and_fts(self, store: ContentStore):
        store.insert(doc_id="d1", title="Deleteme", content="searchable")
        assert store.delete("d1") is True
        assert store.get("d1") is None
        assert store.search("searchable") == []

    def test_delete_nonexistent_returns_false(self, store: ContentStore):
        assert store.delete("nope") is False

    def test_list_documents_unfiltered(self, store: ContentStore):
        store.insert(doc_id="a", title="A", doc_type="fix")
        store.insert(doc_id="b", title="B", doc_type="lesson")
        docs = store.list_documents()
        assert len(docs) == 2

    def test_list_documents_filter_by_type(self, store: ContentStore):
        store.insert(doc_id="a", title="A", doc_type="fix")
        store.insert(doc_id="b", title="B", doc_type="lesson")
        docs = store.list_documents(doc_type="fix")
        assert len(docs) == 1
        assert docs[0]["id"] == "a"

    def test_list_documents_filter_by_project(self, store: ContentStore):
        store.insert(doc_id="a", title="A", project="alpha")
        store.insert(doc_id="b", title="B", project="beta")
        docs = store.list_documents(project="beta")
        assert len(docs) == 1
        assert docs[0]["id"] == "b"

    def test_count_by_type(self, store: ContentStore):
        store.insert(doc_id="1", title="A", doc_type="fix")
        store.insert(doc_id="2", title="B", doc_type="fix")
        store.insert(doc_id="3", title="C", doc_type="lesson")
        counts = store.count_by_type()
        assert counts == {"fix": 2, "lesson": 1}

    def test_total_count(self, store: ContentStore):
        assert store.total_count() == 0
        store.insert(doc_id="1", title="A")
        store.insert(doc_id="2", title="B")
        assert store.total_count() == 2

    def test_insert_with_custom_timestamps(self, store: ContentStore):
        store.insert(
            doc_id="ts-custom",
            title="Custom TS",
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-06-15T12:30:00+00:00",
        )
        doc = store.get("ts-custom")
        assert doc is not None
        assert doc["created_at"] == "2025-01-01T00:00:00+00:00"
        assert doc["updated_at"] == "2025-06-15T12:30:00+00:00"

    def test_insert_defaults_to_now_when_no_timestamps(self, store: ContentStore):
        store.insert(doc_id="ts-default", title="Default TS")
        doc = store.get("ts-default")
        assert doc is not None
        # created_at should be a non-empty ISO datetime string
        assert isinstance(doc["created_at"], str)
        assert len(doc["created_at"]) > 0
        # Starts with a year (basic ISO 8601 sanity check)
        assert doc["created_at"][:4].isdigit()
        assert doc["updated_at"][:4].isdigit()

    def test_insert_duplicate_raises_store_error(self, store: ContentStore):
        store.insert(doc_id="dup", title="First")
        with pytest.raises(StoreError):
            store.insert(doc_id="dup", title="Second")


# ── FTS5 Search ──────────────────────────────────────────────────────


class TestFTS:
    def test_search_word_in_title(self, store: ContentStore):
        store.insert(doc_id="t1", title="Quantum computing")
        results = store.search("quantum")
        assert len(results) == 1
        assert results[0]["id"] == "t1"

    def test_search_word_in_content(self, store: ContentStore):
        store.insert(doc_id="c1", title="Boring", content="The flux capacitor")
        results = store.search("capacitor")
        assert len(results) == 1
        assert results[0]["id"] == "c1"

    def test_search_word_in_tags(self, store: ContentStore):
        store.insert(doc_id="g1", title="Note", tags="kubernetes")
        results = store.search("kubernetes")
        assert len(results) == 1

    def test_search_phrase(self, store: ContentStore):
        store.insert(
            doc_id="p1",
            title="Guide",
            content="The quick brown fox jumps over the lazy dog",
        )
        results = store.search("quick brown fox")
        assert len(results) == 1

    def test_search_no_matches(self, store: ContentStore):
        store.insert(doc_id="n1", title="Something")
        assert store.search("zzzznotfound") == []

    def test_bm25_title_ranks_higher_than_content(self, store: ContentStore):
        # "neutrino" in content only
        store.insert(
            doc_id="content_match",
            title="Generic note",
            content="Neutrino detection research",
        )
        # "neutrino" in title (10x weight)
        store.insert(
            doc_id="title_match",
            title="Neutrino physics overview",
            content="Some other body text",
        )
        results = store.search("neutrino")
        assert len(results) == 2
        assert results[0]["id"] == "title_match"

    def test_sql_injection_treated_as_search_term(self, store: ContentStore):
        store.insert(doc_id="safe", title="Normal document")
        # Classic SQL injection attempts
        payloads = [
            "'; DROP TABLE documents; --",
            '" OR 1=1 --',
            "UNION SELECT * FROM config",
            "title:*",
        ]
        for payload in payloads:
            # Must not raise, must not corrupt the store
            results = store.search(payload)
            assert isinstance(results, list)
        # Store still works after injection attempts
        assert store.get("safe") is not None
        assert store.total_count() == 1

    def test_empty_query_returns_empty(self, store: ContentStore):
        store.insert(doc_id="e1", title="Something")
        assert store.search("") == []
        assert store.search("   ") == []

    def test_very_long_query_handled(self, store: ContentStore):
        store.insert(doc_id="long", title="Test")
        long_q = "a" * 10_000
        results = store.search(long_q)
        assert isinstance(results, list)

    def test_unicode_cjk_does_not_crash(self, store: ContentStore):
        """CJK content is stored and searched without error.

        FTS5 unicode61 tokenizer splits on CJK boundaries, so
        partial-character search may not match. We verify the
        store handles it gracefully and ascii tokens still work.
        """
        store.insert(
            doc_id="cjk",
            title="Chinese",
            content="machine learning is \u673a\u5668\u5b66\u4e60",
        )
        # ASCII word inside the same doc is found
        results = store.search("machine")
        assert len(results) == 1
        assert results[0]["id"] == "cjk"
        # CJK search does not raise
        cjk_results = store.search("\u673a\u5668")
        assert isinstance(cjk_results, list)

    def test_unicode_emoji_does_not_crash(self, store: ContentStore):
        """Emoji in content/title is stored safely; search doesn't error."""
        store.insert(
            doc_id="emoji",
            title="Rocket launch \U0001f680",
            content="To the moon",
        )
        # ASCII part matches
        results = store.search("moon")
        assert len(results) == 1
        # Emoji-only search does not raise
        emoji_results = store.search("\U0001f680")
        assert isinstance(emoji_results, list)


# ── Config ───────────────────────────────────────────────────────────


class TestConfig:
    def test_set_and_get_roundtrip(self, store: ContentStore):
        store.set_config("theme", "dark")
        assert store.get_config("theme") == "dark"

    def test_get_config_default(self, store: ContentStore):
        assert store.get_config("missing", "fallback") == "fallback"
        assert store.get_config("missing") == ""

    def test_set_config_overwrites(self, store: ContentStore):
        store.set_config("key", "first")
        store.set_config("key", "second")
        assert store.get_config("key") == "second"


# ── Query Log ────────────────────────────────────────────────────────


class TestQueryLog:
    def test_log_and_retrieve(self, store: ContentStore):
        store.log_query(
            tool="search",
            params={"q": "test"},
            result_ids=["a", "b"],
            duration_ms=12.5,
            session_id="sess-1",
        )
        logs = store.get_query_log()
        assert len(logs) == 1

        entry = logs[0]
        assert entry["tool"] == "search"
        assert entry["session_id"] == "sess-1"
        assert entry["duration_ms"] == pytest.approx(12.5)
        assert entry["timestamp"] is not None

    def test_result_count_matches(self, store: ContentStore):
        store.log_query(
            tool="list",
            params={},
            result_ids=["x", "y", "z"],
            duration_ms=1.0,
        )
        entry = store.get_query_log()[0]
        assert entry["result_count"] == 3


# ── Embeddings ───────────────────────────────────────────────────────


class TestEmbeddings:
    def test_store_and_get_roundtrip(self, store: ContentStore):
        store.insert(doc_id="emb1", title="Embedded doc")
        blob = b"\x00\x01\x02\x03" * 192  # 768 bytes
        store.store_embedding(
            doc_id="emb1",
            embedding=blob,
            model="all-mpnet-base-v2",
            dimensions=768,
        )
        result = store.get_embedding("emb1")
        assert result == blob

    def test_get_embedding_nonexistent(self, store: ContentStore):
        assert store.get_embedding("no-such-id") is None
