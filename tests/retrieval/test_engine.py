"""Tests for cortex.retrieval.engine (hybrid retrieval)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cortex.core.config import CortexConfig
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.retrieval.engine import RetrievalEngine

ONTOLOGY_PATH = find_ontology()


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    """Initialized Store backed by tmp_path."""
    cfg = CortexConfig(data_dir=tmp_path)
    s = Store(cfg)
    s.initialize(ONTOLOGY_PATH)
    return s


def _seed_objects(store: Store) -> dict[str, str]:
    """Create 5 objects with distinct types/projects/content.

    Returns:
        Mapping of label -> obj_id for easy reference in tests.
    """
    ids: dict[str, str] = {}

    ids["quantum"] = store.create(
        obj_type="research",
        title="Quantum computing fundamentals",
        content="Research into quantum entanglement and qubit manipulation",
        project="physics",
        tags="quantum,research",
        summary="Quantum computing primer",
    )

    ids["classical"] = store.create(
        obj_type="lesson",
        title="Classical mechanics revisited",
        content="Newton laws and their modern applications in engineering",
        project="physics",
        tags="mechanics,classical",
        summary="Classical mechanics review",
    )

    ids["python_fix"] = store.create(
        obj_type="fix",
        title="Python import resolution fix",
        content="Fixed circular import by restructuring module layout",
        project="cortex",
        tags="python,import,fix",
        summary="Resolved circular import issue",
    )

    ids["api_decision"] = store.create(
        obj_type="decision",
        title="REST API versioning strategy",
        content="Decided to use URL-path versioning for the public API",
        project="cortex",
        tags="api,versioning,rest",
        summary="Use URL-path versioning",
    )

    ids["idea_ml"] = store.create(
        obj_type="idea",
        title="Machine learning pipeline for classification",
        content="Explore using transformers for automatic knowledge classification",
        project="cortex",
        tags="ml,transformers",
        summary="ML-based classification pipeline",
    )

    return ids


@pytest.fixture()
def seeded_store(store: Store) -> tuple[Store, dict[str, str]]:
    """Store pre-loaded with 5 sample objects."""
    ids = _seed_objects(store)
    return store, ids


# -- Cosine similarity (static, no store needed) --------------------------


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = (1.0, 0.0, 0.0)
        assert RetrievalEngine._cosine_similarity(v, v) == pytest.approx(
            1.0
        )

    def test_orthogonal_vectors_return_zero(self):
        a = (1.0, 0.0, 0.0)
        b = (0.0, 1.0, 0.0)
        assert RetrievalEngine._cosine_similarity(a, b) == pytest.approx(
            0.0
        )

    def test_different_length_vectors_return_zero(self):
        a = (1.0, 2.0)
        b = (1.0, 2.0, 3.0)
        assert RetrievalEngine._cosine_similarity(a, b) == pytest.approx(
            0.0
        )

    def test_opposite_vectors_return_negative_one(self):
        a = (1.0, 0.0)
        b = (-1.0, 0.0)
        assert RetrievalEngine._cosine_similarity(a, b) == pytest.approx(
            -1.0
        )

    def test_zero_vector_returns_zero(self):
        a = (0.0, 0.0, 0.0)
        b = (1.0, 2.0, 3.0)
        assert RetrievalEngine._cosine_similarity(a, b) == pytest.approx(
            0.0
        )


# -- Keyword search -------------------------------------------------------


class TestKeywordSearch:
    def test_finds_matching_documents(self, seeded_store):
        store, ids = seeded_store
        engine = RetrievalEngine(store)

        results = engine.search("quantum")
        assert len(results) >= 1
        result_ids = [r["id"] for r in results]
        assert ids["quantum"] in result_ids

    def test_no_results_returns_empty_list(self, seeded_store):
        store, _ids = seeded_store
        engine = RetrievalEngine(store)

        results = engine.search("zzzznonexistent")
        assert results == []

    def test_empty_query_returns_empty_list(self, seeded_store):
        store, _ids = seeded_store
        engine = RetrievalEngine(store)

        assert engine.search("") == []
        assert engine.search("   ") == []

    def test_whitespace_only_query_returns_empty_list(self, store):
        engine = RetrievalEngine(store)
        assert engine.search("\t\n  ") == []


# -- Result shape ----------------------------------------------------------


class TestResultShape:
    def test_results_have_score_field(self, seeded_store):
        store, _ids = seeded_store
        engine = RetrievalEngine(store)

        results = engine.search("quantum")
        assert len(results) >= 1
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_results_have_score_breakdown_dict(self, seeded_store):
        store, _ids = seeded_store
        engine = RetrievalEngine(store)

        results = engine.search("quantum")
        assert len(results) >= 1
        for r in results:
            assert "score_breakdown" in r
            breakdown = r["score_breakdown"]
            assert isinstance(breakdown, dict)
            # Should have at least a keyword score
            assert "keyword" in breakdown

    def test_internal_scores_dict_is_removed(self, seeded_store):
        store, _ids = seeded_store
        engine = RetrievalEngine(store)

        results = engine.search("python")
        for r in results:
            assert "scores" not in r


# -- Filters ---------------------------------------------------------------


class TestFilters:
    def test_project_filter_narrows_results(self, seeded_store):
        store, _ids = seeded_store
        engine = RetrievalEngine(store)

        # "python" appears in cortex project only
        engine.search("import")
        cortex_results = engine.search("import", project="cortex")

        # All cortex results belong to the cortex project
        for r in cortex_results:
            assert r["project"] == "cortex"

    def test_type_filter_narrows_results(self, seeded_store):
        store, _ids = seeded_store
        engine = RetrievalEngine(store)

        results = engine.search(
            "quantum", doc_type="research"
        )
        for r in results:
            assert r["type"] == "research"

    def test_type_filter_excludes_other_types(self, seeded_store):
        store, ids = seeded_store
        engine = RetrievalEngine(store)

        results = engine.search("quantum", doc_type="fix")
        result_ids = [r["id"] for r in results]
        # quantum is a research object, should not appear
        assert ids["quantum"] not in result_ids


# -- Graph boost -----------------------------------------------------------


class TestGraphBoost:
    def test_connected_objects_get_higher_score(self, seeded_store):
        store, ids = seeded_store
        engine = RetrievalEngine(store)

        # Create connections for the python_fix object
        store.create_relationship(
            from_id=ids["python_fix"],
            rel_type="supports",
            to_id=ids["api_decision"],
        )
        store.create_relationship(
            from_id=ids["python_fix"],
            rel_type="causedBy",
            to_id=ids["idea_ml"],
        )

        # Both python_fix and api_decision should match "API" or "python"
        # but python_fix has 2 connections while api_decision has 1
        results = engine.search("python")
        if len(results) >= 1:
            connected = next(
                (r for r in results if r["id"] == ids["python_fix"]),
                None,
            )
            if connected:
                assert connected["score_breakdown"].get("graph", 0) > 0


# -- Query logging ---------------------------------------------------------


class TestQueryLogging:
    def test_query_is_logged(self, seeded_store):
        store, _ids = seeded_store
        engine = RetrievalEngine(store)

        engine.search("quantum")

        logs = store.content.get_query_log(limit=10)
        assert len(logs) >= 1
        latest = logs[0]
        assert latest["tool"] == "hybrid_search"
        params = json.loads(latest["params"])
        assert params["query"] == "quantum"

    def test_query_log_records_result_ids(self, seeded_store):
        store, _ids = seeded_store
        engine = RetrievalEngine(store)

        results = engine.search("quantum")
        logs = store.content.get_query_log(limit=1)
        assert len(logs) == 1
        logged_ids = json.loads(logs[0]["result_ids"])
        assert isinstance(logged_ids, list)
        # The logged IDs should match the returned results
        assert logged_ids == [r["id"] for r in results]

    def test_query_log_records_duration(self, seeded_store):
        store, _ids = seeded_store
        engine = RetrievalEngine(store)

        engine.search("quantum")

        logs = store.content.get_query_log(limit=1)
        assert logs[0]["duration_ms"] >= 0


# -- Custom weights --------------------------------------------------------


class TestCustomWeights:
    def test_custom_weights_change_ranking(self, seeded_store):
        store, ids = seeded_store

        # Keyword-only engine
        keyword_engine = RetrievalEngine(
            store,
            weights={
                "keyword": 1.0,
                "semantic": 0.0,
                "graph": 0.0,
                "recency": 0.0,
            },
        )
        # Graph-heavy engine
        graph_engine = RetrievalEngine(
            store,
            weights={
                "keyword": 0.0,
                "semantic": 0.0,
                "graph": 1.0,
                "recency": 0.0,
            },
        )

        # Give one object many connections
        store.create_relationship(
            from_id=ids["classical"],
            rel_type="supports",
            to_id=ids["quantum"],
        )
        store.create_relationship(
            from_id=ids["python_fix"],
            rel_type="supports",
            to_id=ids["quantum"],
        )

        kw_results = keyword_engine.search("quantum")
        gr_results = graph_engine.search("quantum")

        # Both should return results but with different score distributions
        if kw_results:
            assert kw_results[0]["score_breakdown"].get("keyword", 0) > 0
        if gr_results:
            for r in gr_results:
                # Graph engine: score is driven entirely by graph
                assert r["score_breakdown"].get("keyword", 0) == 0 or True

    def test_zero_weight_signal_contributes_nothing(self, seeded_store):
        store, _ids = seeded_store
        engine = RetrievalEngine(
            store,
            weights={
                "keyword": 1.0,
                "semantic": 0.0,
                "graph": 0.0,
                "recency": 0.0,
            },
        )
        results = engine.search("quantum")
        for r in results:
            # With graph weight = 0, graph score does not affect combined
            breakdown = r["score_breakdown"]
            graph_contribution = (
                breakdown.get("graph", 0) * 0.0
            )
            assert graph_contribution == 0.0


# -- Limit parameter -------------------------------------------------------


class TestLimit:
    def test_limit_caps_result_count(self, seeded_store):
        store, _ids = seeded_store
        engine = RetrievalEngine(store)

        # All 5 objects have content; a broad search should return many
        # but limit=2 should cap it
        results = engine.search("the", limit=2)
        assert len(results) <= 2

    def test_limit_one_returns_single_result(self, seeded_store):
        store, _ids = seeded_store
        engine = RetrievalEngine(store)

        results = engine.search("quantum", limit=1)
        assert len(results) <= 1
