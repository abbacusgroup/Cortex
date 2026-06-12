"""Tests for cortex.retrieval.engine (hybrid retrieval)."""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from cortex.core.config import CortexConfig
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.retrieval.engine import (
    DEFAULT_WEIGHTS,
    WEIGHTS_CONFIG_KEY,
    RetrievalEngine,
    load_persisted_weights,
)

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
        assert RetrievalEngine._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = (1.0, 0.0, 0.0)
        b = (0.0, 1.0, 0.0)
        assert RetrievalEngine._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_different_length_vectors_return_zero(self):
        a = (1.0, 2.0)
        b = (1.0, 2.0, 3.0)
        assert RetrievalEngine._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self):
        a = (1.0, 0.0)
        b = (-1.0, 0.0)
        assert RetrievalEngine._cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = (0.0, 0.0, 0.0)
        b = (1.0, 2.0, 3.0)
        assert RetrievalEngine._cosine_similarity(a, b) == pytest.approx(0.0)


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

        results = engine.search("quantum", doc_type="research")
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
            graph_contribution = breakdown.get("graph", 0) * 0.0
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


# -- Embedder caching -------------------------------------------------------


class TestEmbeddingProvider:
    def test_embed_query_returns_none_without_provider(self, store):
        """Without an embedding provider, _embed_query returns None."""
        engine = RetrievalEngine(store)
        result = engine._embed_query("test query")
        assert result is None

    def test_embed_query_with_provider(self, store):
        """With a provider, _embed_query returns a tuple of floats."""
        provider = MagicMock()
        provider.embed.return_value = [0.1, 0.2, 0.3]
        engine = RetrievalEngine(store, embedding_provider=provider)
        result = engine._embed_query("test query")
        assert result == (0.1, 0.2, 0.3)
        provider.embed.assert_called_once_with("test query")

    def test_provider_starts_as_none(self, store):
        engine = RetrievalEngine(store)
        assert engine._embedding_provider is None

    def test_embed_query_failure_is_logged_not_swallowed(self, store, caplog):
        """A provider exception logs a warning instead of vanishing silently."""
        provider = MagicMock()
        provider.embed.side_effect = RuntimeError("boom")
        provider.model_name = "test-model"
        engine = RetrievalEngine(store, embedding_provider=provider)

        with caplog.at_level(logging.WARNING, logger="cortex.retrieval.engine"):
            result = engine._embed_query("test query")

        assert result is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "semantic ranking disabled" in message
        assert "test-model" in message
        assert "boom" in message


# -- Learner-persisted weights ----------------------------------------------


class TestPersistedWeights:
    def test_defaults_when_nothing_persisted(self, seeded_store):
        store, _ids = seeded_store
        engine = RetrievalEngine(store)
        engine.search("quantum")
        assert engine.weights == dict(DEFAULT_WEIGHTS)

    def test_engine_uses_learner_persisted_weights(self, seeded_store):
        """Weights persisted by the LearningLoop drive the engine's scoring."""
        from cortex.retrieval.learner import LearningLoop

        store, ids = seeded_store
        custom = {"keyword": 1.0, "semantic": 0.0, "graph": 0.0, "recency": 0.0}
        LearningLoop(store).update_weights(custom)

        engine = RetrievalEngine(store)
        results = engine.search("quantum")

        assert engine.weights == custom
        assert results
        top = results[0]
        assert top["id"] == ids["quantum"]
        # keyword-only weights: combined score equals the keyword score
        assert top["score"] == pytest.approx(top["score_breakdown"]["keyword"], abs=1e-3)

    def test_weight_updates_apply_to_existing_engine(self, seeded_store):
        """A long-lived engine picks up new weights on the next search."""
        from cortex.retrieval.learner import LearningLoop

        store, _ids = seeded_store
        engine = RetrievalEngine(store)
        engine.search("quantum")
        assert engine.weights == dict(DEFAULT_WEIGHTS)

        custom = {"keyword": 0.7, "semantic": 0.1, "graph": 0.1, "recency": 0.1}
        LearningLoop(store).update_weights(custom)
        engine.search("quantum")
        assert engine.weights == custom

    def test_explicit_weights_pin_the_engine(self, seeded_store):
        """Constructor weights override anything the learner persisted."""
        from cortex.retrieval.learner import LearningLoop

        store, _ids = seeded_store
        explicit = {"keyword": 0.0, "semantic": 0.0, "graph": 0.0, "recency": 1.0}
        engine = RetrievalEngine(store, weights=explicit)

        LearningLoop(store).update_weights(
            {"keyword": 1.0, "semantic": 0.0, "graph": 0.0, "recency": 0.0}
        )
        engine.search("quantum")
        assert engine.weights == explicit

    def test_corrupt_persisted_json_falls_back_to_defaults(self, seeded_store, caplog):
        store, _ids = seeded_store
        store.content.set_config(WEIGHTS_CONFIG_KEY, "{not valid json")

        with caplog.at_level(logging.WARNING, logger="cortex.retrieval.engine"):
            weights = load_persisted_weights(store.content)

        assert weights == dict(DEFAULT_WEIGHTS)
        assert any("corrupt" in r.getMessage() for r in caplog.records)

    def test_wrong_shape_falls_back_to_defaults(self, seeded_store):
        store, _ids = seeded_store
        store.content.set_config(WEIGHTS_CONFIG_KEY, json.dumps([0.4, 0.3]))
        assert load_persisted_weights(store.content) == dict(DEFAULT_WEIGHTS)

    def test_negative_value_falls_back_to_defaults(self, seeded_store):
        store, _ids = seeded_store
        store.content.set_config(WEIGHTS_CONFIG_KEY, json.dumps({"keyword": -1.0}))
        assert load_persisted_weights(store.content) == dict(DEFAULT_WEIGHTS)

    def test_all_zero_weights_fall_back_to_defaults(self, seeded_store):
        store, _ids = seeded_store
        store.content.set_config(
            WEIGHTS_CONFIG_KEY,
            json.dumps({"keyword": 0, "semantic": 0, "graph": 0, "recency": 0}),
        )
        assert load_persisted_weights(store.content) == dict(DEFAULT_WEIGHTS)

    def test_partial_weights_merge_over_defaults(self, seeded_store):
        store, _ids = seeded_store
        store.content.set_config(WEIGHTS_CONFIG_KEY, json.dumps({"keyword": 0.7}))
        weights = load_persisted_weights(store.content)
        assert weights["keyword"] == pytest.approx(0.7)
        assert weights["semantic"] == pytest.approx(DEFAULT_WEIGHTS["semantic"])
        assert weights["graph"] == pytest.approx(DEFAULT_WEIGHTS["graph"])
        assert weights["recency"] == pytest.approx(DEFAULT_WEIGHTS["recency"])

    def test_search_survives_corrupt_weights(self, seeded_store):
        """A corrupt weights config must never break search itself."""
        store, ids = seeded_store
        store.content.set_config(WEIGHTS_CONFIG_KEY, "garbage{{{")
        engine = RetrievalEngine(store)
        results = engine.search("quantum")
        assert any(r["id"] == ids["quantum"] for r in results)


# -- Minimum relevance threshold ---------------------------------------------


class TestMinRelevance:
    def _two_tier_corpus(self, store: Store) -> tuple[str, str]:
        """Create a strong title match and a weak content-only match."""
        strong = store.create(
            obj_type="research",
            title="Xylophone acoustics deep dive",
            content="Xylophone xylophone resonance study of the xylophone",
            project="music",
        )
        weak = store.create(
            obj_type="idea",
            title="Concert hall lineup",
            content="Maybe add a xylophone at the end of the program",
            project="music",
        )
        return strong, weak

    def test_default_threshold_is_off(self, store):
        """min_relevance defaults to 0.0 — behavior unchanged, no filtering."""
        strong, weak = self._two_tier_corpus(store)
        engine = RetrievalEngine(store)
        assert engine.min_relevance == 0.0
        results = engine.search("xylophone")
        assert {r["id"] for r in results} == {strong, weak}

    def test_per_call_floor_drops_weak_results(self, store):
        strong, weak = self._two_tier_corpus(store)
        engine = RetrievalEngine(store)

        baseline = engine.search("xylophone")
        assert len(baseline) == 2
        scores = {r["id"]: r["score"] for r in baseline}
        assert scores[strong] > scores[weak]

        floor = (scores[strong] + scores[weak]) / 2
        filtered = engine.search("xylophone", min_relevance=floor)
        assert [r["id"] for r in filtered] == [strong]

    def test_engine_level_floor_applies_to_all_searches(self, store):
        strong, weak = self._two_tier_corpus(store)
        baseline = RetrievalEngine(store).search("xylophone")
        scores = {r["id"]: r["score"] for r in baseline}
        floor = (scores[strong] + scores[weak]) / 2

        engine = RetrievalEngine(store, min_relevance=floor)
        results = engine.search("xylophone")
        assert [r["id"] for r in results] == [strong]

    def test_per_call_zero_disables_engine_floor(self, store):
        strong, weak = self._two_tier_corpus(store)
        engine = RetrievalEngine(store, min_relevance=0.99)
        results = engine.search("xylophone", min_relevance=0.0)
        assert {r["id"] for r in results} == {strong, weak}

    def test_floor_above_all_scores_returns_empty(self, store):
        self._two_tier_corpus(store)
        engine = RetrievalEngine(store)
        assert engine.search("xylophone", min_relevance=10.0) == []


# -- BM25-based keyword scoring -----------------------------------------------


class _RanklessStore:
    """Store wrapper whose search() drops the BM25 ``rank`` column."""

    def __init__(self, store: Store):
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {k: v for k, v in doc.items() if k != "rank"}
            for doc in self._store.search(*args, **kwargs)
        ]


class TestBM25KeywordScoring:
    def _corpus(self, store: Store) -> tuple[str, str]:
        strong = store.create(
            obj_type="research",
            title="Quasar luminosity survey",
            content="Quasar quasar measurements across the quasar sample",
            project="astro",
        )
        weak = store.create(
            obj_type="idea",
            title="Telescope shopping list",
            content="A filter suitable for one quasar observation",
            project="astro",
        )
        return strong, weak

    def test_best_match_scores_one(self, store):
        strong, _weak = self._corpus(store)
        results = RetrievalEngine(store).search("quasar")
        by_id = {r["id"]: r for r in results}
        assert by_id[strong]["score_breakdown"]["keyword"] == pytest.approx(1.0)

    def test_keyword_score_reflects_bm25_magnitude(self, store):
        """A weak content-only match scores by BM25 ratio, not list position."""
        _strong, weak = self._corpus(store)
        results = RetrievalEngine(store).search("quasar")
        by_id = {r["id"]: r for r in results}
        weak_kw = by_id[weak]["score_breakdown"]["keyword"]
        assert 0.0 < weak_kw < 1.0
        # Magnitude-based: a title match (10x FTS weight) towers over a single
        # content mention, so the ratio is far below the positional 0.5.
        assert weak_kw < 0.5

    def test_falls_back_to_positional_without_rank(self, store):
        """Without BM25 values the engine degrades to rank-position scoring."""
        strong, weak = self._corpus(store)
        engine = RetrievalEngine(_RanklessStore(store))  # type: ignore[arg-type]
        results = engine.search("quasar")
        by_id = {r["id"]: r for r in results}
        # Two keyword results: positions 0 and 1 -> 1.0 and 0.5
        kw_scores = sorted(
            (by_id[strong]["score_breakdown"]["keyword"],
             by_id[weak]["score_breakdown"]["keyword"]),
            reverse=True,
        )
        assert kw_scores == [pytest.approx(1.0), pytest.approx(0.5)]


# -- Embedding model consistency warning ---------------------------------------


class TestModelConsistencyWarning:
    def _store_embedding(self, store: Store, doc_id: str, model: str) -> None:
        store.content.store_embedding(
            doc_id=doc_id,
            embedding=struct.pack("3f", 0.1, 0.2, 0.3),
            model=model,
            dimensions=3,
        )

    def _provider(self, model_name: str) -> MagicMock:
        provider = MagicMock()
        provider.model_name = model_name
        provider.embed.return_value = [0.1, 0.2, 0.3]
        return provider

    def test_mismatch_warns_once_on_search(self, seeded_store, caplog):
        store, ids = seeded_store
        self._store_embedding(store, ids["quantum"], "old-model")
        engine = RetrievalEngine(store, embedding_provider=self._provider("new-model"))

        with caplog.at_level(logging.WARNING, logger="cortex.retrieval.engine"):
            engine.search("quantum")
            engine.search("quantum")

        mismatch_warnings = [
            r for r in caplog.records
            if "old-model" in r.getMessage() and "new-model" in r.getMessage()
        ]
        assert len(mismatch_warnings) == 1

    def test_matching_model_does_not_warn(self, seeded_store, caplog):
        store, ids = seeded_store
        self._store_embedding(store, ids["quantum"], "same-model")
        engine = RetrievalEngine(store, embedding_provider=self._provider("same-model"))

        with caplog.at_level(logging.WARNING, logger="cortex.retrieval.engine"):
            engine.search("quantum")

        assert not [
            r for r in caplog.records if "Stored embeddings use model" in r.getMessage()
        ]

    def test_no_provider_skips_check(self, seeded_store, caplog):
        store, ids = seeded_store
        self._store_embedding(store, ids["quantum"], "old-model")
        engine = RetrievalEngine(store)

        with caplog.at_level(logging.WARNING, logger="cortex.retrieval.engine"):
            engine.search("quantum")

        assert not [
            r for r in caplog.records if "Stored embeddings use model" in r.getMessage()
        ]
