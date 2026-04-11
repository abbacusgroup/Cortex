"""Shared benchmark fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.core.config import CortexConfig
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.retrieval.engine import RetrievalEngine

from benchmarks.corpus.embeddings import SyntheticEmbeddingProvider
from benchmarks.corpus.generator import CorpusGenerator

ONTOLOGY_PATH = find_ontology()


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    """Initialized Store backed by tmp_path."""
    cfg = CortexConfig(data_dir=tmp_path)
    s = Store(cfg)
    s.initialize(ONTOLOGY_PATH)
    return s


@pytest.fixture()
def corpus(store: Store) -> tuple[CorpusGenerator, dict[str, str]]:
    """Fully populated corpus with deterministic data.

    Returns (generator, label_to_id_mapping).
    """
    gen = CorpusGenerator(store, seed=42)
    ids = gen.generate()
    return gen, ids


@pytest.fixture()
def engine(store: Store) -> RetrievalEngine:
    """Hybrid retrieval engine with synthetic embedding provider."""
    return RetrievalEngine(
        store=store,
        embedding_provider=SyntheticEmbeddingProvider(),
    )
