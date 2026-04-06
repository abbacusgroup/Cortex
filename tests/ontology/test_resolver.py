"""Tests for cortex.ontology.resolver (centralized ontology path resolution)."""

from __future__ import annotations

from cortex.db.graph_store import GraphStore
from cortex.ontology.resolver import find_ontology


def test_find_ontology_returns_path():
    """find_ontology() returns a Path that exists on disk."""
    path = find_ontology()
    assert path.exists()


def test_find_ontology_is_ttl():
    """find_ontology() returns a file named cortex.ttl."""
    path = find_ontology()
    assert path.name == "cortex.ttl"


def test_ontology_loads_into_store():
    """The resolved ontology loads into a GraphStore with >0 triples."""
    path = find_ontology()
    store = GraphStore(path=None)  # in-memory
    triples = store.load_ontology(path)
    assert triples > 0
