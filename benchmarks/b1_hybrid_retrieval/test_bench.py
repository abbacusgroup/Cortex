"""Benchmark: hybrid 4-signal retrieval quality.

Runs 50 queries under 7 weight configurations and compares IR metrics
(R@5, R@10, MAP, NDCG@10, MRR). Asserts that the balanced hybrid
config beats single-signal baselines on MAP.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from cortex.retrieval.engine import RetrievalEngine

from benchmarks.corpus.embeddings import SyntheticEmbeddingProvider
from benchmarks.b1_hybrid_retrieval.queries import QUERIES
from benchmarks.metrics.retrieval import (
    recall_at_k,
    mean_average_precision,
    ndcg_at_k,
    mrr,
    percentiles,
)

# ── Weight configurations ──────────────────────────────────────────

CONFIGS: dict[str, dict[str, float]] = {
    "hybrid": {
        "keyword": 0.4,
        "semantic": 0.3,
        "graph": 0.2,
        "recency": 0.1,
    },
    "keyword_only": {
        "keyword": 1.0,
        "semantic": 0.0,
        "graph": 0.0,
        "recency": 0.0,
    },
    "semantic_only": {
        "keyword": 0.0,
        "semantic": 1.0,
        "graph": 0.0,
        "recency": 0.0,
    },
    "graph_only": {
        "keyword": 0.0,
        "semantic": 0.0,
        "graph": 1.0,
        "recency": 0.0,
    },
    "recency_only": {
        "keyword": 0.0,
        "semantic": 0.0,
        "graph": 0.0,
        "recency": 1.0,
    },
    "no_graph": {
        "keyword": 0.5,
        "semantic": 0.4,
        "graph": 0.0,
        "recency": 0.1,
    },
    "no_semantic": {
        "keyword": 0.6,
        "semantic": 0.0,
        "graph": 0.3,
        "recency": 0.1,
    },
}


# ── Helpers ────────────────────────────────────────────────────────

def _run_queries(
    engine: RetrievalEngine,
    id_to_label: dict[str, str],
) -> dict[str, list[str]]:
    """Run all benchmark queries and return label-based result lists.

    Returns:
        Mapping from query text to ordered list of result labels.
    """
    results: dict[str, list[str]] = {}
    for q in QUERIES:
        hits = engine.search(q["query"], limit=10)
        labels = [id_to_label.get(h["id"], "") for h in hits]
        results[q["query"]] = labels
    return results


def _compute_metrics(
    results: dict[str, list[str]],
) -> dict[str, float]:
    """Compute aggregate IR metrics over all 50 queries."""
    all_results: list[list[str]] = []
    all_relevant: list[set[str]] = []

    for q in QUERIES:
        result_labels = results.get(q["query"], [])
        all_results.append(result_labels)
        all_relevant.append(q["relevant"])

    return {
        "R@5": sum(
            recall_at_k(r, rel, 5)
            for r, rel in zip(all_results, all_relevant, strict=True)
        ) / len(QUERIES),
        "R@10": sum(
            recall_at_k(r, rel, 10)
            for r, rel in zip(all_results, all_relevant, strict=True)
        ) / len(QUERIES),
        "MAP": mean_average_precision(all_results, all_relevant),
        "NDCG@10": sum(
            ndcg_at_k(r, rel, 10)
            for r, rel in zip(all_results, all_relevant, strict=True)
        ) / len(QUERIES),
        "MRR": mrr(all_results, all_relevant),
    }


def _print_table(
    all_metrics: dict[str, dict[str, float]],
) -> None:
    """Print a formatted comparison table to stdout."""
    metric_names = ["R@5", "R@10", "MAP", "NDCG@10", "MRR"]
    header = f"{'Config':<16}" + "".join(f"{m:>10}" for m in metric_names)

    separator = "-" * len(header)
    print()
    print(separator)
    print("  B1 Hybrid Retrieval Benchmark Results")
    print(separator)
    print(header)
    print(separator)

    for config_name, metrics in all_metrics.items():
        row = f"{config_name:<16}"
        for m in metric_names:
            row += f"{metrics[m]:>10.4f}"
        print(row)

    print(separator)
    print()


# ── Benchmark test ─────────────────────────────────────────────────

@pytest.mark.bench
def test_hybrid_retrieval_bench(
    corpus: tuple[Any, dict[str, str]],
    store: Any,
) -> None:
    """Benchmark hybrid retrieval across 7 weight configurations.

    Asserts that the balanced hybrid config outperforms keyword-only
    and semantic-only baselines on MAP.
    """
    _gen, label_to_id = corpus
    # Reverse mapping: object ID -> label
    id_to_label: dict[str, str] = {v: k for k, v in label_to_id.items()}

    embedding_provider = SyntheticEmbeddingProvider()

    all_metrics: dict[str, dict[str, float]] = {}

    for config_name, weights in CONFIGS.items():
        engine = RetrievalEngine(
            store=store,
            weights=weights,
            embedding_provider=embedding_provider,
        )
        results = _run_queries(engine, id_to_label)
        metrics = _compute_metrics(results)
        all_metrics[config_name] = metrics

    # Print results table (visible with pytest -s)
    _print_table(all_metrics)

    # Core assertions: hybrid beats single-signal baselines on MAP
    hybrid_map = all_metrics["hybrid"]["MAP"]
    keyword_map = all_metrics["keyword_only"]["MAP"]
    semantic_map = all_metrics["semantic_only"]["MAP"]

    assert hybrid_map > keyword_map, (
        f"hybrid MAP ({hybrid_map:.4f}) should exceed "
        f"keyword_only MAP ({keyword_map:.4f})"
    )
    assert hybrid_map > semantic_map, (
        f"hybrid MAP ({hybrid_map:.4f}) should exceed "
        f"semantic_only MAP ({semantic_map:.4f})"
    )
