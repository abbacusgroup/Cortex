"""Information retrieval metrics: R@K, MAP, NDCG, MRR."""

from __future__ import annotations

import math


def recall_at_k(results: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k results."""
    if not relevant:
        return 1.0
    top_k = set(results[:k])
    return len(top_k & relevant) / len(relevant)


def average_precision(results: list[str], relevant: set[str]) -> float:
    """Average precision for a single query."""
    if not relevant:
        return 1.0
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(results):
        if doc_id in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant)


def mean_average_precision(
    all_results: list[list[str]], all_relevant: list[set[str]]
) -> float:
    """MAP across multiple queries."""
    if not all_results:
        return 0.0
    return sum(
        average_precision(r, rel)
        for r, rel in zip(all_results, all_relevant, strict=True)
    ) / len(all_results)


def dcg_at_k(results: list[str], relevant: set[str], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    score = 0.0
    for i, doc_id in enumerate(results[:k]):
        if doc_id in relevant:
            score += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
    return score


def ndcg_at_k(results: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    actual_dcg = dcg_at_k(results, relevant, k)
    # Ideal: all relevant docs at the top
    ideal_results = list(relevant)[:k]
    ideal_dcg = dcg_at_k(ideal_results, relevant, k)
    if ideal_dcg == 0:
        return 1.0
    return actual_dcg / ideal_dcg


def mrr(all_results: list[list[str]], all_relevant: list[set[str]]) -> float:
    """Mean Reciprocal Rank across multiple queries."""
    if not all_results:
        return 0.0
    rr_sum = 0.0
    for results, relevant in zip(all_results, all_relevant, strict=True):
        for i, doc_id in enumerate(results):
            if doc_id in relevant:
                rr_sum += 1.0 / (i + 1)
                break
    return rr_sum / len(all_results)
