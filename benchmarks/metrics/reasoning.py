"""Reasoning metrics: completeness, convergence."""

from __future__ import annotations


def completeness_ratio(actual: int, expected: int) -> float:
    """Fraction of expected inferred triples that were actually produced."""
    if expected == 0:
        return 1.0
    return actual / expected


def convergence_report(result: dict) -> dict[str, object]:
    """Extract convergence stats from a ReasonStage result."""
    return {
        "total_inferred": result.get("total_inferred", 0),
        "iterations": result.get("iterations", 0),
        "rule_counts": result.get("rule_counts", {}),
    }
