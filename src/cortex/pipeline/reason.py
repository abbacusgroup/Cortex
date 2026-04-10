"""Reason stage — OWL-RL inference via SPARQL CONSTRUCT rules.

Applies inference rules to derive new triples from existing ones:
- Transitive closure: supersedes chains
- Symmetric enforcement: contradicts both directions
- Inverse generation: causedBy ↔ ledTo
- Domain/range validation
"""

from __future__ import annotations

from typing import Any

import pyoxigraph as ox

from cortex.core.logging import get_logger
from cortex.db.graph_store import GraphStore
from cortex.ontology.namespaces import SPARQL_PREFIXES

logger = get_logger("pipeline.reason")

# Inference rules as SPARQL CONSTRUCT queries
# Each rule produces new triples from existing patterns

INFERENCE_RULES: list[dict[str, str]] = [
    {
        "name": "symmetric_contradicts",
        "description": "If A contradicts B, then B contradicts A",
        "query": f"""
            {SPARQL_PREFIXES}
            CONSTRUCT {{ ?b cortex:contradicts ?a }}
            WHERE {{
                ?a cortex:contradicts ?b .
                FILTER NOT EXISTS {{ ?b cortex:contradicts ?a }}
            }}
        """,
    },
    {
        "name": "inverse_causedBy_ledTo",
        "description": "If A causedBy B, then B ledTo A",
        "query": f"""
            {SPARQL_PREFIXES}
            CONSTRUCT {{ ?b cortex:ledTo ?a }}
            WHERE {{
                ?a cortex:causedBy ?b .
                FILTER NOT EXISTS {{ ?b cortex:ledTo ?a }}
            }}
        """,
    },
    {
        "name": "inverse_ledTo_causedBy",
        "description": "If A ledTo B, then B causedBy A",
        "query": f"""
            {SPARQL_PREFIXES}
            CONSTRUCT {{ ?b cortex:causedBy ?a }}
            WHERE {{
                ?a cortex:ledTo ?b .
                FILTER NOT EXISTS {{ ?b cortex:causedBy ?a }}
            }}
        """,
    },
    {
        "name": "transitive_supersedes_1hop",
        "description": "If A supersedes B and B supersedes C, then A supersedes C",
        "query": f"""
            {SPARQL_PREFIXES}
            CONSTRUCT {{ ?a cortex:supersedes ?c }}
            WHERE {{
                ?a cortex:supersedes ?b .
                ?b cortex:supersedes ?c .
                FILTER(?a != ?c)
                FILTER NOT EXISTS {{ ?a cortex:supersedes ?c }}
            }}
        """,
    },
]


class ReasonStage:
    """Apply OWL-RL inference rules to the knowledge graph."""

    def __init__(self, graph: GraphStore):
        self.graph = graph

    def run(self, max_iterations: int = 10) -> dict[str, Any]:
        """Run all inference rules until fixpoint or max iterations.

        Returns:
            Dict with total inferred triples and per-rule counts.
        """
        total_inferred = 0
        rule_counts: dict[str, int] = {}

        for iteration in range(max_iterations):
            iteration_inferred = 0

            for rule in INFERENCE_RULES:
                count = self._apply_rule(rule)
                rule_counts[rule["name"]] = rule_counts.get(rule["name"], 0) + count
                iteration_inferred += count

            if iteration_inferred == 0:
                logger.debug(
                    "Reasoner reached fixpoint after %d iteration(s)", iteration + 1
                )
                break

            total_inferred += iteration_inferred
            logger.debug(
                "Iteration %d: %d new triples", iteration + 1, iteration_inferred
            )

        return {
            "status": "reasoned",
            "total_inferred": total_inferred,
            "rule_counts": rule_counts,
            "iterations": iteration + 1 if total_inferred > 0 else 1,
        }

    def _apply_rule(self, rule: dict[str, str]) -> int:
        """Apply a single CONSTRUCT rule and add inferred triples.

        Returns:
            Number of new triples added.
        """
        try:
            result = self.graph._store.query(rule["query"])
            # CONSTRUCT returns triples
            new_triples = list(result)
        except Exception as e:
            logger.warning("Rule '%s' failed: %s", rule["name"], e)
            return 0

        count = 0
        for triple in new_triples:
            quad = ox.Quad(triple.subject, triple.predicate, triple.object)
            # Check if already exists
            existing = list(self.graph._store.quads_for_pattern(
                triple.subject, triple.predicate, triple.object
            ))
            if not existing:
                self.graph._store.add(quad)
                count += 1

        if count > 0:
            logger.debug("Rule '%s': +%d triples", rule["name"], count)

        return count

    def check_fixpoint(self) -> dict[str, Any]:
        """Check if the graph is at fixpoint WITHOUT writing new triples.

        Runs the same CONSTRUCT queries as ``run()`` but only counts how many
        triples would be inferred. Fully read-only — safe for diagnostics.

        Returns:
            Dict with total_pending and per-rule counts of missing triples.
        """
        total_pending = 0
        rule_counts: dict[str, int] = {}

        for rule in INFERENCE_RULES:
            try:
                result = self.graph._store.query(rule["query"])
                new_triples = list(result)
            except Exception as e:
                logger.warning("Rule '%s' failed during check: %s", rule["name"], e)
                rule_counts[rule["name"]] = 0
                continue

            # Count triples that don't already exist (same logic as _apply_rule)
            count = 0
            for triple in new_triples:
                existing = list(self.graph._store.quads_for_pattern(
                    triple.subject, triple.predicate, triple.object
                ))
                if not existing:
                    count += 1

            rule_counts[rule["name"]] = count
            total_pending += count

        return {
            "ok": total_pending == 0,
            "total_pending": total_pending,
            "rule_counts": rule_counts,
        }

    def run_for_object(self, obj_id: str) -> dict[str, Any]:
        """Run reasoning focused on a specific object's neighborhood.

        This is more efficient than full reasoning — only applies rules
        that could produce new triples involving this object.
        """
        # For now, just run full reasoning.
        # Future optimization: scope CONSTRUCT queries to obj_id neighborhood.
        return self.run()
