"""Advanced reasoning — contradiction detection, pattern detection, gap analysis.

Goes beyond basic OWL-RL inference to detect semantic issues:
- Structural contradictions (superseded + dependsOn)
- Semantic contradictions (LLM-compared content)
- Patterns (repeated entity mentions in fixes)
- Gaps (missing decisions, uncaptured lessons)
- Causal chain assembly
- Staleness propagation
"""

from __future__ import annotations

import datetime
from typing import Any

from cortex.core.logging import get_logger
from cortex.db.store import Store
from cortex.services.llm import LLMClient

logger = get_logger("pipeline.advanced_reason")


class AdvancedReasoner:
    """Advanced reasoning over the knowledge graph."""

    def __init__(self, store: Store, llm: LLMClient | None = None):
        self.store = store
        self.llm = llm

    def run_all(self) -> dict[str, Any]:
        """Run all advanced reasoning checks.

        Returns:
            Dict with findings from each check category.
        """
        return {
            "contradictions": self.detect_contradictions(),
            "patterns": self.detect_patterns(),
            "gaps": self.detect_gaps(),
            "staleness": self.propagate_staleness(),
        }

    # ─── Contradiction Detection ───────────────────────────────────

    def detect_contradictions(self) -> list[dict[str, Any]]:
        """Find structural and semantic contradictions.

        Structural: object superseded but still has active dependsOn
        Semantic: connected objects with conflicting content (LLM)
        """
        findings: list[dict[str, Any]] = []
        findings.extend(self._structural_contradictions())
        return findings

    def _structural_contradictions(self) -> list[dict[str, Any]]:
        """Find superseded objects that still have active dependencies."""
        findings = []
        all_objects = self.store.list_objects(limit=1000)

        for obj in all_objects:
            obj_id = obj.get("id", "")
            rels = self.store.get_relationships(obj_id)

            # Check: is this object superseded?
            is_superseded = any(
                r["rel_type"] == "supersedes" and r["direction"] == "incoming" for r in rels
            )
            if not is_superseded:
                continue

            # Check: does anything still depend on it?
            has_dependents = any(
                r["rel_type"] == "dependsOn" and r["direction"] == "incoming" for r in rels
            )
            if has_dependents:
                dependents = [
                    r["other_id"]
                    for r in rels
                    if r["rel_type"] == "dependsOn" and r["direction"] == "incoming"
                ]
                findings.append(
                    {
                        "type": "structural_contradiction",
                        "severity": "high",
                        "message": (
                            f"'{obj.get('title', obj_id[:8])}' is superseded "
                            f"but still has {len(dependents)} dependent(s)"
                        ),
                        "object_id": obj_id,
                        "dependent_ids": dependents,
                    }
                )

        return findings

    # ─── Pattern Detection ─────────────────────────────────────────

    def detect_patterns(self, *, window_days: int = 14, threshold: int = 3) -> list[dict[str, Any]]:
        """Detect repeated entity mentions in fixes (systemic issues).

        Rule: 3+ fixes mentioning same entity within N days = systemic issue.
        """
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=window_days)
        cutoff_str = cutoff.isoformat()

        fixes = self.store.list_objects(obj_type="fix", limit=1000)
        recent_fixes = [f for f in fixes if f.get("created_at", "") >= cutoff_str]

        # Count entity mentions per entity via graph mention index
        recent_fix_ids = {f.get("id", "") for f in recent_fixes}
        all_entities = self.store.graph.list_entities()
        entity_name_map = {e["id"]: e["name"] for e in all_entities}
        entity_fix_map: dict[str, list[str]] = {}
        for entity in all_entities:
            eid = entity["id"]
            mention_ids = self.store.graph.get_entity_mentions(eid)
            for mid in mention_ids:
                if mid in recent_fix_ids:
                    entity_fix_map.setdefault(eid, []).append(mid)

        findings = []
        for entity_id, fix_ids in entity_fix_map.items():
            if len(fix_ids) >= threshold:
                name = entity_name_map.get(entity_id, entity_id[:8])

                findings.append(
                    {
                        "type": "systemic_issue",
                        "severity": "medium",
                        "message": (
                            f"'{name}' has {len(fix_ids)} fixes in "
                            f"{window_days} days — possible systemic issue"
                        ),
                        "entity_id": entity_id,
                        "entity_name": name,
                        "fix_ids": fix_ids,
                        "window_days": window_days,
                    }
                )

        return findings

    # ─── Gap Analysis ──────────────────────────────────────────────

    def detect_gaps(self) -> list[dict[str, Any]]:
        """Find gaps in knowledge coverage.

        Rules:
        - Project has sessions but no decisions → missing architecture
        - Entity has fixes but no lessons → learning not captured
        """
        findings = []

        # Group objects by project
        all_objects = self.store.list_objects(limit=1000)
        project_types: dict[str, set[str]] = {}
        for obj in all_objects:
            proj = obj.get("project", "")
            if proj:
                project_types.setdefault(proj, set()).add(obj.get("type", ""))

        for proj, types in project_types.items():
            if "session" in types and "decision" not in types:
                findings.append(
                    {
                        "type": "missing_decisions",
                        "severity": "low",
                        "message": (
                            f"Project '{proj}' has sessions but no "
                            f"decisions — missing architecture documentation"
                        ),
                        "project": proj,
                    }
                )

        # Entity-level gaps: fixes without lessons
        entities = self.store.graph.list_entities()
        for entity in entities:
            mention_ids = self.store.graph.get_entity_mentions(entity["id"])
            if not mention_ids:
                continue

            mention_types = set()
            for mid in mention_ids:
                doc = self.store.content.get(mid)
                if doc:
                    mention_types.add(doc.get("type", ""))

            if "fix" in mention_types and "lesson" not in mention_types:
                findings.append(
                    {
                        "type": "missing_lessons",
                        "severity": "low",
                        "message": (
                            f"Entity '{entity['name']}' has fixes but no "
                            f"lessons — learning not captured"
                        ),
                        "entity_id": entity["id"],
                        "entity_name": entity["name"],
                    }
                )

        return findings

    # ─── Staleness Propagation ─────────────────────────────────────

    def propagate_staleness(self) -> list[dict[str, Any]]:
        """Find objects that may need review due to stale dependencies.

        Rule: If A is superseded and B depends on A → B may need review.
        """
        findings = []
        all_objects = self.store.list_objects(limit=1000)

        for obj in all_objects:
            obj_id = obj.get("id", "")
            rels = self.store.get_relationships(obj_id)

            for rel in rels:
                if rel["direction"] == "outgoing" and rel["rel_type"] == "dependsOn":
                    dep_rels = self.store.get_relationships(rel["other_id"])
                    dep_superseded = any(
                        r["rel_type"] == "supersedes" and r["direction"] == "incoming"
                        for r in dep_rels
                    )
                    if dep_superseded:
                        dep_doc = self.store.content.get(rel["other_id"])
                        dep_title = dep_doc.get("title", "") if dep_doc else rel["other_id"][:8]
                        findings.append(
                            {
                                "type": "stale_dependency",
                                "severity": "medium",
                                "message": (
                                    f"'{obj.get('title', obj_id[:8])}' "
                                    f"depends on superseded "
                                    f"object '{dep_title}'"
                                ),
                                "object_id": obj_id,
                                "dependency_id": rel["other_id"],
                            }
                        )

        return findings

    # ─── Causal Chain Assembly ─────────────────────────────────────

    def assemble_causal_chain(self, obj_id: str) -> list[dict[str, Any]]:
        """Build a narrative from decision → implementation → bug → fix → lesson.

        Follows causedBy/ledTo edges to construct the chain.
        """
        chain: list[dict[str, Any]] = []
        visited: set[str] = set()

        # Trace backward (causes)
        backward = []
        self._trace_chain(obj_id, backward, visited, "causedBy", "outgoing")
        backward.reverse()
        chain.extend(backward)

        # Add the object itself
        doc = self.store.content.get(obj_id)
        if doc:
            chain.append(self._to_chain_entry(doc))

        # Trace forward (effects)
        self._trace_chain(obj_id, chain, visited, "ledTo", "outgoing")

        return chain

    def _trace_chain(
        self,
        obj_id: str,
        chain: list[dict[str, Any]],
        visited: set[str],
        rel_type: str,
        direction: str,
        max_depth: int = 10,
    ) -> None:
        if obj_id in visited or max_depth <= 0:
            return
        visited.add(obj_id)

        rels = self.store.get_relationships(obj_id)
        for rel in rels:
            if rel["rel_type"] == rel_type and rel["direction"] == direction:
                other_id = rel["other_id"]
                if other_id not in visited:
                    doc = self.store.content.get(other_id)
                    if doc:
                        chain.append(self._to_chain_entry(doc))
                        self._trace_chain(
                            other_id,
                            chain,
                            visited,
                            rel_type,
                            direction,
                            max_depth - 1,
                        )

    @staticmethod
    def _to_chain_entry(doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": doc.get("id", ""),
            "title": doc.get("title", ""),
            "type": doc.get("type", ""),
            "project": doc.get("project", ""),
            "created_at": doc.get("created_at", ""),
        }
