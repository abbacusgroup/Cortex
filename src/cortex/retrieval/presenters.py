"""Presentation modes — five ways to serve knowledge.

Each mode shapes the same underlying data into a different output format:
- Briefing: summaries only (~100 tokens/doc)
- Dossier: entity/topic-centric intelligence brief
- Document: full content + metadata + relationships
- Synthesis: cross-document narrative over a time period
- Alert: proactive notifications (contradictions, patterns, staleness)
"""

from __future__ import annotations

from typing import Any

from cortex.core.logging import get_logger
from cortex.db.store import Store
from cortex.services.llm import LLMClient

logger = get_logger("retrieval.presenters")


class BriefingPresenter:
    """Summaries + metadata only. Token-efficient for agent context."""

    def render(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Render documents as brief summaries.

        Returns:
            List of dicts with: id, title, type, tags, project, summary, tier.
        """
        results = []
        for doc in documents:
            results.append({
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "type": doc.get("type", ""),
                "tags": doc.get("tags", ""),
                "project": doc.get("project", ""),
                "summary": doc.get("summary", "") or doc.get("title", ""),
                "tier": doc.get("tier", ""),
                "score": doc.get("score"),
            })
        return results


class DossierPresenter:
    """Entity/topic-centric intelligence brief."""

    def __init__(self, store: Store, llm: LLMClient | None = None):
        self.store = store
        self.llm = llm

    def render(self, topic: str) -> dict[str, Any]:
        """Build a dossier around a topic or entity.

        Collects:
        - Key facts (latest state)
        - Recent related objects
        - Contradictions
        - Related entities (1-2 hops)
        - Timeline
        """
        # Find matching entity
        entities = self.store.graph.list_entities()
        entity_match = None
        for e in entities:
            if e["name"].lower() == topic.lower():
                entity_match = e
                break

        # Gather related objects
        related_objects: list[dict[str, Any]] = []
        if entity_match:
            mention_ids = self.store.graph.get_entity_mentions(entity_match["id"])
            for mid in mention_ids:
                doc = self.store.content.get(mid)
                if doc:
                    related_objects.append(doc)
        else:
            # Fall back to text search
            related_objects = self.store.search(topic, limit=20)

        if not related_objects:
            return {
                "topic": topic,
                "entity": entity_match,
                "status": "no_knowledge_found",
                "objects": [],
                "contradictions": [],
                "related_entities": [],
                "timeline": [],
            }

        # Sort by creation date
        related_objects.sort(
            key=lambda x: x.get("created_at", ""), reverse=True
        )

        # Find contradictions
        contradictions = self._find_contradictions(related_objects)

        # Find related entities (entities mentioned in the same objects)
        related_entities = self._find_related_entities(related_objects)

        # Build timeline
        timeline = [
            {
                "id": obj.get("id", ""),
                "title": obj.get("title", ""),
                "type": obj.get("type", ""),
                "created_at": obj.get("created_at", ""),
            }
            for obj in related_objects
        ]

        return {
            "topic": topic,
            "entity": entity_match,
            "status": "ok",
            "object_count": len(related_objects),
            "objects": [
                {
                    "id": obj.get("id", ""),
                    "title": obj.get("title", ""),
                    "type": obj.get("type", ""),
                    "summary": obj.get("summary", "") or obj.get("title", ""),
                    "project": obj.get("project", ""),
                }
                for obj in related_objects[:10]
            ],
            "contradictions": contradictions,
            "related_entities": related_entities,
            "timeline": timeline,
        }

    def _find_contradictions(
        self, objects: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find contradiction relationships among the given objects."""
        contradictions = []
        obj_ids = {obj.get("id") for obj in objects if obj.get("id")}

        for obj in objects:
            obj_id = obj.get("id", "")
            if not obj_id:
                continue
            rels = self.store.get_relationships(obj_id)
            for rel in rels:
                if rel["rel_type"] == "contradicts" and rel["other_id"] in obj_ids:
                    contradictions.append({
                        "object_a": obj_id,
                        "object_b": rel["other_id"],
                        "title_a": obj.get("title", ""),
                    })

        return contradictions

    def _find_related_entities(
        self, objects: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Find entities mentioned across the related objects."""
        obj_ids = {obj.get("id", "") for obj in objects if obj.get("id")}
        all_entities = self.store.graph.list_entities()
        entity_ids: set[str] = set()
        for entity in all_entities:
            mentions = self.store.graph.get_entity_mentions(entity["id"])
            if obj_ids & set(mentions):
                entity_ids.add(entity["id"])
        return [e for e in all_entities if e["id"] in entity_ids]


class DocumentPresenter:
    """Full content + metadata + relationships."""

    def __init__(self, store: Store):
        self.store = store

    def render(self, obj_id: str) -> dict[str, Any] | None:
        """Render a single document with full detail."""
        doc = self.store.read(obj_id)
        if doc is None:
            return None

        # Add entity mentions via dedicated mention index
        all_entities = self.store.graph.list_entities()
        entities = [
            e for e in all_entities
            if obj_id in self.store.graph.get_entity_mentions(e["id"])
        ]
        doc["entities"] = entities
        return doc


class SynthesisPresenter:
    """Cross-document narrative over a time period."""

    def __init__(self, store: Store, llm: LLMClient | None = None):
        self.store = store
        self.llm = llm

    def render(
        self,
        *,
        period_days: int = 7,
        project: str | None = None,
    ) -> dict[str, Any]:
        """Generate a synthesis over the given time period.

        Returns:
            Dict with period, themes, source objects, and narrative.
        """
        import datetime

        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
            days=period_days
        )
        cutoff_str = cutoff.isoformat()

        # Get recent objects
        all_objects = self.store.list_objects(
            obj_type=None, project=project, limit=500
        )
        recent = [
            obj for obj in all_objects
            if obj.get("created_at", "") >= cutoff_str
        ]

        if not recent:
            return {
                "period_days": period_days,
                "project": project,
                "status": "nothing_to_synthesize",
                "object_count": 0,
                "themes": [],
                "sources": [],
                "narrative": "",
            }

        # Group by type as simple theme
        themes: dict[str, int] = {}
        for obj in recent:
            t = obj.get("type", "unknown")
            themes[t] = themes.get(t, 0) + 1

        sources = [
            {"id": obj.get("id", ""), "title": obj.get("title", ""), "type": obj.get("type", "")}
            for obj in recent[:30]
        ]

        # Build narrative (LLM-enhanced if available, otherwise summary)
        narrative = self._build_narrative(recent, themes)

        return {
            "period_days": period_days,
            "project": project,
            "status": "ok",
            "object_count": len(recent),
            "themes": [
                {"name": k, "count": v} for k, v in sorted(
                    themes.items(), key=lambda x: x[1], reverse=True
                )
            ],
            "sources": sources,
            "narrative": narrative,
        }

    def _build_narrative(
        self, objects: list[dict[str, Any]], themes: dict[str, int]
    ) -> str:
        """Build a narrative summary."""
        if self.llm and self.llm.available:
            summaries = "\n".join(
                f"- [{obj.get('type', '?')}] {obj.get('title', '?')}: "
                f"{obj.get('summary', '')}"
                for obj in objects[:20]
            )
            try:
                return self.llm.complete(
                    f"Synthesize these recent knowledge objects into a "
                    f"brief narrative (3-5 sentences):\n\n{summaries}"
                )
            except Exception:
                pass

        # Fallback: simple summary
        lines = [f"Over the past period, {len(objects)} objects were captured."]
        for theme, count in sorted(themes.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {count} {theme}(s)")
        return "\n".join(lines)


class AlertPresenter:
    """Proactive notifications — contradictions, patterns, staleness."""

    def __init__(self, store: Store):
        self.store = store

    def render(self) -> list[dict[str, Any]]:
        """Generate alerts for current issues.

        Returns:
            List of alert dicts: {type, severity, message, object_ids}.
        """
        alerts: list[dict[str, Any]] = []

        # 1. Contradictions
        alerts.extend(self._check_contradictions())

        # 2. Repeated entity patterns (3+ fixes on same entity in 14 days)
        alerts.extend(self._check_patterns())

        # 3. Staleness
        alerts.extend(self._check_staleness())

        return alerts

    def _check_contradictions(self) -> list[dict[str, Any]]:
        """Find active contradiction relationships."""
        alerts = []
        seen: set[tuple[str, str]] = set()

        all_objects = self.store.list_objects(limit=500)
        for obj in all_objects:
            obj_id = obj.get("id", "")
            rels = self.store.get_relationships(obj_id)
            for rel in rels:
                if rel["rel_type"] == "contradicts":
                    pair = tuple(sorted([obj_id, rel["other_id"]]))
                    if pair not in seen:
                        seen.add(pair)
                        alerts.append({
                            "type": "contradiction",
                            "severity": "high",
                            "message": (
                                f"Contradiction between "
                                f"'{obj.get('title', obj_id[:8])}' "
                                f"and object {rel['other_id'][:8]}"
                            ),
                            "object_ids": list(pair),
                        })
        return alerts

    def _check_patterns(self) -> list[dict[str, Any]]:
        """Detect repeated entity mentions in fixes (systemic issues)."""
        import datetime

        alerts = []
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=14)
        cutoff_str = cutoff.isoformat()

        fixes = self.store.list_objects(obj_type="fix", limit=500)
        recent_fixes = [
            f for f in fixes if f.get("created_at", "") >= cutoff_str
        ]

        # Count entity mentions across recent fixes
        recent_fix_ids = {f.get("id", "") for f in recent_fixes}
        entity_counts: dict[str, list[str]] = {}
        for entity in self.store.graph.list_entities():
            mentions = self.store.graph.get_entity_mentions(entity["id"])
            matching = [m for m in mentions if m in recent_fix_ids]
            if matching:
                entity_counts[entity["id"]] = matching

        for entity_id, fix_ids in entity_counts.items():
            if len(fix_ids) >= 3:
                alerts.append({
                    "type": "pattern",
                    "severity": "medium",
                    "message": (
                        f"Systemic issue: {len(fix_ids)} fixes in 14 days "
                        f"mentioning entity {entity_id[:8]}"
                    ),
                    "object_ids": fix_ids,
                })

        return alerts

    def _check_staleness(self) -> list[dict[str, Any]]:
        """Find objects with stale dependencies."""
        alerts = []
        all_objects = self.store.list_objects(limit=500)

        for obj in all_objects:
            obj_id = obj.get("id", "")
            rels = self.store.get_relationships(obj_id)
            for rel in rels:
                if (
                    rel["direction"] == "outgoing"
                    and rel["rel_type"] == "dependsOn"
                ):
                    dep = self.store.content.get(rel["other_id"])
                    if dep and dep.get("tier") == "archive":
                        alerts.append({
                            "type": "staleness",
                            "severity": "low",
                            "message": (
                                f"'{obj.get('title', obj_id[:8])}' depends on "
                                f"archived object {rel['other_id'][:8]}"
                            ),
                            "object_ids": [obj_id, rel["other_id"]],
                        })
        return alerts
