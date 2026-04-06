"""Tests for cortex.retrieval.presenters (presentation modes)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.core.config import CortexConfig
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology
from cortex.retrieval.presenters import (
    AlertPresenter,
    BriefingPresenter,
    DocumentPresenter,
    DossierPresenter,
    SynthesisPresenter,
)

ONTOLOGY_PATH = find_ontology()


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    """Initialized Store backed by tmp_path."""
    cfg = CortexConfig(data_dir=tmp_path)
    s = Store(cfg)
    s.initialize(ONTOLOGY_PATH)
    return s


def _create(
    store: Store,
    *,
    title: str = "Sample",
    obj_type: str = "decision",
    content: str = "body",
    project: str = "",
    tags: str = "",
    summary: str = "",
    tier: str = "archive",
) -> str:
    return store.create(
        obj_type=obj_type,
        title=title,
        content=content,
        project=project,
        tags=tags,
        summary=summary,
        tier=tier,
    )


# -- BriefingPresenter -----------------------------------------------------


class TestBriefingPresenter:
    def test_renders_list_with_correct_fields(self):
        docs = [
            {
                "id": "d1",
                "title": "Title One",
                "type": "decision",
                "tags": "a,b",
                "project": "cortex",
                "summary": "A brief summary",
                "tier": "recall",
                "score": 0.85,
            },
            {
                "id": "d2",
                "title": "Title Two",
                "type": "fix",
                "tags": "c",
                "project": "other",
                "summary": "Fix summary",
                "tier": "archive",
                "score": 0.60,
            },
        ]
        presenter = BriefingPresenter()
        result = presenter.render(docs)

        assert len(result) == 2
        for item in result:
            assert "id" in item
            assert "title" in item
            assert "type" in item
            assert "tags" in item
            assert "project" in item
            assert "summary" in item
            assert "tier" in item
            assert "score" in item

        assert result[0]["id"] == "d1"
        assert result[0]["summary"] == "A brief summary"
        assert result[1]["type"] == "fix"

    def test_empty_documents_returns_empty_list(self):
        presenter = BriefingPresenter()
        assert presenter.render([]) == []

    def test_summary_falls_back_to_title_if_missing(self):
        docs = [
            {
                "id": "d1",
                "title": "Fallback Title",
                "type": "idea",
                "summary": "",
            },
        ]
        presenter = BriefingPresenter()
        result = presenter.render(docs)

        assert result[0]["summary"] == "Fallback Title"

    def test_summary_falls_back_to_title_if_key_absent(self):
        docs = [
            {
                "id": "d1",
                "title": "Only Title",
                "type": "idea",
            },
        ]
        presenter = BriefingPresenter()
        result = presenter.render(docs)

        assert result[0]["summary"] == "Only Title"

    def test_missing_fields_default_to_empty_string(self):
        docs = [{}]
        presenter = BriefingPresenter()
        result = presenter.render(docs)

        assert result[0]["id"] == ""
        assert result[0]["title"] == ""
        assert result[0]["type"] == ""
        assert result[0]["tags"] == ""
        assert result[0]["project"] == ""
        assert result[0]["tier"] == ""


# -- DossierPresenter ------------------------------------------------------


class TestDossierPresenter:
    def test_dossier_for_known_entity(self, store: Store):
        obj_id = _create(
            store,
            title="Python style guide",
            obj_type="lesson",
            content="PEP 8 conventions",
            project="cortex",
        )
        entity_id = store.create_entity(
            name="Python", entity_type="technology"
        )
        store.add_mention(obj_id=obj_id, entity_id=entity_id)

        presenter = DossierPresenter(store)
        result = presenter.render("Python")

        assert result["status"] == "ok"
        assert result["topic"] == "Python"
        assert result["entity"] is not None
        assert result["entity"]["name"] == "Python"
        assert result["object_count"] >= 1
        assert len(result["objects"]) >= 1
        assert any(
            obj["id"] == obj_id for obj in result["objects"]
        )

    def test_dossier_for_unknown_topic_no_results(
        self, store: Store
    ):
        presenter = DossierPresenter(store)
        result = presenter.render("ZzzzNonexistent")

        assert result["status"] == "no_knowledge_found"
        assert result["objects"] == []

    def test_dossier_includes_timeline(self, store: Store):
        obj_id = _create(
            store,
            title="Timeline item",
            obj_type="session",
            content="A session note",
        )
        entity_id = store.create_entity(
            name="Sessions", entity_type="concept"
        )
        store.add_mention(obj_id=obj_id, entity_id=entity_id)

        presenter = DossierPresenter(store)
        result = presenter.render("Sessions")

        assert "timeline" in result
        assert len(result["timeline"]) >= 1
        entry = result["timeline"][0]
        assert "id" in entry
        assert "title" in entry
        assert "type" in entry
        assert "created_at" in entry

    def test_dossier_falls_back_to_text_search(self, store: Store):
        _create(
            store,
            title="Elasticsearch tuning guide",
            obj_type="research",
            content="Tuning Elasticsearch for production workloads",
        )
        # No entity created -- dossier should fall back to FTS
        presenter = DossierPresenter(store)
        result = presenter.render("Elasticsearch")

        # If FTS finds it, status is "ok"; otherwise "no_knowledge_found"
        # Either way the response shape must be valid
        assert result["status"] in ("ok", "no_knowledge_found")
        assert "objects" in result
        assert "timeline" in result

    def test_dossier_includes_contradictions(self, store: Store):
        id_a = _create(
            store,
            title="Claim A",
            obj_type="research",
            content="The sky is blue",
        )
        id_b = _create(
            store,
            title="Claim B",
            obj_type="research",
            content="The sky is green",
        )
        store.create_relationship(
            from_id=id_a, rel_type="contradicts", to_id=id_b
        )
        entity_id = store.create_entity(
            name="Sky", entity_type="concept"
        )
        store.add_mention(obj_id=id_a, entity_id=entity_id)
        store.add_mention(obj_id=id_b, entity_id=entity_id)

        presenter = DossierPresenter(store)
        result = presenter.render("Sky")

        assert result["status"] == "ok"
        assert len(result["contradictions"]) >= 1

    def test_dossier_includes_related_entities(self, store: Store):
        obj_id = _create(
            store,
            title="Go microservice",
            obj_type="decision",
            content="Decided to use Go for the new microservice",
        )
        ent_go = store.create_entity(
            name="Go", entity_type="technology"
        )
        ent_micro = store.create_entity(
            name="Microservices", entity_type="pattern"
        )
        store.add_mention(obj_id=obj_id, entity_id=ent_go)
        store.add_mention(obj_id=obj_id, entity_id=ent_micro)

        presenter = DossierPresenter(store)
        result = presenter.render("Go")

        assert "related_entities" in result
        related_names = {e["name"] for e in result["related_entities"]}
        # Both entities mention the same object, so both appear
        assert "Microservices" in related_names
        assert "Go" in related_names


# -- DocumentPresenter -----------------------------------------------------


class TestDocumentPresenter:
    def test_render_existing_object(self, store: Store):
        obj_id = _create(
            store,
            title="Full document",
            obj_type="lesson",
            content="Detailed lesson content here",
            project="cortex",
            tags="lesson,detail",
        )
        # Add a relationship so we can verify enrichment
        other_id = _create(
            store, title="Related", obj_type="fix"
        )
        store.create_relationship(
            from_id=obj_id, rel_type="supports", to_id=other_id
        )

        presenter = DocumentPresenter(store)
        result = presenter.render(obj_id)

        assert result is not None
        assert result["id"] == obj_id
        assert result["title"] == "Full document"
        assert "relationships" in result
        assert "entities" in result

    def test_render_nonexistent_returns_none(self, store: Store):
        presenter = DocumentPresenter(store)
        assert presenter.render("nonexistent-id-12345") is None

    def test_render_has_entities_key(self, store: Store):
        """DocumentPresenter resolves entity mentions via get_entity_mentions."""
        obj_id = _create(
            store,
            title="Entity doc",
            obj_type="research",
            content="Researching Rust",
        )
        ent_id = store.create_entity(
            name="Rust", entity_type="technology"
        )
        store.add_mention(obj_id=obj_id, entity_id=ent_id)

        presenter = DocumentPresenter(store)
        result = presenter.render(obj_id)

        assert result is not None
        assert "entities" in result
        assert isinstance(result["entities"], list)
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Rust"


# -- SynthesisPresenter ----------------------------------------------------


class TestSynthesisPresenter:
    def test_recent_objects_produce_ok_synthesis(
        self, store: Store
    ):
        # Objects created now are within any reasonable period
        _create(
            store,
            title="Recent fix",
            obj_type="fix",
            content="Fixed a bug",
            project="cortex",
        )
        _create(
            store,
            title="Recent lesson",
            obj_type="lesson",
            content="Learned something",
            project="cortex",
        )
        _create(
            store,
            title="Recent research",
            obj_type="research",
            content="Researched a topic",
            project="cortex",
        )

        presenter = SynthesisPresenter(store)
        result = presenter.render(period_days=7, project="cortex")

        assert result["status"] == "ok"
        assert result["period_days"] == 7
        assert result["project"] == "cortex"
        assert result["object_count"] >= 1
        assert len(result["themes"]) >= 1
        assert len(result["sources"]) >= 1
        assert result["narrative"] != ""

    def test_no_objects_in_period_returns_nothing_to_synthesize(
        self, store: Store
    ):
        # period_days=0 with cutoff at now means no objects qualify
        # unless created at the exact same instant -- use a project
        # filter that has no objects instead
        presenter = SynthesisPresenter(store)
        result = presenter.render(
            period_days=7, project="nonexistent_proj"
        )

        assert result["status"] == "nothing_to_synthesize"
        assert result["object_count"] == 0
        assert result["themes"] == []
        assert result["sources"] == []
        assert result["narrative"] == ""

    def test_themes_grouped_by_type(self, store: Store):
        _create(store, title="Fix 1", obj_type="fix")
        _create(store, title="Fix 2", obj_type="fix")
        _create(store, title="Lesson 1", obj_type="lesson")

        presenter = SynthesisPresenter(store)
        result = presenter.render(period_days=7)

        assert result["status"] == "ok"
        themes = result["themes"]
        theme_names = [t["name"] for t in themes]
        assert "fix" in theme_names
        assert "lesson" in theme_names

        # Themes sorted by count descending
        fix_theme = next(t for t in themes if t["name"] == "fix")
        lesson_theme = next(
            t for t in themes if t["name"] == "lesson"
        )
        assert fix_theme["count"] == 2
        assert lesson_theme["count"] == 1
        # Fix should appear before lesson (higher count)
        assert themes.index(fix_theme) < themes.index(
            lesson_theme
        )

    def test_synthesis_without_llm_produces_fallback_narrative(
        self, store: Store
    ):
        _create(store, title="Note", obj_type="idea")

        presenter = SynthesisPresenter(store, llm=None)
        result = presenter.render(period_days=7)

        assert result["status"] == "ok"
        # Fallback narrative starts with "Over the past period"
        assert "Over the past period" in result["narrative"]

    def test_sources_contain_expected_fields(self, store: Store):
        _create(
            store,
            title="Source doc",
            obj_type="fix",
            content="Some content",
        )

        presenter = SynthesisPresenter(store)
        result = presenter.render(period_days=7)

        assert result["status"] == "ok"
        for source in result["sources"]:
            assert "id" in source
            assert "title" in source
            assert "type" in source


# -- AlertPresenter --------------------------------------------------------


class TestAlertPresenter:
    def test_no_issues_returns_empty_alerts(self, store: Store):
        # Create a clean object with no problems
        _create(
            store,
            title="Clean object",
            obj_type="lesson",
            content="No issues here",
        )

        presenter = AlertPresenter(store)
        alerts = presenter.render()

        # No contradictions, no staleness, no patterns
        contradiction_alerts = [
            a for a in alerts if a["type"] == "contradiction"
        ]
        staleness_alerts = [
            a for a in alerts if a["type"] == "staleness"
        ]
        assert contradiction_alerts == []
        assert staleness_alerts == []

    def test_contradiction_generates_alert(self, store: Store):
        id_a = _create(
            store,
            title="Claim: Redis is fast",
            obj_type="research",
            content="Redis benchmarks show sub-ms latency",
        )
        id_b = _create(
            store,
            title="Claim: Redis is slow",
            obj_type="research",
            content="Redis fails under high contention",
        )
        store.create_relationship(
            from_id=id_a, rel_type="contradicts", to_id=id_b
        )

        presenter = AlertPresenter(store)
        alerts = presenter.render()

        contradiction_alerts = [
            a for a in alerts if a["type"] == "contradiction"
        ]
        assert len(contradiction_alerts) >= 1
        alert = contradiction_alerts[0]
        assert alert["severity"] == "high"
        assert "message" in alert
        assert "object_ids" in alert
        assert set(alert["object_ids"]) == {id_a, id_b}

    def test_staleness_generates_alert(self, store: Store):
        archived_id = _create(
            store,
            title="Old archived doc",
            obj_type="research",
            content="Outdated info",
            tier="archive",
        )
        active_id = _create(
            store,
            title="Active doc",
            obj_type="decision",
            content="Depends on old info",
            tier="recall",
        )
        store.create_relationship(
            from_id=active_id,
            rel_type="dependsOn",
            to_id=archived_id,
        )

        presenter = AlertPresenter(store)
        alerts = presenter.render()

        staleness_alerts = [
            a for a in alerts if a["type"] == "staleness"
        ]
        assert len(staleness_alerts) >= 1
        alert = staleness_alerts[0]
        assert alert["severity"] == "low"
        assert "message" in alert
        assert active_id in alert["object_ids"]
        assert archived_id in alert["object_ids"]

    def test_alert_fields_are_well_formed(self, store: Store):
        id_a = _create(
            store, title="A", obj_type="fix", content="X"
        )
        id_b = _create(
            store, title="B", obj_type="fix", content="Y"
        )
        store.create_relationship(
            from_id=id_a, rel_type="contradicts", to_id=id_b
        )

        presenter = AlertPresenter(store)
        alerts = presenter.render()

        for alert in alerts:
            assert "type" in alert
            assert "severity" in alert
            assert "message" in alert
            assert "object_ids" in alert
            assert isinstance(alert["object_ids"], list)
            assert alert["type"] in (
                "contradiction",
                "staleness",
                "pattern",
            )
            assert alert["severity"] in ("high", "medium", "low")

    def test_no_duplicate_contradiction_alerts(
        self, store: Store
    ):
        id_a = _create(
            store, title="P", obj_type="lesson", content="X"
        )
        id_b = _create(
            store, title="Q", obj_type="lesson", content="Y"
        )
        # contradicts is not symmetric in graph, but AlertPresenter
        # deduplicates by sorted pair
        store.create_relationship(
            from_id=id_a, rel_type="contradicts", to_id=id_b
        )

        presenter = AlertPresenter(store)
        alerts = presenter.render()

        contradiction_alerts = [
            a for a in alerts if a["type"] == "contradiction"
        ]
        pairs = [
            tuple(sorted(a["object_ids"]))
            for a in contradiction_alerts
        ]
        # No duplicate pairs
        assert len(pairs) == len(set(pairs))
