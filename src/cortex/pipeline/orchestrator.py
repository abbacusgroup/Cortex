"""Pipeline orchestrator — coordinates all pipeline stages.

Two execution paths:
1. Sync: INGEST → basic NORMALIZE → return ID (fast)
2. Full: NORMALIZE (full) → LINK → ENRICH → REASON (can run in background)
"""

from __future__ import annotations

from typing import Any

from cortex.core.config import CortexConfig
from cortex.core.logging import get_logger
from cortex.db.store import Store
from cortex.pipeline.enrich import EnrichStage
from cortex.pipeline.link import LinkStage
from cortex.pipeline.normalize import NormalizeStage
from cortex.pipeline.reason import ReasonStage
from cortex.pipeline.templates import get_template
from cortex.services.llm import LLMClient

logger = get_logger("pipeline.orchestrator")


class PipelineOrchestrator:
    """Coordinates the full intelligence pipeline."""

    def __init__(self, store: Store, config: CortexConfig):
        self.store = store
        self.config = config
        self.llm = LLMClient(config)
        self.normalizer = NormalizeStage(store, self.llm)
        self.linker = LinkStage(store, self.llm)
        self.enricher = EnrichStage(store)
        self.reasoner = ReasonStage(store.graph)

    def capture(
        self,
        *,
        title: str,
        content: str = "",
        obj_type: str = "idea",
        project: str = "",
        tags: str = "",
        template: str | None = None,
        template_fields: dict[str, str] | None = None,
        captured_by: str = "",
        summary: str = "",
        entities: list[dict[str, str]] | None = None,
        extra_properties: dict[str, str] | None = None,
        confidence: float = 0.0,
        run_pipeline: bool = True,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        """Capture a knowledge object and optionally run the full pipeline.

        Args:
            title: Object title.
            content: Raw content (overridden by template if provided).
            obj_type: Knowledge type.
            project: Project name.
            tags: Comma-separated tags.
            template: Template name (session, fix, decision, etc.)
            template_fields: Fields for the template.
            captured_by: Source of the capture.
            summary: Pre-classified summary (skips LLM classify if set).
            entities: Pre-extracted entities for the LINK stage.
            extra_properties: Caller-supplied properties (template wins on conflicts).
            confidence: Classification confidence (used when summary is set).
            run_pipeline: If True, run full pipeline. If False, just ingest.

        Returns:
            Dict with id, status, and pipeline results.
        """
        # Apply template if specified
        template_properties: dict[str, str] = {}
        if template:
            tmpl = get_template(template)
            if tmpl:
                obj_type = tmpl.obj_type
                rendered = tmpl.render(template_fields or {})
                if not content:
                    content = rendered["content"]
                template_properties = rendered["properties"]

        # Merge caller properties with template properties (template wins)
        merged_properties: dict[str, str] = {}
        if extra_properties:
            merged_properties.update(extra_properties)
        if template_properties:
            merged_properties.update(template_properties)

        # Step 1: Ingest — create in both stores
        obj_id = self.store.create(
            obj_type=obj_type,
            title=title,
            content=content,
            properties=merged_properties if merged_properties else None,
            project=project,
            tags=tags,
            captured_by=captured_by,
            summary=summary,
            confidence=confidence if summary else 1.0,
            created_at=created_at,
            updated_at=updated_at,
        )

        result: dict[str, Any] = {
            "id": obj_id,
            "status": "ingested",
            "type": obj_type,
        }

        if not run_pipeline:
            return result

        # Step 2: Full pipeline
        pipeline_result = self.run_pipeline(obj_id, pre_entities=entities)
        result.update(pipeline_result)
        return result

    def run_pipeline(
        self, obj_id: str, pre_entities: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Run the full pipeline on an existing object.

        Stages: NORMALIZE → LINK → ENRICH → REASON
        Each stage is resilient — failure in one doesn't block the others.

        Args:
            obj_id: The object ID to process.
            pre_entities: Pre-extracted entities; used instead of normalize output if set.
        """
        result: dict[str, Any] = {"pipeline_stages": {}}

        # NORMALIZE
        try:
            norm_result = self.normalizer.run(obj_id)
            result["pipeline_stages"]["normalize"] = norm_result
            entities = pre_entities if pre_entities is not None else norm_result.get("entities", [])
        except Exception as e:
            logger.warning("Normalize failed for %s: %s", obj_id, e)
            result["pipeline_stages"]["normalize"] = {"status": "failed", "error": str(e)}
            entities = pre_entities if pre_entities is not None else []

        # LINK
        try:
            link_result = self.linker.run(obj_id, entities)
            result["pipeline_stages"]["link"] = link_result
        except Exception as e:
            logger.warning("Link failed for %s: %s", obj_id, e)
            result["pipeline_stages"]["link"] = {"status": "failed", "error": str(e)}

        # ENRICH
        try:
            enrich_result = self.enricher.run(obj_id)
            result["pipeline_stages"]["enrich"] = enrich_result
        except Exception as e:
            logger.warning("Enrich failed for %s: %s", obj_id, e)
            result["pipeline_stages"]["enrich"] = {"status": "failed", "error": str(e)}

        # REASON
        try:
            reason_result = self.reasoner.run()
            result["pipeline_stages"]["reason"] = reason_result
        except Exception as e:
            logger.warning("Reason failed for %s: %s", obj_id, e)
            result["pipeline_stages"]["reason"] = {"status": "failed", "error": str(e)}

        # Determine overall status
        stages = result["pipeline_stages"]
        if all(s.get("status") != "failed" for s in stages.values()):
            result["status"] = "complete"
        else:
            result["status"] = "partial"

        return result
