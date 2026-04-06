"""Capture templates — structured input formats for common knowledge types.

Each template defines expected fields and produces a normalized content string
for ingestion into the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaptureTemplate:
    """A template for structured knowledge capture."""

    name: str
    obj_type: str
    required_fields: list[str]
    optional_fields: list[str] = field(default_factory=list)

    def render(self, fields: dict[str, str]) -> dict[str, Any]:
        """Render the template into a content string and properties dict.

        Returns:
            Dict with: content (rendered markdown), properties (type-specific).
        """
        # Build markdown content from fields
        lines = []
        properties = {}

        for f in self.required_fields + self.optional_fields:
            value = fields.get(f, "").strip()
            if value:
                label = f.replace("_", " ").title()
                lines.append(f"## {label}\n{value}")
                properties[self._to_property_key(f)] = value

        content = "\n\n".join(lines)
        return {"content": content, "properties": properties}

    @staticmethod
    def _to_property_key(field_name: str) -> str:
        """Convert field name to ontology property key (camelCase)."""
        parts = field_name.split("_")
        return parts[0] + "".join(p.capitalize() for p in parts[1:])


# Pre-defined templates

SESSION_TEMPLATE = CaptureTemplate(
    name="session",
    obj_type="session",
    required_fields=["goal"],
    optional_fields=["worked", "failed", "next_steps"],
)

FIX_TEMPLATE = CaptureTemplate(
    name="fix",
    obj_type="fix",
    required_fields=["symptom"],
    optional_fields=["root_cause", "resolution", "severity", "files_affected"],
)

DECISION_TEMPLATE = CaptureTemplate(
    name="decision",
    obj_type="decision",
    required_fields=["chosen"],
    optional_fields=["rationale", "alternatives", "decision_status"],
)

LESSON_TEMPLATE = CaptureTemplate(
    name="lesson",
    obj_type="lesson",
    required_fields=["cause"],
    optional_fields=["impact", "prevention"],
)

RESEARCH_TEMPLATE = CaptureTemplate(
    name="research",
    obj_type="research",
    required_fields=["question"],
    optional_fields=["findings", "sources"],
)

IDEA_TEMPLATE = CaptureTemplate(
    name="idea",
    obj_type="idea",
    required_fields=[],
    optional_fields=["feasibility", "idea_status", "dependencies"],
)

# Template registry
TEMPLATES: dict[str, CaptureTemplate] = {
    t.name: t for t in [
        SESSION_TEMPLATE, FIX_TEMPLATE, DECISION_TEMPLATE,
        LESSON_TEMPLATE, RESEARCH_TEMPLATE, IDEA_TEMPLATE,
    ]
}


def get_template(name: str) -> CaptureTemplate | None:
    """Get a template by name."""
    return TEMPLATES.get(name)
