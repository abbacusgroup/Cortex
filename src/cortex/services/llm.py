"""LLM service — provider-agnostic via litellm.

Handles classification, entity extraction, relationship discovery,
and general-purpose completions with structured output parsing.
"""

from __future__ import annotations

import json
from typing import Any

from cortex.core.config import CortexConfig
from cortex.core.errors import LLMError
from cortex.core.logging import get_logger

logger = get_logger("services.llm")

# Generic words that should never be extracted as entities.
# These are common English words that provide no insight when tracked
# as named entities — "fix" appearing in fix objects is not a pattern.
ENTITY_STOPWORDS = frozenset(
    {
        "fix",
        "bug",
        "error",
        "issue",
        "problem",
        "solution",
        "test",
        "code",
        "change",
        "update",
        "feature",
        "function",
        "file",
        "folder",
        "directory",
        "module",
        "class",
        "method",
        "server",
        "client",
        "request",
        "response",
        "command",
        "terminal",
        "console",
        "log",
        "output",
        "input",
        "data",
        "config",
        "setting",
        "option",
        "parameter",
        "user",
        "admin",
        "system",
        "app",
        "application",
        "task",
        "work",
        "item",
        "object",
        "thing",
        "api",
        "url",
        "endpoint",
        "route",
        "path",
        "build",
        "deploy",
        "run",
        "start",
        "stop",
        "version",
        "release",
        "branch",
        "commit",
        "merge",
    }
)

# Classification prompt template
CLASSIFY_PROMPT = """\
You are a knowledge classifier for a cognitive knowledge system.

Analyze the following text and extract structured metadata.

Title: {title}
Content:
{content}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "type": "<one of: decision, lesson, fix, session, research, source, synthesis, idea>",
  "summary": "<1-2 sentence summary>",
  "tags": "<comma-separated relevant tags>",
  "project": "<project name if identifiable, else empty string>",
  "entities": [
    {{"name": "<specific named entity>", "type": "<technology|project|pattern|concept>"}}
  ],
  // Entities must be SPECIFIC: named technologies (Redis, PostgreSQL, FastAPI),
  // named projects, recognized patterns (circuit-breaker, saga), or domain
  // concepts (authentication, caching). Do NOT extract generic words like:
  // fix, bug, error, issue, test, code, change, update, server, terminal, command.
  "confidence": <0.0-1.0 how confident you are in the classification>,
  "properties": {{
    <type-specific properties as key-value pairs>
  }}
}}

Type-specific properties to extract:
- decision: rationale, alternatives, chosen, decisionStatus
- lesson: cause, impact, prevention
- fix: symptom, rootCause, resolution, severity, filesAffected
- session: goal, worked, failed, nextSteps
- research: question, findings, sources
- source: url, author, credibility
- synthesis: period, themes, sourceIds
- idea: feasibility, ideaStatus, dependencies
"""

RELATIONSHIP_PROMPT = """\
You are analyzing relationships between knowledge objects.

New object:
Title: {new_title}
Type: {new_type}
Content: {new_content}

Existing objects:
{existing_objects}

For each meaningful relationship between the new object and an existing one, output a JSON array:
[
  {{
    "from_id": "<new object ID or existing object ID>",
    "to_id": "<the other object ID>",
    "rel_type": "<causedBy|contradicts|supports|supersedes|dependsOn|implements>",
    "confidence": <0.0-1.0>
  }}
]

Only include relationships you are confident about (>0.5). If none, return [].
Respond with ONLY the JSON array.
"""


class LLMClient:
    """Provider-agnostic LLM client via litellm."""

    def __init__(self, config: CortexConfig):
        self.config = config
        self._litellm_available = self._check_litellm()
        # Ollama runs locally and doesn't require an API key
        needs_key = config.llm_provider not in ("ollama",)
        self._available = bool(
            config.llm_model
            and (config.llm_api_key or not needs_key)
            and self._litellm_available
        )
        if self._available:
            self._model = config.llm_model
        else:
            self._model = ""
            if config.llm_model and not self._litellm_available:
                logger.info("litellm not installed — classification will be skipped")
            else:
                logger.info("No LLM configured — classification will be skipped")

    @staticmethod
    def _check_litellm() -> bool:
        try:
            import litellm  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def available(self) -> bool:
        return self._available

    def classify(self, *, title: str, content: str) -> dict[str, Any]:
        """Classify content and extract structured metadata.

        Returns:
            Dict with type, summary, tags, project, entities, confidence, properties.
            On failure, returns a minimal dict with type="idea" and confidence=0.0.
        """
        if not self._available:
            return self._fallback_classification(title, content)

        prompt = CLASSIFY_PROMPT.format(title=title, content=content[:8000])

        try:
            response = self._complete(prompt)
            parsed = self._parse_json(response)
            return self._validate_classification(parsed)
        except Exception as e:
            logger.warning("Classification failed: %s", e)
            return self._fallback_classification(title, content)

    def discover_relationships(
        self,
        *,
        new_id: str,
        new_title: str,
        new_type: str,
        new_content: str,
        existing: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Discover relationships between a new object and existing ones.

        Returns:
            List of relationship dicts: {from_id, to_id, rel_type, confidence}
        """
        if not self._available or not existing:
            return []

        # Format existing objects for the prompt
        existing_text = "\n".join(
            f"- ID: {obj['id']}, Type: {obj.get('type', '?')}, Title: {obj.get('title', '?')}"
            for obj in existing[:20]  # Limit to 20 for prompt size
        )

        prompt = RELATIONSHIP_PROMPT.format(
            new_title=new_title,
            new_type=new_type,
            new_content=new_content[:4000],
            existing_objects=existing_text,
        )

        try:
            response = self._complete(prompt)
            parsed = self._parse_json(response)
            if not isinstance(parsed, list):
                return []
            return [
                r
                for r in parsed
                if isinstance(r, dict)
                and all(k in r for k in ("from_id", "to_id", "rel_type", "confidence"))
                and r.get("confidence", 0) > 0.5
            ]
        except Exception as e:
            logger.warning("Relationship discovery failed: %s", e)
            return []

    def complete(self, prompt: str) -> str:
        """General-purpose completion.

        Raises:
            LLMError: If the LLM call fails and no fallback is possible.
        """
        if not self._available:
            raise LLMError("No LLM configured")
        return self._complete(prompt)

    def _complete(self, prompt: str) -> str:
        """Call litellm completion."""
        import litellm

        try:
            response = litellm.completion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.config.llm_api_key,
                timeout=30,
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise LLMError(
                f"LLM call failed: {e}",
                context={"model": self._model},
                cause=e,
            ) from e

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Extract and parse JSON from LLM response."""
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse LLM JSON: {e}", cause=e) from e

    @staticmethod
    def _validate_classification(data: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize classification output."""
        valid_types = {
            "decision",
            "lesson",
            "fix",
            "session",
            "research",
            "source",
            "synthesis",
            "idea",
        }

        result = {
            "type": data.get("type", "idea"),
            "summary": data.get("summary", ""),
            "tags": data.get("tags", ""),
            "project": data.get("project", ""),
            "entities": data.get("entities", []),
            "confidence": float(data.get("confidence", 0.5)),
            "properties": data.get("properties", {}),
        }

        # Normalize type
        if result["type"] not in valid_types:
            result["type"] = "idea"
            result["confidence"] = min(result["confidence"], 0.3)

        # Clamp confidence
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))

        # Validate entities
        valid_entity_types = {"technology", "project", "pattern", "concept"}
        result["entities"] = [
            e
            for e in result["entities"]
            if isinstance(e, dict)
            and "name" in e
            and e.get("type", "concept") in valid_entity_types
            and e["name"].lower().strip() not in ENTITY_STOPWORDS
        ]

        return result

    @staticmethod
    def _fallback_classification(title: str, content: str) -> dict[str, Any]:
        """Minimal classification when LLM is unavailable."""
        return {
            "type": "idea",
            "summary": title,
            "tags": "",
            "project": "",
            "entities": [],
            "confidence": 0.0,
            "properties": {},
        }
