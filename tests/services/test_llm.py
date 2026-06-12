"""Tests for cortex.services.llm (LLMClient)."""

from __future__ import annotations

import importlib.util
import sys
import types
from unittest.mock import MagicMock

import pytest

from cortex.core.config import CortexConfig
from cortex.core.errors import LLMError
from cortex.services.llm import RELATIONSHIP_PROMPT, LLMClient

# ``litellm`` is a core dependency (see ``dependencies`` in pyproject.toml),
# so it is always installed and ``_LITELLM_AVAILABLE`` is normally True. This
# guard is a defensive fallback for unusual/minimal environments where the
# import is somehow unavailable, so those tests skip cleanly instead of
# erroring.
_LITELLM_AVAILABLE = importlib.util.find_spec("litellm") is not None
requires_litellm = pytest.mark.skipif(
    not _LITELLM_AVAILABLE,
    reason="litellm not importable in this environment",
)


@pytest.fixture()
def client(tmp_path) -> LLMClient:
    """LLMClient with no API key configured."""
    cfg = CortexConfig(data_dir=tmp_path)
    return LLMClient(cfg)


# -- Availability -----------------------------------------------------------


class TestAvailability:
    def test_no_api_key_means_unavailable(self, client: LLMClient):
        assert client.available is False

    @requires_litellm
    def test_with_model_and_key_is_available(self, tmp_path):
        cfg = CortexConfig(
            data_dir=tmp_path,
            llm_model="gpt-4",
            llm_api_key="sk-test-key",
        )
        c = LLMClient(cfg)
        assert c.available is True


# -- classify ---------------------------------------------------------------


class TestClassify:
    def test_classify_without_llm_returns_fallback(self, client: LLMClient):
        result = client.classify(title="Test", content="Some content")
        assert result["type"] == "idea"
        assert result["confidence"] == 0.0
        assert result["entities"] == []
        assert result["summary"] == "Test"

    def test_classify_fallback_has_all_keys(self, client: LLMClient):
        result = client.classify(title="T", content="C")
        expected_keys = {
            "type",
            "summary",
            "tags",
            "project",
            "entities",
            "confidence",
            "properties",
        }
        assert set(result.keys()) == expected_keys


# -- discover_relationships -------------------------------------------------


class TestDiscoverRelationships:
    def test_returns_empty_when_unavailable(self, client: LLMClient):
        result = client.discover_relationships(
            new_id="abc",
            new_title="Title",
            new_type="idea",
            new_content="Content",
            existing=[{"id": "xyz", "type": "fix", "title": "Other"}],
        )
        assert result == []

    def test_returns_empty_when_no_existing(self, tmp_path):
        cfg = CortexConfig(
            data_dir=tmp_path,
            llm_model="gpt-4",
            llm_api_key="sk-key",
        )
        c = LLMClient(cfg)
        result = c.discover_relationships(
            new_id="abc",
            new_title="Title",
            new_type="idea",
            new_content="Content",
            existing=[],
        )
        assert result == []


# -- _parse_json ------------------------------------------------------------


class TestParseJson:
    def test_valid_json(self):
        result = LLMClient._parse_json('{"type": "fix"}')
        assert result == {"type": "fix"}

    def test_json_array(self):
        result = LLMClient._parse_json('[{"a": 1}]')
        assert result == [{"a": 1}]

    def test_markdown_fenced_json(self):
        text = '```json\n{"type": "lesson"}\n```'
        result = LLMClient._parse_json(text)
        assert result == {"type": "lesson"}

    def test_markdown_fenced_no_language_tag(self):
        text = '```\n{"type": "fix"}\n```'
        result = LLMClient._parse_json(text)
        assert result == {"type": "fix"}

    def test_invalid_json_raises_llm_error(self):
        with pytest.raises(LLMError):
            LLMClient._parse_json("not json at all")

    def test_empty_string_raises_llm_error(self):
        with pytest.raises(LLMError):
            LLMClient._parse_json("")


# -- _validate_classification ----------------------------------------------


class TestValidateClassification:
    def test_normalizes_valid_input(self):
        data = {
            "type": "fix",
            "summary": "A fix",
            "tags": "bug",
            "project": "cortex",
            "entities": [
                {"name": "Python", "type": "technology"},
            ],
            "confidence": 0.9,
            "properties": {"severity": "high"},
        }
        result = LLMClient._validate_classification(data)
        assert result["type"] == "fix"
        assert result["confidence"] == 0.9
        assert len(result["entities"]) == 1

    def test_unknown_type_defaults_to_idea(self):
        data = {"type": "banana", "confidence": 0.8}
        result = LLMClient._validate_classification(data)
        assert result["type"] == "idea"
        # Confidence capped at 0.3 when type is unknown
        assert result["confidence"] == 0.3

    def test_confidence_greater_than_one_clamped(self):
        data = {"type": "lesson", "confidence": 5.0}
        result = LLMClient._validate_classification(data)
        assert result["confidence"] == 1.0

    def test_confidence_negative_clamped_to_zero(self):
        data = {"type": "decision", "confidence": -0.5}
        result = LLMClient._validate_classification(data)
        assert result["confidence"] == 0.0

    def test_bad_entities_filtered_out(self):
        data = {
            "type": "research",
            "entities": [
                {"name": "Good", "type": "technology"},
                "not a dict",
                {"type": "technology"},  # missing name
                {"name": "Bad", "type": "alien"},  # invalid type
                {"name": "OK"},  # no type -> concept
            ],
        }
        result = LLMClient._validate_classification(data)
        names = [e["name"] for e in result["entities"]]
        assert "Good" in names
        assert "OK" in names
        assert len(result["entities"]) == 2

    def test_missing_fields_get_defaults(self):
        result = LLMClient._validate_classification({})
        assert result["type"] == "idea"
        assert result["summary"] == ""
        assert result["tags"] == ""
        assert result["project"] == ""
        assert result["entities"] == []
        assert result["properties"] == {}


# -- _fallback_classification -----------------------------------------------


class TestFallbackClassification:
    def test_returns_correct_structure(self):
        result = LLMClient._fallback_classification("My Title")
        assert result == {
            "type": "idea",
            "summary": "My Title",
            "tags": "",
            "project": "",
            "entities": [],
            "confidence": 0.0,
            "properties": {},
        }

    def test_summary_is_title(self):
        result = LLMClient._fallback_classification("Hello")
        assert result["summary"] == "Hello"


# -- RELATIONSHIP_PROMPT new_id injection -----------------------------------


class TestRelationshipPromptId:
    def test_prompt_template_has_new_id_placeholder(self):
        # Regression: the template previously had no {new_id} slot, so the LLM
        # could never reference the new object and its edges were all dropped.
        assert "{new_id}" in RELATIONSHIP_PROMPT

    def test_formatted_prompt_contains_new_id(self, tmp_path):
        """The new object's real ID must appear in the prompt sent to the LLM."""
        cfg = CortexConfig(
            data_dir=tmp_path,
            llm_model="gpt-4",
            llm_api_key="sk-key",
        )
        c = LLMClient(cfg)

        captured: dict[str, str] = {}

        def fake_complete(prompt: str) -> str:
            captured["prompt"] = prompt
            return "[]"

        c._complete = fake_complete  # type: ignore[assignment]
        c.discover_relationships(
            new_id="the-real-new-id-1234",
            new_title="Title",
            new_type="fix",
            new_content="Content",
            existing=[{"id": "xyz", "type": "fix", "title": "Other"}],
        )
        assert "the-real-new-id-1234" in captured["prompt"]


# -- Keyless (ollama) provider ----------------------------------------------


def _fake_litellm_module() -> types.ModuleType:
    """A stand-in litellm module recording the kwargs of completion()."""
    mod = types.ModuleType("litellm")
    msg = MagicMock()
    msg.content = "ok-response"
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    mod.completion = MagicMock(return_value=response)  # type: ignore[attr-defined]
    return mod


class TestKeylessProvider:
    def test_ollama_without_key_is_available(self, tmp_path, monkeypatch):
        """Keyless ollama (model set, no api key) must report available."""
        monkeypatch.setattr(LLMClient, "_check_litellm", staticmethod(lambda: True))
        cfg = CortexConfig(
            data_dir=tmp_path,
            llm_provider="ollama",
            llm_model="ollama/qwen3:8b",
        )
        c = LLMClient(cfg)
        assert c.available is True

    def test_empty_key_is_not_passed_to_litellm(self, tmp_path, monkeypatch):
        """Regression: empty api_key must be omitted, not passed as '' — an
        empty string makes litellm emit an illegal 'Authorization: Bearer '."""
        monkeypatch.setattr(LLMClient, "_check_litellm", staticmethod(lambda: True))
        fake = _fake_litellm_module()
        monkeypatch.setitem(sys.modules, "litellm", fake)

        cfg = CortexConfig(
            data_dir=tmp_path,
            llm_provider="ollama",
            llm_model="ollama/qwen3:8b",
        )
        c = LLMClient(cfg)
        out = c.complete("hello")

        assert out == "ok-response"
        _, kwargs = fake.completion.call_args
        assert "api_key" not in kwargs  # never '' (no empty Bearer header)
        assert kwargs["model"] == "ollama/qwen3:8b"

    def test_real_key_is_passed_through(self, tmp_path, monkeypatch):
        """Hosted providers must still receive their real key."""
        monkeypatch.setattr(LLMClient, "_check_litellm", staticmethod(lambda: True))
        fake = _fake_litellm_module()
        monkeypatch.setitem(sys.modules, "litellm", fake)

        cfg = CortexConfig(
            data_dir=tmp_path,
            llm_model="gpt-4",
            llm_api_key="sk-real-key",
        )
        c = LLMClient(cfg)
        c.complete("hello")
        _, kwargs = fake.completion.call_args
        assert kwargs["api_key"] == "sk-real-key"


# -- Failure observability --------------------------------------------------


class TestFailureObservability:
    def test_last_error_starts_none(self, tmp_path):
        cfg = CortexConfig(data_dir=tmp_path, llm_model="gpt-4", llm_api_key="k")
        c = LLMClient(cfg)
        assert c.last_error is None

    def test_failure_sets_last_error_and_raises(self, tmp_path, monkeypatch, caplog):
        """A provider failure must be observable (last_error + WARNING log)
        instead of masquerading as a clean fallback."""
        monkeypatch.setattr(LLMClient, "_check_litellm", staticmethod(lambda: True))
        fake = types.ModuleType("litellm")
        fake.completion = MagicMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError("Illegal header value b'Bearer '")
        )
        monkeypatch.setitem(sys.modules, "litellm", fake)

        cfg = CortexConfig(
            data_dir=tmp_path,
            llm_provider="ollama",
            llm_model="ollama/qwen3:8b",
        )
        c = LLMClient(cfg)
        with caplog.at_level("WARNING"), pytest.raises(LLMError):
            c.complete("hi")

        assert c.last_error is not None
        assert "Bearer" in c.last_error
        assert any("LLM call failed" in r.message for r in caplog.records)

    def test_classify_falls_back_on_failure_without_crashing(
        self, tmp_path, monkeypatch
    ):
        """classify() must degrade to the fallback (not crash) when the LLM
        fails, while still recording last_error for observability."""
        monkeypatch.setattr(LLMClient, "_check_litellm", staticmethod(lambda: True))
        fake = types.ModuleType("litellm")
        fake.completion = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "litellm", fake)

        cfg = CortexConfig(
            data_dir=tmp_path,
            llm_provider="ollama",
            llm_model="ollama/qwen3:8b",
        )
        c = LLMClient(cfg)
        result = c.classify(title="My Title", content="body")
        # Did not raise; fell back; recorded the failure.
        assert result["type"] == "idea"
        assert result["confidence"] == 0.0
        assert result["summary"] == "My Title"
        assert c.last_error is not None
