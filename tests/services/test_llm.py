"""Tests for cortex.services.llm (LLMClient)."""

from __future__ import annotations

import pytest

from cortex.core.config import CortexConfig
from cortex.core.errors import LLMError
from cortex.services.llm import LLMClient


@pytest.fixture()
def client(tmp_path) -> LLMClient:
    """LLMClient with no API key configured."""
    cfg = CortexConfig(data_dir=tmp_path)
    return LLMClient(cfg)


# -- Availability -----------------------------------------------------------


class TestAvailability:
    def test_no_api_key_means_unavailable(self, client: LLMClient):
        assert client.available is False

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
    def test_classify_without_llm_returns_fallback(
        self, client: LLMClient
    ):
        result = client.classify(title="Test", content="Some content")
        assert result["type"] == "idea"
        assert result["confidence"] == 0.0
        assert result["entities"] == []
        assert result["summary"] == "Test"

    def test_classify_fallback_has_all_keys(self, client: LLMClient):
        result = client.classify(title="T", content="C")
        expected_keys = {
            "type", "summary", "tags", "project",
            "entities", "confidence", "properties",
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
                {"type": "technology"},           # missing name
                {"name": "Bad", "type": "alien"}, # invalid type
                {"name": "OK"},                   # no type -> concept
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
        result = LLMClient._fallback_classification("My Title", "Body")
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
        result = LLMClient._fallback_classification("Hello", "World")
        assert result["summary"] == "Hello"
