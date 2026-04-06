"""Tests for capture templates."""

from __future__ import annotations

from cortex.pipeline.templates import (
    FIX_TEMPLATE,
    SESSION_TEMPLATE,
    CaptureTemplate,
    get_template,
)


class TestCaptureTemplateRender:
    """Tests for CaptureTemplate.render()."""

    def test_session_render_all_fields(self):
        """Session template with all fields renders full content."""
        result = SESSION_TEMPLATE.render({
            "goal": "Implement auth",
            "worked": "JWT tokens",
            "failed": "Session expiry",
        })

        content = result["content"]
        assert "## Goal" in content
        assert "Implement auth" in content
        assert "## Worked" in content
        assert "JWT tokens" in content
        assert "## Failed" in content
        assert "Session expiry" in content

    def test_fix_render_properties_are_camelcase(self):
        """Fix template properties use camelCase keys."""
        result = FIX_TEMPLATE.render({
            "symptom": "500 on /api/login",
            "root_cause": "Missing DB index",
        })

        props = result["properties"]
        assert "symptom" in props
        assert "rootCause" in props
        assert props["rootCause"] == "Missing DB index"

    def test_missing_optional_fields_skipped(self):
        """Optional fields with no value are excluded from content."""
        result = SESSION_TEMPLATE.render({"goal": "Ship v1"})

        content = result["content"]
        assert "## Goal" in content
        assert "Ship v1" in content
        # Optional fields not provided -> not in output
        assert "## Worked" not in content
        assert "## Failed" not in content
        assert "## Next Steps" not in content

    def test_missing_required_field_still_renders(self):
        """Missing required fields are just empty sections, no crash."""
        result = SESSION_TEMPLATE.render({})

        # Empty fields are skipped (strip() check)
        assert result["content"] == ""
        assert result["properties"] == {}

    def test_extra_fields_ignored(self):
        """Fields not in required or optional are ignored."""
        result = SESSION_TEMPLATE.render({
            "goal": "Ship v1",
            "extra_field": "Should be ignored",
        })

        # Only goal shows up
        assert "## Goal" in result["content"]
        assert "extra_field" not in result["content"]
        assert "extraField" not in result["properties"]


class TestGetTemplate:
    """Tests for the template registry."""

    def test_get_session_template(self):
        tmpl = get_template("session")
        assert tmpl is not None
        assert tmpl.name == "session"
        assert tmpl.obj_type == "session"

    def test_get_fix_template(self):
        tmpl = get_template("fix")
        assert tmpl is not None
        assert tmpl.name == "fix"

    def test_get_nonexistent_template(self):
        assert get_template("nonexistent") is None

    def test_get_all_known_templates(self):
        """All six pre-defined templates are registered."""
        for name in (
            "session", "fix", "decision", "lesson", "research", "idea"
        ):
            assert get_template(name) is not None, (
                f"Template '{name}' missing"
            )


class TestPropertyKeyConversion:
    """Tests for _to_property_key."""

    def test_snake_to_camel(self):
        assert CaptureTemplate._to_property_key("root_cause") == "rootCause"

    def test_single_word_unchanged(self):
        assert CaptureTemplate._to_property_key("goal") == "goal"

    def test_multi_part_snake(self):
        assert (
            CaptureTemplate._to_property_key("files_affected_count")
            == "filesAffectedCount"
        )

    def test_two_part_snake(self):
        assert (
            CaptureTemplate._to_property_key("next_steps") == "nextSteps"
        )
