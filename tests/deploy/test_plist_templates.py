"""Smoke tests for the shipped LaunchAgent plist templates (Bundle 8 / C7).

Parses each plist in ``deploy/`` with Python's stdlib ``plistlib`` to
catch XML syntax errors before they land. Also asserts the minimum set of
keys that launchd needs to actually run the agent.
"""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEPLOY_DIR = _REPO_ROOT / "deploy"

# The plist files we ship as templates. New templates should be appended here.
_PLIST_FILES = [
    "ai.abbacus.cortex.mcp.plist",
    "ai.abbacus.cortex.dashboard.plist",
]


@pytest.mark.parametrize("filename", _PLIST_FILES)
class TestShippedPlistTemplates:
    def test_plist_file_exists(self, filename: str):
        path = _DEPLOY_DIR / filename
        assert path.exists(), f"missing shipped plist: {path}"

    def test_plist_parses_successfully(self, filename: str):
        path = _DEPLOY_DIR / filename
        with path.open("rb") as f:
            data = plistlib.load(f)
        assert isinstance(data, dict)

    def test_plist_has_required_launchd_keys(self, filename: str):
        path = _DEPLOY_DIR / filename
        with path.open("rb") as f:
            data = plistlib.load(f)

        required = {"Label", "ProgramArguments", "RunAtLoad", "KeepAlive"}
        missing = required - set(data.keys())
        assert not missing, f"{filename} missing required keys: {missing}"

    def test_plist_label_matches_filename(self, filename: str):
        """launchctl load fails with a confusing error if the Label
        doesn't match the filename (minus ``.plist``). Catch the mismatch
        at test time instead.
        """
        path = _DEPLOY_DIR / filename
        with path.open("rb") as f:
            data = plistlib.load(f)
        expected_label = filename.removesuffix(".plist")
        assert data["Label"] == expected_label

    def test_plist_uses_yeouser_placeholder(self, filename: str):
        """The shipped templates are not user-specific — they must contain
        ``YOURUSER`` placeholders that a ``sed`` command substitutes at
        install time. This guards against accidentally committing an
        individual developer's home directory into the repo.
        """
        path = _DEPLOY_DIR / filename
        raw = path.read_text()
        assert "YOURUSER" in raw, (
            f"{filename} must use the ``YOURUSER`` placeholder, not a concrete username."
        )
        # And specifically NOT any real username: a very coarse check
        # that catches the most common accident (my own username
        # accidentally committed).
        assert "/Users/fabrizzio" not in raw


class TestPlistCoversBothServices:
    """The deploy directory should cover both the MCP server and the
    dashboard. If a new service is added later, this test reminds
    contributors to ship its plist too.
    """

    def test_mcp_plist_exists(self):
        assert (_DEPLOY_DIR / "ai.abbacus.cortex.mcp.plist").exists()

    def test_dashboard_plist_exists(self):
        assert (_DEPLOY_DIR / "ai.abbacus.cortex.dashboard.plist").exists()

    def test_mcp_plist_launches_mcp_http(self):
        path = _DEPLOY_DIR / "ai.abbacus.cortex.mcp.plist"
        with path.open("rb") as f:
            data = plistlib.load(f)
        args = data["ProgramArguments"]
        assert "serve" in args
        assert "mcp-http" in args

    def test_dashboard_plist_launches_dashboard(self):
        path = _DEPLOY_DIR / "ai.abbacus.cortex.dashboard.plist"
        with path.open("rb") as f:
            data = plistlib.load(f)
        args = data["ProgramArguments"]
        assert "dashboard" in args
