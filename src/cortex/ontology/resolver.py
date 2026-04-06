"""Centralized ontology path resolution."""

from __future__ import annotations

from pathlib import Path

from cortex.core.logging import get_logger

logger = get_logger("ontology.resolver")


def find_ontology() -> Path:
    """Find the cortex.ttl ontology file.

    Resolution order:
    1. Package-relative: same directory as this module
    2. Project-root: ../../ontology/cortex.ttl (development)

    Returns:
        Path to cortex.ttl

    Raises:
        FileNotFoundError if not found in any location.
    """
    # 1. Package-relative (works after pip install)
    pkg_path = Path(__file__).parent / "cortex.ttl"
    if pkg_path.exists():
        return pkg_path

    # 2. Project root (development layout: src/cortex/ontology/ -> ../../ontology/)
    project_path = Path(__file__).parent.parent.parent.parent / "ontology" / "cortex.ttl"
    if project_path.exists():
        return project_path

    raise FileNotFoundError("cortex.ttl not found in package or project root")
