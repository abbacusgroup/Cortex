"""Atomic read-merge-write for ~/.cortex/.env files."""

from __future__ import annotations

import os
import re
from pathlib import Path

_KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def read_env(path: Path) -> dict[str, str]:
    """Read a .env file into a dict, ignoring comments and blank lines."""
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for line in path.read_text().splitlines():
        m = _KV_RE.match(line.strip())
        if m:
            result[m.group(1)] = m.group(2)
    return result


def write_env(path: Path, updates: dict[str, str]) -> None:
    """Merge *updates* into an existing .env file, preserving comments.

    - Existing KEY=VALUE lines are updated in place if KEY is in *updates*.
    - New keys are appended at the end.
    - Comments, blank lines, and unrecognised keys are preserved verbatim.
    - Write is atomic: .env.tmp -> rename -> chmod 0o600.
    """
    if not updates:
        return

    remaining = dict(updates)  # keys we still need to write
    lines: list[str] = []

    if path.is_file():
        for raw_line in path.read_text().splitlines():
            m = _KV_RE.match(raw_line.strip())
            if m and m.group(1) in remaining:
                key = m.group(1)
                lines.append(f"{key}={remaining.pop(key)}")
            else:
                lines.append(raw_line)

    # Append keys that weren't already in the file
    for key, value in remaining.items():
        lines.append(f"{key}={value}")

    # Atomic write
    tmp_path = path.with_suffix(".env.tmp")
    tmp_path.write_text("\n".join(lines) + "\n")
    os.chmod(tmp_path, 0o600)
    tmp_path.rename(path)
