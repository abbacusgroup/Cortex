"""Cortex error hierarchy.

All errors serialize to a consistent JSON structure:
    {"code": "CORTEX_XXX", "message": "...", "context": {...}}
"""

from __future__ import annotations

import json
from typing import Any


class CortexError(Exception):
    """Base error for all Cortex operations."""

    code: str = "CORTEX_ERROR"

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        self.message = message
        self.context = context or {}
        if cause is not None:
            self.__cause__ = cause
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }
        if self.__cause__ is not None:
            if isinstance(self.__cause__, CortexError):
                result["cause"] = self.__cause__.to_dict()
            else:
                result["cause"] = {"code": "EXTERNAL", "message": str(self.__cause__)}
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# --- Config ---

class ConfigError(CortexError):
    code = "CORTEX_CONFIG_ERROR"


class ConfigNotFoundError(ConfigError):
    code = "CORTEX_CONFIG_NOT_FOUND"


class ConfigPermissionError(ConfigError):
    code = "CORTEX_CONFIG_PERMISSION"


# --- Store ---

class StoreError(CortexError):
    code = "CORTEX_STORE_ERROR"


class NotFoundError(StoreError):
    code = "CORTEX_NOT_FOUND"


class DuplicateError(StoreError):
    code = "CORTEX_DUPLICATE"


class SyncError(StoreError):
    code = "CORTEX_SYNC_ERROR"


class StoreLockedError(StoreError):
    """Raised when the graph store cannot be opened because another process holds the lock.

    Carries the holder's PID and command line (when available) so the user can identify
    and stop the conflicting process. Set ``is_stale=True`` when the recorded PID is no
    longer running, indicating an orphaned lock marker.
    """

    code = "CORTEX_STORE_LOCKED"

    def __init__(
        self,
        message: str,
        *,
        holder_pid: int | None = None,
        holder_cmdline: str | None = None,
        is_stale: bool = False,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        if holder_pid is not None and not isinstance(holder_pid, int):
            raise TypeError(f"holder_pid must be int or None, got {type(holder_pid).__name__}")
        self.holder_pid = holder_pid
        self.holder_cmdline = holder_cmdline
        self.is_stale = is_stale
        merged_context = dict(context or {})
        merged_context.setdefault("holder_pid", holder_pid)
        merged_context.setdefault("holder_cmdline", holder_cmdline)
        merged_context.setdefault("is_stale", is_stale)
        super().__init__(message, context=merged_context, cause=cause)

    def __str__(self) -> str:
        parts = [self.message]
        if self.holder_pid is not None:
            holder_desc = f"PID {self.holder_pid}"
            if self.holder_cmdline:
                holder_desc += f" ({self.holder_cmdline})"
            parts.append(f"Lock holder: {holder_desc}.")
            if self.is_stale:
                parts.append(
                    f"This appears to be a stale lock marker — process {self.holder_pid} "
                    f"is no longer running. Manual cleanup may be required."
                )
            else:
                parts.append(f"Stop the conflicting process or kill PID {self.holder_pid}.")
        else:
            parts.append("Lock holder unknown (no marker file).")
        return " ".join(parts)


# --- Ontology ---

class OntologyError(CortexError):
    code = "CORTEX_ONTOLOGY_ERROR"


class ValidationError(CortexError):
    code = "CORTEX_VALIDATION_ERROR"


# --- Pipeline ---

class PipelineError(CortexError):
    code = "CORTEX_PIPELINE_ERROR"


class ClassificationError(PipelineError):
    code = "CORTEX_CLASSIFICATION_ERROR"


class LLMError(PipelineError):
    code = "CORTEX_LLM_ERROR"


# --- Transport ---

class TransportError(CortexError):
    code = "CORTEX_TRANSPORT_ERROR"


class AuthenticationError(TransportError):
    code = "CORTEX_AUTH_ERROR"
