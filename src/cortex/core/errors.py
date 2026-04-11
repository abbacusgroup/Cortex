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
    and stop the conflicting process.

    Four failure modes are distinguished:
    - Normal: another process is holding the lock; user stops it.
    - Stale (``is_stale=True``): the marker's PID is no longer running. Manual
      cleanup of both the marker file AND RocksDB's internal LOCK file may be
      required.
    - PID reuse (``is_pid_reuse=True``): the marker's PID is alive, but the
      process at that PID has a different command line than the marker recorded.
      The marker is probably stale and the OS reused the PID for an unrelated
      process. The actual lock holder cannot be identified — manual cleanup is
      the only path forward.
    - Cmdline unverified (``cmdline_unknown=True``): the marker's PID is alive,
      but ``_process_cmdline`` could not read the live cmdline (e.g. ``ps``
      timeout, transient ``/proc`` race, or missing permissions) so we cannot
      confirm whether this is the same process that recorded the marker. The
      error is raised conservatively (auto-recovery does NOT fire) and the
      user is told that the PID-match is unverified. Closes Bundle 8 / B.2.
    """

    code = "CORTEX_STORE_LOCKED"

    def __init__(
        self,
        message: str,
        *,
        holder_pid: int | None = None,
        holder_cmdline: str | None = None,
        is_stale: bool = False,
        is_pid_reuse: bool = False,
        cmdline_unknown: bool = False,
        db_path: str | None = None,
        marker_path: str | None = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        if holder_pid is not None and not isinstance(holder_pid, int):
            raise TypeError(f"holder_pid must be int or None, got {type(holder_pid).__name__}")
        self.holder_pid = holder_pid
        self.holder_cmdline = holder_cmdline
        self.is_stale = is_stale
        self.is_pid_reuse = is_pid_reuse
        self.cmdline_unknown = cmdline_unknown
        self.db_path = db_path
        self.marker_path = marker_path
        merged_context = dict(context or {})
        merged_context.setdefault("holder_pid", holder_pid)
        merged_context.setdefault("holder_cmdline", holder_cmdline)
        merged_context.setdefault("is_stale", is_stale)
        merged_context.setdefault("is_pid_reuse", is_pid_reuse)
        merged_context.setdefault("cmdline_unknown", cmdline_unknown)
        if db_path is not None:
            merged_context.setdefault("db_path", db_path)
        super().__init__(message, context=merged_context, cause=cause)

    def _cleanup_hint(self) -> str:
        """Return a cleanup hint the user can copy-paste.

        Prefers the guided ``cortex doctor unlock`` command when a marker
        path is known; falls back to raw ``rm`` commands for users who
        want to cleanup by hand.
        """
        parts: list[str] = []
        if self.marker_path or self.db_path:
            parts.append("Run: cortex doctor unlock")
        manual: list[str] = []
        if self.marker_path:
            manual.append(f"rm {self.marker_path}")
        if self.db_path:
            manual.append(f"rm -rf {self.db_path}/LOCK")
        if manual:
            parts.append("Manual cleanup: " + " ; ".join(manual))
        if not parts:
            return "Manual cleanup may be required."
        return " — ".join(parts)

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
                    f"is no longer running."
                )
                if self.context.get("auto_recovery_attempted"):
                    parts.append(
                        "Auto-recovery was already attempted and failed — "
                        "manual cleanup may be required."
                    )
                parts.append(self._cleanup_hint())
            elif self.is_pid_reuse:
                parts.append(
                    f"WARNING: PID {self.holder_pid} is alive but its current command "
                    f"line does NOT match the marker's recorded cmdline. The marker is "
                    f"probably stale and the OS reused the PID for an unrelated process. "
                    f"The actual graph DB lock holder cannot be identified from the marker."
                )
                parts.append(self._cleanup_hint())
            elif self.cmdline_unknown:
                parts.append(
                    f"NOTE: PID {self.holder_pid} is alive, but its current command "
                    f"line could NOT be read (ps/procfs returned no output). We cannot "
                    f"confirm this is the same process that recorded the marker. "
                    f"Auto-recovery was skipped for safety. If you are sure the marker "
                    f"is stale, use: cortex doctor unlock --force"
                )
                parts.append(self._cleanup_hint())
            else:
                parts.append(f"Stop the conflicting process or kill PID {self.holder_pid}.")
        elif self.context.get("marker_unreadable"):
            # The main message already explained the situation — don't add a
            # redundant "no marker file" tail.
            pass
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
