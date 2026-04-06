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
