"""Embedding service — pluggable providers for vector generation.

Supports two backends:
1. sentence-transformers (local, offline, via ``cortex[embeddings]``)
2. litellm (API-based: OpenAI, Ollama, Cohere, Voyage, etc. via ``cortex[llm]``)

The factory ``create_embedding_provider`` reads config and returns the right
provider, or None if the required dependency is missing (graceful degradation).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from cortex.core.config import CortexConfig
from cortex.core.logging import get_logger

logger = get_logger("services.embeddings")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    @property
    def model_name(self) -> str:
        """Canonical model identifier stored in the DB."""
        ...

    @property
    def available(self) -> bool:
        """Whether the provider is ready to produce embeddings."""
        ...

    def embed(self, text: str) -> list[float] | None:
        """Embed a single text string. Returns None on failure."""
        ...


class SentenceTransformerProvider:
    """Local embedding provider via sentence-transformers."""

    def __init__(self, model: str = "all-mpnet-base-v2"):
        self._model_name = model
        self._embedder: Any = None
        self._available: bool | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._check_import()
        return self._available

    def embed(self, text: str) -> list[float] | None:
        embedder = self._get_embedder()
        if embedder is None:
            return None
        try:
            vector = embedder.encode(text, normalize_embeddings=True)
            return [float(x) for x in vector]
        except Exception as e:
            logger.warning("sentence-transformers embed failed: %s", e)
            return None

    def warmup(self) -> bool:
        """Pre-load the model. Returns True on success."""
        return self._get_embedder() is not None

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self._model_name)
                logger.info("Loaded embedding model: %s", self._model_name)
            except Exception as e:
                logger.warning("Failed to load embedding model: %s", e)
                self._available = False
        return self._embedder

    @staticmethod
    def _check_import() -> bool:
        try:
            import sentence_transformers  # noqa: F401

            return True
        except ImportError:
            return False


class LiteLLMProvider:
    """API-based embedding provider via litellm.

    Covers OpenAI, Ollama, Cohere, Voyage, HuggingFace API, and more.
    Model strings follow litellm convention, e.g.:
      - "openai/text-embedding-3-small"
      - "ollama/nomic-embed-text"
      - "cohere/embed-english-v3.0"
      - "voyage/voyage-3"
    """

    def __init__(self, model: str, api_key: str = ""):
        self._model_name = model
        self._api_key = api_key
        self._available: bool | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._check_litellm()
        return self._available

    def embed(self, text: str) -> list[float] | None:
        if not self.available:
            return None
        try:
            import litellm

            kwargs: dict[str, Any] = {
                "model": self._model_name,
                "input": [text],
                "timeout": 30,
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key
            response = litellm.embedding(**kwargs)
            vector = response.data[0]["embedding"]
            return [float(x) for x in vector]
        except Exception as e:
            logger.warning("litellm embed failed: %s", e)
            return None

    def warmup(self) -> bool:
        """Verify connectivity by embedding a test string."""
        if not self.available:
            return False
        result = self.embed("warmup")
        return result is not None

    @staticmethod
    def _check_litellm() -> bool:
        try:
            import litellm  # noqa: F401

            return True
        except ImportError:
            return False


def create_embedding_provider(config: CortexConfig) -> EmbeddingProvider | None:
    """Create the embedding provider based on config.

    Returns None if the requested provider's dependency is missing.
    """
    provider_name = config.embedding_provider

    if provider_name == "litellm":
        provider = LiteLLMProvider(
            model=config.embedding_model,
            api_key=config.embedding_api_key,
        )
        if not provider.available:
            logger.warning("litellm not installed — embeddings disabled")
            return None
        return provider

    # Default: sentence-transformers
    provider = SentenceTransformerProvider(model=config.embedding_model)
    if not provider.available:
        logger.info(
            "sentence-transformers not installed — "
            "install cortex[embeddings] or set CORTEX_EMBEDDING_PROVIDER=litellm"
        )
        return None
    return provider


def check_embedding_model_consistency(
    content_store: Any,
    provider: EmbeddingProvider,
) -> str | None:
    """Check if stored embeddings match the current model.

    Returns a warning message if there's a mismatch, None if consistent.
    """
    row = content_store._db.execute(
        "SELECT model, dimensions FROM embeddings LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    stored_model = row["model"]
    if stored_model and stored_model != provider.model_name:
        return (
            f"Stored embeddings use model '{stored_model}' but current config "
            f"uses '{provider.model_name}'. Semantic search may return poor results. "
            f"Re-capture documents to regenerate embeddings with the new model."
        )
    return None
