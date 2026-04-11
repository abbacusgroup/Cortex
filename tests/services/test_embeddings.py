"""Tests for cortex.services.embeddings (pluggable embedding providers)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cortex.core.config import CortexConfig
from cortex.services.embeddings import (
    EmbeddingProvider,
    LiteLLMProvider,
    SentenceTransformerProvider,
    check_embedding_model_consistency,
    create_embedding_provider,
)

# -- SentenceTransformerProvider --------------------------------------------


class TestSentenceTransformerProvider:
    def test_model_name(self):
        p = SentenceTransformerProvider(model="test-model")
        assert p.model_name == "test-model"

    def test_available_false_without_package(self):
        with patch.object(SentenceTransformerProvider, "_check_import", return_value=False):
            p = SentenceTransformerProvider()
            assert p.available is False

    def test_available_true_with_package(self):
        with patch.object(SentenceTransformerProvider, "_check_import", return_value=True):
            p = SentenceTransformerProvider()
            assert p.available is True

    def test_embed_returns_list_of_floats(self):
        fake_model = MagicMock()
        fake_model.encode.return_value = [0.1, 0.2, 0.3]

        p = SentenceTransformerProvider()
        p._embedder = fake_model
        p._available = True

        result = p.embed("hello")
        assert result == [0.1, 0.2, 0.3]
        fake_model.encode.assert_called_once_with("hello", normalize_embeddings=True)

    def test_embed_returns_none_when_unavailable(self):
        p = SentenceTransformerProvider()
        p._available = False
        p._embedder = None  # Force no embedder
        # Patch _get_embedder to simulate unavailability
        with patch.object(p, "_get_embedder", return_value=None):
            assert p.embed("hello") is None

    def test_embed_returns_none_on_exception(self):
        fake_model = MagicMock()
        fake_model.encode.side_effect = RuntimeError("boom")

        p = SentenceTransformerProvider()
        p._embedder = fake_model
        p._available = True

        assert p.embed("hello") is None

    def test_warmup_returns_true_when_model_loads(self):
        fake_model = MagicMock()
        p = SentenceTransformerProvider()
        p._embedder = fake_model
        assert p.warmup() is True

    def test_warmup_returns_false_when_unavailable(self):
        p = SentenceTransformerProvider()
        p._available = False
        with patch.object(p, "_get_embedder", return_value=None):
            assert p.warmup() is False

    def test_conforms_to_protocol(self):
        p = SentenceTransformerProvider()
        assert isinstance(p, EmbeddingProvider)


# -- LiteLLMProvider --------------------------------------------------------


class TestLiteLLMProvider:
    def test_model_name(self):
        p = LiteLLMProvider(model="openai/text-embedding-3-small")
        assert p.model_name == "openai/text-embedding-3-small"

    def test_available_false_without_litellm(self):
        with patch.object(LiteLLMProvider, "_check_litellm", return_value=False):
            p = LiteLLMProvider(model="test")
            assert p.available is False

    def test_available_true_with_litellm(self):
        with patch.object(LiteLLMProvider, "_check_litellm", return_value=True):
            p = LiteLLMProvider(model="test")
            assert p.available is True

    def test_embed_returns_list_of_floats(self):
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.4, 0.5, 0.6]}]

        with (
            patch.object(LiteLLMProvider, "_check_litellm", return_value=True),
            patch("cortex.services.embeddings.LiteLLMProvider.embed") as mock_embed,
        ):
            mock_embed.return_value = [0.4, 0.5, 0.6]
            p = LiteLLMProvider(model="openai/text-embedding-3-small")
            result = p.embed("hello")
            assert result == [0.4, 0.5, 0.6]

    def test_embed_returns_none_when_unavailable(self):
        with patch.object(LiteLLMProvider, "_check_litellm", return_value=False):
            p = LiteLLMProvider(model="test")
            assert p.embed("hello") is None

    def test_embed_returns_none_on_exception(self):
        with patch.object(LiteLLMProvider, "_check_litellm", return_value=True):
            p = LiteLLMProvider(model="test")
            # litellm not actually installed, so import will fail inside embed()
            # which is caught and returns None
            result = p.embed("hello")
            assert result is None

    def test_api_key_stored(self):
        p = LiteLLMProvider(model="test", api_key="sk-test")
        assert p._api_key == "sk-test"

    def test_conforms_to_protocol(self):
        p = LiteLLMProvider(model="test")
        assert isinstance(p, EmbeddingProvider)


# -- create_embedding_provider factory --------------------------------------


class TestCreateEmbeddingProvider:
    def test_default_returns_sentence_transformer(self, tmp_path):
        with patch.object(SentenceTransformerProvider, "_check_import", return_value=True):
            cfg = CortexConfig(data_dir=tmp_path)
            provider = create_embedding_provider(cfg)
            assert isinstance(provider, SentenceTransformerProvider)
            assert provider.model_name == "all-mpnet-base-v2"

    def test_litellm_returns_litellm_provider(self, tmp_path):
        with patch.object(LiteLLMProvider, "_check_litellm", return_value=True):
            cfg = CortexConfig(
                data_dir=tmp_path,
                embedding_provider="litellm",
                embedding_model="openai/text-embedding-3-small",
                embedding_api_key="sk-test",
            )
            provider = create_embedding_provider(cfg)
            assert isinstance(provider, LiteLLMProvider)
            assert provider.model_name == "openai/text-embedding-3-small"

    def test_missing_deps_returns_none(self, tmp_path):
        with patch.object(SentenceTransformerProvider, "_check_import", return_value=False):
            cfg = CortexConfig(data_dir=tmp_path)
            provider = create_embedding_provider(cfg)
            assert provider is None

    def test_litellm_missing_returns_none(self, tmp_path):
        with patch.object(LiteLLMProvider, "_check_litellm", return_value=False):
            cfg = CortexConfig(
                data_dir=tmp_path,
                embedding_provider="litellm",
                embedding_model="openai/text-embedding-3-small",
            )
            provider = create_embedding_provider(cfg)
            assert provider is None


# -- check_embedding_model_consistency --------------------------------------


class TestModelConsistency:
    def _make_store_mock(self, *, model: str | None = None, dimensions: int = 768):
        """Create a mock ContentStore with optional embedding row."""
        mock = MagicMock()
        if model is None:
            mock._db.execute.return_value.fetchone.return_value = None
        else:
            mock._db.execute.return_value.fetchone.return_value = {
                "model": model,
                "dimensions": dimensions,
            }
        return mock

    def test_no_stored_embeddings_returns_none(self):
        store = self._make_store_mock(model=None)
        provider = SentenceTransformerProvider(model="all-mpnet-base-v2")
        assert check_embedding_model_consistency(store, provider) is None

    def test_matching_model_returns_none(self):
        store = self._make_store_mock(model="all-mpnet-base-v2")
        provider = SentenceTransformerProvider(model="all-mpnet-base-v2")
        assert check_embedding_model_consistency(store, provider) is None

    def test_mismatched_model_returns_warning(self):
        store = self._make_store_mock(model="all-mpnet-base-v2")
        provider = LiteLLMProvider(model="openai/text-embedding-3-small")
        result = check_embedding_model_consistency(store, provider)
        assert result is not None
        assert "all-mpnet-base-v2" in result
        assert "openai/text-embedding-3-small" in result

    def test_empty_stored_model_returns_none(self):
        store = self._make_store_mock(model="")
        provider = SentenceTransformerProvider(model="all-mpnet-base-v2")
        assert check_embedding_model_consistency(store, provider) is None
