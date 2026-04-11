"""Synthetic embedding vectors for benchmarks.

Produces deterministic, controlled embeddings without requiring
sentence-transformers or any model download. Topic centroids ensure
that objects about the same topic cluster together (high cosine
similarity) while different topics diverge.
"""

from __future__ import annotations

import math
import random
import struct

DIMENSIONS = 16


def make_topic_centroid(topic_idx: int, dim: int = DIMENSIONS) -> list[float]:
    """Deterministic centroid for a topic.

    Each topic gets a unique unit vector plus structured noise
    so centroids are well-separated in the embedding space.
    """
    rng = random.Random(topic_idx * 31337)
    raw = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def make_embedding(centroid: list[float], noise: float = 0.1, seed: int = 0) -> list[float]:
    """Generate an embedding vector near a centroid."""
    rng = random.Random(seed)
    return [c + rng.gauss(0, noise) for c in centroid]


def pack_embedding(vector: list[float]) -> bytes:
    """Pack a float vector into bytes for ContentStore.store_embedding."""
    return struct.pack(f"{len(vector)}f", *vector)


# Pre-computed topic centroids (10 topics)
TOPIC_NAMES = [
    "caching",        # 0
    "authentication", # 1
    "deployment",     # 2
    "databases",      # 3
    "api_design",     # 4
    "testing",        # 5
    "observability",  # 6
    "architecture",   # 7
    "performance",    # 8
    "security",       # 9
]

TOPIC_CENTROIDS = {name: make_topic_centroid(i) for i, name in enumerate(TOPIC_NAMES)}


class SyntheticEmbeddingProvider:
    """Embedding provider for benchmarks.

    Fulfills the EmbeddingProvider protocol. Maps query text to
    a deterministic embedding based on keyword matching to topics.
    """

    @property
    def model_name(self) -> str:
        return "bench-synthetic"

    @property
    def available(self) -> bool:
        return True

    def embed(self, text: str) -> list[float] | None:
        """Embed text by matching keywords to topic centroids."""
        text_lower = text.lower()

        # Find best-matching topic by keyword overlap
        best_topic = "architecture"  # fallback
        best_score = 0
        keyword_map = {
            "caching": ["redis", "cache", "memcach", "ttl", "evict"],
            "authentication": ["auth", "login", "token", "oauth", "session", "jwt", "password"],
            "deployment": ["deploy", "docker", "kubernetes", "k8s", "ci/cd", "pipeline", "container"],
            "databases": ["postgres", "sql", "database", "migration", "query", "index", "sqlite"],
            "api_design": ["api", "endpoint", "rest", "graphql", "schema", "route"],
            "testing": ["test", "pytest", "mock", "fixture", "assert", "coverage"],
            "observability": ["log", "metric", "monitor", "alert", "trace", "observ"],
            "architecture": ["architect", "pattern", "design", "refactor", "monorepo", "microservice"],
            "performance": ["perf", "latency", "throughput", "bottleneck", "optim", "slow", "fast"],
            "security": ["secur", "vulnerab", "inject", "xss", "csrf", "encrypt"],
        }

        for topic, keywords in keyword_map.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_topic = topic

        centroid = TOPIC_CENTROIDS[best_topic]
        # Use hash of text as seed for reproducible noise
        seed = hash(text) % (2**31)
        return make_embedding(centroid, noise=0.05, seed=seed)
