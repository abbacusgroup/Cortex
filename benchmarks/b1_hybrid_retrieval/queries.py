"""Benchmark queries for hybrid 4-signal retrieval.

50 queries organized into 5 categories (10 each). Each query carries
ground-truth relevant object labels that correspond to labels produced
by the CorpusGenerator.

Object label index:
    decision_0..9, fix_10..21, lesson_22..29, session_30..34,
    research_35..39, idea_40..44, source_45..47, synthesis_48..49

Temporal model: age_days = (idx * 90) / 50, so lower idx = more recent.
"""

from __future__ import annotations

# ── Category 1: Keyword (exact terms in object content) ────────────

KEYWORD_QUERIES: list[dict] = [
    {
        "query": "Redis connection pool",
        "relevant": {"fix_10", "fix_18"},
        "category": "keyword",
    },
    {
        "query": "PostgreSQL migration deadlock",
        "relevant": {"fix_12", "decision_0"},
        "category": "keyword",
    },
    {
        "query": "JWT bearer tokens stateless auth",
        "relevant": {"decision_3", "synthesis_48"},
        "category": "keyword",
    },
    {
        "query": "Docker layer cache invalidation",
        "relevant": {"fix_14", "decision_6"},
        "category": "keyword",
    },
    {
        "query": "GraphQL mutations CSRF protection",
        "relevant": {"fix_20", "decision_4"},
        "category": "keyword",
    },
    {
        "query": "Kubernetes readiness probe crash loop",
        "relevant": {"fix_21", "decision_6"},
        "category": "keyword",
    },
    {
        "query": "PgBouncer connection pooling sizing",
        "relevant": {"decision_8", "source_45"},
        "category": "keyword",
    },
    {
        "query": "OpenTelemetry distributed tracing vendor",
        "relevant": {"research_39", "source_47"},
        "category": "keyword",
    },
    {
        "query": "circuit breaker fail fast timeout",
        "relevant": {"decision_5", "lesson_24"},
        "category": "keyword",
    },
    {
        "query": "N+1 query problem eager loading",
        "relevant": {"fix_19", "session_33"},
        "category": "keyword",
    },
]

# ── Category 2: Semantic (different wording, same topic) ──────────

SEMANTIC_QUERIES: list[dict] = [
    {
        "query": "database performance issues",
        "relevant": {
            "decision_0", "decision_8", "fix_12", "lesson_22",
            "lesson_29", "research_35", "session_31", "source_45",
            "synthesis_49",
        },
        "category": "semantic",
    },
    {
        "query": "securing API endpoints from abuse",
        "relevant": {
            "decision_9", "fix_20", "lesson_28", "session_34",
            "source_46",
        },
        "category": "semantic",
    },
    {
        "query": "improving system reliability with monitoring",
        "relevant": {
            "decision_7", "fix_16", "lesson_25", "research_39",
            "source_47",
        },
        "category": "semantic",
    },
    {
        "query": "container orchestration and scaling",
        "relevant": {
            "decision_6", "fix_14", "fix_21", "research_37",
            "session_32", "idea_41", "idea_42",
        },
        "category": "semantic",
    },
    {
        "query": "user login and identity management",
        "relevant": {
            "decision_3", "fix_11", "session_30", "research_36",
            "synthesis_48",
        },
        "category": "semantic",
    },
    {
        "query": "speeding up slow API responses",
        "relevant": {
            "fix_13", "fix_19", "session_33",
        },
        "category": "semantic",
    },
    {
        "query": "API schema design and client integration",
        "relevant": {
            "decision_4", "fix_15", "lesson_23", "idea_44",
        },
        "category": "semantic",
    },
    {
        "query": "automated quality assurance strategy",
        "relevant": {
            "fix_17", "lesson_26", "idea_43",
        },
        "category": "semantic",
    },
    {
        "query": "in-memory store for temporary data",
        "relevant": {
            "fix_10", "fix_18", "lesson_27", "research_38",
        },
        "category": "semantic",
    },
    {
        "query": "software design philosophy and modularity",
        "relevant": {
            "decision_1", "decision_2", "decision_5", "lesson_24",
            "idea_40",
        },
        "category": "semantic",
    },
]

# ── Category 3: Graph-boosted (high relationship counts) ──────────
# Objects wired with many relationships should rank higher.
# Relationship wiring: fixes causedBy decisions, lessons causedBy fixes,
# research supports decisions, supersedes between decisions.
# decision_0 (databases, alpha): has fix_12 causedBy, research_35 supports,
#   superseded by decision_8, contradicted by decision_3 => 4+ rels
# decision_3 (authentication, alpha): fix_11 causedBy, research_36 supports,
#   contradicts decision_0, fix_10 dependsOn => 4+ rels
# decision_6 (deployment, alpha): fix_14 causedBy, fix_21 causedBy,
#   fix_11 dependsOn => 3+ rels

GRAPH_BOOSTED_QUERIES: list[dict] = [
    {
        "query": "session storage database choice",
        "relevant": {"decision_0", "decision_8", "synthesis_49"},
        "category": "graph_boosted",
    },
    {
        "query": "API authentication token strategy",
        "relevant": {"decision_3", "synthesis_48", "fix_11"},
        "category": "graph_boosted",
    },
    {
        "query": "container deployment orchestration",
        "relevant": {"decision_6", "fix_14", "fix_21"},
        "category": "graph_boosted",
    },
    {
        "query": "caching layer connection management",
        "relevant": {"fix_10", "fix_18", "lesson_27"},
        "category": "graph_boosted",
    },
    {
        "query": "database migration and reliability",
        "relevant": {"fix_12", "lesson_22", "lesson_29"},
        "category": "graph_boosted",
    },
    {
        "query": "circuit breaker pattern for resilience",
        "relevant": {"decision_5", "lesson_24"},
        "category": "graph_boosted",
    },
    {
        "query": "event-driven notification architecture",
        "relevant": {"decision_1", "idea_40"},
        "category": "graph_boosted",
    },
    {
        "query": "graph database evaluation for relationships",
        "relevant": {"research_35", "decision_0", "decision_8"},
        "category": "graph_boosted",
    },
    {
        "query": "rate limiting and API abuse prevention",
        "relevant": {"decision_9", "lesson_28", "fix_20"},
        "category": "graph_boosted",
    },
    {
        "query": "structured logging and tracing infrastructure",
        "relevant": {"decision_7", "fix_16", "lesson_25"},
        "category": "graph_boosted",
    },
]

# ── Category 4: Recency (newer objects should rank higher) ────────
# age_days = (idx * 90) / 50  →  idx 0 ≈ 0 days, idx 49 ≈ 88 days
# Lower-index objects are more recent.

RECENCY_QUERIES: list[dict] = [
    {
        "query": "latest database decisions",
        "relevant": {"decision_0", "decision_8"},
        "category": "recency",
    },
    {
        "query": "recent architecture changes",
        "relevant": {"decision_1", "decision_2", "decision_5"},
        "category": "recency",
    },
    {
        "query": "newest caching fixes",
        "relevant": {"fix_10", "fix_18"},
        "category": "recency",
    },
    {
        "query": "current authentication approach",
        "relevant": {"decision_3", "fix_11", "session_30"},
        "category": "recency",
    },
    {
        "query": "freshest deployment configuration",
        "relevant": {"decision_6", "fix_14", "fix_21"},
        "category": "recency",
    },
    {
        "query": "most recent security findings",
        "relevant": {"decision_9", "fix_20", "session_34"},
        "category": "recency",
    },
    {
        "query": "up to date API design patterns",
        "relevant": {"decision_4", "fix_15", "lesson_23"},
        "category": "recency",
    },
    {
        "query": "latest performance investigation",
        "relevant": {"fix_13", "fix_19", "session_33"},
        "category": "recency",
    },
    {
        "query": "new observability tooling evaluation",
        "relevant": {"decision_7", "fix_16", "lesson_25"},
        "category": "recency",
    },
    {
        "query": "recent testing improvements",
        "relevant": {"fix_17", "lesson_26"},
        "category": "recency",
    },
]

# ── Category 5: Adversarial (one signal fails, others rescue) ─────
# These use synonyms / indirect phrasing so BM25 alone fails,
# or target objects without graph edges so graph alone fails.

ADVERSARIAL_QUERIES: list[dict] = [
    {
        # No keyword "Redis" / "cache" — BM25 fails; semantic rescues
        "query": "ephemeral key-value store connection issues",
        "relevant": {"fix_10", "fix_18", "lesson_27"},
        "category": "adversarial",
    },
    {
        # Uses synonym "credentials" instead of "auth/token" — BM25 weak
        "query": "credentials rotation race condition",
        "relevant": {"fix_11", "decision_3"},
        "category": "adversarial",
    },
    {
        # "OCI images" instead of "Docker" — BM25 fails
        "query": "OCI image build layer caching",
        "relevant": {"fix_14", "research_37"},
        "category": "adversarial",
    },
    {
        # "telemetry" instead of "logging" / "observability"
        "query": "telemetry correlation across services",
        "relevant": {"lesson_25", "decision_7", "source_47"},
        "category": "adversarial",
    },
    {
        # "RPC" instead of "API" — BM25 misses
        "query": "RPC pagination offset calculation bug",
        "relevant": {"fix_15"},
        "category": "adversarial",
    },
    {
        # Idea objects have few graph edges — graph signal fails
        "query": "offline synchronization conflict resolution",
        "relevant": {"idea_40"},
        "category": "adversarial",
    },
    {
        # Source objects have few graph edges — graph signal fails
        "query": "OWASP top vulnerabilities for web services",
        "relevant": {"source_46", "lesson_28"},
        "category": "adversarial",
    },
    {
        # "horizontal pod autoscaler" instead of "auto-scaling"
        "query": "horizontal pod autoscaler message queue depth",
        "relevant": {"idea_41", "idea_42"},
        "category": "adversarial",
    },
    {
        # "data persistence during failover" — indirect phrasing
        "query": "data persistence guarantees during failover",
        "relevant": {"lesson_27", "decision_0", "synthesis_49"},
        "category": "adversarial",
    },
    {
        # "embedding cosine" not in content — BM25 fails; semantic works
        "query": "embedding cosine duplicate detection",
        "relevant": {"idea_43"},
        "category": "adversarial",
    },
]

# ── Combined query set ─────────────────────────────────────────────

QUERIES: list[dict] = (
    KEYWORD_QUERIES
    + SEMANTIC_QUERIES
    + GRAPH_BOOSTED_QUERIES
    + RECENCY_QUERIES
    + ADVERSARIAL_QUERIES
)

assert len(QUERIES) == 50, f"Expected 50 queries, got {len(QUERIES)}"
