"""Deterministic corpus generator for benchmarks.

Creates a realistic knowledge base of software engineering objects with
controlled relationships, entities, and embeddings. Fully seeded —
produces identical output every run.
"""

from __future__ import annotations

import datetime
import random
import struct
from typing import Any

from cortex.db.store import Store

from .embeddings import (
    DIMENSIONS,
    TOPIC_CENTROIDS,
    make_embedding,
    pack_embedding,
)

# ─── Object Templates ─────────────────────────────────────────────

TEMPLATES: dict[str, list[dict[str, str]]] = {
    "decision": [
        {"title": "Use PostgreSQL for session storage", "content": "Decided to migrate session store from Redis to PostgreSQL for durability and simplified ops. Evaluated Redis Cluster vs PostgreSQL with row-level locking. PostgreSQL wins on durability guarantees.", "topic": "databases", "project": "alpha"},
        {"title": "Adopt event-driven architecture for notifications", "content": "Moving from synchronous API calls to an event bus for notification delivery. This decouples services and improves resilience when downstream systems are unavailable.", "topic": "architecture", "project": "alpha"},
        {"title": "Switch to monorepo for frontend packages", "content": "Consolidating 12 frontend packages into a single monorepo using Turborepo. Reduces CI complexity and makes cross-package changes atomic.", "topic": "architecture", "project": "beta"},
        {"title": "Use JWT tokens for API authentication", "content": "Replacing session cookies with JWT bearer tokens for the public API. Enables stateless auth and simplifies horizontal scaling of API servers.", "topic": "authentication", "project": "alpha"},
        {"title": "Adopt GraphQL for mobile API", "content": "Mobile clients need flexible queries. Switching from REST to GraphQL for the mobile-facing API to reduce over-fetching and number of round-trips.", "topic": "api_design", "project": "beta"},
        {"title": "Implement circuit breaker for external APIs", "content": "Adding circuit breaker pattern around all external API calls. When downstream services fail, we fail fast instead of cascading timeouts.", "topic": "architecture", "project": "alpha"},
        {"title": "Use Kubernetes for container orchestration", "content": "Migrating from docker-compose to Kubernetes for production deployment. Enables auto-scaling, rolling updates, and better resource utilization.", "topic": "deployment", "project": "alpha"},
        {"title": "Adopt structured logging with JSON format", "content": "Switching from plain text logs to structured JSON logging. Makes log aggregation in Elasticsearch much more effective for debugging.", "topic": "observability", "project": "beta"},
        {"title": "Use connection pooling for database access", "content": "Implementing PgBouncer connection pooling to manage PostgreSQL connections. Reduces connection overhead and prevents pool exhaustion under load.", "topic": "databases", "project": "alpha"},
        {"title": "Implement rate limiting on public API", "content": "Adding token bucket rate limiting to prevent abuse of public endpoints. Configurable per-client and per-endpoint limits with Redis as the counter store.", "topic": "security", "project": "beta"},
    ],
    "fix": [
        {"title": "Fix Redis connection pool exhaustion", "content": "Connection pool was leaking connections when timeout exceptions occurred in the retry logic. Added proper cleanup in the finally block and set max_connections=50.", "topic": "caching", "project": "alpha"},
        {"title": "Fix race condition in auth token refresh", "content": "Multiple concurrent requests would all trigger token refresh simultaneously, causing 401 cascades. Added a mutex lock around the refresh logic.", "topic": "authentication", "project": "alpha"},
        {"title": "Fix PostgreSQL migration deadlock", "content": "ALTER TABLE was deadlocking with long-running read queries. Changed to use CREATE INDEX CONCURRENTLY and applied DDL changes in smaller batches.", "topic": "databases", "project": "alpha"},
        {"title": "Fix memory leak in WebSocket handler", "content": "Event listeners were not being removed when WebSocket connections closed. Added proper cleanup in the disconnect handler to prevent memory growth.", "topic": "performance", "project": "beta"},
        {"title": "Fix Docker build cache invalidation", "content": "Changes to package.json were invalidating the entire Docker layer cache. Restructured Dockerfile to copy package files before source code.", "topic": "deployment", "project": "alpha"},
        {"title": "Fix incorrect pagination in search results", "content": "Off-by-one error in the OFFSET calculation caused duplicate results when paginating. Fixed the offset formula and added boundary tests.", "topic": "api_design", "project": "beta"},
        {"title": "Fix log rotation causing data loss", "content": "Logrotate was configured with copytruncate which could lose lines during rotation. Switched to create mode with proper signal handling.", "topic": "observability", "project": "beta"},
        {"title": "Fix slow test suite from database setup", "content": "Each test was creating and migrating a fresh database. Switched to transaction rollback isolation pattern — 10x speedup in test execution.", "topic": "testing", "project": "alpha"},
        {"title": "Fix Redis sentinel failover not detected", "content": "Client library was caching the master address and not re-resolving after sentinel failover. Updated Redis client configuration to enable sentinel discovery.", "topic": "caching", "project": "alpha"},
        {"title": "Fix API response time regression", "content": "N+1 query problem introduced in the user profile endpoint refactor. Added eager loading with SELECT ... JOIN and response times dropped from 800ms to 40ms.", "topic": "performance", "project": "alpha"},
        {"title": "Fix CSRF token validation on GraphQL endpoint", "content": "GraphQL mutations were bypassing CSRF protection because the middleware only checked POST forms, not JSON bodies. Extended validation to all mutation requests.", "topic": "security", "project": "beta"},
        {"title": "Fix Kubernetes pod crash loop on startup", "content": "Readiness probe was hitting an endpoint that required database connection, but the DB wasn't ready yet. Added a simple health endpoint that doesn't need DB.", "topic": "deployment", "project": "alpha"},
    ],
    "lesson": [
        {"title": "Connection pools need bounded lifetimes", "content": "Learned that connection pools must have max lifetime settings, not just max idle. Long-lived connections accumulate server-side state and eventually cause issues.", "topic": "databases", "project": "alpha"},
        {"title": "Never trust client-side timestamps", "content": "Using client-provided timestamps for ordering events caused data inconsistency. Always use server-generated timestamps for anything authoritative.", "topic": "api_design", "project": "beta"},
        {"title": "Circuit breakers need half-open state testing", "content": "Our initial circuit breaker implementation went from open to closed without testing. The half-open state that sends probe requests is essential for safe recovery.", "topic": "architecture", "project": "alpha"},
        {"title": "Structured logs must include correlation IDs", "content": "JSON logs without request correlation IDs are just organized noise. Every log line must carry a trace ID to enable end-to-end request tracking.", "topic": "observability", "project": "beta"},
        {"title": "Test isolation beats test speed", "content": "Sharing database state between tests saved CI time but caused intermittent failures. Transaction rollback isolation is the right trade-off: fast AND reliable.", "topic": "testing", "project": "alpha"},
        {"title": "Redis is not a durable data store", "content": "Using Redis as the primary store for session data caused data loss during failover. Redis is a cache — anything that must survive restarts belongs in PostgreSQL.", "topic": "caching", "project": "alpha"},
        {"title": "Rate limits need graceful degradation", "content": "Hard rate limits with 429 errors frustrated legitimate users. Implemented graduated throttling that slows responses before hard-blocking.", "topic": "security", "project": "beta"},
        {"title": "Database migrations must be backward compatible", "content": "A column rename broke the running application during deployment. All migrations must work with both old and new code versions simultaneously.", "topic": "databases", "project": "alpha"},
    ],
    "session": [
        {"title": "Sprint 14 planning — auth refactor scope", "content": "Scoped the authentication refactor to 3 sprints. Sprint 14 handles token migration, Sprint 15 handles session management, Sprint 16 handles cleanup.", "topic": "authentication", "project": "alpha"},
        {"title": "Incident review — production database outage", "content": "Reviewed the 45-minute production outage caused by a runaway migration. Established new migration review process and automated rollback procedures.", "topic": "databases", "project": "alpha"},
        {"title": "Architecture review — API gateway evaluation", "content": "Evaluated Kong, Traefik, and custom Nginx for API gateway. Chose Traefik for its Kubernetes-native integration and automatic certificate management.", "topic": "deployment", "project": "beta"},
        {"title": "Performance review — Q3 latency targets", "content": "Reviewed p99 latency targets for Q3. Current p99 is 340ms, target is 200ms. Identified N+1 queries and missing cache layers as primary bottlenecks.", "topic": "performance", "project": "alpha"},
        {"title": "Security audit — penetration test findings", "content": "Reviewed results from external penetration test. Found 2 high-severity issues: CSRF bypass on GraphQL and missing rate limiting on login endpoint.", "topic": "security", "project": "beta"},
    ],
    "research": [
        {"title": "Evaluation of graph databases for relationship queries", "content": "Compared Neo4j, Oxigraph, and PostgreSQL with recursive CTEs for relationship-heavy queries. Oxigraph wins for RDF/SPARQL workloads; PostgreSQL adequate for simpler graphs.", "topic": "databases", "project": "alpha"},
        {"title": "Survey of authentication protocols for APIs", "content": "Reviewed OAuth 2.0, API keys, and mutual TLS for service-to-service auth. OAuth 2.0 with client credentials grant is the recommended pattern.", "topic": "authentication", "project": "alpha"},
        {"title": "Analysis of container runtime alternatives", "content": "Compared Docker, Podman, and containerd for production workloads. containerd is lighter and Kubernetes-native. Docker adds unnecessary daemon overhead.", "topic": "deployment", "project": "beta"},
        {"title": "Benchmark of caching strategies for API responses", "content": "Tested Redis, in-memory LRU, and CDN caching for API response caching. Redis best for multi-instance consistency; CDN best for public read-heavy endpoints.", "topic": "caching", "project": "alpha"},
        {"title": "Study of observability platforms for distributed tracing", "content": "Evaluated Jaeger, Zipkin, and OpenTelemetry for distributed tracing. OpenTelemetry provides vendor-neutral instrumentation with export to any backend.", "topic": "observability", "project": "beta"},
    ],
    "idea": [
        {"title": "Explore CRDT-based conflict resolution for offline sync", "content": "Could use CRDTs to handle conflicting edits when mobile clients go offline. Would eliminate the need for manual conflict resolution in the sync protocol.", "topic": "architecture", "project": "beta"},
        {"title": "Auto-scaling based on queue depth instead of CPU", "content": "Current CPU-based auto-scaling is reactive. Scaling based on message queue depth would be predictive — scaling up before requests hit the servers.", "topic": "deployment", "project": "alpha"},
        {"title": "Implement canary deployments with traffic splitting", "content": "Instead of rolling updates, route 5% of traffic to new version first. If error rate stays flat, gradually increase. Catches regressions before full rollout.", "topic": "deployment", "project": "alpha"},
        {"title": "Use embedding similarity for duplicate detection", "content": "Instead of exact-match deduplication, use embedding cosine similarity to find near-duplicate content. Would catch paraphrased duplicates that exact match misses.", "topic": "testing", "project": "beta"},
        {"title": "Build a query cost estimator for GraphQL", "content": "Complex nested GraphQL queries can accidentally generate expensive database operations. A cost estimator could reject or paginate queries exceeding a cost threshold.", "topic": "api_design", "project": "beta"},
    ],
    "source": [
        {"title": "PostgreSQL documentation on connection pooling", "content": "Official PostgreSQL docs on connection management, pool sizing, and PgBouncer integration. Key insight: pool size should be roughly 2x CPU cores.", "topic": "databases", "project": "alpha"},
        {"title": "OWASP API Security Top 10", "content": "Reference document for API security best practices. Covers broken authentication, injection, excessive data exposure, and rate limiting recommendations.", "topic": "security", "project": "beta"},
        {"title": "Google SRE book on monitoring and alerting", "content": "Chapter on monitoring philosophy: the four golden signals (latency, traffic, errors, saturation) and why symptom-based alerting beats cause-based.", "topic": "observability", "project": "alpha"},
    ],
    "synthesis": [
        {"title": "Q3 authentication overhaul summary", "content": "Synthesized all auth-related decisions, fixes, and lessons from Q3. Key theme: migration from session cookies to JWT tokens revealed several edge cases in token refresh and CSRF protection.", "topic": "authentication", "project": "alpha"},
        {"title": "Database reliability patterns learned in 2025", "content": "Compiled lessons from 4 database incidents. Common thread: connection management and migration safety are the two highest-leverage areas for reliability improvement.", "topic": "databases", "project": "alpha"},
    ],
}

# ─── Entity Templates ──────────────────────────────────────────────

ENTITY_TEMPLATES: list[dict[str, str]] = [
    {"name": "Redis", "type": "technology"},
    {"name": "PostgreSQL", "type": "technology"},
    {"name": "Kubernetes", "type": "technology"},
    {"name": "Docker", "type": "technology"},
    {"name": "FastAPI", "type": "technology"},
    {"name": "GraphQL", "type": "technology"},
    {"name": "JWT", "type": "technology"},
    {"name": "OAuth", "type": "technology"},
    {"name": "Elasticsearch", "type": "technology"},
    {"name": "OpenTelemetry", "type": "technology"},
    {"name": "PgBouncer", "type": "technology"},
    {"name": "Traefik", "type": "technology"},
    {"name": "circuit-breaker", "type": "pattern"},
    {"name": "connection-pooling", "type": "pattern"},
    {"name": "event-driven", "type": "pattern"},
    {"name": "caching", "type": "concept"},
    {"name": "authentication", "type": "concept"},
    {"name": "observability", "type": "concept"},
    {"name": "rate-limiting", "type": "concept"},
    {"name": "deployment-pipeline", "type": "concept"},
    {"name": "alpha", "type": "project"},
    {"name": "beta", "type": "project"},
]

# ─── Relationship Templates ────────────────────────────────────────

# Relationships are defined as (from_label_prefix, rel_type, to_label_prefix)
# These get wired by the generator based on matching topic/project

ENTITY_MENTION_MAP: dict[str, list[str]] = {
    "caching": ["Redis"],
    "authentication": ["JWT", "OAuth"],
    "deployment": ["Kubernetes", "Docker", "Traefik"],
    "databases": ["PostgreSQL", "PgBouncer"],
    "api_design": ["GraphQL", "FastAPI"],
    "observability": ["Elasticsearch", "OpenTelemetry"],
    "architecture": ["circuit-breaker", "event-driven"],
    "performance": ["connection-pooling"],
    "security": ["rate-limiting"],
    "testing": [],
}


class CorpusGenerator:
    """Deterministic corpus generator for benchmarks.

    Args:
        store: The Store to populate.
        seed: Random seed for reproducibility.
    """

    def __init__(self, store: Store, seed: int = 42):
        self.store = store
        self.rng = random.Random(seed)
        self.obj_ids: dict[str, str] = {}       # label -> obj_id
        self.entity_ids: dict[str, str] = {}     # entity_name -> entity_id
        self.obj_topics: dict[str, str] = {}     # label -> topic
        self.obj_projects: dict[str, str] = {}   # label -> project

    def generate(self) -> dict[str, str]:
        """Generate the full corpus. Returns label -> obj_id mapping."""
        self._create_entities()
        self._create_objects()
        self._create_embeddings()
        self._create_relationships()
        self._create_entity_mentions()
        return dict(self.obj_ids)

    def _create_entities(self) -> None:
        """Create all entity nodes."""
        for tmpl in ENTITY_TEMPLATES:
            eid, _ = self.store.create_entity(
                name=tmpl["name"], entity_type=tmpl["type"]
            )
            self.entity_ids[tmpl["name"]] = eid

    def _create_objects(self) -> None:
        """Create all knowledge objects with temporal spread."""
        now = datetime.datetime.now(datetime.UTC)
        idx = 0

        for obj_type, templates in TEMPLATES.items():
            for tmpl in templates:
                # Spread objects over 90 days using deterministic ordering
                age_days = (idx * 90) / self._total_object_count()
                created = now - datetime.timedelta(days=age_days)
                created_str = created.isoformat()

                label = f"{obj_type}_{idx}"
                obj_id = self.store.create(
                    obj_type=obj_type,
                    title=tmpl["title"],
                    content=tmpl["content"],
                    project=tmpl.get("project", ""),
                    tags=f"{obj_type},{tmpl.get('topic', '')}",
                    summary=tmpl["title"],
                    confidence=0.9,
                    created_at=created_str,
                    updated_at=created_str,
                )
                self.obj_ids[label] = obj_id
                self.obj_topics[label] = tmpl.get("topic", "architecture")
                self.obj_projects[label] = tmpl.get("project", "")
                idx += 1

    def _create_embeddings(self) -> None:
        """Create synthetic embeddings for all objects."""
        for label, obj_id in self.obj_ids.items():
            topic = self.obj_topics[label]
            centroid = TOPIC_CENTROIDS.get(topic, TOPIC_CENTROIDS["architecture"])
            seed = hash(label) % (2**31)
            vector = make_embedding(centroid, noise=0.08, seed=seed)
            self.store.content.store_embedding(
                doc_id=obj_id,
                embedding=pack_embedding(vector),
                model="bench-synthetic",
                dimensions=DIMENSIONS,
            )

    def _create_relationships(self) -> None:
        """Wire relationships between objects based on realistic patterns."""
        labels = list(self.obj_ids.keys())

        # Fixes causedBy decisions (same topic, same project)
        for label in labels:
            if not label.startswith("fix_"):
                continue
            topic = self.obj_topics[label]
            project = self.obj_projects[label]
            # Find a decision with same topic and project
            for other in labels:
                if (
                    other.startswith("decision_")
                    and self.obj_topics[other] == topic
                    and self.obj_projects[other] == project
                ):
                    self.store.create_relationship(
                        from_id=self.obj_ids[label],
                        rel_type="causedBy",
                        to_id=self.obj_ids[other],
                    )
                    break

        # Lessons causedBy fixes (same topic)
        for label in labels:
            if not label.startswith("lesson_"):
                continue
            topic = self.obj_topics[label]
            for other in labels:
                if (
                    other.startswith("fix_")
                    and self.obj_topics[other] == topic
                ):
                    self.store.create_relationship(
                        from_id=self.obj_ids[label],
                        rel_type="causedBy",
                        to_id=self.obj_ids[other],
                    )
                    break

        # Research supports decisions (same topic)
        for label in labels:
            if not label.startswith("research_"):
                continue
            topic = self.obj_topics[label]
            for other in labels:
                if (
                    other.startswith("decision_")
                    and self.obj_topics[other] == topic
                ):
                    self.store.create_relationship(
                        from_id=self.obj_ids[label],
                        rel_type="supports",
                        to_id=self.obj_ids[other],
                    )
                    break

        # Supersedes: later decisions supersede earlier ones (same topic)
        decisions_by_topic: dict[str, list[str]] = {}
        for label in labels:
            if label.startswith("decision_"):
                topic = self.obj_topics[label]
                decisions_by_topic.setdefault(topic, []).append(label)

        for topic, dec_labels in decisions_by_topic.items():
            if len(dec_labels) >= 2:
                # Later supersedes earlier
                self.store.create_relationship(
                    from_id=self.obj_ids[dec_labels[-1]],
                    rel_type="supersedes",
                    to_id=self.obj_ids[dec_labels[0]],
                )

        # Contradicts: plant 3 contradictions for B3
        contradiction_pairs = [
            ("decision_3", "decision_0"),   # JWT vs PostgreSQL sessions
            ("decision_4", "decision_5"),   # GraphQL vs circuit breaker
        ]
        for from_label, to_label in contradiction_pairs:
            if from_label in self.obj_ids and to_label in self.obj_ids:
                self.store.create_relationship(
                    from_id=self.obj_ids[from_label],
                    rel_type="contradicts",
                    to_id=self.obj_ids[to_label],
                )

        # dependsOn: some fixes depend on decisions
        depends_pairs = [
            ("fix_10", "decision_3"),  # fix depends on JWT decision
            ("fix_11", "decision_6"),  # fix depends on K8s decision
        ]
        for from_label, to_label in depends_pairs:
            if from_label in self.obj_ids and to_label in self.obj_ids:
                self.store.create_relationship(
                    from_id=self.obj_ids[from_label],
                    rel_type="dependsOn",
                    to_id=self.obj_ids[to_label],
                )

    def _create_entity_mentions(self) -> None:
        """Link objects to entities based on topic → entity mapping."""
        for label, obj_id in self.obj_ids.items():
            topic = self.obj_topics[label]
            entity_names = ENTITY_MENTION_MAP.get(topic, [])
            for name in entity_names:
                if name in self.entity_ids:
                    self.store.add_mention(
                        obj_id=obj_id, entity_id=self.entity_ids[name]
                    )

    def _total_object_count(self) -> int:
        return sum(len(v) for v in TEMPLATES.values())

    def labels_by_type(self, obj_type: str) -> list[str]:
        """Get all labels for a given object type."""
        return [l for l in self.obj_ids if l.startswith(f"{obj_type}_")]

    def labels_by_topic(self, topic: str) -> list[str]:
        """Get all labels matching a topic."""
        return [l for l, t in self.obj_topics.items() if t == topic]

    def labels_by_project(self, project: str) -> list[str]:
        """Get all labels matching a project."""
        return [l for l, p in self.obj_projects.items() if p == project]
