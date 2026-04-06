"""Oxigraph-backed RDF graph store for Cortex.

Handles CRUD for knowledge objects, relationships, and entities as RDF triples.
Persistence via Oxigraph's built-in storage to ~/.cortex/graph.db.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyoxigraph as ox

from cortex.core.errors import NotFoundError, OntologyError, StoreError, ValidationError
from cortex.core.logging import get_logger
from cortex.ontology.namespaces import (
    CLASS_MAP,
    CORTEX,
    ENTITY_CLASS,
    ENTITY_CLASS_MAP,
    KNOWLEDGE_OBJECT_CLASS,
    RDF_TYPE,
    RELATIONSHIP_MAP,
    SPARQL_PREFIXES,
    cortex_iri,
)
from cortex.ontology.resolver import find_ontology

logger = get_logger("db.graph")

# Maximum content size (10 MB)
MAX_CONTENT_SIZE = 10 * 1024 * 1024


class GraphStore:
    """Oxigraph RDF store for knowledge objects, relationships, and entities."""

    def __init__(self, path: Path | None = None):
        """Initialize the store.

        Args:
            path: Directory for persistent storage. If None, uses in-memory store.
        """
        self._read_only = False
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._store = ox.Store(str(path))
            except OSError:
                logger.warning(
                    "Graph DB locked by another process — using in-memory fallback"
                )
                self._store = ox.Store()
                self._read_only = True
        else:
            self._store = ox.Store()
        self._ontology_loaded = False

    def load_ontology(self, ontology_path: Path | None = None) -> int:
        """Load the Cortex OWL ontology into the store.

        Args:
            ontology_path: Path to cortex.ttl. If None, uses the bundled ontology.

        Returns:
            Number of triples loaded.

        Raises:
            OntologyError: If the ontology file cannot be parsed.
        """
        if ontology_path is None:
            # Use bundled ontology from package
            ontology_path = find_ontology()

        if not ontology_path.exists():
            raise OntologyError(
                f"Ontology file not found: {ontology_path}",
                context={"path": str(ontology_path)},
            )

        before = len(self._store)
        try:
            with open(ontology_path, "rb") as f:
                self._store.load(f, ox.RdfFormat.TURTLE, base_iri=CORTEX)
        except Exception as e:
            raise OntologyError(
                f"Failed to parse ontology: {e}",
                context={"path": str(ontology_path)},
                cause=e,
            )

        loaded = len(self._store) - before
        self._ontology_loaded = True
        logger.info("Ontology loaded: %d triples", loaded)
        return loaded

    @property
    def triple_count(self) -> int:
        return len(self._store)

    # -------------------------------------------------------------------------
    # Knowledge Object CRUD
    # -------------------------------------------------------------------------

    def create_object(
        self,
        *,
        obj_type: str,
        title: str,
        content: str = "",
        properties: dict[str, str] | None = None,
        project: str = "",
        tags: str = "",
        tier: str = "archive",
        captured_by: str = "",
        confidence: float = 1.0,
    ) -> str:
        """Create a knowledge object in the graph.

        Args:
            obj_type: One of the 8 knowledge types (decision, lesson, fix, etc.)
            title: Object title.
            content: Full text content.
            properties: Type-specific properties (e.g., rationale, symptom).
            project: Project name.
            tags: Comma-separated tags.
            tier: Memory tier (archive, recall, reflex).
            captured_by: Who/what captured this.
            confidence: Classification confidence 0.0-1.0.

        Returns:
            Generated object ID (UUID).

        Raises:
            ValidationError: If obj_type is invalid or content exceeds size limit.
        """
        if obj_type not in CLASS_MAP:
            raise ValidationError(
                f"Invalid knowledge type: {obj_type}",
                context={"type": obj_type, "valid_types": sorted(CLASS_MAP.keys())},
            )
        if len(content.encode("utf-8")) > MAX_CONTENT_SIZE:
            raise ValidationError(
                "Content exceeds maximum size of 10MB",
                context={"size": len(content.encode("utf-8"))},
            )

        obj_id = str(uuid4())
        subject = cortex_iri(f"obj/{obj_id}")
        now = datetime.now(UTC).isoformat()

        triples: list[ox.Quad] = [
            # Type assertions
            ox.Quad(subject, RDF_TYPE, CLASS_MAP[obj_type]),
            ox.Quad(subject, RDF_TYPE, KNOWLEDGE_OBJECT_CLASS),
            # Common properties
            ox.Quad(subject, cortex_iri("title"), ox.Literal(title)),
            ox.Quad(subject, cortex_iri("capturedAt"), ox.Literal(now, datatype=ox.NamedNode("http://www.w3.org/2001/XMLSchema#dateTime"))),
            ox.Quad(subject, cortex_iri("tier"), ox.Literal(tier)),
            ox.Quad(subject, cortex_iri("confidence"), ox.Literal(str(confidence), datatype=ox.NamedNode("http://www.w3.org/2001/XMLSchema#float"))),
        ]

        if content:
            triples.append(ox.Quad(subject, cortex_iri("content"), ox.Literal(content)))
        if project:
            triples.append(ox.Quad(subject, cortex_iri("project"), ox.Literal(project)))
        if tags:
            triples.append(ox.Quad(subject, cortex_iri("tags"), ox.Literal(tags)))
        if captured_by:
            triples.append(ox.Quad(subject, cortex_iri("capturedBy"), ox.Literal(captured_by)))

        # Type-specific properties
        if properties:
            for key, value in properties.items():
                if value:
                    triples.append(ox.Quad(subject, cortex_iri(key), ox.Literal(value)))

        for quad in triples:
            self._store.add(quad)

        logger.debug("Created object %s (type=%s)", obj_id, obj_type)
        return obj_id

    def read_object(self, obj_id: str) -> dict[str, Any] | None:
        """Read a knowledge object by ID.

        Returns:
            Dict with all properties, or None if not found.
        """
        subject = cortex_iri(f"obj/{obj_id}")
        query = f"""
        {SPARQL_PREFIXES}
        SELECT ?pred ?obj WHERE {{
            <{subject.value}> ?pred ?obj .
        }}
        """
        results = list(self._store.query(query))
        if not results:
            return None

        props: dict[str, Any] = {"id": obj_id}
        obj_type = None
        for row in results:
            pred_str = str(row["pred"].value)
            obj_val = row["obj"]

            if pred_str == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                type_str = str(obj_val.value)
                if type_str.startswith(CORTEX) and type_str != f"{CORTEX}KnowledgeObject":
                    obj_type = type_str.split("#")[-1].lower()
            else:
                prop_name = pred_str.split("#")[-1] if "#" in pred_str else pred_str.split("/")[-1]
                if isinstance(obj_val, ox.Literal):
                    props[prop_name] = obj_val.value
                else:
                    props[prop_name] = str(obj_val.value)

        if obj_type:
            props["type"] = obj_type
        return props

    def update_object(self, obj_id: str, **updates: str) -> bool:
        """Update properties of a knowledge object.

        Args:
            obj_id: Object ID.
            **updates: Property name → new value pairs.

        Returns:
            True if object existed and was updated.

        Raises:
            NotFoundError: If object doesn't exist.
        """
        subject = cortex_iri(f"obj/{obj_id}")

        # Verify object exists
        existing = self.read_object(obj_id)
        if existing is None:
            raise NotFoundError(
                f"Object not found: {obj_id}",
                context={"id": obj_id},
            )

        for key, value in updates.items():
            pred = cortex_iri(key)
            # Remove old value(s) for this predicate
            old_quads = list(self._store.quads_for_pattern(subject, pred, None))
            for quad in old_quads:
                self._store.remove(quad)
            # Add new value
            if value:
                self._store.add(ox.Quad(subject, pred, ox.Literal(value)))

        return True

    def delete_object(self, obj_id: str) -> bool:
        """Delete a knowledge object and all its triples (including relationships TO it).

        Returns:
            True if any triples were removed.
        """
        subject = cortex_iri(f"obj/{obj_id}")

        # Remove all triples where this is the subject
        outgoing = list(self._store.quads_for_pattern(subject, None, None))
        for quad in outgoing:
            self._store.remove(quad)

        # Remove all triples where this is the object (incoming relationships)
        incoming = list(self._store.quads_for_pattern(None, None, subject))
        for quad in incoming:
            self._store.remove(quad)

        removed = len(outgoing) + len(incoming)
        if removed:
            logger.debug("Deleted object %s (%d triples)", obj_id, removed)
        return removed > 0

    def list_objects(
        self,
        *,
        obj_type: str | None = None,
        project: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List knowledge objects with optional filters.

        Returns:
            List of object dicts (id, type, title, project, tags, capturedAt).
        """
        filters = []
        if obj_type and obj_type in CLASS_MAP:
            filters.append(f"?s a cortex:{obj_type.capitalize()} .")
        else:
            filters.append("?s a cortex:KnowledgeObject .")

        if project:
            filters.append(f'?s cortex:project "{project}" .')

        filter_block = "\n            ".join(filters)

        query = f"""
        {SPARQL_PREFIXES}
        SELECT ?s ?title ?type ?project ?tags ?capturedAt ?tier WHERE {{
            {filter_block}
            ?s cortex:title ?title .
            ?s a ?type .
            FILTER(?type != cortex:KnowledgeObject)
            OPTIONAL {{ ?s cortex:project ?project }}
            OPTIONAL {{ ?s cortex:tags ?tags }}
            OPTIONAL {{ ?s cortex:capturedAt ?capturedAt }}
            OPTIONAL {{ ?s cortex:tier ?tier }}
        }}
        ORDER BY DESC(?capturedAt)
        LIMIT {limit}
        OFFSET {offset}
        """
        results = list(self._store.query(query))

        objects = []
        for row in results:
            subj = str(row["s"].value)
            obj_id = subj.split("/")[-1] if "obj/" in subj else subj
            type_val = str(row["type"].value).split("#")[-1].lower()

            obj = {
                "id": obj_id,
                "type": type_val,
                "title": row["title"].value if row["title"] else "",
            }
            if row["project"]:
                obj["project"] = row["project"].value
            if row["tags"]:
                obj["tags"] = row["tags"].value
            if row["capturedAt"]:
                obj["capturedAt"] = row["capturedAt"].value
            if row["tier"]:
                obj["tier"] = row["tier"].value
            objects.append(obj)

        return objects

    # -------------------------------------------------------------------------
    # Relationship CRUD
    # -------------------------------------------------------------------------

    def create_relationship(
        self,
        *,
        from_id: str,
        rel_type: str,
        to_id: str,
        confidence: float = 1.0,
        inferred_by: str = "",
    ) -> bool:
        """Create a typed relationship between two knowledge objects.

        Args:
            from_id: Source object ID.
            rel_type: Relationship type name (causedBy, contradicts, etc.)
            to_id: Target object ID.
            confidence: Relationship confidence.
            inferred_by: If set, marks this as an inferred triple.

        Returns:
            True if created successfully.

        Raises:
            ValidationError: If rel_type is invalid or self-referential.
        """
        if rel_type not in RELATIONSHIP_MAP:
            raise ValidationError(
                f"Invalid relationship type: {rel_type}",
                context={"type": rel_type, "valid_types": sorted(RELATIONSHIP_MAP.keys())},
            )
        if from_id == to_id:
            raise ValidationError(
                "Self-referential relationships are not allowed",
                context={"id": from_id, "rel_type": rel_type},
            )

        subject = cortex_iri(f"obj/{from_id}")
        predicate = RELATIONSHIP_MAP[rel_type]
        obj = cortex_iri(f"obj/{to_id}")

        # Idempotent — check if already exists
        existing = list(self._store.quads_for_pattern(subject, predicate, obj))
        if existing:
            return True

        self._store.add(ox.Quad(subject, predicate, obj))
        return True

    def delete_relationship(self, *, from_id: str, rel_type: str, to_id: str) -> bool:
        """Delete a specific relationship."""
        if rel_type not in RELATIONSHIP_MAP:
            return False

        subject = cortex_iri(f"obj/{from_id}")
        predicate = RELATIONSHIP_MAP[rel_type]
        obj = cortex_iri(f"obj/{to_id}")

        quads = list(self._store.quads_for_pattern(subject, predicate, obj))
        for quad in quads:
            self._store.remove(quad)
        return len(quads) > 0

    def get_relationships(self, obj_id: str) -> list[dict[str, str]]:
        """Get all relationships for an object (both directions).

        Returns:
            List of dicts: {direction, rel_type, other_id}
        """
        subject = cortex_iri(f"obj/{obj_id}")
        rels = []

        # Outgoing
        for rel_name, pred_iri in RELATIONSHIP_MAP.items():
            if rel_name == "mentions":
                continue  # entity links handled separately
            quads = list(self._store.quads_for_pattern(subject, pred_iri, None))
            for quad in quads:
                target = str(quad.object.value)
                if "obj/" in target:
                    rels.append({
                        "direction": "outgoing",
                        "rel_type": rel_name,
                        "other_id": target.split("/")[-1],
                    })

        # Incoming
        for rel_name, pred_iri in RELATIONSHIP_MAP.items():
            if rel_name == "mentions":
                continue
            quads = list(self._store.quads_for_pattern(None, pred_iri, subject))
            for quad in quads:
                source = str(quad.subject.value)
                if "obj/" in source:
                    rels.append({
                        "direction": "incoming",
                        "rel_type": rel_name,
                        "other_id": source.split("/")[-1],
                    })

        return rels

    # -------------------------------------------------------------------------
    # Entity CRUD
    # -------------------------------------------------------------------------

    def create_entity(
        self,
        *,
        name: str,
        entity_type: str = "concept",
        aliases: str = "",
    ) -> str:
        """Create or get an entity node.

        Returns:
            Entity ID.
        """
        if entity_type not in ENTITY_CLASS_MAP:
            entity_type = "concept"

        # Check for existing entity with same name (case-insensitive)
        existing = self._find_entity_by_name(name)
        if existing:
            return existing

        entity_id = str(uuid4())
        subject = cortex_iri(f"entity/{entity_id}")

        quads = [
            ox.Quad(subject, RDF_TYPE, ENTITY_CLASS_MAP[entity_type]),
            ox.Quad(subject, RDF_TYPE, ENTITY_CLASS),
            ox.Quad(subject, cortex_iri("entityName"), ox.Literal(name)),
        ]
        if aliases:
            quads.append(ox.Quad(subject, cortex_iri("entityAliases"), ox.Literal(aliases)))

        for quad in quads:
            self._store.add(quad)

        return entity_id

    def _find_entity_by_name(self, name: str) -> str | None:
        """Find entity by name (case-insensitive)."""
        query = f"""
        {SPARQL_PREFIXES}
        SELECT ?s ?name WHERE {{
            ?s a cortex:Entity .
            ?s cortex:entityName ?name .
        }}
        """
        for row in self._store.query(query):
            if row["name"].value.lower() == name.lower():
                subj = str(row["s"].value)
                return subj.split("/")[-1] if "entity/" in subj else subj
        return None

    def add_mention(self, *, obj_id: str, entity_id: str) -> None:
        """Link a knowledge object to an entity via mentions."""
        subject = cortex_iri(f"obj/{obj_id}")
        entity = cortex_iri(f"entity/{entity_id}")
        self._store.add(ox.Quad(subject, RELATIONSHIP_MAP["mentions"], entity))

    def get_entity_mentions(self, entity_id: str) -> list[str]:
        """Get all object IDs that mention an entity."""
        entity = cortex_iri(f"entity/{entity_id}")
        quads = list(self._store.quads_for_pattern(None, RELATIONSHIP_MAP["mentions"], entity))
        ids = []
        for quad in quads:
            subj = str(quad.subject.value)
            if "obj/" in subj:
                ids.append(subj.split("/")[-1])
        return ids

    def list_entities(self, entity_type: str | None = None) -> list[dict[str, str]]:
        """List all entities, optionally filtered by type."""
        type_filter = ""
        if entity_type and entity_type in ENTITY_CLASS_MAP:
            type_filter = f"?s a cortex:{entity_type.capitalize()} ."

        query = f"""
        {SPARQL_PREFIXES}
        SELECT ?s ?name ?type WHERE {{
            ?s a cortex:Entity .
            ?s cortex:entityName ?name .
            ?s a ?type .
            FILTER(?type != cortex:Entity)
            {type_filter}
        }}
        ORDER BY ?name
        """
        entities = []
        for row in self._store.query(query):
            subj = str(row["s"].value)
            entities.append({
                "id": subj.split("/")[-1] if "entity/" in subj else subj,
                "name": row["name"].value,
                "type": str(row["type"].value).split("#")[-1].lower(),
            })
        return entities

    # -------------------------------------------------------------------------
    # SPARQL Query
    # -------------------------------------------------------------------------

    def query(self, sparql: str) -> list[dict[str, Any]]:
        """Execute a SPARQL SELECT query.

        The cortex/owl/rdf/rdfs/xsd prefixes are auto-prepended.

        Raises:
            StoreError: If SPARQL is malformed.
        """
        full_query = f"{SPARQL_PREFIXES}\n{sparql}" if "PREFIX" not in sparql else sparql
        try:
            solutions = self._store.query(full_query)
        except SyntaxError as e:
            raise StoreError(
                f"SPARQL syntax error: {e}",
                context={"query": sparql},
                cause=e,
            )
        variables = solutions.variables
        rows = []
        for row in solutions:
            d: dict[str, Any] = {}
            for var in variables:
                var_name = var.value
                val = row[var]
                if val is None:
                    d[var_name] = None
                elif isinstance(val, ox.Literal):
                    d[var_name] = val.value
                else:
                    d[var_name] = str(val.value)
            rows.append(d)
        return rows

    def count_by_type(self) -> dict[str, int]:
        """Count knowledge objects by type."""
        query = f"""
        {SPARQL_PREFIXES}
        SELECT ?type (COUNT(?s) AS ?count) WHERE {{
            ?s a cortex:KnowledgeObject .
            ?s a ?type .
            FILTER(?type != cortex:KnowledgeObject)
        }}
        GROUP BY ?type
        """
        counts = {}
        for row in self._store.query(query):
            type_str = str(row["type"].value).split("#")[-1].lower()
            counts[type_str] = int(row["count"].value)
        return counts
