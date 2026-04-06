"""RDF namespace helpers for Cortex ontology."""

from __future__ import annotations

import pyoxigraph as ox

from cortex.core.constants import ONTOLOGY_NAMESPACE

# Namespace IRIs
CORTEX = ONTOLOGY_NAMESPACE
OWL = "http://www.w3.org/2002/07/owl#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"

# Common SPARQL prefix block
SPARQL_PREFIXES = f"""\
PREFIX cortex: <{CORTEX}>
PREFIX owl: <{OWL}>
PREFIX rdf: <{RDF}>
PREFIX rdfs: <{RDFS}>
PREFIX xsd: <{XSD}>
"""


def cortex_iri(local: str) -> ox.NamedNode:
    """Create an IRI in the cortex namespace."""
    return ox.NamedNode(f"{CORTEX}{local}")


def rdf_iri(local: str) -> ox.NamedNode:
    return ox.NamedNode(f"{RDF}{local}")


def rdfs_iri(local: str) -> ox.NamedNode:
    return ox.NamedNode(f"{RDFS}{local}")


def xsd_iri(local: str) -> ox.NamedNode:
    return ox.NamedNode(f"{XSD}{local}")


def owl_iri(local: str) -> ox.NamedNode:
    return ox.NamedNode(f"{OWL}{local}")


# Pre-built common nodes
RDF_TYPE = rdf_iri("type")
RDFS_LABEL = rdfs_iri("label")

# Cortex class IRIs
CLASS_MAP: dict[str, ox.NamedNode] = {
    "decision": cortex_iri("Decision"),
    "lesson": cortex_iri("Lesson"),
    "fix": cortex_iri("Fix"),
    "session": cortex_iri("Session"),
    "research": cortex_iri("Research"),
    "source": cortex_iri("Source"),
    "synthesis": cortex_iri("Synthesis"),
    "idea": cortex_iri("Idea"),
}

ENTITY_CLASS_MAP: dict[str, ox.NamedNode] = {
    "technology": cortex_iri("Technology"),
    "project": cortex_iri("Project"),
    "pattern": cortex_iri("Pattern"),
    "concept": cortex_iri("Concept"),
}

# Relationship IRI map
RELATIONSHIP_MAP: dict[str, ox.NamedNode] = {
    "causedBy": cortex_iri("causedBy"),
    "contradicts": cortex_iri("contradicts"),
    "supports": cortex_iri("supports"),
    "supersedes": cortex_iri("supersedes"),
    "dependsOn": cortex_iri("dependsOn"),
    "ledTo": cortex_iri("ledTo"),
    "implements": cortex_iri("implements"),
    "mentions": cortex_iri("mentions"),
}

# Knowledge object base class
KNOWLEDGE_OBJECT_CLASS = cortex_iri("KnowledgeObject")
ENTITY_CLASS = cortex_iri("Entity")
