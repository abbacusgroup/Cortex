"""Tests for the Cortex OWL ontology loaded via pyoxigraph.

Verifies class hierarchy, property definitions, relationship characteristics,
and adversarial edge cases against ontology/cortex.ttl.
"""

from __future__ import annotations

import tempfile

import pyoxigraph as ox
import pytest

from cortex.ontology.resolver import find_ontology

ONTOLOGY_PATH = find_ontology()
CORTEX_NS = "https://cortex.abbacus.ai/ontology#"

PREFIXES = f"""\
PREFIX cortex: <{CORTEX_NS}>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


@pytest.fixture()
def store() -> ox.Store:
    """Load the ontology into a fresh in-memory Oxigraph store."""
    s = ox.Store()
    with open(ONTOLOGY_PATH, "rb") as f:
        s.load(f, ox.RdfFormat.TURTLE, base_iri=CORTEX_NS)
    return s


def _query_values(store: ox.Store, sparql: str, var: str) -> set[str]:
    """Run a SPARQL query and collect a single variable's values as a set."""
    results = list(store.query(f"{PREFIXES}\n{sparql}"))
    return {str(row[var].value) for row in results}


# =========================================================================
# Loading
# =========================================================================


class TestOntologyLoading:
    """Ontology file loads correctly into Oxigraph."""

    def test_loads_without_error(self, store: ox.Store) -> None:
        assert len(store) > 0

    def test_triple_count_reasonable(self, store: ox.Store) -> None:
        # The ontology defines many classes, properties, and annotations.
        # Expect at least 100 triples.
        assert len(store) >= 100


# =========================================================================
# Knowledge Object Subclasses
# =========================================================================


KNOWLEDGE_SUBCLASSES = {
    "Decision",
    "Lesson",
    "Fix",
    "Session",
    "Research",
    "Source",
    "Synthesis",
    "Idea",
}


class TestKnowledgeObjectSubclasses:
    """All 8 knowledge object subclasses are declared."""

    def test_all_eight_subclasses_defined(self, store: ox.Store) -> None:
        query = """
        SELECT ?cls WHERE {
            ?cls rdfs:subClassOf cortex:KnowledgeObject .
            ?cls a owl:Class .
        }
        """
        found = _query_values(store, query, "cls")
        local_names = {uri.split("#")[-1] for uri in found}
        assert local_names == KNOWLEDGE_SUBCLASSES

    @pytest.mark.parametrize("cls_name", sorted(KNOWLEDGE_SUBCLASSES))
    def test_each_subclass_is_owl_class(self, store: ox.Store, cls_name: str) -> None:
        query = f"""
        ASK {{
            cortex:{cls_name} a owl:Class .
        }}
        """
        assert store.query(f"{PREFIXES}\n{query}")

    @pytest.mark.parametrize("cls_name", sorted(KNOWLEDGE_SUBCLASSES))
    def test_each_subclass_has_label(self, store: ox.Store, cls_name: str) -> None:
        query = f"""
        ASK {{
            cortex:{cls_name} rdfs:label ?label .
        }}
        """
        assert store.query(f"{PREFIXES}\n{query}")


# =========================================================================
# Properties — domain and range
# =========================================================================


class TestPropertyDomainRange:
    """Datatype and object properties have correct domain/range."""

    @pytest.mark.parametrize(
        ("prop", "domain", "range_"),
        [
            ("title", "KnowledgeObject", "xsd:string"),
            ("content", "KnowledgeObject", "xsd:string"),
            ("summary", "KnowledgeObject", "xsd:string"),
            ("tags", "KnowledgeObject", "xsd:string"),
            ("project", "KnowledgeObject", "xsd:string"),
            ("confidence", "KnowledgeObject", "xsd:float"),
            ("capturedAt", "KnowledgeObject", "xsd:dateTime"),
            ("capturedBy", "KnowledgeObject", "xsd:string"),
            ("tier", "KnowledgeObject", "xsd:string"),
            ("rationale", "Decision", "xsd:string"),
            ("symptom", "Fix", "xsd:string"),
            ("rootCause", "Fix", "xsd:string"),
            ("goal", "Session", "xsd:string"),
            ("question", "Research", "xsd:string"),
            ("url", "Source", "xsd:anyURI"),
            ("period", "Synthesis", "xsd:string"),
            ("feasibility", "Idea", "xsd:string"),
            ("entityName", "Entity", "xsd:string"),
        ],
    )
    def test_datatype_property_domain_range(
        self, store: ox.Store, prop: str, domain: str, range_: str
    ) -> None:
        # Build expected range IRI
        if range_.startswith("xsd:"):
            range_iri = f"http://www.w3.org/2001/XMLSchema#{range_[4:]}"
        else:
            range_iri = f"{CORTEX_NS}{range_}"

        query = f"""
        ASK {{
            cortex:{prop} rdfs:domain cortex:{domain} .
            cortex:{prop} rdfs:range <{range_iri}> .
        }}
        """
        assert store.query(f"{PREFIXES}\n{query}"), (
            f"cortex:{prop} missing domain={domain} or range={range_}"
        )

    @pytest.mark.parametrize(
        ("prop", "domain", "range_"),
        [
            ("causedBy", "KnowledgeObject", "KnowledgeObject"),
            ("ledTo", "KnowledgeObject", "KnowledgeObject"),
            ("contradicts", "KnowledgeObject", "KnowledgeObject"),
            ("supports", "KnowledgeObject", "KnowledgeObject"),
            ("supersedes", "KnowledgeObject", "KnowledgeObject"),
            ("dependsOn", "KnowledgeObject", "KnowledgeObject"),
            ("implements", "Session", "Decision"),
            ("mentions", "KnowledgeObject", "Entity"),
        ],
    )
    def test_object_property_domain_range(
        self, store: ox.Store, prop: str, domain: str, range_: str
    ) -> None:
        query = f"""
        ASK {{
            cortex:{prop} a owl:ObjectProperty .
            cortex:{prop} rdfs:domain cortex:{domain} .
            cortex:{prop} rdfs:range cortex:{range_} .
        }}
        """
        assert store.query(f"{PREFIXES}\n{query}"), (
            f"cortex:{prop} missing ObjectProperty/domain/range"
        )


# =========================================================================
# Relationship Characteristics
# =========================================================================


class TestRelationshipCharacteristics:
    """OWL property characteristics for special relationships."""

    def test_contradicts_is_symmetric(self, store: ox.Store) -> None:
        query = "ASK { cortex:contradicts a owl:SymmetricProperty . }"
        assert store.query(f"{PREFIXES}\n{query}")

    def test_supersedes_is_transitive(self, store: ox.Store) -> None:
        query = "ASK { cortex:supersedes a owl:TransitiveProperty . }"
        assert store.query(f"{PREFIXES}\n{query}")

    def test_causedby_ledto_are_inverses(self, store: ox.Store) -> None:
        query = "ASK { cortex:ledTo owl:inverseOf cortex:causedBy . }"
        assert store.query(f"{PREFIXES}\n{query}")

    def test_contradicts_is_not_transitive(self, store: ox.Store) -> None:
        query = "ASK { cortex:contradicts a owl:TransitiveProperty . }"
        assert not store.query(f"{PREFIXES}\n{query}")

    def test_supports_is_plain_object_property(self, store: ox.Store) -> None:
        """supports has no special OWL characteristic."""
        for char in ["SymmetricProperty", "TransitiveProperty"]:
            query = f"ASK {{ cortex:supports a owl:{char} . }}"
            assert not store.query(f"{PREFIXES}\n{query}")


# =========================================================================
# Class Hierarchy via SPARQL
# =========================================================================


class TestClassHierarchy:
    """Class hierarchy assertions verified through SPARQL."""

    def test_fix_is_a_knowledge_object(self, store: ox.Store) -> None:
        query = """
        ASK {
            cortex:Fix rdfs:subClassOf cortex:KnowledgeObject .
        }
        """
        assert store.query(f"{PREFIXES}\n{query}")

    def test_knowledge_object_is_base_class(self, store: ox.Store) -> None:
        """KnowledgeObject is not a subclass of anything in cortex ns."""
        query = """
        SELECT ?parent WHERE {
            cortex:KnowledgeObject rdfs:subClassOf ?parent .
            FILTER(STRSTARTS(STR(?parent), STR(cortex:)))
        }
        """
        results = list(store.query(f"{PREFIXES}\n{query}"))
        assert len(results) == 0

    def test_entity_subtypes_exist(self, store: ox.Store) -> None:
        query = """
        SELECT ?cls WHERE {
            ?cls rdfs:subClassOf cortex:Entity .
            ?cls a owl:Class .
        }
        """
        found = _query_values(store, query, "cls")
        local_names = {uri.split("#")[-1] for uri in found}
        assert local_names == {"Technology", "Project", "Pattern", "Concept"}

    @pytest.mark.parametrize(
        "entity_type",
        ["Technology", "Project", "Pattern", "Concept"],
    )
    def test_entity_subtype_has_label(self, store: ox.Store, entity_type: str) -> None:
        query = f"""
        ASK {{
            cortex:{entity_type} rdfs:label ?label .
            cortex:{entity_type} rdfs:subClassOf cortex:Entity .
        }}
        """
        assert store.query(f"{PREFIXES}\n{query}")


# =========================================================================
# Adversarial
# =========================================================================


class TestAdversarialOntologyLoading:
    """Malformed and degenerate ontology inputs are handled safely."""

    def test_malformed_turtle_raises_parse_error(self) -> None:
        """Loading invalid Turtle produces a clear error, not a crash."""
        malformed = b"@prefix cortex: <bad> . cortex:Foo a ;;;; ."
        s = ox.Store()
        with tempfile.NamedTemporaryFile(suffix=".ttl") as f:
            f.write(malformed)
            f.flush()
            with pytest.raises(SyntaxError), open(f.name, "rb") as fp:
                s.load(fp, ox.RdfFormat.TURTLE)

    def test_empty_turtle_loads_zero_triples(self) -> None:
        """An empty Turtle file loads without error but adds nothing."""
        s = ox.Store()
        with tempfile.NamedTemporaryFile(suffix=".ttl") as f:
            f.write(b"")
            f.flush()
            with open(f.name, "rb") as fp:
                s.load(fp, ox.RdfFormat.TURTLE)
        assert len(s) == 0

    def test_empty_with_prefixes_only(self) -> None:
        """A Turtle file with only prefix declarations loads cleanly."""
        s = ox.Store()
        content = b"@prefix cortex: <https://cortex.abbacus.ai/ontology#> .\n"
        with tempfile.NamedTemporaryFile(suffix=".ttl") as f:
            f.write(content)
            f.flush()
            with open(f.name, "rb") as fp:
                s.load(fp, ox.RdfFormat.TURTLE)
        assert len(s) == 0
