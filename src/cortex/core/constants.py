"""Global constants for Cortex."""

from pathlib import Path

# Default data directory
DEFAULT_DATA_DIR = Path.home() / ".cortex"

# Store files
GRAPH_DB_FILE = "graph.db"
SQLITE_DB_FILE = "cortex.db"

# Ontology
ONTOLOGY_NAMESPACE = "https://cortex.abbacus.ai/ontology#"

# Server defaults
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 1314

# Knowledge object types (must match ontology classes)
KNOWLEDGE_TYPES = frozenset(
    {
        "decision",
        "lesson",
        "fix",
        "session",
        "research",
        "source",
        "synthesis",
        "idea",
    }
)

# Relationship types (must match ontology object properties)
RELATIONSHIP_TYPES = frozenset(
    {
        "causedBy",
        "contradicts",
        "supports",
        "supersedes",
        "dependsOn",
        "ledTo",
        "implements",
        "mentions",
    }
)

# Config env prefix
ENV_PREFIX = "CORTEX_"
