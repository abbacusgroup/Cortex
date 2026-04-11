"""Graph topology generators for OWL-RL reasoning benchmarks.

Each function seeds a Store with objects and relationships forming a specific
graph shape, then returns metadata describing what was created and what the
reasoner should infer.
"""

from __future__ import annotations

from cortex.db.store import Store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_node(store: Store, index: int) -> str:
    """Create a single decision node and return its ID."""
    return store.create(
        obj_type="decision",
        title=f"Node {index}",
        content=f"Node {index}",
        summary=f"Node {index}",
    )


# ---------------------------------------------------------------------------
# Topologies
# ---------------------------------------------------------------------------

def chain(store: Store, depth: int) -> dict:
    """Create a supersedes chain: N0 -> N1 -> N2 -> ... -> N(depth-1).

    Transitive closure should add depth*(depth-1)/2 new edges beyond the
    direct ones (every ancestor-descendant pair that isn't already a direct
    parent-child link).

    Returns:
        {"obj_ids": list[str], "expected_transitive": int}
    """
    obj_ids: list[str] = []
    for i in range(depth):
        obj_ids.append(_create_node(store, i))

    # Direct edges: obj_ids[i] supersedes obj_ids[i+1]
    for i in range(depth - 1):
        store.create_relationship(
            from_id=obj_ids[i], rel_type="supersedes", to_id=obj_ids[i + 1]
        )

    # Expected transitive = total pairs - direct edges
    # Total ancestor-descendant pairs = depth*(depth-1)/2
    # Direct edges = depth-1
    # New transitive edges = depth*(depth-1)/2 - (depth-1)
    #                      = (depth-1)*(depth-2)/2
    expected_transitive = (depth - 1) * (depth - 2) // 2

    return {"obj_ids": obj_ids, "expected_transitive": expected_transitive}


def star(store: Store, size: int) -> dict:
    """Center node supersedes N leaf nodes.

    No transitive closure is needed because there's no chain longer than one
    hop. The `size` direct edges are the complete set.

    Returns:
        {"center_id": str, "leaf_ids": list[str], "expected_edges": int}
    """
    center_id = _create_node(store, 0)
    leaf_ids: list[str] = []

    for i in range(1, size + 1):
        leaf_id = _create_node(store, i)
        leaf_ids.append(leaf_id)
        store.create_relationship(
            from_id=center_id, rel_type="supersedes", to_id=leaf_id
        )

    return {"center_id": center_id, "leaf_ids": leaf_ids, "expected_edges": size}


def binary_tree(store: Store, depth: int) -> dict:
    """Binary tree where each parent supersedes its two children.

    Depth 1 = root only (0 edges). Depth 2 = root + 2 children (2 edges).

    Expected transitive edges: all ancestor-descendant pairs that are NOT
    direct parent-child links.

    Returns:
        {"obj_ids": list[str], "expected_transitive": int}
    """
    if depth < 1:
        return {"obj_ids": [], "expected_transitive": 0}

    # Build level by level
    levels: list[list[str]] = []
    node_counter = 0

    for level in range(depth):
        level_nodes: list[str] = []
        count_at_level = 2 ** level
        for _ in range(count_at_level):
            node_id = _create_node(store, node_counter)
            level_nodes.append(node_id)
            node_counter += 1
        levels.append(level_nodes)

    # Create parent-child supersedes edges
    direct_edges = 0
    for level in range(depth - 1):
        for i, parent_id in enumerate(levels[level]):
            left_child = levels[level + 1][2 * i]
            right_child = levels[level + 1][2 * i + 1]
            store.create_relationship(
                from_id=parent_id, rel_type="supersedes", to_id=left_child
            )
            store.create_relationship(
                from_id=parent_id, rel_type="supersedes", to_id=right_child
            )
            direct_edges += 2

    # Count all ancestor-descendant pairs
    # For each node at level L, it has L ancestors (root is level 0).
    # Total ancestor-descendant pairs = sum over all nodes of (depth of that node)
    # = sum_{L=1}^{depth-1} L * 2^L
    total_pairs = 0
    for level in range(1, depth):
        nodes_at_level = 2 ** level
        ancestors_per_node = level  # each node at level L has L ancestors
        total_pairs += nodes_at_level * ancestors_per_node

    expected_transitive = total_pairs - direct_edges

    all_ids = [nid for level_nodes in levels for nid in level_nodes]
    return {"obj_ids": all_ids, "expected_transitive": expected_transitive}


def cycle(store: Store, size: int) -> dict:
    """Circular supersedes chain: N0 -> N1 -> ... -> N(size-1) -> N0.

    Cycles are pathological for naive transitive closure. The reasoner must
    terminate safely (no infinite loop, no self-edges).

    Returns:
        {"obj_ids": list[str], "expected_safe": True}
    """
    obj_ids: list[str] = []
    for i in range(size):
        obj_ids.append(_create_node(store, i))

    for i in range(size):
        store.create_relationship(
            from_id=obj_ids[i],
            rel_type="supersedes",
            to_id=obj_ids[(i + 1) % size],
        )

    return {"obj_ids": obj_ids, "expected_safe": True}


def diamond(store: Store) -> dict:
    """Diamond: A supersedes B and C, both B and C supersede D.

    Expected inference: A supersedes D (1 new transitive edge).

    Returns:
        {"obj_ids": list[str], "expected_transitive": 1}
    """
    a = _create_node(store, 0)
    b = _create_node(store, 1)
    c = _create_node(store, 2)
    d = _create_node(store, 3)

    store.create_relationship(from_id=a, rel_type="supersedes", to_id=b)
    store.create_relationship(from_id=a, rel_type="supersedes", to_id=c)
    store.create_relationship(from_id=b, rel_type="supersedes", to_id=d)
    store.create_relationship(from_id=c, rel_type="supersedes", to_id=d)

    return {"obj_ids": [a, b, c, d], "expected_transitive": 1}


def contradicts_batch(store: Store, size: int) -> dict:
    """Create N one-directional contradicts edges: A_i contradicts B_i.

    The symmetric-contradicts rule should generate the reverse for each pair.

    Returns:
        {"pairs": list[tuple[str, str]], "expected_symmetric": int}
    """
    pairs: list[tuple[str, str]] = []
    node_counter = 0

    for _ in range(size):
        a = _create_node(store, node_counter)
        node_counter += 1
        b = _create_node(store, node_counter)
        node_counter += 1

        store.create_relationship(from_id=a, rel_type="contradicts", to_id=b)
        pairs.append((a, b))

    return {"pairs": pairs, "expected_symmetric": size}


def causal_batch(store: Store, size: int) -> dict:
    """Create N causedBy edges: A_i causedBy B_i.

    The inverse rule should generate B_i ledTo A_i for each pair.

    Returns:
        {"pairs": list[tuple[str, str]], "expected_inverse": int}
    """
    pairs: list[tuple[str, str]] = []
    node_counter = 0

    for _ in range(size):
        a = _create_node(store, node_counter)
        node_counter += 1
        b = _create_node(store, node_counter)
        node_counter += 1

        store.create_relationship(from_id=a, rel_type="causedBy", to_id=b)
        pairs.append((a, b))

    return {"pairs": pairs, "expected_inverse": size}
