from __future__ import annotations

"""
Generate a Cypher export for a Neighborhood subgraph so it matches the current
builder/neighborhood schema (Function/Class/Module/External/VirtualCall nodes
with CALLS/CALLS_VIRTUAL/OVERRIDES/etc. edges).
"""

from pathlib import Path
from typing import Dict, List, Sequence

from algotracer.graph.neighborhood import Neighborhood


def _escape(val: object) -> str:
    """Escape single quotes and newlines for Cypher string literals."""
    if val is None:
        return ""
    s = str(val)
    s = s.replace("\\", "\\\\").replace("'", "\\'")
    s = s.replace("\n", "\\n").replace("\r", "")
    return s


def _primary_label(labels: Sequence[str] | str | None) -> str:
    if isinstance(labels, str):
        labels = [labels]
    labels_set = set(labels or [])
    for candidate in ("Function", "Class", "Module", "External", "VirtualCall"):
        if candidate in labels_set:
            return candidate
    return "Node"


def _node_props(node: Dict[str, object]) -> str:
    props: Dict[str, object] = {}
    for key in (
        "id",
        "name",
        "qualname",
        "path",
        "abs_path",
        "lineno",
        "kind",
        "signature",
        "owner_class",
        "owner_class_id",
        "namespace",
        "category",
        "side_effect_category",
        "side_effect_confidence",
        "side_effect_evidence",
        "callee_attr",
        "receiver_kind",
        "receiver_name",
        "full_text",
        "call_kind",
        "repo_id",
    ):
        val = node.get(key)
        if val is None:
            continue
        props[key] = val

    parts: List[str] = []
    for k, v in props.items():
        if isinstance(v, (int, float)):
            parts.append(f"{k}: {v}")
        else:
            parts.append(f"{k}: '{_escape(v)}'")
    return ", ".join(parts)


def _write_node(f, node: Dict[str, object]) -> None:
    label = _primary_label(node.get("labels"))
    props = _node_props(node)
    f.write(f"MERGE (:{label} {{{props}}});\n")


def _write_edge(f, edge: Dict[str, object]) -> None:
    src = edge.get("src")
    dst = edge.get("dst")
    typ = edge.get("type") or "REL"
    if not (src and dst):
        return
    rel = _escape(typ.upper())
    f.write(
        "MATCH (a {id: '%s'}), (b {id: '%s'}) "
        "MERGE (a)-[:%s]->(b);\n" % (_escape(src), _escape(dst), rel)
    )


def generate_cypher(neighborhood: Neighborhood, output_dir: Path, filename: str = "graph.cypher") -> Path:
    """
    Export a Neighborhood (nodes/edges) to a Cypher file compatible with the
    builder’s graph schema.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename

    nodes = neighborhood.nodes
    edges = neighborhood.edges

    with file_path.open("w", encoding="utf-8") as f:
        # Indexes for id lookups
        for label in ("Module", "Class", "Function", "External", "VirtualCall"):
            f.write(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.id);\n")

        for node in nodes:
            _write_node(f, node)
        for edge in edges:
            _write_edge(f, edge)

    return file_path
