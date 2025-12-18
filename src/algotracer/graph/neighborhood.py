from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from gqlalchemy import Memgraph


@dataclass(frozen=True)
class NeighborhoodConfig:
    depth_up: int = 2
    depth_down: int = 2
    max_nodes: int = 200
    max_edges: int = 400
    max_paths: int = 200


@dataclass(frozen=True)
class Neighborhood:
    target: Dict[str, object]
    nodes: List[Dict[str, object]]
    edges: List[Dict[str, object]]
    upstream_paths: List[str]
    downstream_paths: List[str]
    callers: List[str]
    callees: List[str]
    externals: List[str]


def _fetch_paths(
    mg: Memgraph,
    query: str,
    params: dict,
) -> List[Dict[str, Sequence[Dict[str, object]]]]:
    rows = mg.execute_and_fetch(query, params)
    out: List[Dict[str, Sequence[Dict[str, object]]]] = []
    for row in rows:
        out.append({"nodes": row.get("nodes", []), "rels": row.get("rels", [])})
    return out


def _path_to_string(nodes: Sequence[Dict[str, object]]) -> str:
    ids = [str(n.get("id")) for n in nodes if n.get("id")]
    return " -> ".join(ids)


def _cap_sequence(items: Sequence[Dict[str, object]], max_items: int) -> List[Dict[str, object]]:
    return list(items[:max_items])


def _unique_in_order(items: Iterable[Dict[str, object]], key_fields: Sequence[str] = ("id",)) -> List[Dict[str, object]]:
    seen: set[str] = set()
    out: List[Dict[str, object]] = []
    for item in items:
        key = "|".join(str(item.get(field)) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def fetch_neighborhood(
    *,
    mg: Memgraph,
    repo_id: str,
    func_id: str,
    config: NeighborhoodConfig,
) -> Neighborhood:
    target_query = (
        "MATCH (f:Function {repo_id: $repo_id, id: $id}) "
        "RETURN f.id AS id, f.name AS name, f.qualname AS qualname, f.path AS path, "
        "f.lineno AS lineno, f.kind AS kind, f.signature AS signature, labels(f) AS labels"
    )
    target_rows = list(mg.execute_and_fetch(target_query, {"repo_id": repo_id, "id": func_id}))
    if not target_rows:
        raise RuntimeError(f"Function not found: {func_id}")
    target = target_rows[0]

    downstream_query = (
        "MATCH p=(f:Function {repo_id: $repo_id, id: $id})-[:CALLS*1..$depth]->(n) "
        "RETURN [x IN nodes(p) | {id: x.id, labels: labels(x), name: x.name, qualname: x.qualname, "
        "path: x.path, lineno: x.lineno, kind: x.kind, signature: x.signature}] AS nodes, "
        "[r IN relationships(p) | {src: startNode(r).id, dst: endNode(r).id, type: type(r)}] AS rels "
        "LIMIT $limit"
    )
    upstream_query = (
        "MATCH p=(c)-[:CALLS*1..$depth]->(f:Function {repo_id: $repo_id, id: $id}) "
        "RETURN [x IN nodes(p) | {id: x.id, labels: labels(x), name: x.name, qualname: x.qualname, "
        "path: x.path, lineno: x.lineno, kind: x.kind, signature: x.signature}] AS nodes, "
        "[r IN relationships(p) | {src: startNode(r).id, dst: endNode(r).id, type: type(r)}] AS rels "
        "LIMIT $limit"
    )

    downstream_paths = _fetch_paths(
        mg,
        downstream_query,
        {"repo_id": repo_id, "id": func_id, "depth": config.depth_down, "limit": config.max_paths},
    )
    upstream_paths = _fetch_paths(
        mg,
        upstream_query,
        {"repo_id": repo_id, "id": func_id, "depth": config.depth_up, "limit": config.max_paths},
    )

    nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []
    downstream_strings: List[str] = []
    upstream_strings: List[str] = []

    for path in downstream_paths:
        nodes.extend(path["nodes"])
        edges.extend(path["rels"])
        downstream_strings.append(_path_to_string(path["nodes"]))

    for path in upstream_paths:
        nodes.extend(path["nodes"])
        edges.extend(path["rels"])
        upstream_strings.append(_path_to_string(path["nodes"]))

    if not any(n.get("id") == target.get("id") for n in nodes):
        nodes.insert(0, target)

    nodes = _unique_in_order(nodes, ("id",))
    edges = _unique_in_order(edges, ("src", "dst", "type"))

    nodes = _cap_sequence(nodes, config.max_nodes)
    edges = _cap_sequence(edges, config.max_edges)

    callers_query = (
        "MATCH (c)-[:CALLS {repo_id: $repo_id}]->(f:Function {repo_id: $repo_id, id: $id}) "
        "RETURN c.id AS id ORDER BY id"
    )
    callees_query = (
        "MATCH (f:Function {repo_id: $repo_id, id: $id})-[:CALLS {repo_id: $repo_id}]->(c) "
        "RETURN c.id AS id ORDER BY id"
    )
    callers = [row.get("id") for row in mg.execute_and_fetch(callers_query, {"repo_id": repo_id, "id": func_id})]
    callees = [row.get("id") for row in mg.execute_and_fetch(callees_query, {"repo_id": repo_id, "id": func_id})]

    externals = sorted(
        {
            str(node.get("id"))
            for node in nodes
            if isinstance(node.get("labels"), list) and "External" in node.get("labels")
        }
    )

    return Neighborhood(
        target=target,
        nodes=nodes,
        edges=edges,
        upstream_paths=upstream_strings,
        downstream_paths=downstream_strings,
        callers=callers,
        callees=callees,
        externals=externals,
    )
