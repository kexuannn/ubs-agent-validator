from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

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
    virtual_targets: List[Dict[str, object]]


def _node_to_dict(n: object) -> Dict[str, object]:
    props = getattr(n, "properties", None) or getattr(n, "_properties", {}) or {}
    labels = getattr(n, "labels", None) or getattr(n, "_labels", []) or []
    stable_id = props.get("id")

    out: Dict[str, object] = {
        "id": stable_id,
        "labels": list(labels) if isinstance(labels, (list, set, tuple)) else labels,
        "name": props.get("name"),
        "qualname": props.get("qualname"),
        "path": props.get("path"),
        "abs_path": props.get("abs_path"),  # used for source-snippet extraction
        "lineno": props.get("lineno"),
        "kind": props.get("kind"),
        "signature": props.get("signature"),
        "repo_id": props.get("repo_id"),
        # dispatch-related
        "owner_class": props.get("owner_class"),
        "owner_class_id": props.get("owner_class_id"),
    }

    # Surface VirtualCall metadata so explainer can see it
    if "VirtualCall" in (out["labels"] or []):
        out["callee_attr"] = props.get("callee_attr")
        out["receiver_kind"] = props.get("receiver_kind")
        out["receiver_name"] = props.get("receiver_name")
        out["full_text"] = props.get("full_text")
        out["call_kind"] = props.get("call_kind")

    # Surface External metadata so explainer can classify stdlib / third_party / unresolved
    if "External" in (out["labels"] or []):
        out["category"] = props.get("category")
        out["namespace"] = props.get("namespace")
        out["side_effect_category"] = props.get("side_effect_category")
        out["side_effect_confidence"] = props.get("side_effect_confidence")
        out["side_effect_evidence"] = props.get("side_effect_evidence")

    return out


def _rel_to_dict(r: object, *, node_by_internal: Dict[int, Dict[str, object]]) -> Dict[str, object]:
    s_internal = getattr(r, "start_id", None) or getattr(r, "start", None)
    e_internal = getattr(r, "end_id", None) or getattr(r, "end", None)
    typ = getattr(r, "type", None) or getattr(r, "relationship_type", None)

    src = None
    dst = None
    if isinstance(s_internal, int):
        src = (node_by_internal.get(s_internal) or {}).get("id")
    if isinstance(e_internal, int):
        dst = (node_by_internal.get(e_internal) or {}).get("id")

    return {"src": src, "dst": dst, "type": typ}


def _fetch_paths(mg: Memgraph, query: str, params: dict) -> List[Dict[str, Sequence[Dict[str, object]]]]:
    rows = mg.execute_and_fetch(query, params)
    out: List[Dict[str, Sequence[Dict[str, object]]]] = []

    for row in rows:
        raw_nodes = row.get("nodes", []) or []
        raw_rels = row.get("rels", []) or []

        nodes: List[Dict[str, object]] = []
        node_by_internal: Dict[int, Dict[str, object]] = {}

        for n in raw_nodes:
            nd = _node_to_dict(n)
            nodes.append(nd)
            internal_id = getattr(n, "id", None)  # Memgraph internal numeric node id
            if isinstance(internal_id, int):
                node_by_internal[internal_id] = nd

        rels = [_rel_to_dict(r, node_by_internal=node_by_internal) for r in raw_rels]
        rels = [r for r in rels if r.get("src") and r.get("dst")]
        out.append({"nodes": nodes, "rels": rels})

    return out


def _path_to_string(nodes: Sequence[Dict[str, object]]) -> str:
    parts: List[str] = []
    for n in nodes:
        label = n.get("qualname") or n.get("name") or n.get("id")
        if label:
            parts.append(str(label))
    return " -> ".join(parts)


def _cap_sequence(items: Sequence[Dict[str, object]], max_items: int) -> List[Dict[str, object]]:
    return list(items[:max_items])


def _unique_in_order(items: Iterable[Dict[str, object]], key_fields: Sequence[str]) -> List[Dict[str, object]]:
    seen: set[str] = set()
    out: List[Dict[str, object]] = []
    for item in items:
        key = "|".join(str(item.get(field)) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def fetch_neighborhood(*, mg: Memgraph, repo_id: str, func_id: str, config: NeighborhoodConfig) -> Neighborhood:
    # 1) Target
    target_query = (
        "MATCH (f:Function) "
        "WHERE f.repo_id = $repo_id AND f.id = $id "
        "RETURN f.id AS id, f.name AS name, f.qualname AS qualname, f.path AS path, "
        "f.lineno AS lineno, f.kind AS kind, f.signature AS signature, "
        "f.owner_class AS owner_class, f.owner_class_id AS owner_class_id, labels(f) AS labels"
    )
    target_rows = list(mg.execute_and_fetch(target_query, {"repo_id": repo_id, "id": func_id}))
    if not target_rows:
        raise RuntimeError(f"Function not found: {func_id}")
    target = target_rows[0]

    down_depth = int(config.depth_down)
    up_depth = int(config.depth_up)

    # 2) Paths
    #
    # IMPORTANT: include OVERRIDES hops so callers->base->impl becomes visible to the neighborhood
    downstream_query = (
        "MATCH (f:Function) "
        "WHERE f.repo_id = $repo_id AND f.id = $id "
        f"MATCH p=(f)-[:CALLS|CALLS_VIRTUAL|OVERRIDES*1..{down_depth}]->(n) "
        "WHERE ALL(x IN nodes(p) WHERE x.repo_id = $repo_id) "
        "RETURN nodes(p) AS nodes, relationships(p) AS rels "
        "LIMIT $limit"
    )
    upstream_query = (
        "MATCH (f:Function) "
        "WHERE f.repo_id = $repo_id AND f.id = $id "
        f"MATCH p=(c)-[:CALLS|CALLS_VIRTUAL|OVERRIDES*1..{up_depth}]->(f) "
        "WHERE ALL(x IN nodes(p) WHERE x.repo_id = $repo_id) "
        "RETURN nodes(p) AS nodes, relationships(p) AS rels "
        "LIMIT $limit"
    )

    downstream_paths = _fetch_paths(mg, downstream_query, {"repo_id": repo_id, "id": func_id, "limit": config.max_paths})
    upstream_paths = _fetch_paths(mg, upstream_query, {"repo_id": repo_id, "id": func_id, "limit": config.max_paths})

    # 3) Accumulate
    nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []
    downstream_strings: List[str] = []
    upstream_strings: List[str] = []
    virtual_call_ids: Set[str] = set()

    for path in downstream_paths:
        nodes.extend(path["nodes"])
        edges.extend(path["rels"])
        downstream_strings.append(_path_to_string(path["nodes"]))
        for n in path["nodes"]:
            if "VirtualCall" in (n.get("labels") or []) and n.get("id"):
                virtual_call_ids.add(str(n["id"]))

    for path in upstream_paths:
        nodes.extend(path["nodes"])
        edges.extend(path["rels"])
        upstream_strings.append(_path_to_string(path["nodes"]))
        for n in path["nodes"]:
            if "VirtualCall" in (n.get("labels") or []) and n.get("id"):
                virtual_call_ids.add(str(n["id"]))

    # Ensure target present
    if not any(n.get("id") == target.get("id") for n in nodes):
        nodes.insert(0, target)

    # 4) Dedupe + cap
    nodes = _unique_in_order(nodes, ("id",))
    edges = _unique_in_order(edges, ("src", "dst", "type"))

    nodes = _cap_sequence(nodes, config.max_nodes)
    node_ids = {str(n.get("id")) for n in nodes if n.get("id") is not None}
    edges = [e for e in edges if str(e.get("src")) in node_ids and str(e.get("dst")) in node_ids]
    edges = _cap_sequence(edges, config.max_edges)

    # 5) Callers/callees
    # Direct callers: (caller)-[:CALLS]->(this)
    callers_direct_query = (
        "MATCH (c:Function)-[:CALLS]->(f:Function) "
        "WHERE c.repo_id = $repo_id AND f.repo_id = $repo_id AND f.id = $id "
        "RETURN c.id AS id"
    )
    # Effective callers via override dispatch:
    # caller CALLS base, and this impl OVERRIDES base
    callers_effective_query = (
        "MATCH (caller:Function)-[:CALLS]->(base:Function)<-[:OVERRIDES]-(impl:Function) "
        "WHERE caller.repo_id = $repo_id AND base.repo_id = $repo_id AND impl.repo_id = $repo_id "
        "  AND impl.id = $id "
        "RETURN caller.id AS id"
    )

    # Direct callees: (this)-[:CALLS]->(callee)
    callees_direct_query = (
        "MATCH (f:Function)-[:CALLS]->(c:Function) "
        "WHERE f.repo_id = $repo_id AND f.id = $id AND c.repo_id = $repo_id "
        "RETURN c.id AS id"
    )
    # Effective callees via override dispatch:
    # this CALLS base, and some impl OVERRIDES base
    callees_effective_query = (
        "MATCH (src:Function)-[:CALLS]->(base:Function)<-[:OVERRIDES]-(impl:Function) "
        "WHERE src.repo_id = $repo_id AND base.repo_id = $repo_id AND impl.repo_id = $repo_id "
        "  AND src.id = $id "
        "RETURN impl.id AS id"
    )

    callers: List[str] = []
    callers.extend([row.get("id") for row in mg.execute_and_fetch(callers_direct_query, {"repo_id": repo_id, "id": func_id})])
    callers.extend([row.get("id") for row in mg.execute_and_fetch(callers_effective_query, {"repo_id": repo_id, "id": func_id})])
    callers = sorted({c for c in callers if c})

    callees: List[str] = []
    callees.extend([row.get("id") for row in mg.execute_and_fetch(callees_direct_query, {"repo_id": repo_id, "id": func_id})])
    callees.extend([row.get("id") for row in mg.execute_and_fetch(callees_effective_query, {"repo_id": repo_id, "id": func_id})])
    callees = sorted({c for c in callees if c})

    externals = sorted({str(node.get("id")) for node in nodes if "External" in (node.get("labels") or []) and node.get("id")})

    # ------------------------
    # Virtual call resolution
    # ------------------------
    virtual_targets: List[Dict[str, object]] = []
    if virtual_call_ids:
        # parent map by class_id
        class_edges = list(
            mg.execute_and_fetch(
                "MATCH (c:Class)-[:SUBCLASS_OF]->(p:Class) "
                "WHERE c.repo_id = $repo_id AND p.repo_id = $repo_id "
                "RETURN c.id AS child, p.id AS parent",
                {"repo_id": repo_id},
            )
        )
        parents_by_child: Dict[str, Set[str]] = {}
        for row in class_edges:
            child = row.get("child")
            parent = row.get("parent")
            if child and parent:
                parents_by_child.setdefault(str(child), set()).add(str(parent))

        def _closure_types(class_id: str | None) -> Set[str]:
            if not class_id:
                return set()
            types = {str(class_id)}
            frontier = [str(class_id)]
            while frontier:
                cur = frontier.pop()
                for parent in parents_by_child.get(cur, set()):
                    if parent not in types:
                        types.add(parent)
                        frontier.append(parent)
            return types

        for vc_id in virtual_call_ids:
            vc_row = next(
                mg.execute_and_fetch(
                    "MATCH (v:VirtualCall) WHERE v.repo_id = $repo_id AND v.id = $id "
                    "RETURN v.callee_attr AS callee_attr, v.receiver_kind AS receiver_kind, "
                    "v.receiver_name AS receiver_name, v.full_text AS full_text, v.lineno AS lineno",
                    {"repo_id": repo_id, "id": vc_id},
                ),
                None,
            )
            if not vc_row:
                continue

            callee_attr = vc_row.get("callee_attr")
            receiver_kind = vc_row.get("receiver_kind")
            receiver_name = vc_row.get("receiver_name")

            # source function(s) for this virtual call
            src_rows = list(
                mg.execute_and_fetch(
                    "MATCH (src:Function)-[:CALLS_VIRTUAL]->(v:VirtualCall) "
                    "WHERE src.repo_id = $repo_id AND v.repo_id = $repo_id AND v.id = $vc_id "
                    "RETURN src.owner_class_id AS owner_class_id, src.owner_class AS owner_class",
                    {"repo_id": repo_id, "vc_id": vc_id},
                )
            )

            candidate_type_ids: Set[str] = set()

            if receiver_kind in {"self", "cls", "super"}:
                for sr in src_rows:
                    candidate_type_ids |= _closure_types(sr.get("owner_class_id"))
            elif receiver_kind == "name" and receiver_name:
                cls_rows = list(
                    mg.execute_and_fetch(
                        "MATCH (c:Class) WHERE c.repo_id = $repo_id AND c.name = $name RETURN c.id AS id",
                        {"repo_id": repo_id, "name": receiver_name},
                    )
                )
                for cr in cls_rows:
                    candidate_type_ids |= _closure_types(cr.get("id"))

            targets: Set[str] = set()

            if callee_attr:
                # Primary: method on candidate types (GLOBAL query)
                for cid in candidate_type_ids:
                    rows = mg.execute_and_fetch(
                        "MATCH (f:Function) "
                        "WHERE f.repo_id = $repo_id AND f.owner_class_id = $cid AND f.name = $name "
                        "RETURN f.id AS id",
                        {"repo_id": repo_id, "cid": cid, "name": callee_attr},
                    )
                    for r in rows:
                        if r.get("id"):
                            targets.add(str(r["id"]))

                # Fallback: any function with that name (limit)
                if not targets:
                    rows = mg.execute_and_fetch(
                        "MATCH (f:Function) WHERE f.repo_id = $repo_id AND f.name = $name "
                        "RETURN f.id AS id LIMIT 200",
                        {"repo_id": repo_id, "name": callee_attr},
                    )
                    for r in rows:
                        if r.get("id"):
                            targets.add(str(r["id"]))

            if targets:
                virtual_targets.append(
                    {
                        "virtual_call_id": vc_id,
                        "targets": sorted(targets),
                        "callee_attr": callee_attr,
                        "receiver_kind": receiver_kind,
                        "receiver_name": receiver_name,
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
        virtual_targets=virtual_targets,
    )
