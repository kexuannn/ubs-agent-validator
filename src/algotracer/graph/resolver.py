from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from gqlalchemy import Memgraph


@dataclass(frozen=True)
class FunctionCandidate:
    id: str
    path: str
    qualname: str
    name: str
    lineno: int | None
    kind: str | None
    signature: str | None
    callers: int


def normalize_path(path: Path, repo_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def _fetch_candidates(mg: Memgraph, query: str, params: dict) -> List[FunctionCandidate]:
    rows = mg.execute_and_fetch(query, params)
    out: List[FunctionCandidate] = []
    for row in rows:
        out.append(
            FunctionCandidate(
                id=row.get("id"),
                path=row.get("path"),
                qualname=row.get("qualname"),
                name=row.get("name"),
                lineno=row.get("lineno"),
                kind=row.get("kind"),
                signature=row.get("signature"),
                callers=row.get("callers", 0),
            )
        )
    return out


def resolve_by_id(mg: Memgraph, repo_id: str, func_id: str) -> FunctionCandidate | None:
    query = (
        "MATCH (f:Function {repo_id: $repo_id, id: $id}) "
        "OPTIONAL MATCH (c)-[:CALLS {repo_id: $repo_id}]->(f) "
        "RETURN f.id AS id, f.path AS path, f.qualname AS qualname, f.name AS name, "
        "f.lineno AS lineno, f.kind AS kind, f.signature AS signature, count(c) AS callers"
    )
    candidates = _fetch_candidates(mg, query, {"repo_id": repo_id, "id": func_id})
    return candidates[0] if candidates else None


def resolve_by_path_and_name(
    mg: Memgraph,
    repo_id: str,
    path: str,
    qualname: str,
) -> FunctionCandidate | None:
    query = (
        "MATCH (f:Function {repo_id: $repo_id, path: $path, qualname: $qualname}) "
        "OPTIONAL MATCH (c)-[:CALLS {repo_id: $repo_id}]->(f) "
        "RETURN f.id AS id, f.path AS path, f.qualname AS qualname, f.name AS name, "
        "f.lineno AS lineno, f.kind AS kind, f.signature AS signature, count(c) AS callers"
    )
    candidates = _fetch_candidates(
        mg,
        query,
        {"repo_id": repo_id, "path": path, "qualname": qualname},
    )
    return candidates[0] if candidates else None


def resolve_by_name(
    mg: Memgraph,
    repo_id: str,
    qualname: str,
) -> List[FunctionCandidate]:
    query = (
        "MATCH (f:Function {repo_id: $repo_id, qualname: $qualname}) "
        "OPTIONAL MATCH (c)-[:CALLS {repo_id: $repo_id}]->(f) "
        "RETURN f.id AS id, f.path AS path, f.qualname AS qualname, f.name AS name, "
        "f.lineno AS lineno, f.kind AS kind, f.signature AS signature, count(c) AS callers"
    )
    return _fetch_candidates(mg, query, {"repo_id": repo_id, "qualname": qualname})


def resolve_by_path_and_lineno(
    mg: Memgraph,
    repo_id: str,
    path: str,
    lineno: int,
) -> List[FunctionCandidate]:
    query = (
        "MATCH (f:Function {repo_id: $repo_id, path: $path, lineno: $lineno}) "
        "OPTIONAL MATCH (c)-[:CALLS {repo_id: $repo_id}]->(f) "
        "RETURN f.id AS id, f.path AS path, f.qualname AS qualname, f.name AS name, "
        "f.lineno AS lineno, f.kind AS kind, f.signature AS signature, count(c) AS callers"
    )
    return _fetch_candidates(
        mg,
        query,
        {"repo_id": repo_id, "path": path, "lineno": lineno},
    )


def disambiguate(candidates: List[FunctionCandidate]) -> FunctionCandidate | None:
    if not candidates:
        return None

    def _score(c: FunctionCandidate) -> tuple[int, int, int, str]:
        path_len = len(c.path or "")
        qual_len = len(c.qualname or "")
        callers = c.callers or 0
        return (path_len, qual_len, -callers, c.id)

    return sorted(candidates, key=_score)[0]
