from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

from gqlalchemy import Memgraph


@dataclass(frozen=True)
class MemgraphConfig:
    host: str = "127.0.0.1"
    port: int = 7687
    user: str | None = None
    password: str | None = None

    @classmethod
    def from_env(cls) -> "MemgraphConfig":
        host = os.getenv("MEMGRAPH_HOST", "127.0.0.1")
        port = int(os.getenv("MEMGRAPH_PORT", "7687"))
        user = os.getenv("MEMGRAPH_USER") or None
        password = os.getenv("MEMGRAPH_PASSWORD") or None
        return cls(host=host, port=port, user=user, password=password)


def connect_memgraph(config: MemgraphConfig) -> Memgraph:
    kwargs = {"host": config.host, "port": config.port}
    if config.user:
        kwargs["username"] = config.user
    if config.password:
        kwargs["password"] = config.password
    return Memgraph(**kwargs)


def ensure_schema(mg: Memgraph) -> None:
    statements: Iterable[str] = (
        "CREATE INDEX ON :Repo(repo_id)",
        "CREATE INDEX ON :Module(repo_id)",
        "CREATE INDEX ON :Module(path)",
        "CREATE INDEX ON :Function(repo_id)",
        "CREATE INDEX ON :Function(id)",
        "CREATE INDEX ON :Function(name)",
        "CREATE INDEX ON :Function(path)",
        "CREATE INDEX ON :External(repo_id)",
        "CREATE INDEX ON :External(id)",
    )
    for stmt in statements:
        try:
            mg.execute(stmt)
        except Exception:
            # Memgraph may throw if the index already exists.
            continue


def clear_repo(mg: Memgraph, repo_id: str) -> None:
    mg.execute(
        "MATCH (n {repo_id: $repo_id}) DETACH DELETE n",
        {"repo_id": repo_id},
    )
