from __future__ import annotations

"""Lightweight graph tracing utilities.

Instructions:
- Use trace_from_entrypoints(graph, entrypoints, max_depth, ...) to walk outgoing edges from entrypoint symbols.
- Nodes are expected to use the typed IDs from deps.py (mod:, sym:, ext:).
- Adjust max_depth to control traversal size; cycles are avoided by keeping the current path acyclic.
- Use max_paths/max_expansions to prevent path explosions on dense graphs.

Explanation:
- Provides small DFS-based tracing to surface reachable calls from detected entrypoints.
- Returns per-entrypoint path lists, plus structured summaries for reporting/reasoning.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from algotracer.analysis.deps import DependencyGraph
from algotracer.analysis.entrypoints import EntryPoint


@dataclass(frozen=True)
class TraceSummary:
    """Structured trace output for a single entrypoint."""

    entry: str
    internal_symbols: Set[str]
    external_nodes: Set[str]
    visited_edges: Set[Tuple[str, str]]
    example_paths: List[str]  # list of path strings like "a -> b -> c"


def _walk(
    graph: DependencyGraph,
    start: str,
    max_depth: int,
    *,
    max_paths: Optional[int] = None,
    max_expansions: Optional[int] = None,
) -> List[List[str]]:
    """Depth-limited DFS that records acyclic paths starting from `start`.

    Guardrails:
    - max_paths: stop once we've recorded this many paths
    - max_expansions: stop once we've expanded this many (node -> neighbor) steps
    """
    paths: List[List[str]] = []
    stack: List[tuple[str, List[str], int]] = [(start, [start], 0)]
    expansions = 0

    def _paths_full() -> bool:
        return max_paths is not None and len(paths) >= max_paths

    def _expansions_full() -> bool:
        return max_expansions is not None and expansions >= max_expansions

    while stack:
        if _paths_full() or _expansions_full():
            break

        node, path, depth = stack.pop()
        if depth >= max_depth:
            continue

        for nbr in sorted(graph.neighbors(node)):
            if _paths_full() or _expansions_full():
                break

            expansions += 1

            if nbr in path:  # avoid cycles within the current path
                continue

            next_path = path + [nbr]
            paths.append(next_path)
            stack.append((nbr, next_path, depth + 1))

    return paths


def trace_from_entrypoints(
    graph: DependencyGraph,
    entrypoints: Sequence[EntryPoint],
    max_depth: int = 3,
    *,
    max_paths: Optional[int] = None,
    max_expansions: Optional[int] = None,
) -> Dict[str, List[List[str]]]:
    """Return raw path traces for each entrypoint (keyed by ep.sym_id)."""
    traces: Dict[str, List[List[str]]] = {}
    for ep in entrypoints:
        print(f"AlgoTracer: tracing {ep.sym_id} to depth={max_depth} ...")
        traces[ep.sym_id] = _walk(
            graph,
            ep.sym_id,
            max_depth=max_depth,
            max_paths=max_paths,
            max_expansions=max_expansions,
        )
    return traces


def summarize_traces(traces: Dict[str, List[List[str]]]) -> Dict[str, List[str]]:
    """Convert path lists to human-readable summaries.

    Example output entry:
      "sym:...:A.fit -> sym:...:A._helper -> ext:np.dot"
    """
    summaries: Dict[str, List[str]] = {}
    for sym_id, paths in traces.items():
        summaries[sym_id] = [" -> ".join(path) for path in paths]
    return summaries


def summarize_trace_sets(
    traces: Dict[str, List[List[str]]],
    *,
    max_examples: int = 10,
) -> Dict[str, TraceSummary]:
    """Summarize raw traces into structured sets for reporting.

    Produces, per entrypoint:
    - internal_symbols: all sym:* nodes reached (including entry)
    - external_nodes: all ext:* (and mod:* if present) nodes seen
    - visited_edges: all (src, dest) edges encountered in any path
    - example_paths: a small number of shortest "A -> B -> C" strings
    """
    out: Dict[str, TraceSummary] = {}

    for entry, paths in traces.items():
        internal: Set[str] = {entry}
        external: Set[str] = set()
        edges: Set[Tuple[str, str]] = set()

        for path in paths:
            # classify nodes
            for node in path:
                if node.startswith("sym:"):
                    internal.add(node)
                elif node.startswith("ext:") or node.startswith("mod:"):
                    external.add(node)

            # record edges along the path
            for a, b in zip(path, path[1:]):
                edges.add((a, b))

        # keep a few shortest paths for readability
        example_paths = [" -> ".join(p) for p in sorted(paths, key=len)[:max_examples]]

        out[entry] = TraceSummary(
            entry=entry,
            internal_symbols=internal,
            external_nodes=external,
            visited_edges=edges,
            example_paths=example_paths,
        )
        print(f"AlgoTracer: summarized traces for {entry}: paths={len(paths)}, examples={len(example_paths)}")

    return out


def trace_and_summarize(
    graph: DependencyGraph,
    entrypoints: Sequence[EntryPoint],
    max_depth: int = 3,
    *,
    max_paths: Optional[int] = None,
    max_expansions: Optional[int] = None,
    max_examples: int = 10,
) -> Dict[str, TraceSummary]:
    """Convenience wrapper: trace from entrypoints then return structured summaries."""
    traces = trace_from_entrypoints(
        graph,
        entrypoints,
        max_depth=max_depth,
        max_paths=max_paths,
        max_expansions=max_expansions,
    )
    return summarize_trace_sets(traces, max_examples=max_examples)
