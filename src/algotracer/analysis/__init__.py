"""Analysis utilities for graphs and entrypoints.

Instructions:
- Import build_dependency_graph and find_entrypoints for downstream processing.
- Re-export types so callers can depend on algotracer.analysis.*.

Explanation:
- Packages together the dependency graph and entrypoint detection helpers.
"""

from .deps import DependencyGraph, build_dependency_graph, summarize_callers
from .entrypoints import EntryPoint, find_entrypoints
from .trace import trace_from_entrypoints, summarize_traces, summarize_trace_sets, trace_and_summarize

__all__ = [
    "DependencyGraph",
    "build_dependency_graph",
    "summarize_callers",
    "EntryPoint",
    "find_entrypoints",
    "trace_from_entrypoints",
    "summarize_traces",
    "summarize_trace_sets",
    "trace_and_summarize",
]
