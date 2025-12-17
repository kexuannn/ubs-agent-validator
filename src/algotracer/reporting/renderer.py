from __future__ import annotations

"""Markdown report rendering for AlgoTracer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from algotracer.analysis.deps import DependencyGraph, summarize_callers
from algotracer.analysis.entrypoints import EntryPoint
from algotracer.analysis.trace import TraceSummary
from algotracer.ingest.ast_parser import ModuleInfo
from algotracer.reasoning.flow_explainer import explain_entrypoints


@dataclass
class MarkdownReport:
    path: Path
    content: str


REPORT_FILENAME = "algotrace-report.md"


def render_markdown_report(
    modules: Iterable[ModuleInfo],
    graph: DependencyGraph,
    entrypoints: List[EntryPoint],
    traces: Dict[str, TraceSummary] | None = None,
    callers: Dict[str, List[str]] | None = None,
    output_dir: Path | None = None,
    llm=None,  # optional LLM callable
) -> MarkdownReport:
    output_dir = output_dir or Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = traces or {}
    callers = callers or {}
    print(f"AlgoTracer: compiling report for {len(entrypoints)} entrypoints.")

    parts: List[str] = []

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------
    parts.append("# AlgoTracer Report")

    # ------------------------------------------------------------
    # Modules overview
    # ------------------------------------------------------------
    parts.append("\n## Modules")
    for module in modules:
        parts.append(
            f"- `{module.path}`: "
            f"{len(module.classes)} classes, {len(module.functions)} functions"
        )

    # ------------------------------------------------------------
    # Entrypoints summary
    # ------------------------------------------------------------
    parts.append("\n## Entrypoints")
    for ep in entrypoints:
        parts.append(
            f"- **{ep.kind}** `{ep.qualname}` "
            f"({ep.module}:{ep.lineno}, score={ep.score})"
        )

    # ------------------------------------------------------------
    # Defensive warning if tracing produced nothing
    # ------------------------------------------------------------
    if not traces:
        parts.append(
            "\n> ⚠️ **No downstream traces available.**  \n"
            "> This usually means no entrypoints were traced, "
            "or `max_depth` was too low to discover calls."
        )

    # ------------------------------------------------------------
    # Reasoning (heuristic or LLM)
    # ------------------------------------------------------------
    reasoning = explain_entrypoints(entrypoints, traces, llm=llm)

    # ------------------------------------------------------------
    # Entrypoint details
    # ------------------------------------------------------------
    parts.append("\n## Entrypoint Details")

    for ep in entrypoints:
        parts.append(f"\n### `{ep.qualname}`")
        parts.append(f"- **Module:** `{ep.module}`")

        ep_callers = callers.get(ep.sym_id) or summarize_callers(graph, ep.sym_id, limit=10)
        parts.append(f"- **Callers:** {', '.join(ep_callers) if ep_callers else '(none)'}")

        ts = traces.get(ep.sym_id)
        if ts and ts.example_paths:
            parts.append("- **Downstream paths:**")
            for p in ts.example_paths:
                parts.append(f"  - `{p}`")
        else:
            parts.append("- **Downstream paths:** (none at chosen depth)")

        if ep.sym_id in reasoning:
            parts.append("- **Structural reasoning:**")
            parts.append(reasoning[ep.sym_id])

    # ------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------
    content = "\n".join(parts) + "\n"
    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(content, encoding="utf-8")

    return MarkdownReport(path=report_path, content=content)
