from __future__ import annotations

"""Lightweight structural explainer with optional LLM summarization (Gemini-ready).

Use:
- explain_entrypoints(traces, llm=None) -> dict[sym_id, str]
  where each value is a short markdown section.
- build_gemini_llm(api_key=None, model=\"gemini-1.5-flash\") returns a callable `llm(prompt)->str`.

Design:
- Prompt is bounded: we summarize each TraceSummary (top nodes + sample paths),
  NOT the full graph edges.
- If llm is None or fails, returns a deterministic heuristic summary.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import os
from typing import Callable, Dict, Iterable, List, Sequence

from algotracer.analysis.trace import TraceSummary
from algotracer.analysis.entrypoints import EntryPoint


GEMINI_API_KEY = None


SYSTEM_PROMPT = """You are AlgoTracer.

You will be given a static call trace summary for one entrypoint.
Write a brief reasoning summary (max 6 bullets) of what the entrypoint likely does.

Rules:
- Describe likely stages (validation, preprocessing, core algorithm, post-processing, outputs).
- Mention only the most important external libraries.
- If the trace is shallow or mostly ext:* (unresolved), explicitly say so as a caveat.
- Output MUST be markdown bullets only (lines starting with "- ").
"""


def _group_ext_prefix(ext_nodes: Iterable[str], top_n: int = 6) -> List[str]:
    # ext:numpy.linalg.lstsq -> numpy
    counts = defaultdict(int)
    for n in ext_nodes:
        raw = n.removeprefix("ext:")
        prefix = raw.split(".", 1)[0] if raw else "unknown"
        counts[prefix] += 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [f"{k} ({v})" for k, v in ranked]


def _heuristic_summary(ts: TraceSummary) -> str:
    # Deterministic fallback: short, readable
    internal = sorted(ts.internal_symbols)
    external = sorted(ts.external_nodes)

    # Rank nodes by appearance in example_paths
    freq = Counter()
    for p in ts.example_paths:
        for node in p.split(" -> "):
            freq[node] += 1

    top_internal = [n for n, _ in freq.most_common() if n.startswith("sym:")][:6]
    top_ext = [n for n, _ in freq.most_common() if n.startswith("ext:")][:12]

    ext_groups = _group_ext_prefix(top_ext or external, top_n=6)

    lines = []
    lines.append(f"- Entrypoint: `{ts.entry}`")
    if top_internal:
        lines.append("- Key internal symbols:")
        for n in top_internal[:5]:
            lines.append(f"  - `{n}`")
    if ext_groups:
        lines.append(f"- Key external dependencies: {', '.join(ext_groups)}")
    if not ts.example_paths:
        lines.append("- Caveat: trace has no paths at the chosen depth (graph may be sparse).")
    internal_other = [n for n in internal if n != ts.entry]
    if external and not any(n.startswith("sym:") for n in internal_other):
        lines.append("- Caveat: many calls are unresolved (`ext:*`); internal sym→sym resolution may be limited.")
    return "\n".join(lines)


def build_trace_prompt(entrypoint: EntryPoint, ts: TraceSummary, max_paths: int = 10) -> str:
    """Small, bounded prompt for one entrypoint trace."""
    internal = sorted(ts.internal_symbols)
    external = sorted(ts.external_nodes)

    # Keep it small
    internal_small = internal[:40]
    external_small = external[:80]
    paths_small = ts.example_paths[:max_paths]

    parts = [
        SYSTEM_PROMPT,
        "",
        f"EntryPoint: {entrypoint.kind} {entrypoint.qualname} (module: {entrypoint.module}, score: {entrypoint.score})",
        f"EntrypointSymId: {entrypoint.sym_id}",
        "",
        "Internal symbols (sym:*):",
        "\n".join(f"- {x}" for x in internal_small) or "- (none)",
        "",
        "External nodes (ext:*):",
        "\n".join(f"- {x}" for x in external_small) or "- (none)",
        "",
        "Example paths:",
    ]
    if paths_small:
        for p in paths_small:
            parts.append(f"- {p}")
    else:
        parts.append("- (none)")
    return "\n".join(parts)


def explain_entrypoints(
    entrypoints: Sequence[EntryPoint],
    traces: Dict[str, TraceSummary],
    llm: Callable[[str], str] | None = None,
) -> Dict[str, str]:
    """Return per-entrypoint markdown reasoning blocks keyed by ep.sym_id."""
    # Map sym_id -> EntryPoint for easy lookup
    ep_by_id = {ep.sym_id: ep for ep in entrypoints}

    out: Dict[str, str] = {}
    for sym_id, ts in traces.items():
        ep = ep_by_id.get(sym_id)
        if ep is None:
            # fallback if mismatch
            out[sym_id] = _heuristic_summary(ts)
            continue

        if llm is None:
            out[sym_id] = _heuristic_summary(ts)
            continue

        prompt = build_trace_prompt(ep, ts)
        try:
            print(f"AlgoTracer LLM: calling for {sym_id} ...")
            resp = llm(prompt).strip()
            print(f"AlgoTracer LLM: received response for {sym_id}: {resp[:200]!r}")
            # If the model didn't follow bullet format, still don't crash
            out[sym_id] = resp if resp else _heuristic_summary(ts)
        except Exception as e:
            print(f"AlgoTracer LLM: failed for {sym_id}: {type(e).__name__}: {e}")
            out[sym_id] = _heuristic_summary(ts)

    return out


def build_gemini_llm(api_key: str | None = None, model: str = "gemini-2.5-flash") -> Callable[[str], str]:
    """Return an llm(prompt)->str callable backed by Gemini.

    Requires `google-generativeai` and GEMINI_API_KEY in the environment (or provided api_key).
    """
    def _call(prompt: str) -> str:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - external dep
            raise RuntimeError("google-generativeai is not installed") from exc

        key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "Gemini API key not found. "
                "Set GEMINI_API_KEY at top of file or in environment."
            )

        genai.configure(api_key=key)
        client = genai.GenerativeModel(model)
        print(f"AlgoTracer LLM: invoking Gemini model={model}")
        resp = client.generate_content(prompt)
        text = getattr(resp, "text", None)
        return text.strip() if isinstance(text, str) else str(resp)

    return _call
