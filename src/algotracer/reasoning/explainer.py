from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Callable, Dict, List


SYSTEM_PROMPT = """You are AlgoTracer.

Use only the provided evidence. Do not speculate.
Return Markdown with these sections:
1) What the function is (location, signature-ish info if available)
2) What triggers it (top callers / upstream)
3) What it does (key downstream calls + external deps)
4) Notable side effects (optional, only if supported by evidence)

Cite graph references as (node:<id>) or (edge:<src>-><dst>).
"""


@dataclass(frozen=True)
class EvidencePack:
    target: Dict[str, object]
    upstream: Dict[str, object]
    downstream: Dict[str, object]
    externals: List[str]
    edges: List[Dict[str, object]]
    nodes: List[Dict[str, object]]


def build_prompt(evidence: EvidencePack) -> str:
    payload = {
        "target": evidence.target,
        "upstream": evidence.upstream,
        "downstream": evidence.downstream,
        "externals": evidence.externals,
        "nodes": evidence.nodes,
        "edges": evidence.edges,
    }
    return "\n".join(
        [
            SYSTEM_PROMPT,
            "Evidence JSON:",
            json.dumps(payload, indent=2, sort_keys=True),
        ]
    )


def deterministic_summary(evidence: EvidencePack) -> str:
    target = evidence.target
    callers = evidence.upstream.get("callers", [])
    callees = evidence.downstream.get("callees", [])
    upstream_paths = evidence.upstream.get("paths", [])
    downstream_paths = evidence.downstream.get("paths", [])
    edge_lookup = {
        (edge.get("src"), edge.get("dst")): edge
        for edge in evidence.edges
        if edge.get("src") and edge.get("dst")
    }
    target_id = target.get("id")

    lines: List[str] = []
    lines.append(
        f"- Target: `{target.get('qualname')}` in `{target.get('path')}` "
        f"(line {target.get('lineno')}) (node:{target_id})"
    )
    if target.get("signature"):
        lines.append(f"- Signature: `{target.get('signature')}` (node:{target_id})")
    if callers:
        formatted = []
        for c in callers:
            edge_key = (c, target_id)
            if edge_key in edge_lookup:
                formatted.append(f"`{c}` (edge:{c}->{target_id})")
            else:
                formatted.append(f"`{c}` (node:{c})")
        lines.append(f"- Upstream callers: {', '.join(formatted)}")
    if callees:
        formatted = []
        for c in callees:
            edge_key = (target_id, c)
            if edge_key in edge_lookup:
                formatted.append(f"`{c}` (edge:{target_id}->{c})")
            else:
                formatted.append(f"`{c}` (node:{c})")
        lines.append(f"- Downstream calls: {', '.join(formatted)}")
    if evidence.externals:
        formatted = ", ".join(f"`{c}` (node:{c})" for c in evidence.externals)
        lines.append(f"- External deps: {formatted}")
    if upstream_paths:
        lines.append("- Example upstream paths:")
        for p in upstream_paths[:5]:
            lines.append(f"  - `{p}`")
    if downstream_paths:
        lines.append("- Example downstream paths:")
        for p in downstream_paths[:5]:
            lines.append(f"  - `{p}`")

    return "\n".join(lines)


def explain(evidence: EvidencePack, llm: Callable[[str], str] | None = None) -> str:
    if llm is None:
        return deterministic_summary(evidence)

    prompt = build_prompt(evidence)
    try:
        return llm(prompt).strip()
    except Exception:
        return deterministic_summary(evidence)


def build_gemini_llm(api_key: str | None = None, model: str = "gemini-2.5-flash") -> Callable[[str], str]:
    def _call(prompt: str) -> str:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - external dep
            raise RuntimeError("google-generativeai is not installed") from exc

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "Gemini API key not found. Set GEMINI_API_KEY in the environment."
            )

        genai.configure(api_key=key)
        client = genai.GenerativeModel(model)
        resp = client.generate_content(prompt)
        text = getattr(resp, "text", None)
        return text.strip() if isinstance(text, str) else str(resp)

    return _call
