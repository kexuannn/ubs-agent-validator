from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
from typing import Callable, Dict, List, Optional, Tuple


SYSTEM_PROMPT = """You are AlgoTracer.

HARD RULES (non-negotiable):
- Use only the provided Evidence JSON. Do not speculate.
- NEVER describe an OVERRIDES edge as a call. OVERRIDES != CALLS.
- Only say “X triggers Y” or “X calls Y” if there is a CALLS edge X->Y in the evidence.
- If something is a hypothesis, you MUST label it clearly as **GUESS** and it must be grounded in evidence (names/types) — but do not invent callers.

Output Markdown with these sections:
1) What the function is (location, signature-ish info if available)
2) What triggers it (explicit upstream CALLS into this function; if none, say so)
3) What it does (explicit downstream CALLS from this function; externals; virtual calls)
4) Inheritance / overrides (only from OVERRIDES edges; do not imply calling)
5) Notable side effects (only if supported by evidence; otherwise omit)

Citations:
- Cite graph references exactly as (node:<id>) or (edge:<src>-><dst>).
- Any claim about calling/triggering MUST include an (edge:...->...) citation of type CALLS.
"""


@dataclass(frozen=True)
class EvidencePack:
    target: Dict[str, object]
    upstream: Dict[str, object]
    downstream: Dict[str, object]
    externals: List[str]
    edges: List[Dict[str, object]]
    nodes: List[Dict[str, object]]
    # Optional new field (safe default so it won't break callers)
    virtual_targets: List[Dict[str, object]] = field(default_factory=list)


def build_prompt(evidence: EvidencePack) -> str:
    payload = {
        "target": evidence.target,
        "upstream": evidence.upstream,
        "downstream": evidence.downstream,
        "externals": evidence.externals,
        "nodes": evidence.nodes,
        "edges": evidence.edges,
        "virtual_targets": evidence.virtual_targets,
    }
    return "\n".join(
        [
            SYSTEM_PROMPT,
            "Evidence JSON:",
            json.dumps(payload, indent=2, sort_keys=True),
        ]
    )


def _edge_type_lookup(edges: List[Dict[str, object]]) -> Dict[Tuple[str, str], str]:
    out: Dict[Tuple[str, str], str] = {}
    for e in edges:
        s = e.get("src")
        d = e.get("dst")
        t = e.get("type")
        if isinstance(s, str) and isinstance(d, str) and isinstance(t, str):
            out[(s, d)] = t
    return out


def _find_override_edges_for_target(
    target_id: str,
    edges: List[Dict[str, object]],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Return (overrides_outgoing, overrides_incoming) as (src,dst) pairs."""
    out: List[Tuple[str, str]] = []
    inc: List[Tuple[str, str]] = []
    for e in edges:
        if e.get("type") != "OVERRIDES":
            continue
        s = e.get("src")
        d = e.get("dst")
        if not isinstance(s, str) or not isinstance(d, str):
            continue
        if s == target_id:
            out.append((s, d))
        if d == target_id:
            inc.append((s, d))
    return out, inc


def _find_virtual_calls_from_target(
    target_id: str,
    edges: List[Dict[str, object]],
) -> List[Tuple[str, str]]:
    """Return CALLS_VIRTUAL edges (src,dst) starting at target_id."""
    out: List[Tuple[str, str]] = []
    for e in edges:
        if e.get("type") != "CALLS_VIRTUAL":
            continue
        s = e.get("start") or e.get("src")
        d = e.get("end") or e.get("dst")
        if isinstance(s, str) and isinstance(d, str) and s == target_id:
            out.append((s, d))
    return out


def deterministic_summary(evidence: EvidencePack) -> str:
    target = evidence.target
    target_id = str(target.get("id") or "")

    # These are usually produced by neighborhood.py
    callers = evidence.upstream.get("callers", []) or []
    callees = evidence.downstream.get("callees", []) or []
    upstream_paths = evidence.upstream.get("paths", []) or []
    downstream_paths = evidence.downstream.get("paths", []) or []

    edge_type = _edge_type_lookup(evidence.edges)

    # Node lookup by id (so we can inspect VirtualCall / External metadata)
    nodes_by_id: Dict[str, Dict[str, object]] = {}
    for n in evidence.nodes or []:
        nid = n.get("id")
        if isinstance(nid, str):
            nodes_by_id[nid] = n

    lines: List[str] = []

    # 1) What the function is
    qual = target.get("qualname")
    path = target.get("path")
    lineno = target.get("lineno")
    sig = target.get("signature")
    lines.append("1) What the function is")
    lines.append(
        f"The function is `{qual}` (node:{target_id}), located in `{path}` at line {lineno}."
    )
    if sig:
        lines.append(f"Its signature is `{sig}`. (node:{target_id})")
    lines.append("")

    # 2) What triggers it (ONLY explicit CALLS edges into target)
    lines.append("2) What triggers it")
    call_edges_in: List[str] = []
    for c in callers:
        if not isinstance(c, str):
            continue
        if edge_type.get((c, target_id)) == "CALLS":
            call_edges_in.append(c)

    if call_edges_in:
        formatted = ", ".join(f"`{c}` (edge:{c}->{target_id})" for c in sorted(call_edges_in))
        lines.append(f"Explicit callers (CALLS edges into target): {formatted}.")
    else:
        # Important: do NOT infer framework callbacks here
        lines.append(
            "No explicit callers were found in the provided graph evidence "
            "(no CALLS edges into this function within the queried neighborhood)."
        )
    lines.append("")

    # 3) What it does (ONLY CALLS edges out of target; externals; virtual calls)
    lines.append("3) What it does")

    call_edges_out: List[str] = []
    for c in callees:
        if not isinstance(c, str):
            continue
        if edge_type.get((target_id, c)) == "CALLS":
            call_edges_out.append(c)

    if call_edges_out:
        formatted = ", ".join(f"`{c}` (edge:{target_id}->{c})" for c in sorted(call_edges_out))
        lines.append(f"It makes the following direct calls (CALLS edges): {formatted}.")
    else:
        lines.append("No direct downstream CALLS edges were found from this function in the evidence.")

    # Externals (already computed by neighborhood)
    if evidence.externals:
        formatted = ", ".join(f"`{x}` (node:{x})" for x in evidence.externals)
        lines.append(f"External dependencies present in the neighborhood: {formatted}.")

    # -------------------------
    # Virtual / dynamic dispatch
    # -------------------------
    vc_edges = _find_virtual_calls_from_target(target_id, evidence.edges)
    vc_ids_from_target = [dst for (_, dst) in vc_edges]

    # Map vc_id -> virtual_targets entry, but only keep those with at least one internal target
    vc_with_internal: Dict[str, Dict[str, object]] = {}
    for vt in evidence.virtual_targets or []:
        vc_id = vt.get("virtual_call_id")
        if not isinstance(vc_id, str):
            continue
        if vc_id not in vc_ids_from_target:
            continue
        targets = [t for t in (vt.get("targets") or []) if isinstance(t, str)]
        if targets:
            vc_with_internal[vc_id] = {**vt, "targets": targets}

    # 3a) VC sites that have INTERNAL candidate targets:
    # explain the *relationship* (dynamic dispatch) but do NOT talk about VC nodes themselves.
    if vc_with_internal:
        all_targets = sorted(
            {t for vt in vc_with_internal.values() for t in vt["targets"]}  # type: ignore[index]
        )
        formatted = ", ".join(f"`{t}` (node:{t})" for t in all_targets[:20])
        lines.append(
            "This function has dynamic dispatch sites that may call the following internal implementations "
            f"(derived from CALLS_VIRTUAL edges plus type/inheritance analysis): {formatted}."
        )

    # 3b) VC sites with NO internal candidates:
    # treat them as unresolved ONLY if they ALSO do not have a CALLS edge to a matching External.
    unresolved_vc_ids: List[str] = []

    for vc_id in vc_ids_from_target:
        # already resolved to internal
        if vc_id in vc_with_internal:
            continue

        node = nodes_by_id.get(vc_id) or {}
        full_text = node.get("full_text")

        # If we know the call text, check whether there is a CALLS edge to the matching External.
        if isinstance(full_text, str) and full_text:
            external_id = f"ext:{full_text}"
            if edge_type.get((target_id, external_id)) == "CALLS":
                # Consider this virtual call "resolved to an external"; no need to mention it as unresolved VC.
                continue

        # Otherwise this really is unresolved from the explainer's POV.
        unresolved_vc_ids.append(vc_id)

    if unresolved_vc_ids:
        descs: List[str] = []
        for vc_id in unresolved_vc_ids[:10]:
            n = nodes_by_id.get(vc_id) or {}
            callee_attr = n.get("callee_attr")
            full_text = n.get("full_text")
            if isinstance(full_text, str) and full_text:
                descs.append(
                    f"`{vc_id}` calling `{full_text}` (edge:{target_id}->{vc_id})"
                )
            elif isinstance(callee_attr, str) and callee_attr:
                descs.append(
                    f"`{vc_id}` calling attribute `{callee_attr}` (edge:{target_id}->{vc_id})"
                )
            else:
                descs.append(f"`{vc_id}` (edge:{target_id}->{vc_id})")

        lines.append(
            "There are dynamic callsites that the graph could not resolve to specific internal implementations "
            "or to known external functions; they remain as unresolved virtual calls in the graph: "
            + ", ".join(descs)
            + "."
        )

    lines.append("")

    # 4) Inheritance / overrides (OVERRIDES edges only; DO NOT imply calls)
    lines.append("4) Inheritance / overrides")
    ov_out, ov_in = _find_override_edges_for_target(target_id, evidence.edges)
    if ov_out:
        formatted = ", ".join(f"`{dst}` (edge:{src}->{dst})" for src, dst in ov_out)
        lines.append(f"This method overrides: {formatted}.")
    elif ov_in:
        formatted = ", ".join(f"`{src}` (edge:{src}->{dst})" for src, dst in ov_in)
        lines.append(f"This method is overridden by: {formatted}.")
    else:
        lines.append("No OVERRIDES edges involving this function were found in the evidence.")
    lines.append("")

    # 5) Notable side effects (only if evidence-based)
    lines.append("5) Notable side effects")
    # We cannot prove side effects from calls alone; keep conservative.
    # But we can at least mention obvious mutators if present as externals (e.g., .pop/.append) without asserting impact.
    mutators = [
        x
        for x in evidence.externals
        if isinstance(x, str) and re.search(r"\.(pop|append|remove|clear|update|setdefault)$", x)
    ]
    if mutators:
        formatted = ", ".join(f"`{m}` (node:{m})" for m in mutators[:20])
        lines.append(
            "The evidence shows calls to common mutator methods (which may imply state changes), "
            f"but the exact state impact is not provable from the graph alone: {formatted}."
        )
    else:
        lines.append("No side effects are provable from the provided graph evidence.")
    lines.append("")

    # Optional: paths (useful for debugging, but keep short)
    if upstream_paths:
        lines.append("Evidence: example upstream paths")
        for p in upstream_paths[:5]:
            lines.append(f"- `{p}`")
        lines.append("")

    if downstream_paths:
        # Hide paths that only go through VCs we consider "resolved"
        resolved_vc_ids = set(vc_ids_from_target) - set(unresolved_vc_ids)

        def _path_has_resolved_vc(path: str) -> bool:
            return any(vc_id in path for vc_id in resolved_vc_ids)

        filtered_downstream = [
            p for p in downstream_paths if not _path_has_resolved_vc(p)
        ]

        if filtered_downstream:
            lines.append(
                "Evidence: example downstream paths "
                "(functions plus any unresolved dynamic calls)"
            )
            for p in filtered_downstream[:5]:
                lines.append(f"- `{p}`")
            lines.append("")

    return "\n".join(lines).strip()


def _llm_output_is_evidence_safe(llm_text: str, evidence: EvidencePack) -> bool:
    """
    A strict gate:
    - If the model uses the word 'calls' or 'triggered by' without citing an edge:...->...,
      reject.
    - If it treats OVERRIDES as a call, reject.
    """
    t = llm_text.lower()

    # OVERRIDES misuse
    if "overrides and then calls" in t:
        return False
    if "calls the base class" in t and "edge:" not in t:
        return False

    # Require edge citations for call/trigger language
    risky_words = ["trigger", "triggered", "calls", "invoked by", "called by"]
    if any(w in t for w in risky_words):
        # Must have at least one edge citation if it uses these words
        if "edge:" not in t:
            return False

    # If it contains an edge citation, ensure it is a real CALLS edge (not OVERRIDES)
    edge_type = _edge_type_lookup(evidence.edges)
    cited_edges = re.findall(r"edge:([^\s\)]+)->([^\s\)]+)", llm_text)
    for src, dst in cited_edges:
        typ = edge_type.get((src, dst))
        if typ is None:
            # unknown edge cited
            return False
        # If they describe calling with an OVERRIDES edge, reject.
        if typ == "OVERRIDES" and ("call" in t or "trigger" in t):
            return False

    return True


def explain(evidence: EvidencePack, llm: Optional[Callable[[str], str]] = None) -> str:
    """
    Deterministic-first:
    - Always return a correct evidence summary.
    - If LLM is provided, use it only if the output passes a strict evidence safety gate.
      Otherwise return deterministic output.
    """
    base = deterministic_summary(evidence)

    if llm is None:
        return base

    prompt = build_prompt(evidence)
    try:
        llm_text = llm(prompt).strip()
    except Exception:
        return base

    if not llm_text:
        return base

    if not _llm_output_is_evidence_safe(llm_text, evidence):
        return base

    return llm_text


def build_gemini_llm(api_key: str | None = None, model: str = "gemini-2.5-flash") -> Callable[[str], str]:
    def _call(prompt: str) -> str:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("google-generativeai is not installed") from exc

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Gemini API key not found. Set GEMINI_API_KEY in the environment.")

        genai.configure(api_key=key)
        client = genai.GenerativeModel(model)
        resp = client.generate_content(prompt)
        text = getattr(resp, "text", None)
        return text.strip() if isinstance(text, str) else str(resp)

    return _call
