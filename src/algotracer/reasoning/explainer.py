from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

from algotracer.graph.neighborhood import Neighborhood

SYSTEM_PROMPT = """You are AlgoTracer, an explainer for Python call graphs grounded in concrete graph evidence and code context.

ROLE & SOURCES OF TRUTH
- Your ONLY factual sources of truth are:
  - The Evidence JSON (graph nodes/edges, externals, virtual_targets)
  - The provided source_snippet for the target function (if any)
- Do NOT invent functions, files, or edges that are not present there.
- You MAY infer *intent* and *semantics* of the function from:
  - Its name and qualname (e.g. EquityDemoStrategy.on_trade)
  - The surrounding source_snippet
  - Names of callees/callers and external dependencies
  but every such inference MUST be clearly labeled as **GUESS**.

HARD RULES ABOUT CALLS
- ALL statements about who calls whom MUST be backed by CALLS edges from the evidence.
- NEVER describe an OVERRIDES edge as a call. OVERRIDES != CALLS.
- Only say “X triggers Y”, “X calls Y”, or “X is invoked by Y” if there is a CALLS edge X->Y in the evidence, and you cite it as (edge:<src>-><dst>).
- You may describe override relationships (base vs. override) but MUST NOT say they call each other unless a CALLS edge exists.

DOMAIN-AWARE INTENT (VERY IMPORTANT)
In section **6) High-level intent (GUESS)**, you should write a human-friendly, repo-aware explanation of what the function is trying to do.

Use the following cues aggressively when forming your GUESS:
- Trading / strategy style:
  - Class or qualname includes words like `Strategy`, `Algo`, `Portfolio`, or lives under modules like `vnpy`, `alpha`, `trade`, `execution`.
  - Method names like `on_trade`, `on_bar`, `on_bars`, `on_tick`, `on_init`, `on_start`, `on_stop`.
  - Types or identifiers like `TradeData`, `BarData`, `Order`, `position`, `holding_days`, `cash`, `volume`, `price`, `signal`.
  → In these cases, explain in plain language things like:
    - “This appears to be a trade-event callback that updates per-symbol state when a trade fills.”
    - “This function looks like a bar-driven rebalance step that builds sell and buy lists based on ranking signals and current positions.”
- Data / analytics style:
  - Mentions `DataFrame`, `polars`, `pandas`, `numpy`, etc.
  → Explain that it likely transforms or filters tabular data for analysis or downstream steps.
- Infrastructure / networking style:
  - Mentions `request`, `session`, `client`, `http`, etc.
  → Explain that it likely coordinates external service calls or HTTP requests.

When you GUESS, always:
- Ground the guess explicitly in observed names / identifiers / patterns.
- Use phrases like “This appears to…”, “This likely…”, “In this strategy, this probably…” and end with (**GUESS**).
- Do NOT contradict the hard evidence (e.g. don’t say it places orders if there is no sign of order APIs in snippet/externals).

OUTPUT FORMAT
Output Markdown with these sections:

1) What the function is
   - Location, qualname, path, line number, and signature if available.
   - You may restate or briefly comment on the source snippet (but do not rewrite all of it).

2) What triggers it
   - List explicit upstream CALLS edges into this function.
   - If none, clearly say no callers were found in the evidence.

3) What it does
   - Describe explicit downstream CALLS edges to other functions or externals.
   - Describe dynamic/virtual callsites based on the provided virtual_targets (only as “may call…”).
   - Use citations (edge:<src>-><dst>) for all concrete calling relationships.

4) Inheritance / overrides
   - Describe base/override relationships using OVERRIDES edges only.
   - Do NOT imply that overrides call bases unless there is an explicit CALLS edge.

5) Notable side effects
   - Use explicit side_effect_* fields on External nodes if available.
   - Otherwise you may mention that calls to obvious mutators (e.g. `.append`, `.pop`) *may* imply state changes, but clarify that this is not proven.

6) High-level intent (GUESS)
   - Provide a short, human-understandable summary of what the function is *trying* to do in the context of the repo.
   - Use domain cues (trading, data processing, networking, etc.) plus the source snippet and external dependencies.
   - Make it specific when possible, e.g.:
     - “This looks like a trade callback that clears per-symbol holding-day state whenever a short/closing trade is executed, so the strategy stops tracking that position. (**GUESS**)"
   - Always clearly mark this section as speculative using **GUESS**.

CITATIONS
- Cite graph references exactly as (node:<id>) or (edge:<src>-><dst>).
- Any factual claim about calling/triggering MUST include an (edge:...->...) citation of type CALLS.
"""


@dataclass(frozen=True)
class EvidencePack:
    target: Dict[str, object]
    upstream: Dict[str, object]
    downstream: Dict[str, object]
    externals: List[str]
    edges: List[Dict[str, object]]
    nodes: List[Dict[str, object]]
    # Optional: resolved virtual call targets
    virtual_targets: List[Dict[str, object]] = field(default_factory=list)
    # Optional: a code snippet for the target function (filled by caller if desired)
    source_snippet: str | None = None


def build_prompt(evidence: EvidencePack) -> str:
    payload = {
        "target": evidence.target,
        "upstream": evidence.upstream,
        "downstream": evidence.downstream,
        "externals": evidence.externals,
        "nodes": evidence.nodes,
        "edges": evidence.edges,
        "virtual_targets": evidence.virtual_targets,
        "source_snippet": evidence.source_snippet,
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
    """
    Build a fully deterministic, report-style explanation.

    Sections:
      1) Integrated explanation (GUESS)      [heuristic fallback; overwritten by LLM if provided]
      2) Proven call-graph facts             [pure graph evidence]
      3) External interactions & side effects
      4) Code excerpt                        [trimmed source snippet]
    """
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

    # Precompute external nodes (we’ll reuse in side-effects / externals section)
    external_nodes: List[Dict[str, object]] = []
    for eid in evidence.externals or []:
        if not isinstance(eid, str):
            continue
        node = nodes_by_id.get(eid)
        if node:
            external_nodes.append(node)

    lines: List[str] = []

    # ---------- Header: what the function is ----------
    qual = target.get("qualname")
    path = target.get("path")
    lineno = target.get("lineno")
    sig = target.get("signature")
    lines.append(f"Function: `{qual}` ({path}:{lineno}) (node:{target_id})")
    if sig:
        lines.append(f"Signature: `{sig}`")
    lines.append("")

    # Upstream/downstream CALLS edges
    call_edges_in: List[str] = []
    for c in callers:
        if not isinstance(c, str):
            continue
        if edge_type.get((c, target_id)) == "CALLS":
            call_edges_in.append(c)

    call_edges_out: List[str] = []
    for c in callees:
        if not isinstance(c, str):
            continue
        if edge_type.get((target_id, c)) == "CALLS":
            call_edges_out.append(c)

    # Virtual / dynamic dispatch
    vc_edges = _find_virtual_calls_from_target(target_id, evidence.edges)
    vc_ids_from_target = [dst for (_, dst) in vc_edges]

    # Map vc_id -> virtual_targets entry, but only keep those with at least one internal target
    vc_with_internal: Dict[str, Dict[str, object]] = {}
    vc_with_external: Dict[str, List[str]] = {}
    for vt in evidence.virtual_targets or []:
        vc_id = vt.get("virtual_call_id")
        if not isinstance(vc_id, str):
            continue
        if vc_id not in vc_ids_from_target:
            continue
        targets = [t for t in (vt.get("targets") or []) if isinstance(t, str)]
        external_targets = [t for t in (vt.get("external_targets") or []) if isinstance(t, str)]
        if targets:
            vc_with_internal[vc_id] = {**vt, "targets": targets}
        if external_targets:
            vc_with_external[vc_id] = external_targets

    # VC sites that have INTERNAL candidate targets
    vc_resolved_targets: List[str] = []
    if vc_with_internal:
        vc_resolved_targets = sorted(
            {t for vt in vc_with_internal.values() for t in vt["targets"]}  # type: ignore[index]
        )
    vc_resolved_external: List[str] = []
    if vc_with_external:
        vc_resolved_external = sorted({t for targets in vc_with_external.values() for t in targets})

    # VC sites with NO internal candidates (and not resolved to externals)
    unresolved_vc_ids: List[str] = []
    for vc_id in vc_ids_from_target:
        if vc_id in vc_with_internal or vc_id in vc_with_external:
            continue
        node = nodes_by_id.get(vc_id) or {}
        full_text = node.get("full_text")
        if isinstance(full_text, str) and full_text:
            external_id = f"ext:{full_text}"
            if edge_type.get((target_id, external_id)) == "CALLS":
                continue
        unresolved_vc_ids.append(vc_id)

    # Build a human-friendly label for each VC based on its node metadata.
    # We prefer:
    #   - full_text (e.g. "self.pos_data.items")
    #   - or receiver_name.callee_attr
    #   - or just callee_attr
    vc_labels: Dict[str, str] = {}
    for vc_id in vc_ids_from_target:
        node = nodes_by_id.get(vc_id) or {}
        label: Optional[str] = None

        full_text = node.get("full_text")
        if isinstance(full_text, str) and full_text.strip():
            label = full_text.strip()
        else:
            recv = node.get("receiver_name")
            attr = node.get("callee_attr")
            if isinstance(recv, str) and isinstance(attr, str):
                label = f"{recv}.{attr}"
            elif isinstance(attr, str):
                label = attr

        if label:
            vc_labels[vc_id] = label

    # 4) Inheritance / overrides (used later in facts; OVERRIDES edges only)
    ov_out, ov_in = _find_override_edges_for_target(target_id, evidence.edges)

    # 5) Notable side effects (only if evidence-based)
    side_effect_nodes = [
        n for n in external_nodes if n.get("side_effect_category")
    ]

    # Heuristic high-level intent setup (fallback if no LLM)
    qual_str = str(qual) if qual is not None else ""
    snippet = evidence.source_snippet or ""

    # Add semantic hints from neighbor node names (callers, callees, externals, overrides, side-effect externals)
    def _node_label(nid: str) -> str:
        n = nodes_by_id.get(nid) or {}
        return str(n.get("qualname") or n.get("name") or n.get("id") or nid)

    neighbor_labels: List[str] = []
    neighbor_labels.extend(_node_label(c) for c in callers if isinstance(c, str))
    neighbor_labels.extend(_node_label(c) for c in callees if isinstance(c, str))
    neighbor_labels.extend(_node_label(e) for e in evidence.externals if isinstance(e, str))
    neighbor_labels.extend(_node_label(dst) for (_, dst) in ov_out)
    neighbor_labels.extend(_node_label(src) for (src, _) in ov_in)
    for n in side_effect_nodes:
        nid = n.get("id")
        if isinstance(nid, str):
            neighbor_labels.append(_node_label(nid))

    context_text = " ".join([qual_str, snippet] + neighbor_labels)
    lc_context = context_text.lower()

    guess_bits: List[str] = []

    # --- Trading / strategy-specific heuristics ---
    is_strategy = "strategy" in lc_context
    mentions_trade = "on_trade" in lc_context or "trade" in lc_context
    mentions_bars = "on_bars" in lc_context or "on_bar" in lc_context or "bars" in lc_context
    mentions_positions = any(tok in lc_context for tok in ["pos", "position", "holding_days", "get_pos"])
    mentions_price = any(tok in lc_context for tok in ["price", "volume", "cash", "turnover"])
    mentions_vnpy = "vnpy" in lc_context

    if is_strategy and mentions_trade:
        detail_bits = []
        if "holding_days" in lc_context:
            detail_bits.append("updates or clears per-symbol holding period state")
        if "direction.short" in lc_context or "direction.short" in lc_context.replace(" ", ""):
            detail_bits.append("reacts specifically to short/closing trades")

        detail_text = ""
        if detail_bits:
            detail_text = " — " + ", ".join(detail_bits)

        guess_bits.append(
            "Trade-event callback in a strategy that adjusts internal position bookkeeping "
            f"when trades arrive{detail_text}. (**GUESS**)"
        )
    elif is_strategy and mentions_bars:
        extra = []
        if "signal" in lc_context:
            extra.append("sorting or filtering assets by signal score")
        if "buy" in lc_context or "sell" in lc_context:
            extra.append("building instruments to buy/sell")
        if "cash" in lc_context or "turnover" in lc_context:
            extra.append("allocating cash across symbols")

        extra_txt = ""
        if extra:
            extra_txt = " It appears to be " + ", ".join(extra) + "."

        guess_bits.append(
            "Bar-driven rebalancing step in an equity strategy, deciding which symbols to hold, buy, or exit "
            f"based on signals and positions.{extra_txt} (**GUESS**)"
        )
    elif is_strategy and (mentions_positions or mentions_price or mentions_vnpy):
        guess_bits.append(
            "Part of a trading strategy coordinating positions, prices, and cash during the strategy lifecycle. (**GUESS**)"
        )

    # --- Data / analytics style heuristics ---
    if not guess_bits and any(tok in lc_context for tok in ["dataframe", "polars", "pandas", "numpy"]):
        guess_bits.append(
            "Performs data transformation or analysis over table-like data structures (e.g. DataFrames). (**GUESS**)"
        )

    # --- Networking / HTTP heuristics ---
    if not guess_bits and any(tok in lc_context for tok in ["request", "http", "client", "session"]):
        guess_bits.append(
            "Coordinates network or HTTP requests to external services. (**GUESS**)"
        )

    # Generic fallback if nothing else matched
    if not guess_bits:
        guess_bits.append(
            "Normal function; not enough context to infer a specific role. (**GUESS**)"
        )

    # ---------- Section 1: Integrated explanation (heuristic fallback) ----------
    lines.append("1) Integrated explanation (GUESS)")
    for g in guess_bits:
        lines.append(f"- {g}")
    lines.append("")

    # ---------- Section 2: Proven call-graph facts ----------
    lines.append("2) Proven call-graph facts")

    if call_edges_in:
        formatted_callers = ", ".join(
            f"`{_node_label(c)}` (edge:{c}->{target_id})" for c in sorted(call_edges_in)
        )
        lines.append(f"- **Callers**: {formatted_callers}")
    else:
        lines.append("- **Callers**: none in graph.")

    if call_edges_out:
        formatted_callees = ", ".join(
            f"`{_node_label(c)}` (edge:{target_id}->{c})" for c in sorted(call_edges_out)
        )
        lines.append(f"- **Callees**: {formatted_callees}")
    else:
        lines.append("- **Callees**: none in graph.")

    # Virtual dispatch facts
    if vc_resolved_targets:
        formatted_vc = ", ".join(f"`{t}` (node:{t})" for t in vc_resolved_targets[:20])
        lines.append(
            f"- **Virtual dispatch**: may target {formatted_vc} based on CALLS_VIRTUAL resolution."
        )
    elif unresolved_vc_ids:
        descs: List[str] = []
        for vc_id in unresolved_vc_ids[:10]:
            n = nodes_by_id.get(vc_id) or {}
            callee_attr = n.get("callee_attr")
            full_text = n.get("full_text")
            if isinstance(full_text, str) and full_text:
                descs.append(f"`{vc_id}` -> `{full_text}` (edge:{target_id}->{vc_id})")
            elif isinstance(callee_attr, str) and callee_attr:
                descs.append(f"`{vc_id}` attribute `{callee_attr}` (edge:{target_id}->{vc_id})")
            else:
                descs.append(f"`{vc_id}` (edge:{target_id}->{vc_id})")
        lines.append(f"- **Virtual dispatch**: unresolved sites {', '.join(descs)}")
    else:
        lines.append("- **Virtual dispatch**: none noted.")

    # Overrides / inheritance
    if ov_out:
        formatted_ov = ", ".join(f"`{_node_label(dst)}` (edge:{src}->{dst})" for src, dst in ov_out)
        lines.append(f"- **Overrides**: overrides {formatted_ov}")
    elif ov_in:
        formatted_ov = ", ".join(f"`{_node_label(src)}` (edge:{src}->{dst})" for src, dst in ov_in)
        lines.append(f"- **Overrides**: overridden by {formatted_ov}")
    else:
        lines.append("- **Overrides**: none.")
    lines.append("")

    # ---------- Section 3: External interactions & side effects ----------
    lines.append("3) External interactions & side effects")

    if external_nodes:
        by_cat: Dict[str, List[Dict[str, object]]] = {}
        for n in external_nodes:
            cat = n.get("category") or "unclassified"
            by_cat.setdefault(str(cat), []).append(n)

        if by_cat:
            parts: List[str] = []
            for cat, group in sorted(by_cat.items(), key=lambda kv: kv[0]):
                items: List[str] = []
                for n in group[:10]:
                    nid = n.get("id")
                    nm = n.get("name") or nid
                    ns = n.get("namespace")
                    if ns:
                        items.append(f"`{nm}` [ns={ns}] (node:{nid})")
                    else:
                        items.append(f"`{nm}` (node:{nid})")
                joined = ", ".join(items) if items else "none"
                parts.append(f"{cat}: {joined}")
            lines.append(f"- **Externals**: {'; '.join(parts)}")
    else:
        lines.append("- **Externals**: none in neighborhood.")

    if side_effect_nodes:
        side_effect_items: List[str] = []
        for n in side_effect_nodes[:20]:
            nid = n.get("id")
            nm = n.get("name") or nid
            cat = n.get("side_effect_category")
            conf = n.get("side_effect_confidence")
            side_effect_items.append(
                f"`{nm}` (node:{nid}, side_effect_category={cat}, confidence={conf})"
            )
        lines.append(f"- **Side effects (modelled)**: {', '.join(side_effect_items)}")
    else:
        mutators = [
            x
            for x in evidence.externals
            if isinstance(x, str) and re.search(r"\.(pop|append|remove|clear|update|setdefault)$", x)
        ]
        if mutators:
            mutator_text = ", ".join(f"`{m}` (node:{m})" for m in mutators[:20])
            lines.append(
                "- **Side effects (heuristic)**: uses common mutators (may imply state changes) "
                f"{mutator_text}"
            )
        else:
            lines.append("- **Side effects**: none provable from evidence.")

    # ---------- Paths (with VirtualCall annotations) ----------
    #
    # We KEEP vc:<id> in the path strings, but:
    # - If a VC has resolved targets, we rewrite it as:
    #       vc:<id>[callsite]→[TargetFunc1, TargetFunc2, …]
    # - If it has no internal targets, we mark it as:
    #       vc:<id>[callsite][unresolved]

    def _annotate_path(p: str) -> str:
        out_p = p

        # First, annotate resolved VCs with their INTERNAL targets
        for vc_id, vt in vc_with_internal.items():
            if vc_id not in out_p:
                continue
            target_ids = vt.get("targets") or []
            target_ids = [t for t in target_ids if isinstance(t, str)]
            if not target_ids:
                continue

            target_labels = [_node_label(tid) for tid in target_ids]
            if not target_labels:
                continue

            shown = ", ".join(target_labels[:3])
            if len(target_labels) > 3:
                shown += ", …"

            vc_label = vc_labels.get(vc_id)
            # Example: vc:a3a4db0bf2b3[self.pos_data.items]→[EquityDemoStrategy.on_bars, …]
            call_part = f"[{vc_label}]" if vc_label else ""
            replacement = f"{vc_id}{call_part}→[{shown}]"
            out_p = out_p.replace(vc_id, replacement)

        # Next, annotate VCs resolved to external targets
        for vc_id, ext_ids in vc_with_external.items():
            if vc_id not in out_p:
                continue
            labels = [_node_label(eid) for eid in ext_ids if isinstance(eid, str)]
            if not labels:
                continue

            shown = ", ".join(labels[:3])
            if len(labels) > 3:
                shown += ", …"

            vc_label = vc_labels.get(vc_id)
            call_part = f"[{vc_label}]" if vc_label else ""
            replacement = f"{vc_id}{call_part}→[{shown}]"
            out_p = out_p.replace(vc_id, replacement)

        # Then, mark any remaining VCs as unresolved, but still show the callsite name if we know it.
        for vc_id in unresolved_vc_ids:
            if vc_id and vc_id in out_p:
                vc_label = vc_labels.get(vc_id)
                call_part = f"[{vc_label}]" if vc_label else ""
                out_p = out_p.replace(vc_id, f"{vc_id}{call_part}[unresolved]")

        return out_p

    def _non_empty_paths(paths: List[str]) -> List[str]:
        return [p for p in paths if p.strip()]

    upstream_paths = _non_empty_paths(upstream_paths)
    downstream_paths = _non_empty_paths(downstream_paths)

    if upstream_paths:
        lines.append("- **Upstream paths**:")
        for p in upstream_paths:
            lines.append(f"  - `{_annotate_path(p)}`")

    if downstream_paths:
        lines.append("- **Downstream paths**:")
        for p in downstream_paths:
            lines.append(f"  - `{_annotate_path(p)}`")

    lines.append("")

    # ---------- Section 4: Code excerpt ----------
    def _trim_snippet(snippet_text: str, target_name: str) -> str:
        snippet_lines = snippet_text.splitlines()
        target_short = target_name.split(".")[-1]
        start_idx = 0
        for i, line in enumerate(snippet_lines):
            if re.match(rf"\s*def\s+{re.escape(target_short)}\b", line):
                start_idx = max(i - 1, 0)  # include potential decorator
                break
        trimmed = snippet_lines[start_idx:]
        end_idx = len(trimmed)
        for j, line in enumerate(trimmed[1:], start=1):
            if re.match(r"\s*(def |class )", line):
                end_idx = j
                break
        return "\n".join(trimmed[:end_idx]).strip()

    if isinstance(evidence.source_snippet, str) and evidence.source_snippet.strip():
        trimmed = _trim_snippet(evidence.source_snippet, qual_str or "")
        lines.append("4) Code excerpt")
        lines.append("```python")
        lines.append(trimmed or evidence.source_snippet.rstrip())
        lines.append("```")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# LLM helper: build Section 1 (Integrated explanation) from deterministic info
# ---------------------------------------------------------------------------


def _build_intent_from_deterministic(base: str, llm: Callable[[str], str]) -> str | None:
    """
    Use the deterministic report (sections 2–4) as context and ask the LLM
    to generate a *justified* Section 1 (Integrated explanation (GUESS)).

    We ignore the existing body of section 1 in `base` and treat it as replaceable.
    """
    lines = base.splitlines()

    # Find section anchors
    try:
        s1_idx = lines.index("1) Integrated explanation (GUESS)")
    except ValueError:
        s1_idx = None

    next_anchor = None
    for i, line in enumerate(lines):
        if line.strip() == "2) Proven call-graph facts":
            next_anchor = i
            break

    # Build context *without* the old section-1 body.
    if next_anchor is None or s1_idx is None or s1_idx > next_anchor:
        # Unexpected shape; fall back to giving the whole thing.
        context_text = base
    else:
        # Keep the heading line for context, plus everything from section 2 onwards.
        context_text = "\n".join(
            lines[: s1_idx + 1] +  # up to and including "1) Integrated explanation (GUESS)"
            lines[next_anchor:]    # start at "2) Proven call-graph facts"
        )

    prompt = "\n".join(
        [
            "You are AlgoTracer-Intent.",
            "",
            "You are given a structured, deterministic report about a Python function.",
            "It includes:",
            "- A heading with the function's location and signature.",
            "- Section 2: Proven call-graph facts (callers, callees, virtual dispatch, overrides).",
            "- Section 3: External interactions & side effects.",
            "- Section 4: A trimmed code excerpt.",
            "",
            "Your task:",
            "- Rewrite **Section 1) Integrated explanation (GUESS)** as a short report-style",
            "  explanation that ties these facts together.",
            "- Explicitly ground your explanation in the facts: when you mention who calls",
            "  this function, what it calls, externals it uses, or side effects, they MUST",
            "  correspond to entries present in Sections 2 or 3.",
            "- Do NOT invent new function/class/module names or new call relationships.",
            "- Do NOT introduce new edge:src->dst citations.",
            "- Do NOT contradict any stated facts; this section is an intuitive narrative",
            "  built strictly on the given facts.",
            "- End your final sentence with (**GUESS**).",
            "",
            "Style:",
            "- Either a short paragraph or 3–6 bullets is fine.",
            "- Make it read like a mini report: start from what the function is,",
            "  then how it is used (callers), what it does internally (callees / externals),",
            "  and what role it plays overall.",
            "",
            "Format:",
            "- Output ONLY the body of Section 1, WITHOUT the heading line",
            "  '1) Integrated explanation (GUESS)'.",
            "",
            "Here is the deterministic report (ignore any existing text under Section 1):",
            "<BEGIN_REPORT>",
            context_text,
            "<END_REPORT>",
        ]
    )

    try:
        text = llm(prompt).strip()
    except Exception as exc:
        print(f"[AlgoTracer][LLM] intent exception: {exc!r}")
        return None

    if not text:
        print("[AlgoTracer][LLM] intent returned empty output")
        return None

    # Light safety: don't allow raw edge: citations in the GUESS text.
    if "edge:" in text:
        print("[AlgoTracer][LLM] intent contains raw edge citations – rejecting.")
        return None

    return text


def explain(evidence: EvidencePack, llm: Optional[Callable[[str], str]] = None) -> str:
    """
    Final behavior:

    1. Build a deterministic report from graph evidence (sections 1–4).
    2. If no LLM is provided, return that directly.
    3. If LLM is provided, use it ONLY to regenerate Section 1
       (Integrated explanation (GUESS)), based on the deterministic report.
    4. If anything looks off, fall back to the deterministic report.
    """
    base = deterministic_summary(evidence)

    if llm is None:
        return base

    print("AlgoTracer: LLM enabled (integrated intent mode).")
    print("[AlgoTracer][LLM] building Section 1 (Integrated explanation)...")

    intent = _build_intent_from_deterministic(base, llm)
    if intent is None:
        print("[AlgoTracer][LLM] intent rejected or failed – using deterministic report")
        return base

    print("[AlgoTracer][LLM] intent accepted, stitching into output.")

    lines = base.splitlines()
    try:
        s1_idx = lines.index("1) Integrated explanation (GUESS)")
    except ValueError:
        # No section-1 header found; just append a new section at the end.
        return base + "\n\n1) Integrated explanation (GUESS)\n" + intent

    # Find where section 1 ends (start of section 2 or end of doc).
    end_idx = len(lines)
    for i in range(s1_idx + 1, len(lines)):
        if lines[i].startswith("2) "):
            end_idx = i
            break

    before = lines[: s1_idx + 1]   # keep the heading itself
    after = lines[end_idx:]        # everything from section 2 onwards

    new_lines: List[str] = []
    new_lines.extend(before)
    new_lines.extend(intent.splitlines())
    new_lines.append("")           # blank line after section-1 body
    new_lines.extend(after)

    return "\n".join(new_lines).strip()


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


# ---------- glue: Neighborhood -> EvidencePack with code snippet ----------


def _extract_source_snippet_from_nodes(
    target: Dict[str, object],
    nodes: List[Dict[str, object]],
    *,
    repo_root: Path | None = None,
    context_lines: int = 15,
) -> str | None:
    """
    Best-effort: find the file and grab a window of source around the target's lineno.
    Priority:
    1) Use Module.abs_path if present in the neighborhood.
    2) Fallback to repo_root / Function.path if repo_root is provided.
    """
    func_path = target.get("path")
    lineno = target.get("lineno")
    if not func_path or not isinstance(lineno, int):
        return None

    abs_path: str | None = None

    # Try to find a Module node with matching path and an abs_path
    for n in nodes:
        if "Module" in (n.get("labels") or []):
            if n.get("path") == func_path and n.get("abs_path"):
                abs_path = str(n["abs_path"])
                break

    # Fallback: repo_root + relative path
    if not abs_path and repo_root is not None:
        abs_path = str((repo_root / str(func_path)).resolve())

    if not abs_path:
        return None

    try:
        lines = Path(abs_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None

    start = max(0, lineno - 1 - context_lines)
    end = min(len(lines), lineno - 1 + context_lines)
    return "\n".join(lines[start:end])


def evidence_from_neighborhood(
    neighborhood: Neighborhood,
    *,
    repo_root: Path | None = None,
) -> EvidencePack:
    """
    Turn a Neighborhood into an EvidencePack the explainer understands,
    automatically attaching a source snippet.
    """
    target = neighborhood.target
    nodes = neighborhood.nodes
    edges = neighborhood.edges

    upstream = {
        "callers": neighborhood.callers,
        "paths": neighborhood.upstream_paths,
    }
    downstream = {
        "callees": neighborhood.callees,
        "paths": neighborhood.downstream_paths,
    }

    source_snippet = _extract_source_snippet_from_nodes(
        target,
        nodes,
        repo_root=repo_root,
    )

    # 🔍 DEBUG: check whether the snippet is empty
    preview = (source_snippet or "").strip()
    print(
        "[AlgoTracer] source_snippet empty? "
        f"{not bool(preview)} "
        f"for {target.get('qualname')} at {target.get('path')}:{target.get('lineno')}"
    )
    if preview:
        print("[AlgoTracer] source_snippet preview:\n", preview[:200])

    return EvidencePack(
        target=target,
        upstream=upstream,
        downstream=downstream,
        externals=neighborhood.externals,
        edges=edges,
        nodes=nodes,
        virtual_targets=neighborhood.virtual_targets,
        source_snippet=source_snippet,
    )
