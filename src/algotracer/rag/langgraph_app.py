from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, NotRequired

from dotenv import load_dotenv
from langsmith import traceable

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, MessagesState, StateGraph

from algotracer.rag.config import RagConfig
from algotracer.rag.retriever import retrieve_with_scores
from algotracer.memgraph.client import MemgraphConfig, connect_memgraph
from algotracer.graph.neighborhood import NeighborhoodConfig, fetch_neighborhood
from algotracer.graph.resolver import resolve_by_id

load_dotenv()

# =============================================================================
# State
# =============================================================================


class AlgoState(MessagesState):
    retrieved_context: NotRequired[str]
    chunks: NotRequired[List[dict]]
    scores: NotRequired[List[float]]  # FAISS distance (lower is better)

    graph_context: NotRequired[str]
    assumption: NotRequired[str]
    target_id: NotRequired[str]
    target_qualname: NotRequired[str]
    repo_id: NotRequired[str]
    selection_reason: NotRequired[dict]


# =============================================================================
# State helpers (handles LangGraph Response objects safely)
# =============================================================================


def _state_get(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _state_messages(state: Any) -> List[BaseMessage]:
    return _state_get(state, "messages", []) or []


# =============================================================================
# General helpers
# =============================================================================


def _latest_user_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _config_from_env() -> RagConfig:
    repo_root = Path(os.getenv("ALGOTRACER_REPO_ROOT", ".")).resolve()
    index_path = Path(os.getenv("ALGOTRACER_RAG_INDEX", repo_root / ".algotracer_rag"))
    embedding_model = os.getenv("ALGOTRACER_RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k = int(os.getenv("ALGOTRACER_RAG_TOP_K", "8"))
    return RagConfig(
        repo_root=repo_root,
        index_path=index_path,
        embedding_model=embedding_model,
        top_k=top_k,
    )


def _repo_id_from_env(config: RagConfig) -> str:
    return os.getenv("ALGOTRACER_REPO_ID", config.repo_root.name)


def _doc_metadata(doc: Document) -> dict:
    meta = dict(doc.metadata or {})
    meta["preview"] = doc.page_content[:200]
    return meta


def _format_context(docs: List[Document]) -> str:
    out: List[str] = []
    for d in docs:
        meta = d.metadata or {}
        label = meta.get("qualname") or meta.get("path") or "chunk"
        out.append(f"[{label}]\n{d.page_content}")
    return "\n\n".join(out)


# =============================================================================
# Graph formatting
# =============================================================================


def _format_graph_evidence(neighborhood) -> str:
    nodes = {n["id"]: n for n in neighborhood.nodes}

    def lbl(i: str) -> str:
        n = nodes.get(i, {})
        return n.get("qualname") or n.get("name") or i

    tgt = neighborhood.target
    tid = tgt["id"]
    tlabel = tgt.get("qualname") or tgt.get("name") or tid

    edge_types = {(e["src"], e["dst"]): e["type"] for e in neighborhood.edges}

    callers = [lbl(c) for c in neighborhood.callers if edge_types.get((c, tid)) == "CALLS"]
    callees = [lbl(c) for c in neighborhood.callees if edge_types.get((tid, c)) == "CALLS"]

    virtuals: list[str] = []
    for vt in neighborhood.virtual_targets:
        vc_id = vt.get("virtual_call_id")
        targets = vt.get("targets") or []
        if vc_id and targets:
            shown = ", ".join(lbl(t) for t in targets[:5])
            virtuals.append(f"{vc_id} -> [{shown}]")

    overrides_out = [
        f"{lbl(dst)} (edge:{src}->{dst})"
        for (src, dst), typ in edge_types.items()
        if typ == "OVERRIDES" and src == tid
    ]
    overrides_in = [
        f"{lbl(src)} (edge:{src}->{dst})"
        for (src, dst), typ in edge_types.items()
        if typ == "OVERRIDES" and dst == tid
    ]

    subclasses = [
        f"{lbl(src)} -> {lbl(dst)}"
        for (src, dst), typ in edge_types.items()
        if typ == "SUBCLASS_OF"
    ]

    return "\n".join(
        [
            f"Target: {tlabel} ({tid})",
            f"Callers: {', '.join(callers) if callers else 'none'}",
            f"Callees: {', '.join(callees) if callees else 'none'}",
            f"Virtual dispatch: {', '.join(virtuals) if virtuals else 'none'}",
            f"Overrides (outgoing): {', '.join(overrides_out) if overrides_out else 'none'}",
            f"Overrides (incoming): {', '.join(overrides_in) if overrides_in else 'none'}",
            f"Inheritance (subclass_of): {', '.join(subclasses) if subclasses else 'none'}",
        ]
    )


# =============================================================================
# Target ranking (lexical + distance, plus reason payload)
# =============================================================================


def _desired_symbol(question: str) -> Optional[str]:
    m = re.search(r"`([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)`", question)
    if m:
        return m.group(1)
    # fallback: a bare token (useful for "what does on_trade do")
    m = re.search(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\b", question)
    return m.group(1) if m else None


def _pick_best_target(question: str, chunks: List[dict], distances: List[float], mg, repo_id: str):
    want = (_desired_symbol(question) or "").lower()
    ql = question.lower()

    scored = []

    for i, ch in enumerate(chunks):
        sid = ch.get("stable_id")
        if not isinstance(sid, str):
            continue

        t = resolve_by_id(mg, repo_id, sid)
        if not t:
            continue

        qn = (t.qualname or "").lower()
        name = qn.split(".")[-1] if qn else ""
        dist = float(distances[i]) if i < len(distances) else 1e9

        lex = 0.0
        if want:
            if name == want or qn.endswith("." + want):
                lex += 10.0
            elif want in qn:
                lex += 4.0

        if name and name in ql:
            lex += 2.0

        score = lex - dist  # smaller distance => better

        scored.append(
            (
                score,
                t,
                {
                    "score": score,
                    "lex": lex,
                    "dist": dist,
                    "qualname": t.qualname,
                    "stable_id": sid,
                },
            )
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    reason = {"want": want or None, "candidates": [s[2] for s in scored[:8]]}
    return (scored[0][1], reason) if scored else (None, reason)


# =============================================================================
# Intermediate steps formatter (NO <details>; clean separation)
# =============================================================================


def _format_intermediate_steps(
    *,
    chunks: List[dict],
    distances: List[float],
    target_qualname: Optional[str],
    target_id: Optional[str],
    selection_reason: Optional[dict],
    graph_context: str,
) -> str:
    lines: List[str] = []

    lines.append("## 🔍 Intermediate reasoning (RAG + Graph grounding)")
    lines.append(
        "_This section shows the retrieval, target selection, and graph evidence used to derive the final answer._"
    )
    lines.append("")

    # Retrieval
    lines.append("### Retrieval (top semantic matches)")
    if not chunks:
        lines.append("- (no chunks retrieved)")
    else:
        for i, ch in enumerate(chunks[:8]):
            label = ch.get("qualname") or ch.get("path") or "chunk"
            dist = distances[i] if i < len(distances) else None
            dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else "n/a"
            lines.append(f"- {i+1}. dist={dist_str} · **{label}**")

    # Target selection
    lines.append("")
    lines.append("### Target selection")
    if target_qualname and target_id:
        lines.append(f"- Selected target: **`{target_qualname}`**")
        lines.append(f"- Stable ID: `{target_id}`")
    else:
        lines.append("- Selected target: (none)")

    if selection_reason:
        want = selection_reason.get("want")
        lines.append(f"- Symbol hint from question: `{want}`" if want else "- Symbol hint from question: (none)")
        cands = selection_reason.get("candidates", [])
        if cands:
            lines.append("- Candidate ranking:")
            for c in cands[:5]:
                lines.append(
                    f"  - score={c['score']:.2f} (lex={c['lex']:.1f}, dist={c['dist']:.3f}) · `{c['qualname']}`"
                )

    # Graph evidence
    lines.append("")
    lines.append("### Graph evidence (deterministic)")
    if graph_context.strip():
        lines.append("```text")
        lines.append(graph_context.strip())
        lines.append("```")
    else:
        lines.append("- (no graph context)")

    return "\n".join(lines)


def _compose_final_message(*, final_answer: str, show_steps: bool, steps_block: str) -> str:
    if not show_steps:
        return final_answer

    return "\n".join(
        [
            steps_block,
            "",
            "---",
            "",
            "## ✅ Final Answer",
            "",
            final_answer,
        ]
    )


# =============================================================================
# Nodes
# =============================================================================


@traceable(name="rag_retrieve", run_type="tool")
def retrieve(state: AlgoState) -> Dict[str, Any]:
    q = _latest_user_text(_state_messages(state)).strip()
    if not q:
        return {}

    cfg = _config_from_env()
    results = retrieve_with_scores(
        index_path=cfg.index_path,
        embedding_model=cfg.embedding_model,
        query=q,
        top_k=cfg.top_k,
    )

    docs = [d for d, _ in results]
    distances = [float(s) for _, s in results]

    return {
        "retrieved_context": _format_context(docs),
        "chunks": [_doc_metadata(d) for d in docs],
        "scores": distances,
        "repo_id": _repo_id_from_env(cfg),
    }


@traceable(name="graph_select_target", run_type="tool")
def select_target(state: AlgoState) -> Dict[str, Any]:
    chunks = _state_get(state, "chunks", []) or []
    scores = _state_get(state, "scores", []) or []
    repo_id = _state_get(state, "repo_id")
    q = _latest_user_text(_state_messages(state)).strip()

    mg = connect_memgraph(MemgraphConfig.from_env())

    target, reason = _pick_best_target(q, chunks, scores, mg, repo_id)
    if not target:
        return {"assumption": "No graph target resolved.", "selection_reason": reason}

    nb = fetch_neighborhood(
        mg=mg,
        repo_id=repo_id,
        func_id=target.id,
        config=NeighborhoodConfig(),
    )

    return {
        "target_id": target.id,
        "target_qualname": target.qualname,
        "graph_context": _format_graph_evidence(nb),
        "selection_reason": reason,
    }


@traceable(name="rag_answer", run_type="chain")
def answer(state: AlgoState) -> Dict[str, Any]:
    q = _latest_user_text(_state_messages(state)).strip()
    if not q:
        return {"messages": [AIMessage(content="Ask a question to start.")]}

    llm = ChatOpenAI(
        model=os.getenv("ALGOTRACER_CHAT_MODEL", "gemini-2.5-flash"),
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url=os.getenv("ALGOTRACER_OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        temperature=0.2,
    )

    graph_context = _state_get(state, "graph_context", "") or ""
    retrieved_context = _state_get(state, "retrieved_context", "") or ""

    prompt = f"""
You are AlgoTracer.

Explain what the target function does using:
- Graph Evidence (who calls whom) for STRUCTURAL FACTS
- Retrieved Context (code semantics) for BEHAVIORAL INTERPRETATION

Structure:
1. What this function is
2. When it runs / what triggers it
3. Step-by-step execution flow
4. Decision logic & conditions
5. Side effects

IMPORTANT:
- If the graph evidence says "Callees: none", do not invent callees.
- If a behavior claim is based on semantics rather than graph, label it as interpretation.

Question:
{q}

Graph Evidence:
{graph_context}

Retrieved Context:
{retrieved_context}
""".strip()

    resp = llm.invoke(prompt)
    final_answer = resp.content if hasattr(resp, "content") else str(resp)

    show_steps = os.getenv("ALGOTRACER_SHOW_STEPS", "false").lower() in {"1", "true", "yes"}

    steps_block = _format_intermediate_steps(
        chunks=_state_get(state, "chunks", []) or [],
        distances=_state_get(state, "scores", []) or [],
        target_qualname=_state_get(state, "target_qualname"),
        target_id=_state_get(state, "target_id"),
        selection_reason=_state_get(state, "selection_reason"),
        graph_context=graph_context,
    )

    text = _compose_final_message(final_answer=final_answer, show_steps=show_steps, steps_block=steps_block)

    return {"messages": [AIMessage(content=text)]}


# =============================================================================
# Graph
# =============================================================================


def build_graph():
    g = StateGraph(AlgoState)
    g.add_node("retrieve", retrieve)
    g.add_node("select_target", select_target)
    g.add_node("answer", answer)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "select_target")
    g.add_edge("select_target", "answer")
    g.add_edge("answer", END)
    return g.compile()


graph = build_graph()
