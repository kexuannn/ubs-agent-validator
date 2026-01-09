from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Literal, Optional
import hashlib

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from algotracer.rag.embeddings import build_embeddings


def load_faiss_store(
    *,
    index_path: Path,
    embedding_model: str,
) -> FAISS:
    """
    Load a persisted FAISS vector store.
    """
    embeddings = build_embeddings(model=embedding_model)
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_faiss_retriever(
    *,
    index_path: Path,
    embedding_model: str,
    top_k: int,
    mode: Literal["similarity", "mmr"] = "similarity",
    fetch_k: Optional[int] = None,
    lambda_mult: float = 0.5,
):
    """
    Build a LangChain retriever on top of FAISS.

    mode="similarity":
        Standard top-k nearest neighbors.
    mode="mmr":
        Maximal Marginal Relevance. Retrieves a larger candidate pool (fetch_k)
        then selects a diverse top_k set.
    """
    store = load_faiss_store(index_path=index_path, embedding_model=embedding_model)

    if mode == "mmr":
        fk = fetch_k if fetch_k is not None else max(top_k * 4, 20)
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": fk, "lambda_mult": lambda_mult},
        )

    return store.as_retriever(search_kwargs={"k": top_k})


def retrieve_documents(
    *,
    index_path: Path,
    embedding_model: str,
    query: str,
    top_k: int,
    mode: Literal["similarity", "mmr"] = "similarity",
) -> List[Document]:
    """
    Retrieve documents only (no scores). Kept for backward compatibility.
    """
    retriever = load_faiss_retriever(
        index_path=index_path,
        embedding_model=embedding_model,
        top_k=top_k,
        mode=mode,
    )
    return list(retriever.invoke(query))


def retrieve_with_scores(
    *,
    index_path: Path,
    embedding_model: str,
    query: str,
    top_k: int,
) -> List[Tuple[Document, float]]:
    """
    Retrieve (Document, score) pairs.

    Fixes:
    - Overfetch from FAISS then prefer symbol-backed chunks (stable_id/qualname).
    - Choose up to `top_k` UNIQUE SYMBOLS (stable_id), not just chunks.
    - Expand each chosen symbol to ALL of its chunks (sorted by chunk_index),
      so long functions come back complete.
    """
    store = load_faiss_store(index_path=index_path, embedding_model=embedding_model)

    # Overfetch (does NOT change external top_k)
    fetch_k = max(top_k * 8, 80)
    raw = store.similarity_search_with_score(query, k=fetch_k)

    def is_symbol(doc: Document) -> bool:
        meta = doc.metadata or {}
        return bool(meta.get("stable_id")) or bool(meta.get("qualname"))

    def content_hash(doc: Document) -> str:
        return hashlib.md5((doc.page_content or "").encode("utf-8", errors="ignore")).hexdigest()

    # --- 1) Pick up to top_k unique stable_ids (symbols), keep best score per symbol
    picked_symbol_best_score: dict[str, float] = {}
    fallback_docs: List[Tuple[Document, float]] = []  # if nothing has stable_id

    for doc, score in raw:
        meta = doc.metadata or {}
        sid = meta.get("stable_id")
        if sid and is_symbol(doc):
            if sid not in picked_symbol_best_score:
                picked_symbol_best_score[sid] = float(score)
                if len(picked_symbol_best_score) >= top_k:
                    break
        else:
            fallback_docs.append((doc, float(score)))

    # If we somehow found no symbol-backed docs at all, just return best unique docs.
    if not picked_symbol_best_score:
        out: List[Tuple[Document, float]] = []
        seen = set()
        for doc, score in raw:
            k = (meta := (doc.metadata or {}))
            # try to dedupe with metadata; fallback to content hash
            dk = (
                meta.get("stable_id"),
                meta.get("chunk_index"),
                meta.get("path"),
                meta.get("lineno"),
            )
            if all(v is None for v in dk):
                dk = ("__no_meta__", content_hash(doc))
            if dk in seen:
                continue
            seen.add(dk)
            out.append((doc, float(score)))
            if len(out) >= top_k:
                break
        return out

    # --- 2) Expand each picked symbol to ALL chunks from the docstore
    docs_dict = getattr(getattr(store, "docstore", None), "_dict", None) or {}
    expanded: List[Tuple[Document, float]] = []

    for sid, best_score in picked_symbol_best_score.items():
        symbol_docs: List[Document] = []
        for d in docs_dict.values():
            m = d.metadata or {}
            if m.get("stable_id") == sid:
                symbol_docs.append(d)

        # Return symbol chunks in order
        symbol_docs.sort(key=lambda d: int((d.metadata or {}).get("chunk_index") or 0))

        for d in symbol_docs:
            expanded.append((d, best_score))

    # Keep output deterministic: best symbols first, then chunk order
    expanded.sort(
        key=lambda pair: (
            pair[1],  # lower distance first
            (pair[0].metadata or {}).get("stable_id") or "",
            int((pair[0].metadata or {}).get("chunk_index") or 0),
        )
    )

    return expanded
