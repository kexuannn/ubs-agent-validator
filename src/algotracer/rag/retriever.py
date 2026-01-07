from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Literal, Optional

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

    Note: LangChain's FAISS similarity_search_with_score typically returns an L2
    distance (lower is better) unless you have configured a different distance strategy.
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
        then selects a diverse top_k set. This is useful when many chunks are
        near-duplicates (e.g., lots of similar "trade" code paths).
    """
    store = load_faiss_store(index_path=index_path, embedding_model=embedding_model)

    if mode == "mmr":
        # Pull more candidates, then diversify down to k
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

    For LangChain FAISS, the returned `score` is typically an L2 distance:
    - lower = more similar
    - higher = less similar
    """
    store = load_faiss_store(index_path=index_path, embedding_model=embedding_model)
    results = store.similarity_search_with_score(query, k=top_k)
    return [(doc, float(score)) for doc, score in results]
