from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings


def build_embeddings(*, model: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model)
