from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RagConfig:
    repo_root: Path
    index_path: Path
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    include_docs: bool = True
    docs_globs: Iterable[str] = field(default_factory=lambda: ("README.md", "docs/**/*.md", "*.md"))
    code_fallback_context_lines: int = 25
    top_k: int = 8

    @classmethod
    def with_defaults(cls, repo_root: Path, *, index_path: Path | None = None) -> "RagConfig":
        base = index_path or (repo_root / ".algotracer_rag")
        return cls(repo_root=repo_root, index_path=base)
