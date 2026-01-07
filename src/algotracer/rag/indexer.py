from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from algotracer.ingest.ast_parser import ClassInfo, FunctionInfo, ModuleInfo
from algotracer.rag.config import RagConfig
from algotracer.rag.embeddings import build_embeddings


@dataclass(frozen=True)
class SymbolChunk:
    kind: str
    qualname: str
    path: str
    lineno: int
    end_lineno: int | None
    content: str
    docstring: str | None


def build_rag_index(modules: Sequence[ModuleInfo], *, config: RagConfig) -> None:
    documents = collect_documents(modules, config=config)
    if not documents:
        print("AlgoTracer[RAG]: no documents found to index.")
        return

    embeddings = build_embeddings(model=config.embedding_model)
    store = FAISS.from_documents(documents, embeddings)
    config.index_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(config.index_path))
    print(f"AlgoTracer[RAG]: indexed {len(documents)} chunks into {config.index_path}")


def collect_documents(modules: Sequence[ModuleInfo], *, config: RagConfig) -> List[Document]:
    docs: List[Document] = []
    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
    )
    doc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    code_docs = list(_documents_from_ast(modules, config=config))
    for d in code_docs:
        docs.extend(_split_document(d, splitter=code_splitter))

    if config.include_docs:
        docs.extend(_documents_from_markdown(config, splitter=doc_splitter))

    return docs


def _documents_from_ast(modules: Sequence[ModuleInfo], *, config: RagConfig) -> Iterator[Document]:
    """
    Build symbol-level Documents from AST.

    Key best-practice upgrade:
    - Embed *identity* + *docstring* + *code* in page_content
      so symbol-name queries like "on_trade" retrieve the right chunks.
    """
    file_cache: Dict[Path, List[str]] = {}

    for module in modules:
        module_path = module.path
        lines = file_cache.get(module_path)
        if lines is None:
            try:
                lines = module_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            file_cache[module_path] = lines

        rel_path = _relative_path(module_path, config.repo_root)

        for symbol in _iter_symbols(module):
            code = _slice_source(
                lines,
                lineno=symbol.lineno,
                end_lineno=symbol.end_lineno,
                fallback_lines=config.code_fallback_context_lines,
            )
            if not code:
                continue

            stable_id = _stable_id(rel_path, symbol.qualname)

            # --- NEW: Add a consistent "header" that gets embedded with the code.
            # This makes symbol-name queries (e.g., "on_trade") much more reliable.
            header_lines: List[str] = [
                f"SYMBOL: {symbol.qualname}",
                f"KIND: {symbol.kind}",
                f"PATH: {rel_path}:{symbol.lineno}",
                f"STABLE_ID: {stable_id}",
            ]

            docstring = (symbol.docstring or "").strip()
            if docstring:
                header_lines.append("DOCSTRING:")
                header_lines.append(docstring)

            header = "\n".join(header_lines).rstrip()

            page_content = f"{header}\n\nCODE:\n{code}".rstrip()

            metadata = {
                "stable_id": stable_id,
                "kind": symbol.kind,
                "qualname": symbol.qualname,
                "path": rel_path,
                "lineno": symbol.lineno,
                "end_lineno": symbol.end_lineno,
            }
            yield Document(page_content=page_content, metadata=metadata)


def _documents_from_markdown(config: RagConfig, *, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    docs: List[Document] = []
    for path in _iter_markdown_files(config):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        base_doc = Document(
            page_content=text,
            metadata={"kind": "doc", "path": _relative_path(path, config.repo_root)},
        )
        docs.extend(_split_document(base_doc, splitter=splitter))
    return docs


def _iter_symbols(module: ModuleInfo) -> Iterator[SymbolChunk]:
    for cls in module.classes:
        yield SymbolChunk(
            kind="class",
            qualname=cls.name,
            path=str(module.path),
            lineno=cls.lineno,
            end_lineno=cls.end_lineno,
            content="",
            docstring=cls.docstring,
        )
        for method in cls.methods:
            yield SymbolChunk(
                kind="function",
                qualname=f"{cls.name}.{method.name}",
                path=str(module.path),
                lineno=method.lineno,
                end_lineno=method.end_lineno,
                content="",
                docstring=method.docstring,
            )

    for func in module.functions:
        yield SymbolChunk(
            kind="function",
            qualname=func.name,
            path=str(module.path),
            lineno=func.lineno,
            end_lineno=func.end_lineno,
            content="",
            docstring=func.docstring,
        )


def _split_document(doc: Document, *, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    chunks = splitter.split_text(doc.page_content)
    out: List[Document] = []
    for i, chunk in enumerate(chunks):
        metadata = dict(doc.metadata)
        metadata["chunk_index"] = i
        out.append(Document(page_content=chunk, metadata=metadata))
    return out


def _slice_source(lines: Sequence[str], *, lineno: int, end_lineno: int | None, fallback_lines: int) -> str:
    start = max(lineno - 1, 0)
    if end_lineno and end_lineno > 0:
        end = min(end_lineno, len(lines))
    else:
        end = min(start + fallback_lines, len(lines))
    snippet = "\n".join(lines[start:end]).rstrip()
    return snippet


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _stable_id(rel_path: str, qualname: str) -> str:
    return f"stable:sym:{rel_path}:{qualname}"


def _iter_markdown_files(config: RagConfig) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in config.docs_globs:
        for path in config.repo_root.glob(pattern):
            if path in seen:
                continue
            if _should_skip_path(path):
                continue
            seen.add(path)
            yield path


def _should_skip_path(path: Path) -> bool:
    parts = set(path.parts)
    if ".git" in parts or ".venv" in parts or "build" in parts or "dist" in parts:
        return True
    if ".algotracer_rag" in parts or ".algotracer_notebooks" in parts:
        return True
    return False
