from __future__ import annotations

import json
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, List

from algotracer.graph.builder import GraphBuildConfig, write_graph
from algotracer.graph.neighborhood import NeighborhoodConfig, fetch_neighborhood
from algotracer.graph.resolver import (
    disambiguate,
    normalize_path,
    resolve_by_id,
    resolve_by_name,
    resolve_by_path_and_name,
)
from algotracer.ingest.ast_parser import parse_python_files
from algotracer.memgraph.client import MemgraphConfig, clear_repo, connect_memgraph, ensure_schema
from algotracer.reasoning.explainer import build_gemini_llm, evidence_from_neighborhood, explain
from algotracer.graph.cypher import generate_cypher


@dataclass(frozen=True)
class AnalyzeConfig:
    repo_path: Path
    repo_id: str
    include_tests: bool = False


@dataclass(frozen=True)
class ExplainConfig:
    repo_path: Path
    repo_id: str
    function_id: str | None = None
    function_name: str | None = None
    function_file: Path | None = None
    depth_up: int = 2
    depth_down: int = 2
    max_nodes: int = 200
    max_edges: int = 400
    max_paths: int = 200
    debug_subgraph_path: Path | None = None
    use_llm: bool = True
    output_path: Path | None = None


class AlgoTracerPipeline:
    def __init__(self, memgraph_config: MemgraphConfig):
        self.memgraph_config = memgraph_config
        self._notebook_cache: Path | None = None

    def analyze(self, config: AnalyzeConfig) -> None:
        repo_path = config.repo_path
        self._notebook_cache = repo_path / ".algotracer_notebooks"
        python_files = list(self._iter_python_files(repo_path, include_tests=config.include_tests))
        print(f"AlgoTracer: collected {len(python_files)} Python files.")

        modules = parse_python_files(python_files)
        print(f"AlgoTracer: parsed {len(modules)} modules.")

        mg = connect_memgraph(self.memgraph_config)
        ensure_schema(mg)
        clear_repo(mg, config.repo_id)

        write_graph(
            mg=mg,
            modules=modules,
            config=GraphBuildConfig(repo_id=config.repo_id, repo_root=repo_path),
        )

        print(f"AlgoTracer: graph written to Memgraph for repo_id={config.repo_id}.")

    def explain(self, config: ExplainConfig) -> str:
        mg = connect_memgraph(self.memgraph_config)

        target = None
        if config.function_id:
            target = resolve_by_id(mg, config.repo_id, config.function_id)
        elif config.function_file and config.function_name:
            stable_path = normalize_path(config.function_file, config.repo_path)
            target = resolve_by_path_and_name(mg, config.repo_id, stable_path, config.function_name)
        elif config.function_name:
            candidates = resolve_by_name(mg, config.repo_id, config.function_name)
            target = disambiguate(candidates)

        if target is None:
            raise RuntimeError("Unable to resolve target function. Provide --id or --file/--name.")

        neighborhood = fetch_neighborhood(
            mg=mg,
            repo_id=config.repo_id,
            func_id=target.id,
            config=NeighborhoodConfig(
                depth_up=config.depth_up,
                depth_down=config.depth_down,
                max_nodes=config.max_nodes,
                max_edges=config.max_edges,
                max_paths=config.max_paths,
            ),
        )

        evidence = evidence_from_neighborhood(
            neighborhood,
            repo_root=config.repo_path,
        )

        if config.debug_subgraph_path is not None:
            payload = {"nodes": neighborhood.nodes, "edges": neighborhood.edges}
            config.debug_subgraph_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        llm = None
        if config.use_llm:
            try:
                llm = build_gemini_llm()
                print("AlgoTracer: LLM enabled (Gemini).")
            except Exception as exc:
                print(f"AlgoTracer: LLM unavailable ({type(exc).__name__}: {exc}); using deterministic summary.")

        result = explain(evidence, llm=llm)

        output_path = config.output_path or (config.repo_path / "reports" / "explanation.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result, encoding="utf-8")

        # Export neighborhood as Cypher alongside the report
        generate_cypher(neighborhood, output_path.parent)

        return result

    def _iter_python_files(self, repo_path: Path, *, include_tests: bool) -> Iterable[Path]:
        for path in repo_path.rglob("*.py"):
            if not include_tests and "tests" in path.parts:
                continue
            if self._notebook_cache in path.parents:
                continue
            yield path

        for nb in repo_path.rglob("*.ipynb"):
            if not include_tests and "tests" in nb.parts:
                continue
            converted = self._convert_notebook(nb)
            if converted:
                yield converted

    def _convert_notebook(self, path: Path) -> Path | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"AlgoTracer: failed to read notebook {path}: {e}")
            return None

        code_blocks: List[str] = []
        for cell in data.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source", [])
            if isinstance(src, str):
                src = [src]
            cleaned: List[str] = []
            for line in src:
                stripped = line.lstrip()
                if stripped.startswith("%") or stripped.startswith("!") or stripped.startswith("%%"):
                    cleaned.append(f"# NOTE: skipped notebook magic: {line}")
                else:
                    cleaned.append(line)
            code_blocks.append("".join(cleaned))

        if not code_blocks:
            print(f"AlgoTracer: notebook {path} has no usable code cells; skipping.")
            return None

        if self._notebook_cache is None:
            self._notebook_cache = Path(".algotracer_notebooks")
        self._notebook_cache.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
        dest = self._notebook_cache / f"{path.stem}.{h}.py"
        dest.write_text("\n\n".join(code_blocks), encoding="utf-8")
        print(f"AlgoTracer: converted notebook {path} -> {dest}")
        return dest
