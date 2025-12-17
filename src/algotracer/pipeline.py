from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from algotracer.ingest.ast_parser import ModuleInfo, parse_python_files
from algotracer.analysis.deps import (
    DependencyGraph,
    build_dependency_graph,
    summarize_callers,
)
from algotracer.analysis.entrypoints import (
    EntryPoint,
    EntryPointRules,
    find_entrypoints,
    find_entrypoints_auto,
)
from algotracer.analysis.trace import TraceSummary, trace_and_summarize
from algotracer.reporting.renderer import MarkdownReport, render_markdown_report
from algotracer.reasoning.flow_explainer import build_gemini_llm


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

@dataclass
class PipelineConfig:
    sources: Sequence[Path]
    report_dir: Path

    # Entrypoint detection
    entrypoint_names: Sequence[str] | None = ("fit", "predict", "transform")
    auto_entrypoints: bool = False

    # Rules-based tuning
    add_sklearn_defaults: bool = False
    base_requires_name_match: bool = True

    # Source selection
    include_tests: bool = False

    # Tracing controls
    max_depth: int = 3
    max_paths: int | None = 2000
    max_expansions: int | None = 20000
    max_examples: int = 10

    # How many entrypoints to trace
    top_k_entrypoints: int = 1


# -------------------------------------------------------------------
# Artifacts
# -------------------------------------------------------------------

@dataclass
class PipelineArtifacts:
    modules: List[ModuleInfo] = field(default_factory=list)
    dependency_graph: DependencyGraph | None = None

    entrypoints: List[EntryPoint] = field(default_factory=list)
    traces: Dict[str, TraceSummary] = field(default_factory=dict)
    callers: Dict[str, List[str]] = field(default_factory=dict)

    report: MarkdownReport | None = None


# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

class AlgoTracerPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.artifacts = PipelineArtifacts()
        self._llm = None
        self._notebook_cache = Path(".algotracer_notebooks")

    # ------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------

    def run(self) -> PipelineArtifacts:
        python_files = list(self._iter_source_files())
        print(f"AlgoTracer: collected {len(python_files)} Python files.")

        print("AlgoTracer: parsing modules...")
        self.artifacts.modules = parse_python_files(python_files)
        print(f"AlgoTracer: parsed {len(self.artifacts.modules)} modules.")

        print("AlgoTracer: building dependency graph...")
        graph = build_dependency_graph(self.artifacts.modules)
        self.artifacts.dependency_graph = graph
        print(f"AlgoTracer: graph nodes={len(graph.nodes)}, edges={sum(len(v) for v in graph.edges.values())}.")

        # Entrypoints
        if self.config.auto_entrypoints or not self.config.entrypoint_names:
            print("AlgoTracer: detecting entrypoints (auto mode)...")
            self.artifacts.entrypoints = find_entrypoints_auto(
                self.artifacts.modules,
                graph,
                top_k=max(self.config.top_k_entrypoints, 10),
            )
        else:
            print("AlgoTracer: detecting entrypoints (rules mode)...")
            rules = EntryPointRules(
                base_requires_name_match=self.config.base_requires_name_match
            )
            self.artifacts.entrypoints = find_entrypoints(
                self.artifacts.modules,
                names=set(self.config.entrypoint_names),
                rules=rules,
                add_sklearn_defaults=self.config.add_sklearn_defaults,
            )

        if not self.artifacts.entrypoints:
            raise RuntimeError("No entrypoints detected.")

        print(f"AlgoTracer: detected {len(self.artifacts.entrypoints)} entrypoints.")

        selected = self.artifacts.entrypoints[: self.config.top_k_entrypoints]
        print(f"AlgoTracer: tracing {len(selected)} entrypoints (max_depth={self.config.max_depth}).")

        self.artifacts.traces = trace_and_summarize(
            graph,
            selected,
            max_depth=self.config.max_depth,
            max_paths=self.config.max_paths,
            max_expansions=self.config.max_expansions,
            max_examples=self.config.max_examples,
        )

        print("AlgoTracer: collecting callers...")
        self.artifacts.callers = {
            ep.sym_id: summarize_callers(graph, ep.sym_id)
            for ep in selected
        }

        self.config.report_dir.mkdir(parents=True, exist_ok=True)
        print(f"AlgoTracer: rendering report to {self.config.report_dir} ...")

        self.artifacts.report = render_markdown_report(
            modules=self.artifacts.modules,
            graph=graph,
            entrypoints=self.artifacts.entrypoints,
            traces=self.artifacts.traces,
            callers=self.artifacts.callers,
            output_dir=self.config.report_dir,
            llm=self._get_llm(),
        )

        return self.artifacts

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _include_path(self, path: Path) -> bool:
        if not self.config.include_tests and "tests" in path.parts:
            return False
        if self._notebook_cache in path.parents:
            return False
        return True

    def _iter_source_files(self) -> Iterable[Path]:
        for path in self.config.sources:
            if path.is_dir():
                for p in path.rglob("*.py"):
                    if self._include_path(p):
                        yield p
                for nb in path.rglob("*.ipynb"):
                    if self._include_path(nb):
                        converted = self._convert_notebook(nb)
                        if converted:
                            yield converted
            elif path.suffix == ".py":
                if self._include_path(path):
                    yield path
            elif path.suffix == ".ipynb":
                converted = self._convert_notebook(path)
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

            cleaned = []
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

        self._notebook_cache.mkdir(parents=True, exist_ok=True)

        h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
        dest = self._notebook_cache / f"{path.stem}.{h}.py"
        dest.write_text("\n\n".join(code_blocks), encoding="utf-8")

        print(f"AlgoTracer: converted notebook {path} -> {dest}")
        return dest

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        try:
            self._llm = build_gemini_llm()
            print("AlgoTracer: LLM enabled via Gemini.")
        except Exception:
            self._llm = None
            print("AlgoTracer: LLM unavailable; using heuristic reasoning.")
        return self._llm
