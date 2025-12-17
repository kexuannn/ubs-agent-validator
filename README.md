# AlgoTracer

AlgoTracer is an agent-based static + structural analyzer for Python projects. It ingests code, builds dependency/entrypoint traces, and emits Markdown reports with callers, downstream paths, and reasoning (LLM-backed if configured).

## What it does
- **Parse**: Walk Python files; capture imports, classes, functions, calls, decorators via AST.
- **Graph**: Build mod/sym/ext graphs with forward + reverse edges and intra-module sym→sym resolution.
- **Entrypoints**: Detect likely entrypoints (rules or auto), rank them, and trace downstream paths.
- **Reasoning**: Summarize traces heuristically or via Gemini LLM (if available).
- **Report**: Emit `algotrace-report.md` with modules, entrypoints, callers, downstream paths, and reasoning.

## Quickstart
```bash
python -m algotracer.cli analyze path/to/code \
  --report-dir /tmp/algotrace \
  --auto-entrypoints \
  --top-k-entrypoints 5
```
See `instructions.md` for full CLI options.

## Pipeline (Milestone 1)
1) Collect source files (optionally skip tests)
2) Parse AST → ModuleInfo
3) Build dependency graph (mod/sym/ext nodes, reverse edges, intra-module resolution)
4) Detect entrypoints (rules or auto)
5) Trace downstream flows and gather callers
6) Reason over traces (LLM if available; otherwise heuristic)
7) Render Markdown report

## Configuration highlights
- **Entrypoints**: rules mode (`--entrypoints ...` with optional `--add-sklearn-defaults`) or auto mode (`--auto-entrypoints`, `--top-k-entrypoints N`).
- **Tracing**: `--max-depth`, `--max-paths`, `--max-expansions`, `--max-examples`.
- **LLM**: set `GEMINI_API_KEY` and install `google-generativeai` to enable Gemini-backed reasoning; otherwise falls back to deterministic summaries.
- **Notebooks**: `.ipynb` files are converted automatically; code cells are extracted into temp `.py` files under `.algotracer_notebooks/`.

## Project layout
- `src/algotracer/ingest/ast_parser.py` — AST extraction (imports/classes/functions/calls/decorators).
- `src/algotracer/analysis/deps.py` — dependency graph (forward + reverse edges, intra-module resolution).
- `src/algotracer/analysis/entrypoints.py` — rules + auto entrypoint detection.
- `src/algotracer/analysis/trace.py` — DFS tracing, summaries, path limits.
- `src/algotracer/reasoning/flow_explainer.py` — reasoning (heuristic or Gemini-backed).
- `src/algotracer/reporting/renderer.py` — Markdown report (entrypoints, callers, paths, reasoning).
- `src/algotracer/pipeline.py` — orchestrator; logs pipeline stages.
- `src/algotracer/cli.py` — CLI entrypoint.
- `instructions.md` — detailed CLI usage examples.

## Notes
- Default behavior skips tests; add `--include-tests` to include them.
- If no LLM key is present, reports still render with heuristic reasoning.
- Reports are written to the specified `--report-dir` (default `reports`).
