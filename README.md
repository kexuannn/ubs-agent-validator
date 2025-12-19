# AlgoTracer

AlgoTracer builds a call graph for a Python repository directly in Memgraph, then explains the neighborhood around a specific function using only that graph evidence.

## What it does
- **Parse**: Walk Python files (and notebooks), capture imports, classes, functions, calls, decorators via AST.
- **Graph (Memgraph-first)**: Write Repo/Module/Function/External nodes + CALLS edges into Memgraph.
- **Resolve**: Pick a function by stable id or file/name and disambiguate if needed.
- **Neighborhood**: Fetch bounded upstream/downstream subgraphs with caps.
- **Explain**: Generate a short explanation grounded in that subgraph (LLM optional; deterministic fallback).

## Quickstart

1) **Start Memgraph via Docker (no auth):**
```bash
docker run -it --rm \
  -p 7687:7687 -p 3000:3000 \
  memgraph/memgraph-platform
```
This exposes Bolt on `127.0.0.1:7687`. Adjust ports as needed.

2) **Install AlgoTracer (editable):**
```bash
pip install -e .
```

3) **Analyze a repo (build graph in Memgraph):**
```bash
algotracer analyze path/to/repo --repo-id my-repo
```

4) **Explain a function:**
```bash
algotracer explain path/to/repo \
  --name Class.fit \
  --file src/model.py \
  --output reports/explanation.md
```

Memgraph connection is configurable via environment variables:
- `MEMGRAPH_HOST` (default `127.0.0.1`)
- `MEMGRAPH_PORT` (default `7687`)
- `MEMGRAPH_USER`, `MEMGRAPH_PASSWORD` (optional)

You can also override via CLI flags: `--memgraph-host/--memgraph-port/--memgraph-user/--memgraph-password`.

## CLI
Run `python -m algotracer.cli --help` for full options.

## Project layout
- `src/algotracer/ingest/ast_parser.py` — AST extraction (imports/classes/functions/calls/decorators).
- `src/algotracer/graph/builder.py` — build and write the Memgraph graph.
- `src/algotracer/graph/resolver.py` — resolve target functions by id/path/name.
- `src/algotracer/graph/neighborhood.py` — traverse upstream/downstream neighborhood.
- `src/algotracer/memgraph/client.py` — Memgraph connection + schema helpers.
- `src/algotracer/reasoning/explainer.py` — explanation prompt + deterministic fallback.
- `src/algotracer/pipeline.py` — analyze/explain orchestration.
- `src/algotracer/cli.py` — CLI entrypoint.

## Notes
- Default behavior skips tests; add `--include-tests` to include them.
- Notebooks (`.ipynb`) are converted to temp `.py` files under `.algotracer_notebooks/` before parsing.
- If no LLM key is present, explanations fall back to deterministic summaries.
- Multiple repos can coexist in Memgraph by using distinct `repo_id` values.
- If Memgraph is not running, `analyze`/`explain` will fail to connect; start the Docker container first.
