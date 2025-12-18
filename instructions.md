# AlgoTracer CLI Instructions

Run the CLI from the repo root:

```bash
python -m algotracer.cli <command> [options]
```

## Commands

### analyze
Analyze a repository and write the call graph into Memgraph.

Required:
- `repo_path` — repository root to analyze.

Common options:
- `--repo-id ID` — override repo_id (default: repo folder name).
- `--include-tests` — include files under `tests/` (default: skip).
- `--memgraph-host HOST` — Memgraph host (default: env or 127.0.0.1).
- `--memgraph-port PORT` — Memgraph port (default: env or 7687).
- `--memgraph-user USER` — Memgraph username (default: env).
- `--memgraph-password PASS` — Memgraph password (default: env).

Notes:
- Notebooks (`.ipynb`) are converted to temporary `.py` files under `.algotracer_notebooks/` during analysis.

### explain
Explain a function neighborhood from Memgraph.

Required:
- `repo_path` — repository root (used for path resolution).
- One of:
  - `--id <function_id>` (preferred)
  - `--name <qualname>` (optionally with `--file <path>`)

Common options:
- `--depth-up N` — caller traversal depth (default: 2).
- `--depth-down N` — callee traversal depth (default: 2).
- `--max-nodes N` — max nodes returned (default: 200).
- `--max-edges N` — max edges returned (default: 400).
- `--max-paths N` — max paths sampled per direction (default: 200).
- `--debug-subgraph PATH` — write `subgraph.json` for debugging.
- `--no-llm` — disable LLM usage (deterministic summary only).

## Examples

- Build graph for a repo:
  ```bash
  python -m algotracer.cli analyze path/to/repo --repo-id my-repo
  ```

- Explain a function by file + name:
  ```bash
  python -m algotracer.cli explain path/to/repo --file src/model.py --name Class.fit
  ```

- Explain by stable id:
  ```bash
  python -m algotracer.cli explain path/to/repo --id stable:sym:src/model.py:Class.fit
  ```
