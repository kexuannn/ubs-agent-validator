# AlgoTracer CLI Instructions

Run the CLI from the repo root:

```bash
python -m algotracer.cli <command> [options]
```

## Commands

### analyze
Analyze one or more Python files/directories and emit a Markdown report (`algotrace-report.md`).

Required:
- `paths...` — one or more files or directories to analyze.

Common options:
- `--report-dir PATH` — where to write the report (default: `reports`).
- `--include-tests` — include files under `tests/` (default: skip).

Entrypoint detection:
- Rules mode (default): `--entrypoints name1 name2 ...` (default: `fit predict`).
- Auto mode: `--auto-entrypoints` (ignores `--entrypoints`), `--top-k-entrypoints N` (default: 1).
- Tuning: `--add-sklearn-defaults` (adds sklearn-ish heuristics in rules mode); `--no-base-requires-name-match` (let base-class hints apply without name match).

Tracing controls:
- `--max-depth N` — DFS depth (default: 3).
- `--max-paths N` — cap number of paths per entrypoint (default: unlimited).
- `--max-expansions N` — cap neighbor expansions per entrypoint (default: unlimited).
- `--max-examples N` — number of example paths to show (default: 10).

## Examples

- Rules mode with defaults:
  ```bash
  python -m algotracer.cli analyze src --report-dir /tmp/algotrace
  ```

- Auto entrypoints, trace top 5, include tests:
  ```bash
  python -m algotracer.cli analyze src --auto-entrypoints --top-k-entrypoints 5 --include-tests --report-dir /tmp/algotrace
  ```

- Custom entrypoints and deeper trace:
  ```bash
  python -m algotracer.cli analyze src --entrypoints main run serve --max-depth 4 --max-paths 500 --report-dir /tmp/algotrace
  ```
