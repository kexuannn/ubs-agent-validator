from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

"""Command-line interface for AlgoTracer Memgraph workflow."""

import argparse
from pathlib import Path

from algotracer.memgraph.client import MemgraphConfig
from algotracer.pipeline import AlgoTracerPipeline, AnalyzeConfig, ExplainConfig


def _add_memgraph_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--memgraph-host", default=None, help="Memgraph host (default: MEMGRAPH_HOST or 127.0.0.1)")
    parser.add_argument("--memgraph-port", type=int, default=None, help="Memgraph port (default: MEMGRAPH_PORT or 7687)")
    parser.add_argument("--memgraph-user", default=None, help="Memgraph username (default: MEMGRAPH_USER)")
    parser.add_argument("--memgraph-password", default=None, help="Memgraph password (default: MEMGRAPH_PASSWORD)")


def _build_memgraph_config(args: argparse.Namespace) -> MemgraphConfig:
    base = MemgraphConfig.from_env()
    return MemgraphConfig(
        host=args.memgraph_host or base.host,
        port=args.memgraph_port or base.port,
        user=args.memgraph_user if args.memgraph_user is not None else base.user,
        password=args.memgraph_password if args.memgraph_password is not None else base.password,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlgoTracer Memgraph-first workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a repository and build the Memgraph call graph")
    analyze.add_argument("repo_path", type=Path, help="Repository root to analyze")
    analyze.add_argument("--repo-id", default=None, help="Override repo_id (default: repo folder name)")
    analyze.add_argument("--include-tests", action="store_true", help="Include files under tests/")
    _add_memgraph_flags(analyze)

    explain = sub.add_parser("explain", help="Explain a function neighborhood from Memgraph")
    explain.add_argument("repo_path", type=Path, help="Repository root (used for path resolution)")
    explain.add_argument("--repo-id", default=None, help="Override repo_id (default: repo folder name)")

    target = explain.add_mutually_exclusive_group(required=True)
    target.add_argument("--id", dest="function_id", help="Stable function id (preferred)")
    target.add_argument("--name", dest="function_name", help="Function qualname (e.g., Class.method or func)")

    explain.add_argument("--file", dest="function_file", type=Path, help="File path for --name resolution")

    explain.add_argument("--depth-up", type=int, default=2, help="Caller traversal depth (default: 2)")
    explain.add_argument("--depth-down", type=int, default=2, help="Callee traversal depth (default: 2)")
    explain.add_argument("--max-nodes", type=int, default=200, help="Max nodes to return (default: 200)")
    explain.add_argument("--max-edges", type=int, default=400, help="Max edges to return (default: 400)")
    explain.add_argument("--max-paths", type=int, default=200, help="Max paths to sample per direction (default: 200)")
    explain.add_argument("--debug-subgraph", type=Path, default=None, help="Write subgraph JSON to this path")
    explain.add_argument("--no-llm", action="store_true", help="Disable LLM usage")
    _add_memgraph_flags(explain)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    memgraph_config = _build_memgraph_config(args)
    pipeline = AlgoTracerPipeline(memgraph_config)

    if args.command == "analyze":
        repo_path: Path = args.repo_path
        repo_id = args.repo_id or repo_path.resolve().name
        pipeline.analyze(
            AnalyzeConfig(
                repo_path=repo_path,
                repo_id=repo_id,
                include_tests=args.include_tests,
            )
        )
        return 0

    if args.command == "explain":
        if args.function_file and not args.function_name:
            parser.error("--file requires --name")

        repo_path: Path = args.repo_path
        repo_id = args.repo_id or repo_path.resolve().name
        explanation = pipeline.explain(
            ExplainConfig(
                repo_path=repo_path,
                repo_id=repo_id,
                function_id=args.function_id,
                function_name=args.function_name,
                function_file=args.function_file,
                depth_up=args.depth_up,
                depth_down=args.depth_down,
                max_nodes=args.max_nodes,
                max_edges=args.max_edges,
                max_paths=args.max_paths,
                debug_subgraph_path=args.debug_subgraph,
                use_llm=not args.no_llm,
            )
        )
        print(explanation)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
