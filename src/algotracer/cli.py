from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

"""Command-line interface for running AlgoTracer analyses.

Instructions:
- Run `python -m algotracer.cli analyze <paths> --report-dir reports/` to analyze files or directories.
- Use --entrypoints to override entrypoint names and --include-tests to include test files.
- Extend build_parser if you add new subcommands.

Explanation:
- Parses CLI args into a PipelineConfig and executes AlgoTracerPipeline.run().
- Prints the report directory path after completion.
"""

import argparse
from pathlib import Path

from algotracer.pipeline import AlgoTracerPipeline, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlgoTracer agent-based analysis pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze one or more Python paths")
    analyze.add_argument("paths", nargs="+", type=Path, help="Files or directories to analyze")
    analyze.add_argument("--report-dir", type=Path, default=Path("reports"), help="Output directory for reports")

    # Entrypoint detection
    analyze.add_argument("--entrypoints", nargs="*", default=["fit", "predict"], help="Entrypoint function names to highlight (rules mode)")
    analyze.add_argument("--auto-entrypoints", action="store_true", help="Enable auto entrypoint discovery (ignores --entrypoints)")
    analyze.add_argument("--top-k-entrypoints", type=int, default=1, help="Number of entrypoints to trace/report")
    analyze.add_argument("--add-sklearn-defaults", action="store_true", help="Add sklearn-ish defaults to rules mode")
    analyze.add_argument("--no-base-requires-name-match", dest="base_requires_name_match", action="store_false", help="Allow class base hints without name match")
    analyze.set_defaults(base_requires_name_match=True)

    # Tracing knobs
    analyze.add_argument("--max-depth", type=int, default=3, help="Max trace depth")
    analyze.add_argument("--max-paths", type=int, default=None, help="Max number of paths to record per entrypoint")
    analyze.add_argument("--max-expansions", type=int, default=None, help="Max neighbor expansions to perform per entrypoint")
    analyze.add_argument("--max-examples", type=int, default=10, help="Number of example paths to include per entrypoint")

    # Source selection
    analyze.add_argument("--include-tests", action="store_true", help="Include test files in analysis")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        config = PipelineConfig(
            sources=args.paths,
            report_dir=args.report_dir,
            entrypoint_names=tuple(args.entrypoints) if args.entrypoints else None,
            auto_entrypoints=args.auto_entrypoints,
            top_k_entrypoints=args.top_k_entrypoints,
            add_sklearn_defaults=args.add_sklearn_defaults,
            base_requires_name_match=args.base_requires_name_match,
            include_tests=args.include_tests,
            max_depth=args.max_depth,
            max_paths=args.max_paths,
            max_expansions=args.max_expansions,
            max_examples=args.max_examples,
        )
        pipeline = AlgoTracerPipeline(config)
        pipeline.run()
        print(f"Report written to {args.report_dir}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
