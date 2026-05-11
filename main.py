"""CLI entrypoint for the creative search pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json

from pipeline.runner import load_problem_from_file, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the creative search pipeline.")
    parser.add_argument("--problem", help="Problem text to run directly.")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Problem index from data/problems.json when --problem is not provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.problem:
        print("[main] Using problem from --problem input.")
    else:
        print(f"[main] No direct problem provided. Loading problem at index {args.index}.")
    problem = args.problem or load_problem_from_file(args.index)
    print(f"[main] Problem selected: {problem}")
    print("[main] Working/output language: Korean")
    print("[main] Starting creative-search pipeline.")
    result = asyncio.run(run_pipeline(problem))
    print("[main] Pipeline finished. Final JSON result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
