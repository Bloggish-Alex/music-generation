#!/usr/bin/env python3
"""CLI for artifact-only evaluation runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation" / "src"))

from evaluation_framework.evaluation_registry import DEFAULT_MODULE_REGISTRY
from evaluation_framework.evaluation_runner import EvaluationMode, EvaluationRunRequest, EvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and evaluate registered music evaluation modules.")
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--run-dir", type=Path, help="Write flat evaluation artifacts directly into this existing generation run directory.")
    parser.add_argument("--code-revision", type=str, default="unknown")
    parser.add_argument("--modules", default="all", help="Comma-separated test-point names or 'all'.")
    parser.add_argument("--mode", choices=[mode.value for mode in EvaluationMode], default=EvaluationMode.ALL.value)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-modules", action="store_true")
    args = parser.parse_args()
    if args.list_modules:
        print(json.dumps(DEFAULT_MODULE_REGISTRY.names(), indent=2))
        return
    if args.run_dir is None and (args.output_root is None or args.run_id is None):
        parser.error("--output-root and --run-id are required unless --list-modules is used.")
    if args.run_dir is not None and (args.output_root is not None or args.run_id is not None):
        parser.error("--run-dir cannot be combined with --output-root or --run-id.")
    if args.run_dir is not None and args.input_root is not None and args.input_root.resolve() != args.run_dir.resolve():
        parser.error("--run-dir uses the same directory as its public artifact input; do not provide a different --input-root.")
    input_root = args.run_dir if args.run_dir is not None else (args.input_root or Path("."))
    modules = "all" if args.modules == "all" else tuple(value.strip() for value in args.modules.split(","))
    store = EvaluationRunner(DEFAULT_MODULE_REGISTRY).run(
        EvaluationRunRequest(
            input_root=input_root,
            output_root=args.output_root,
            run_id=args.run_id or args.run_dir.name,
            modules=modules,
            mode=EvaluationMode(args.mode),
            resume=args.resume,
            run_dir=args.run_dir,
            options={"code_revision": args.code_revision},
        )
    )
    index_name = "evaluation_index.json" if args.run_dir else "index.json"
    index_path = store.run_dir / index_name
    failures = {
        name: {phase: value for phase, value in entry.items() if isinstance(value, dict) and value.get("status") == "FAIL"}
        for name, entry in getattr(store, "index", {}).get("modules", {}).items()
    }
    failures = {name: value for name, value in failures.items() if value}
    print(json.dumps({"run_directory": str(store.run_dir), "index": str(index_path), "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
