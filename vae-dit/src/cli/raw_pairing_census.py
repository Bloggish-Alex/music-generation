from __future__ import annotations

import argparse
from pathlib import Path

from diagnostics.raw_pairing_census import run_census


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only raw SMF pairing census.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-revision", default="unknown")
    args = parser.parse_args()
    print(run_census(args.dataset_root, args.output_dir, args.code_revision))


if __name__ == "__main__": main()
