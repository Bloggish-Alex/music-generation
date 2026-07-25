#!/usr/bin/env python3
"""CLI entrypoint for aggregating Latent-Transformer fold reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.latent_transformer_fold_report import FoldReportConfig, LatentTransformerFoldReport


class FoldReportCLI:
    """Command-line adapter for fold report aggregation."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI parser."""
        parser = argparse.ArgumentParser(description="Aggregate Latent-Transformer fold summaries.")
        parser.add_argument(
            "--input-dir",
            type=Path,
            action="append",
            required=True,
            help="Fold directory or root directory. Can be passed multiple times.",
        )
        parser.add_argument("--output-dir", type=Path, required=True)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run fold report aggregation."""
        args = self.build_parser().parse_args(argv)
        report = LatentTransformerFoldReport().run(
            FoldReportConfig(
                input_dirs=tuple(Path(path) for path in args.input_dir),
                output_dir=Path(args.output_dir),
            )
        )
        print(f"Fold report JSON -> {report['paths']['json']}")
        print(f"Fold report Markdown -> {report['paths']['markdown']}")
        print(f"Fold count: {report['fold_count']}")


def main() -> None:
    """Run CLI."""
    FoldReportCLI().run()


if __name__ == "__main__":
    main()
