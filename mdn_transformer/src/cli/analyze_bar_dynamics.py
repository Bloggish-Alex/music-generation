#!/usr/bin/env python3
"""CLI entrypoint for VAR-style bar dynamics diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.bar_dynamics_analysis import BarDynamicsAnalysisConfig, BarDynamicsAnalyzer


class BarDynamicsAnalysisCLI:
    """Command-line adapter for bar dynamics analysis."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI parser."""
        parser = argparse.ArgumentParser(description="Analyze latent and explicit bar feature dynamics with VAR-style probes.")
        parser.add_argument("--model-dir", type=Path, required=True, help="Model directory containing latent/ and encoded/ artifacts.")
        parser.add_argument("--latent-dir", type=Path, default=None, help="Defaults to --model-dir/latent.")
        parser.add_argument("--encoded-dir", type=Path, default=None, help="Defaults to --model-dir/encoded.")
        parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to --model-dir/bar_dynamics_analysis.")
        parser.add_argument("--max-lag", type=int, default=16)
        parser.add_argument("--validation-ratio", type=float, default=0.2)
        parser.add_argument("--ridge-alpha", type=float, default=1.0)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--redundancy-threshold", type=float, default=0.95)
        parser.add_argument("--drift-threshold", type=float, default=0.35)
        parser.add_argument("--low-lag-correlation-threshold", type=float, default=0.05)
        parser.add_argument("--max-rows", type=int, default=None)
        parser.add_argument("--max-songs", type=int, default=None)
        parser.add_argument("--no-figures", action="store_true", help="Skip optional PNG heatmaps.")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run analysis."""
        args = self.build_parser().parse_args(argv)
        config = BarDynamicsAnalysisConfig(
            model_dir=Path(args.model_dir),
            latent_dir=Path(args.latent_dir) if args.latent_dir else None,
            encoded_dir=Path(args.encoded_dir) if args.encoded_dir else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            max_lag=int(args.max_lag),
            validation_ratio=float(args.validation_ratio),
            ridge_alpha=float(args.ridge_alpha),
            random_seed=int(args.seed),
            redundancy_threshold=float(args.redundancy_threshold),
            drift_threshold=float(args.drift_threshold),
            low_lag_correlation_threshold=float(args.low_lag_correlation_threshold),
            max_rows=int(args.max_rows) if args.max_rows is not None else None,
            max_songs=int(args.max_songs) if args.max_songs is not None else None,
            write_figures=not bool(args.no_figures),
        )
        report = BarDynamicsAnalyzer(config).run()
        output_dir = Path(report["output_dir"])
        print(f"Bar dynamics diagnostics -> {output_dir / 'bar_dynamics_diagnostics.json'}")
        print(f"Bar dynamics report -> {output_dir / 'bar_dynamics_report.md'}")


def main() -> None:
    """Run CLI."""
    BarDynamicsAnalysisCLI().run()


if __name__ == "__main__":
    main()
