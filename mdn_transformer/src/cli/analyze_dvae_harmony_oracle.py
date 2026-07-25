#!/usr/bin/env python3
"""CLI for the frozen DVAE physical-harmony oracle diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.dvae_harmony_oracle import DVAEHarmonyOracleAnalyzer, DVAEHarmonyOracleConfig


class DVAEHarmonyOracleCLI:
    """Parse paths and execute the read-only harmony oracle."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Analyze frozen DVAE latent harmony preservation.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--latent-dir", type=Path, default=None)
        parser.add_argument("--encoded-dir", type=Path, default=None)
        parser.add_argument("--dvae-path", type=Path, default=None)
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--batch-size", type=int, default=256)
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--pitch-scale", type=float, default=24.0)
        parser.add_argument("--pitch-class-sigma", type=float, default=0.35)
        parser.add_argument("--posterior-samples", type=int, default=4)
        parser.add_argument("--max-rows", type=int, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        report = DVAEHarmonyOracleAnalyzer(DVAEHarmonyOracleConfig(
            model_dir=args.model_dir,
            latent_dir=args.latent_dir,
            encoded_dir=args.encoded_dir,
            dvae_path=args.dvae_path,
            output_dir=args.output_dir,
            batch_size=int(args.batch_size),
            device=str(args.device),
            pitch_scale=float(args.pitch_scale),
            pitch_class_sigma=float(args.pitch_class_sigma),
            posterior_samples=max(1, int(args.posterior_samples)),
            max_rows=None if args.max_rows is None else int(args.max_rows),
        )).run()
        output_dir = Path(report["output_dir"])
        print(f"DVAE harmony diagnostics -> {output_dir / 'dvae_harmony_oracle_diagnostics.json'}")
        print(f"DVAE harmony report -> {output_dir / 'dvae_harmony_oracle_report.md'}")


def main() -> None:
    DVAEHarmonyOracleCLI().run()


if __name__ == "__main__":
    main()
