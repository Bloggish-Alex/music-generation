#!/usr/bin/env python3
"""CLI for the semantic BarCodec harmony oracle diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.bar_codec_harmony_oracle import BarCodecHarmonyOracleAnalyzer, BarCodecHarmonyOracleConfig


class BarCodecHarmonyOracleCLI:
    """Parse inputs and run the read-only semantic-codec harmony probe."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Analyze source-harmony retention by the semantic BarCodec.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--encoded-dir", type=Path, default=None)
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--pitch-scale", type=float, default=24.0)
        parser.add_argument("--pitch-class-sigma", type=float, default=0.35)
        parser.add_argument("--max-rows", type=int, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        report = BarCodecHarmonyOracleAnalyzer(BarCodecHarmonyOracleConfig(
            model_dir=args.model_dir,
            encoded_dir=args.encoded_dir,
            output_dir=args.output_dir,
            pitch_scale=float(args.pitch_scale),
            pitch_class_sigma=float(args.pitch_class_sigma),
            max_rows=None if args.max_rows is None else int(args.max_rows),
        )).run()
        output_dir = Path(report["output_dir"])
        print(f"BarCodec harmony diagnostics -> {output_dir / 'bar_codec_harmony_oracle_diagnostics.json'}")
        print(f"BarCodec harmony report -> {output_dir / 'bar_codec_harmony_oracle_report.md'}")


def main() -> None:
    BarCodecHarmonyOracleCLI().run()


if __name__ == "__main__":
    main()
