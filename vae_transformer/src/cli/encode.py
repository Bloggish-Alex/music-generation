#!/usr/bin/env python3
"""CLI entrypoint for parsing, tensor encoding, and action labeling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader
from pipeline.encoding_pipeline import EncodingPipeline


class EncodingCLI:
    """Command-line adapter for the encoding pipeline."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI argument parser."""
        parser = argparse.ArgumentParser(description="Encode symbolic music bars into tensor datasets.")
        parser.add_argument("--music-dir", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--config", type=Path, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run CLI from parsed arguments."""
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        result = EncodingPipeline(config).run(args.music_dir, args.output_dir)
        print(f"Parsed songs: {len(result.songs)}")
        print(f"Encoded bars: {len(result.tensors)}")
        print(f"Output -> {args.output_dir}")


def main() -> None:
    """Run the encoding CLI."""
    EncodingCLI().run()


if __name__ == "__main__":
    main()
