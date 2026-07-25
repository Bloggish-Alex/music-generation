#!/usr/bin/env python3
"""CLI for rendering a continuous DVAE encoder-to-decoder reconstruction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader
from pipeline.dvae_reconstruction_pipeline import DVAEReconstructionPipeline


class DVAEReconstructionRenderCLI:
    """Command-line adapter for continuous DVAE reconstruction listening tests."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Render at least 16 contiguous bars through DVAE Encoder -> Decoder.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--output-json", type=Path, required=True)
        parser.add_argument("--output-midi", type=Path, required=True)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--dvae-path", type=Path, default=None)
        parser.add_argument("--bars", type=int, default=None, help="Must be at least 16; defaults to config or 32.")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--base-pitch", type=int, default=None)
        parser.add_argument("--tempo-bpm", type=int, default=None)
        parser.add_argument("--source-song-id", type=str, default=None)
        parser.add_argument("--start-bar", type=int, default=None)
        parser.add_argument("--fixed-base-pitch", action="store_true")
        parser.add_argument("--audio-quality", action="store_true")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        overrides = {
            "bars": args.bars,
            "seed": args.seed,
            "device": args.device,
            "base_pitch": args.base_pitch,
            "tempo_bpm": args.tempo_bpm,
            "source_song_id": args.source_song_id,
            "start_bar": args.start_bar,
            "use_source_base_pitch": False if args.fixed_base_pitch else None,
            "audio_quality_enabled": True if args.audio_quality else None,
        }
        result = DVAEReconstructionPipeline(config, overrides=overrides).run(
            model_dir=args.model_dir,
            output_json=args.output_json,
            output_midi=args.output_midi,
            dvae_path=args.dvae_path,
        )
        print(f"Reconstructed {result['reconstructed_bars']} contiguous bars -> {args.output_midi}")
        print(f"Source song: {result['source_song_id']} bars {result['start_bar']}..{result['end_bar']}")
        print(f"Diagnostics -> {args.output_json}")


def main() -> None:
    """Run the DVAE reconstruction rendering CLI."""
    DVAEReconstructionRenderCLI().run()


if __name__ == "__main__":
    main()
