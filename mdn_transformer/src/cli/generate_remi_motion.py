#!/usr/bin/env python3
"""CLI for REMI-motion DVAE generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader
from pipeline.remi_motion_pipeline import RemiMotionGenerationPipeline


class RemiMotionGenerationCLI:
    """Generate MIDI with REMI motion context and DVAE decoder."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Generate with REMI motion + DVAE decoder.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--output-json", type=Path, required=True)
        parser.add_argument("--output-midi", type=Path, required=True)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--checkpoint-path", type=Path, default=None)
        parser.add_argument("--dvae-path", type=Path, default=None)
        parser.add_argument("--bars", type=int, default=None)
        parser.add_argument("--primer-bars", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--base-pitch", type=int, default=None)
        parser.add_argument("--tempo-bpm", type=int, default=None)
        parser.add_argument("--seed-song-id", type=str, default=None)
        parser.add_argument("--skip-audio-quality", action="store_true")
        parser.add_argument("--disable-feedback-tokenization", action="store_true")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        overrides = {
            "bars": args.bars,
            "primer_bars": args.primer_bars,
            "seed": args.seed,
            "device": args.device,
            "base_pitch": args.base_pitch,
            "tempo_bpm": args.tempo_bpm,
            "seed_song_id": args.seed_song_id,
            "audio_quality_enabled": False if args.skip_audio_quality else None,
            "feedback_tokenization_enabled": False if args.disable_feedback_tokenization else None,
        }
        result = RemiMotionGenerationPipeline(config, overrides=overrides).run(
            model_dir=args.model_dir,
            output_json=args.output_json,
            output_midi=args.output_midi,
            checkpoint_path=args.checkpoint_path,
            dvae_path=args.dvae_path,
        )
        print(f"Generated {result['generated_bars']} bars -> {args.output_midi}")
        print(f"Diagnostics -> {args.output_json}")


def main() -> None:
    RemiMotionGenerationCLI().run()


if __name__ == "__main__":
    main()
