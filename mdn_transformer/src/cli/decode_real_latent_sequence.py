#!/usr/bin/env python3
"""CLI for decoding real latent sequences through the DVAE decoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader
from pipeline.dvae_latent_oracle_pipeline import DVAELatentOraclePipeline


class DVAELatentOracleCLI:
    """Decode true training latents to isolate DVAE decoder behavior."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Decode real latent sequences with the DVAE decoder.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--output-json", type=Path, required=True)
        parser.add_argument("--output-midi", type=Path, required=True)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--dvae-path", type=Path, default=None)
        parser.add_argument("--latent-dir", type=Path, default=None)
        parser.add_argument("--bars", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--base-pitch", type=int, default=None)
        parser.add_argument("--tempo-bpm", type=int, default=None)
        parser.add_argument("--seed-song-id", type=str, default=None)
        parser.add_argument("--start-bar", type=int, default=None)
        parser.add_argument("--fixed-base-pitch", action="store_true")
        parser.add_argument("--skip-audio-quality", action="store_true")
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
            "seed_song_id": args.seed_song_id,
            "start_bar": args.start_bar,
            "use_source_base_pitch": False if args.fixed_base_pitch else None,
            "audio_quality_enabled": False if args.skip_audio_quality else None,
        }
        result = DVAELatentOraclePipeline(config, overrides=overrides).run(
            model_dir=args.model_dir,
            output_json=args.output_json,
            output_midi=args.output_midi,
            dvae_path=args.dvae_path,
            latent_dir=args.latent_dir,
        )
        print(f"Decoded {result['generated_bars']} real latent bars -> {args.output_midi}")
        print(f"Source song: {result['source_song_id']} start_bar={result['start_bar']}")
        print(f"Diagnostics -> {args.output_json}")


def main() -> None:
    DVAELatentOracleCLI().run()


if __name__ == "__main__":
    main()
