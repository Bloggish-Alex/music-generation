#!/usr/bin/env python3
"""CLI for hybrid MidiTok retrieval generation."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader
from pipeline.hybrid_miditok_retrieval_pipeline import (
    HybridMidiTokGenerationConfig,
    HybridMidiTokGenerationPipeline,
)


class HybridMidiTokRetrievalGenerationCLI:
    """Generate MIDI from a trained hybrid retrieval model."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI parser."""
        parser = argparse.ArgumentParser(description="Generate MIDI with hybrid latent + MidiTok retrieval.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--latent-dir", type=Path, default=None)
        parser.add_argument("--encoded-dir", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--checkpoint-path", type=Path, default=None)
        parser.add_argument("--dvae-path", type=Path, default=None)
        parser.add_argument("--output-json", type=Path, required=True)
        parser.add_argument("--output-midi", type=Path, required=True)
        parser.add_argument("--bars", type=int, default=None)
        parser.add_argument("--primer-bars", type=int, default=None)
        parser.add_argument("--top-k", type=int, default=None)
        parser.add_argument("--temperature", type=float, default=None)
        parser.add_argument("--recent-penalty", type=float, default=None)
        parser.add_argument("--recent-window", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--base-pitch", type=int, default=None)
        parser.add_argument("--tempo", type=int, default=None)
        parser.add_argument("--seed-song-id", type=str, default=None)
        parser.add_argument("--base-pitch-mode", type=str, default=None, help="source / fixed / learned.")
        parser.add_argument("--base-pitch-motion-path", type=str, default=None)
        parser.add_argument("--render-base-pitch-min", type=int, default=None)
        parser.add_argument("--render-base-pitch-max", type=int, default=None)
        parser.add_argument("--base-pitch-delta-min", type=int, default=None)
        parser.add_argument("--base-pitch-delta-max", type=int, default=None)
        parser.add_argument("--candidate-limit", type=int, default=None, help="Debug only. Limit generation candidate rows.")
        parser.add_argument(
            "--candidate-transpose-mode",
            type=str,
            default=None,
            help="all / canonical_only. canonical_only excludes *_T+N transpose augmentation rows during generation.",
        )
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run generation."""
        args = self.build_parser().parse_args(argv)
        config = self._generation_config(args)
        result = HybridMidiTokGenerationPipeline(config).run(
            model_dir=args.model_dir,
            latent_dir=args.latent_dir,
            encoded_dir=args.encoded_dir,
            checkpoint_path=args.checkpoint_path,
            dvae_path=args.dvae_path,
            output_json=args.output_json,
            output_midi=args.output_midi,
            seed_song_id=args.seed_song_id,
        )
        print(f"Generated {config.bars} bars -> {result.midi_path}")
        print(f"Diagnostics -> {result.json_path}")
        print(f"Tensors -> {result.tensor_path}")

    def _generation_config(self, args: argparse.Namespace) -> HybridMidiTokGenerationConfig:
        """Build generation config."""
        value = self._config_file_generation_config(args.config)
        updates = {}
        for arg_name, field_name in (
            ("bars", "bars"),
            ("primer_bars", "primer_bars"),
            ("top_k", "top_k"),
            ("temperature", "temperature"),
            ("recent_penalty", "recent_penalty"),
            ("recent_window", "recent_window"),
            ("seed", "seed"),
            ("device", "device"),
            ("base_pitch", "base_pitch"),
            ("tempo", "tempo_bpm"),
            ("candidate_limit", "candidate_limit"),
            ("candidate_transpose_mode", "candidate_transpose_mode"),
            ("base_pitch_mode", "base_pitch_mode"),
            ("base_pitch_motion_path", "base_pitch_motion_path"),
            ("render_base_pitch_min", "render_base_pitch_min"),
            ("render_base_pitch_max", "render_base_pitch_max"),
            ("base_pitch_delta_min", "base_pitch_delta_min"),
            ("base_pitch_delta_max", "base_pitch_delta_max"),
        ):
            override = getattr(args, arg_name)
            if override is not None:
                updates[field_name] = override
        return replace(value, **updates)

    def _config_file_generation_config(self, config_path: Optional[Path]) -> HybridMidiTokGenerationConfig:
        """Build base generation config from style config defaults."""
        config = ConfigLoader().load(config_path)
        section = config.get("latent_generation", {})
        if not isinstance(section, dict):
            section = {}
        value = HybridMidiTokGenerationConfig()
        updates = {}
        for config_name, field_name in (
            ("bars", "bars"),
            ("primer_bars", "primer_bars"),
            ("retrieval_top_k", "top_k"),
            ("retrieval_temperature", "temperature"),
            ("retrieval_recent_penalty", "recent_penalty"),
            ("retrieval_recent_window", "recent_window"),
            ("seed", "seed"),
            ("device", "device"),
            ("base_pitch", "base_pitch"),
            ("tempo_bpm", "tempo_bpm"),
            ("candidate_limit", "candidate_limit"),
            ("candidate_transpose_mode", "candidate_transpose_mode"),
            ("base_pitch_mode", "base_pitch_mode"),
            ("base_pitch_motion_path", "base_pitch_motion_path"),
            ("render_base_pitch_min", "render_base_pitch_min"),
            ("render_base_pitch_max", "render_base_pitch_max"),
            ("base_pitch_delta_min", "base_pitch_delta_min"),
            ("base_pitch_delta_max", "base_pitch_delta_max"),
        ):
            if config_name in section and section[config_name] is not None:
                updates[field_name] = section[config_name]
        return replace(value, **updates)


def main() -> None:
    """Run CLI."""
    HybridMidiTokRetrievalGenerationCLI().run()


if __name__ == "__main__":
    main()
