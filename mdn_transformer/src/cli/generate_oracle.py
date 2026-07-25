#!/usr/bin/env python3
"""CLI entrypoint for Nearest Neighbor Oracle generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader, ConfigView
from pipeline.oracle_generation_pipeline import OracleGenerationConfig, OracleGenerationPipeline


class OracleGenerationCLI:
    """Command-line adapter for nearest-neighbor oracle generation."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI arguments."""
        parser = argparse.ArgumentParser(description="Generate a data upper-bound MIDI with Nearest Neighbor Oracle.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--latent-dir", type=Path, default=None, help="Defaults to --model-dir/latent.")
        parser.add_argument("--output-json", type=Path, required=True)
        parser.add_argument("--output-midi", type=Path, required=True)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--bars", type=int, default=None)
        parser.add_argument("--primer-bars", type=int, default=None)
        parser.add_argument("--selection-mode", type=str, default=None, choices=["nearest_neighbor", "sequential"])
        parser.add_argument("--source-scope", type=str, default=None, choices=["same_base", "same_song_strict", "free"])
        parser.add_argument("--top-k", type=int, default=None)
        parser.add_argument("--temperature", type=float, default=None)
        parser.add_argument("--query-context-bars", type=int, default=None)
        parser.add_argument("--position-vocab-size", type=int, default=None)
        parser.add_argument("--default-action", type=str, default=None)
        parser.add_argument("--action-plan", type=str, default=None, help="source / sections / cycle.")
        parser.add_argument("--action-sections", type=str, default=None, help="Example: INTRODUCE:8,VARY:8,DEVELOP:8,RETURN:6,CADENCE:2")
        parser.add_argument("--seed-song-id", type=str, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--base-pitch", type=int, default=None)
        parser.add_argument("--tempo", type=int, default=None)
        parser.add_argument("--form", type=str, default=None, help="Accepted for script compatibility; not used by this oracle.")
        parser.add_argument("--style", type=str, default=None, help="Accepted for script compatibility; not used by this oracle.")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run oracle generation from CLI arguments."""
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        section = ConfigView(config).section("latent_oracle")
        generation_config = OracleGenerationConfig(
            bars=int(args.bars if args.bars is not None else section.get("bars", 32)),
            primer_bars=int(args.primer_bars if args.primer_bars is not None else section.get("primer_bars", 8)),
            selection_mode=str(args.selection_mode if args.selection_mode is not None else section.get("selection_mode", "nearest_neighbor")),
            source_scope=str(args.source_scope if args.source_scope is not None else section.get("source_scope", "same_base")),
            top_k=int(args.top_k if args.top_k is not None else section.get("top_k", 16)),
            temperature=float(args.temperature if args.temperature is not None else section.get("temperature", 0.25)),
            query_context_bars=int(args.query_context_bars if args.query_context_bars is not None else section.get("query_context_bars", 1)),
            position_vocab_size=int(args.position_vocab_size if args.position_vocab_size is not None else section.get("position_vocab_size", 8)),
            default_action=str(args.default_action if args.default_action is not None else section.get("default_action", "VARY")),
            action_plan=str(args.action_plan if args.action_plan is not None else section.get("action_plan", "sections")),
            action_sections=str(args.action_sections if args.action_sections is not None else section.get("action_sections", "INTRODUCE:8,VARY:8,DEVELOP:8,RETURN:6,CADENCE:2")),
            seed=int(args.seed if args.seed is not None else section.get("seed", 42)),
            base_pitch=int(args.base_pitch if args.base_pitch is not None else section.get("base_pitch", 60)),
            tempo_bpm=int(args.tempo if args.tempo is not None else section.get("tempo_bpm", 120)),
        )
        model_dir = Path(args.model_dir)
        latent_dir = Path(args.latent_dir) if args.latent_dir else model_dir / "latent"
        result = OracleGenerationPipeline(generation_config).run(
            model_dir=model_dir,
            latent_dir=latent_dir,
            output_json=args.output_json,
            output_midi=args.output_midi,
            seed_song_id=args.seed_song_id,
        )
        print(f"Generated {generation_config.bars} oracle bars -> {result.midi_path}")
        print(f"Diagnostics -> {result.json_path}")
        print(f"Tensors -> {result.tensor_path}")


def main() -> None:
    """Run CLI."""
    OracleGenerationCLI().run()


if __name__ == "__main__":
    main()
