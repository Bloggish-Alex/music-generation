#!/usr/bin/env python3
"""CLI entrypoint for Latent-Transformer + DVAE generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader, ConfigView
from pipeline.anchor_motion_composer_pipeline import AnchorMotionComposerGenerationPipeline
from pipeline.hybrid_miditok_retrieval_pipeline import HybridMidiTokGenerationConfig, HybridMidiTokGenerationPipeline
from pipeline.latent_generation_pipeline import LatentGenerationConfig, LatentGenerationPipeline


class GenerationCLI:
    """Command-line adapter for the latent generation pipeline."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI arguments."""
        parser = argparse.ArgumentParser(description="Generate MIDI from the configured generation backend.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--latent-dir", type=Path, default=None, help="Defaults to --model-dir/latent.")
        parser.add_argument("--transformer-path", type=Path, default=None, help="Advanced override. By default, the generator resolves the checkpoint from --model-dir and backend.")
        parser.add_argument("--composer-path", type=Path, default=None, help="Advanced override for anchor_motion_composer. Defaults to --model-dir/anchor_motion_composer.pt.")
        parser.add_argument("--dvae-path", type=Path, default=None, help="Defaults to --model-dir/dvae.pt.")
        parser.add_argument("--output-json", type=Path, required=True)
        parser.add_argument("--output-midi", type=Path, required=True)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--bars", type=int, default=None)
        parser.add_argument("--primer-bars", type=int, default=None)
        parser.add_argument("--temperature", type=float, default=None)
        parser.add_argument("--sample-std-scale", type=float, default=None)
        parser.add_argument("--backend", type=str, default=None, help="hybrid_miditok_retrieval / direct_mdn / retrieval_mdn / memory_latent / anchor_motion_composer.")
        parser.add_argument("--model-variant", type=str, default=None, help="root / theme_fusion. Defaults to latent_generation.model_variant or root.")
        parser.add_argument("--default-action", type=str, default=None)
        parser.add_argument("--action-plan", type=str, default=None, help="source / sections / cycle.")
        parser.add_argument("--action-sections", type=str, default=None, help="Example: INTRODUCE:8,VARY:8,DEVELOP:8,RETURN:6,CADENCE:2")
        parser.add_argument("--seed-song-id", type=str, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--base-pitch", type=int, default=None)
        parser.add_argument("--tempo", type=int, default=None)
        parser.add_argument("--retrieval-top-k", type=int, default=None)
        parser.add_argument("--retrieval-temperature", type=float, default=None)
        parser.add_argument("--retrieval-distance-weight", type=float, default=None)
        parser.add_argument("--retrieval-energy-weight", type=float, default=None)
        parser.add_argument("--retrieval-position-weight", type=float, default=None)
        parser.add_argument("--retrieval-recent-penalty", type=float, default=None)
        parser.add_argument("--retrieval-recent-window", type=int, default=None)
        parser.add_argument("--candidate-transpose-mode", type=str, default=None, help="Hybrid backend: all / canonical_only.")
        parser.add_argument("--base-pitch-mode", type=str, default=None, help="Hybrid backend: source / fixed / learned.")
        parser.add_argument("--base-pitch-delta-min", type=int, default=None)
        parser.add_argument("--base-pitch-delta-max", type=int, default=None)
        parser.add_argument("--render-base-pitch-min", type=int, default=None)
        parser.add_argument("--render-base-pitch-max", type=int, default=None)
        parser.add_argument("--retrieval-use-retrieved-tensors", action="store_true")
        parser.add_argument("--retrieval-decode-latent", action="store_true")
        parser.add_argument("--energy-curve", type=str, default=None)
        parser.add_argument("--energy-arc-strength", type=float, default=None)
        parser.add_argument("--memory-scope-disabled", action="store_true")
        parser.add_argument("--memory-scope-top-n", type=int, default=None)
        parser.add_argument("--memory-scope-develop-top-n", type=int, default=None)
        parser.add_argument("--form", type=str, default=None, help="Accepted for script compatibility; not used by this generator yet.")
        parser.add_argument("--style", type=str, default=None, help="Accepted for script compatibility; not used by this generator yet.")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run generation from CLI arguments."""
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        section = ConfigView(config).section("latent_generation")
        generation_config = LatentGenerationConfig(
            bars=int(args.bars if args.bars is not None else section.get("bars", 32)),
            primer_bars=int(args.primer_bars if args.primer_bars is not None else section.get("primer_bars", 8)),
            temperature=float(args.temperature if args.temperature is not None else section.get("temperature", 0.9)),
            sample_std_scale=float(args.sample_std_scale if args.sample_std_scale is not None else section.get("sample_std_scale", 0.0)),
            backend=str(args.backend if args.backend is not None else section.get("backend", "direct_mdn")),
            model_variant=str(args.model_variant if args.model_variant is not None else section.get("model_variant", "root")),
            default_action=str(args.default_action if args.default_action is not None else section.get("default_action", "VARY")),
            action_plan=str(args.action_plan if args.action_plan is not None else section.get("action_plan", "source")),
            action_sections=str(args.action_sections if args.action_sections is not None else section.get("action_sections", "")),
            seed=int(args.seed if args.seed is not None else section.get("seed", 42)),
            device=str(args.device if args.device is not None else section.get("device", "cpu")),
            base_pitch=int(args.base_pitch if args.base_pitch is not None else section.get("base_pitch", 60)),
            tempo_bpm=int(args.tempo if args.tempo is not None else section.get("tempo_bpm", 120)),
            retrieval_top_k=int(args.retrieval_top_k if args.retrieval_top_k is not None else section.get("retrieval_top_k", 24)),
            retrieval_temperature=float(args.retrieval_temperature if args.retrieval_temperature is not None else section.get("retrieval_temperature", 0.35)),
            retrieval_distance_weight=float(args.retrieval_distance_weight if args.retrieval_distance_weight is not None else section.get("retrieval_distance_weight", 1.0)),
            retrieval_energy_weight=float(args.retrieval_energy_weight if args.retrieval_energy_weight is not None else section.get("retrieval_energy_weight", 1.25)),
            retrieval_position_weight=float(args.retrieval_position_weight if args.retrieval_position_weight is not None else section.get("retrieval_position_weight", 0.25)),
            retrieval_recent_penalty=float(args.retrieval_recent_penalty if args.retrieval_recent_penalty is not None else section.get("retrieval_recent_penalty", 2.0)),
            retrieval_recent_window=int(args.retrieval_recent_window if args.retrieval_recent_window is not None else section.get("retrieval_recent_window", 8)),
            retrieval_use_retrieved_tensors=(
                False if args.retrieval_decode_latent
                else (True if args.retrieval_use_retrieved_tensors else section.get("retrieval_use_retrieved_tensors"))
            ),
            energy_curve=str(args.energy_curve if args.energy_curve is not None else section.get("energy_curve", "INTRODUCE:0.35,VARY:0.55,DEVELOP:0.9,RETURN:0.7,CADENCE:0.45")),
            energy_arc_strength=float(args.energy_arc_strength if args.energy_arc_strength is not None else section.get("energy_arc_strength", 0.15)),
            memory_scope_enabled=(False if args.memory_scope_disabled else bool(section.get("memory_scope_enabled", True))),
            memory_scope_top_n=int(args.memory_scope_top_n if args.memory_scope_top_n is not None else section.get("memory_scope_top_n", 4)),
            memory_scope_develop_top_n=int(args.memory_scope_develop_top_n if args.memory_scope_develop_top_n is not None else section.get("memory_scope_develop_top_n", 12)),
        )
        model_dir = Path(args.model_dir)
        latent_dir = Path(args.latent_dir) if args.latent_dir else model_dir / "latent"
        backend = str(generation_config.backend).strip().lower()
        if backend in {"hybrid_miditok_retrieval", "hybrid_miditok", "hybrid"}:
            result = HybridMidiTokGenerationPipeline(self._hybrid_generation_config(args, section)).run(
                model_dir=model_dir,
                latent_dir=latent_dir,
                encoded_dir=model_dir / "encoded",
                output_json=args.output_json,
                output_midi=args.output_midi,
                seed_song_id=args.seed_song_id,
            )
        elif backend == "anchor_motion_composer":
            result = AnchorMotionComposerGenerationPipeline(generation_config).run(
                model_dir=model_dir,
                latent_dir=latent_dir,
                output_json=args.output_json,
                output_midi=args.output_midi,
                seed_song_id=args.seed_song_id,
                composer_path=args.composer_path,
                dvae_path=args.dvae_path,
            )
        else:
            result = LatentGenerationPipeline(generation_config).run(
                model_dir=model_dir,
                latent_dir=latent_dir,
                output_json=args.output_json,
                output_midi=args.output_midi,
                seed_song_id=args.seed_song_id,
                transformer_path=args.transformer_path,
                dvae_path=args.dvae_path,
            )
        print(f"Generated {generation_config.bars} bars -> {result.midi_path}")
        print(f"Diagnostics -> {result.json_path}")
        print(f"Tensors -> {result.tensor_path}")

    def _hybrid_generation_config(self, args: argparse.Namespace, section: dict) -> HybridMidiTokGenerationConfig:
        """Build hybrid MidiTok retrieval generation config from the common generation section."""
        return HybridMidiTokGenerationConfig(
            bars=int(args.bars if args.bars is not None else section.get("bars", 32)),
            primer_bars=int(args.primer_bars if args.primer_bars is not None else section.get("primer_bars", 8)),
            top_k=int(args.retrieval_top_k if args.retrieval_top_k is not None else section.get("retrieval_top_k", 24)),
            temperature=float(args.retrieval_temperature if args.retrieval_temperature is not None else section.get("retrieval_temperature", 0.35)),
            recent_penalty=float(args.retrieval_recent_penalty if args.retrieval_recent_penalty is not None else section.get("retrieval_recent_penalty", 2.0)),
            recent_window=int(args.retrieval_recent_window if args.retrieval_recent_window is not None else section.get("retrieval_recent_window", 8)),
            seed=int(args.seed if args.seed is not None else section.get("seed", 42)),
            device=str(args.device if args.device is not None else section.get("device", "cpu")),
            base_pitch=int(args.base_pitch if args.base_pitch is not None else section.get("base_pitch", 60)),
            tempo_bpm=int(args.tempo if args.tempo is not None else section.get("tempo_bpm", 100)),
            candidate_transpose_mode=str(args.candidate_transpose_mode if args.candidate_transpose_mode is not None else section.get("candidate_transpose_mode", "canonical_only")),
            base_pitch_mode=str(args.base_pitch_mode if args.base_pitch_mode is not None else section.get("base_pitch_mode", "learned")),
            render_base_pitch_min=int(args.render_base_pitch_min if args.render_base_pitch_min is not None else section.get("render_base_pitch_min", 36)),
            render_base_pitch_max=int(args.render_base_pitch_max if args.render_base_pitch_max is not None else section.get("render_base_pitch_max", 84)),
            base_pitch_delta_min=(
                int(args.base_pitch_delta_min)
                if args.base_pitch_delta_min is not None
                else (None if section.get("base_pitch_delta_min") is None else int(section.get("base_pitch_delta_min")))
            ),
            base_pitch_delta_max=(
                int(args.base_pitch_delta_max)
                if args.base_pitch_delta_max is not None
                else (None if section.get("base_pitch_delta_max") is None else int(section.get("base_pitch_delta_max")))
            ),
        )


def main() -> None:
    """Run CLI."""
    GenerationCLI().run()


if __name__ == "__main__":
    main()
