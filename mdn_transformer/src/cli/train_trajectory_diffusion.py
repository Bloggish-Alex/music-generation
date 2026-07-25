#!/usr/bin/env python3
"""CLI for the independent joint REMI/latent/register trajectory diffusion route."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader
from pipeline.trajectory_diffusion_pipeline import TrajectoryDiffusionTrainingPipeline


class TrajectoryDiffusionTrainingCLI:
    """Train four-bar joint diffusion from aligned three-stream history."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Train joint REMI trajectory diffusion.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--latent-dir", type=Path, default=None)
        parser.add_argument("--encoded-dir", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--weight-decay", type=float, default=None)
        parser.add_argument("--validation-ratio", type=float, default=None)
        parser.add_argument("--validation-split-unit", choices=["base_song_id", "sample"], default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--max-songs", type=int, default=None)
        parser.add_argument("--force-rebuild-tokens", action="store_true")
        parser.add_argument("--context-bars", type=int, default=None)
        parser.add_argument("--trajectory-bars", type=int, default=None)
        parser.add_argument("--diffusion-steps", type=int, default=None)
        parser.add_argument("--sampling-steps", type=int, default=None)
        parser.add_argument("--d-model", type=int, default=None)
        parser.add_argument("--token-layers", type=int, default=None)
        parser.add_argument("--bar-layers", type=int, default=None)
        parser.add_argument("--denoiser-layers", type=int, default=None)
        parser.add_argument("--n-heads", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument("--predictor-hidden-dim", type=int, default=None)
        parser.add_argument("--enable-flow-matching", dest="flow_matching_enabled", action="store_const", const=True, default=None)
        parser.add_argument("--disable-flow-matching", dest="flow_matching_enabled", action="store_const", const=False)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        overrides = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "validation_ratio": args.validation_ratio,
            "validation_split_unit": args.validation_split_unit,
            "device": args.device,
            "random_seed": args.seed,
            "max_songs": args.max_songs,
            "force_rebuild_tokens": True if args.force_rebuild_tokens else None,
            "context_bars": args.context_bars,
            "trajectory_bars": args.trajectory_bars,
            "diffusion_steps": args.diffusion_steps,
            "sampling_steps": args.sampling_steps,
            "d_model": args.d_model,
            "token_layers": args.token_layers,
            "bar_layers": args.bar_layers,
            "denoiser_layers": args.denoiser_layers,
            "n_heads": args.n_heads,
            "dropout": args.dropout,
            "predictor_hidden_dim": args.predictor_hidden_dim,
            "flow_matching_enabled": args.flow_matching_enabled,
        }
        result = TrajectoryDiffusionTrainingPipeline(config, overrides=overrides).run(
            model_dir=args.model_dir,
            latent_dir=args.latent_dir,
            encoded_dir=args.encoded_dir,
        )
        print(f"Saved trajectory diffusion model -> {result['model_path']}")
        print(f"Diagnostics -> {Path(args.model_dir) / 'trajectory_diffusion' / 'trajectory_diffusion_training_diagnostics.json'}")
        print(f"Best validation velocity loss -> {result['best_val_loss']:.6f}")


def main() -> None:
    TrajectoryDiffusionTrainingCLI().run()


if __name__ == "__main__":
    main()
