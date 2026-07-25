#!/usr/bin/env python3
"""CLI for training REMI-motion latent predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader
from pipeline.remi_motion_pipeline import RemiMotionTrainingPipeline


class RemiMotionTrainingCLI:
    """Train REMI-context to next-latent prediction."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Train REMI motion latent predictor.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--latent-dir", type=Path, default=None)
        parser.add_argument("--encoded-dir", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--weight-decay", type=float, default=None)
        parser.add_argument("--validation-ratio", type=float, default=None)
        parser.add_argument("--validation-split-unit", type=str, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--max-songs", type=int, default=None)
        parser.add_argument("--force-rebuild-tokens", action="store_true")
        parser.add_argument("--context-bars", type=int, default=None)
        parser.add_argument("--max-context-tokens", type=int, default=None)
        parser.add_argument("--d-model", type=int, default=None)
        parser.add_argument("--n-layers", type=int, default=None)
        parser.add_argument("--n-heads", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument("--predictor-hidden-dim", type=int, default=None)
        parser.add_argument("--context-pooling", type=str, choices=["last", "mean", "attention"], default=None)
        parser.add_argument("--vocab-size", type=int, default=None)
        parser.add_argument("--decode-aware-state-loss", dest="decode_aware_state_loss", action="store_true", default=None)
        parser.add_argument("--disable-decode-aware-state-loss", dest="decode_aware_state_loss", action="store_false")
        parser.add_argument("--decode-aware-density-loss", dest="decode_aware_density_loss", action="store_true", default=None)
        parser.add_argument("--disable-decode-aware-density-loss", dest="decode_aware_density_loss", action="store_false")
        parser.add_argument("--latent-loss-weight", type=float, default=None)
        parser.add_argument("--state-loss-weight", type=float, default=None)
        parser.add_argument("--density-loss-weight", type=float, default=None)
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
            "max_context_tokens": args.max_context_tokens,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "dropout": args.dropout,
            "predictor_hidden_dim": args.predictor_hidden_dim,
            "context_pooling": args.context_pooling,
            "vocab_size": args.vocab_size,
            "decode_aware_state_loss": args.decode_aware_state_loss,
            "decode_aware_density_loss": args.decode_aware_density_loss,
            "latent_loss_weight": args.latent_loss_weight,
            "state_loss_weight": args.state_loss_weight,
            "density_loss_weight": args.density_loss_weight,
        }
        result = RemiMotionTrainingPipeline(config, overrides=overrides).run(
            model_dir=args.model_dir,
            latent_dir=args.latent_dir,
            encoded_dir=args.encoded_dir,
        )
        print(f"Saved REMI motion model -> {result['model_path']}")
        print(f"Diagnostics -> {Path(args.model_dir) / 'remi_motion' / 'remi_motion_training_diagnostics.json'}")
        print(f"Best val loss -> {result['best_val_loss']:.6f}")


def main() -> None:
    RemiMotionTrainingCLI().run()


if __name__ == "__main__":
    main()
