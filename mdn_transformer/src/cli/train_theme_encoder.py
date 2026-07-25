#!/usr/bin/env python3
"""CLI entrypoint for contrastive Theme Encoder training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader
from pipeline.theme_encoder_training_pipeline import ThemeEncoderTrainingPipeline


class ThemeEncoderTrainingCLI:
    """Command-line adapter for offline Theme Encoder training."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI arguments."""
        parser = argparse.ArgumentParser(description="Train a contrastive Theme Encoder from exported DVAE latents.")
        parser.add_argument("--model-dir", type=Path, required=True, help="Directory for output theme encoder files.")
        parser.add_argument("--latent-dir", type=Path, default=None, help="Defaults to --model-dir/latent.")
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--theme-bars", type=int, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--temperature", type=float, default=None)
        parser.add_argument("--jitter-std", type=float, default=None)
        parser.add_argument("--time-mask-ratio", type=float, default=None)
        parser.add_argument("--validation-ratio", type=float, default=None)
        parser.add_argument("--device", type=str, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run training from CLI arguments."""
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        overrides = {
            "theme_bars": args.theme_bars,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "jitter_std": args.jitter_std,
            "time_mask_ratio": args.time_mask_ratio,
            "validation_ratio": args.validation_ratio,
            "device": args.device,
        }
        model_dir = Path(args.model_dir)
        latent_dir = Path(args.latent_dir) if args.latent_dir else model_dir / "latent"
        result = ThemeEncoderTrainingPipeline(config, overrides=overrides).run(latent_dir, model_dir)
        print(f"Saved Theme Encoder -> {result.model_path}")
        print(f"Embeddings -> {result.embeddings_path}")
        print(f"Diagnostics -> {result.diagnostics_path}")
        print(f"Report -> {result.report_path}")


def main() -> None:
    """Run CLI."""
    ThemeEncoderTrainingCLI().run()


if __name__ == "__main__":
    main()
