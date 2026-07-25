#!/usr/bin/env python3
"""CLI entrypoint for Memory Latent Transformer training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader
from pipeline.memory_latent_training_pipeline import MemoryLatentTrainingPipeline


class MemoryLatentTrainingCLI:
    """Command-line adapter for memory latent training."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI parser."""
        parser = argparse.ArgumentParser(description="Train Memory Latent Transformer from exported DVAE latents.")
        parser.add_argument("--model-dir", type=Path, required=True, help="Main model directory. Internal paths are resolved from this directory.")
        parser.add_argument("--latent-dir", type=Path, default=None, help="Advanced override. Defaults to --model-dir/latent.")
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--validation-fold-count", type=int, default=None)
        parser.add_argument("--validation-fold-index", type=int, default=None)
        parser.add_argument("--validation-split-unit", type=str, default=None)
        parser.add_argument("--contrastive-temperature", type=float, default=None)
        parser.add_argument("--early-stopping-patience", type=int, default=None)
        parser.add_argument("--early-stopping-min-delta", type=float, default=None)
        parser.add_argument("--enable-theme-fusion", action="store_true")
        parser.add_argument("--theme-encoder-path", type=Path, default=None, help="Advanced override. Defaults to --model-dir/theme_encoder/theme_encoder.pt when theme fusion is enabled.")
        parser.add_argument("--theme-fusion-mode", type=str, default=None)
        parser.add_argument("--theme-project-dim", type=int, default=None)
        parser.add_argument("--theme-gate-init", type=float, default=None)
        parser.add_argument("--theme-dropout", type=float, default=None)
        parser.add_argument("--theme-embedding-noise-std", type=float, default=None)
        parser.add_argument("--theme-token-bars", type=int, default=None)
        parser.add_argument("--theme-cross-attention-heads", type=int, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run training."""
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        overrides = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "validation_fold_count": args.validation_fold_count,
            "validation_fold_index": args.validation_fold_index,
            "validation_split_unit": args.validation_split_unit,
            "contrastive_temperature": args.contrastive_temperature,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
        }
        if args.enable_theme_fusion:
            config.setdefault("theme_fusion", {})["enabled"] = True
        if args.theme_encoder_path is not None:
            config.setdefault("theme_fusion", {})["theme_encoder_path"] = str(args.theme_encoder_path)
        if args.theme_fusion_mode is not None:
            config.setdefault("theme_fusion", {})["mode"] = str(args.theme_fusion_mode)
        if args.theme_project_dim is not None:
            config.setdefault("theme_fusion", {})["project_dim"] = int(args.theme_project_dim)
        if args.theme_gate_init is not None:
            config.setdefault("theme_fusion", {})["gate_init"] = float(args.theme_gate_init)
        if args.theme_dropout is not None:
            config.setdefault("theme_fusion", {})["theme_dropout"] = float(args.theme_dropout)
        if args.theme_embedding_noise_std is not None:
            config.setdefault("theme_fusion", {})["embedding_noise_std"] = float(args.theme_embedding_noise_std)
        if args.theme_token_bars is not None:
            config.setdefault("theme_fusion", {})["token_bars"] = int(args.theme_token_bars)
        if args.theme_cross_attention_heads is not None:
            config.setdefault("theme_fusion", {})["cross_attention_heads"] = int(args.theme_cross_attention_heads)
        model_dir = Path(args.model_dir)
        latent_dir = Path(args.latent_dir) if args.latent_dir else model_dir / "latent"
        if args.enable_theme_fusion and args.theme_encoder_path is None:
            config.setdefault("theme_fusion", {})["theme_encoder_path"] = str(model_dir / "theme_encoder" / "theme_encoder.pt")
        result = MemoryLatentTrainingPipeline(config, overrides=overrides).run(latent_dir, model_dir)
        print(f"Saved Memory Latent Transformer -> {result.model_path}")
        print(f"Diagnostics -> {result.diagnostics_path}")
        print(f"Summary -> {result.summary_path}")


def main() -> None:
    """Run CLI."""
    MemoryLatentTrainingCLI().run()


if __name__ == "__main__":
    main()
