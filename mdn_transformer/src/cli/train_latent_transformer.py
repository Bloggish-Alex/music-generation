#!/usr/bin/env python3
"""CLI entrypoint for Latent-Transformer + MDN training."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader
from pipeline.anchor_motion_composer_pipeline import AnchorMotionComposerTrainingPipeline
from pipeline.latent_transformer_training_pipeline import LatentTransformerTrainingPipeline
from pipeline.theme_encoder_training_pipeline import ThemeEncoderTrainingPipeline


class LatentTransformerTrainingCLI:
    """Command-line adapter for latent Transformer training."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI arguments."""
        parser = argparse.ArgumentParser(description="Train latent sequence models from exported DVAE latents.")
        parser.add_argument("--model-dir", type=Path, required=True, help="Main model directory. Internal subdirectories are managed by training mode.")
        parser.add_argument("--latent-dir", type=Path, default=None, help="Advanced override. Defaults to --model-dir/latent.")
        parser.add_argument("--training-mode", type=str, default="direct", help="direct / theme_fusion / anchor_motion_composer.")
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--context-bars", type=int, default=None)
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--validation-fold-count", type=int, default=None)
        parser.add_argument("--validation-fold-index", type=int, default=None)
        parser.add_argument("--validation-split-unit", type=str, default=None)
        parser.add_argument("--early-stopping-patience", type=int, default=None)
        parser.add_argument("--early-stopping-min-delta", type=float, default=None)
        parser.add_argument("--pi-entropy-weight", type=float, default=None)
        parser.add_argument("--enable-theme-fusion", action="store_true", help="Compatibility alias for --training-mode theme_fusion.")
        parser.add_argument("--theme-encoder-path", type=Path, default=None, help="Advanced override. Defaults to --model-dir/theme_encoder/theme_encoder.pt.")
        parser.add_argument("--theme-bars", type=int, default=None)
        parser.add_argument("--theme-fusion-mode", type=str, default=None)
        parser.add_argument("--theme-fusion-target", type=str, default=None)
        parser.add_argument("--theme-project-dim", type=int, default=None)
        parser.add_argument("--theme-gate-init", type=float, default=None)
        parser.add_argument("--theme-dropout", type=float, default=None)
        parser.add_argument("--theme-embedding-noise-std", type=float, default=None)
        parser.add_argument("--theme-token-bars", type=int, default=None)
        parser.add_argument("--theme-cross-attention-heads", type=int, default=None)
        parser.add_argument("--composer-hidden-dim", type=int, default=None)
        parser.add_argument("--composer-layers", type=int, default=None)
        parser.add_argument("--composer-model-hidden-dim", type=int, default=None)
        parser.add_argument("--composer-model-layers", type=int, default=None)
        parser.add_argument("--composer-dropout", type=float, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run training from CLI arguments."""
        args = self.build_parser().parse_args(argv)
        base_config = ConfigLoader().load(args.config)
        overrides = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "validation_fold_count": args.validation_fold_count,
            "validation_fold_index": args.validation_fold_index,
            "validation_split_unit": args.validation_split_unit,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
        }
        model_dir = Path(args.model_dir)
        latent_dir = Path(args.latent_dir) if args.latent_dir else model_dir / "latent"
        training_mode = self._training_mode(args)
        if training_mode == "theme_fusion":
            result = self._run_theme_fusion(args, base_config, overrides, latent_dir, model_dir)
        elif training_mode == "direct":
            config = copy.deepcopy(base_config)
            self._apply_common_config_overrides(config, args)
            result = LatentTransformerTrainingPipeline(config, overrides=overrides).run(latent_dir, model_dir)
        elif training_mode == "anchor_motion_composer":
            config = copy.deepcopy(base_config)
            self._apply_common_config_overrides(config, args)
            self._apply_anchor_motion_overrides(config, args)
            result = AnchorMotionComposerTrainingPipeline(config, overrides=overrides).run(latent_dir, model_dir)
        else:
            raise ValueError(f"Unsupported --training-mode: {args.training_mode}")
        print(f"Saved latent sequence model -> {result.model_path}")
        print(f"Diagnostics -> {result.diagnostics_path}")
        print(f"Summary -> {result.summary_path}")

    def _run_theme_fusion(
        self,
        args: argparse.Namespace,
        base_config: dict,
        overrides: dict,
        latent_dir: Path,
        model_dir: Path,
    ):
        """Train theme encoder and theme-fusion latent transformer under the main model directory."""
        theme_encoder_dir = model_dir / "theme_encoder"
        theme_encoder_path = Path(args.theme_encoder_path) if args.theme_encoder_path else theme_encoder_dir / "theme_encoder.pt"
        if args.theme_encoder_path is None:
            theme_config = copy.deepcopy(base_config)
            theme_overrides = {
                "theme_bars": args.theme_bars if args.theme_bars is not None else args.theme_token_bars,
                "device": args.device,
            }
            theme_result = ThemeEncoderTrainingPipeline(theme_config, overrides=theme_overrides).run(latent_dir, theme_encoder_dir)
            theme_encoder_path = theme_result.model_path
            print(f"Saved Theme Encoder -> {theme_result.model_path}")
            print(f"Theme diagnostics -> {theme_result.diagnostics_path}")
        config = copy.deepcopy(base_config)
        self._apply_common_config_overrides(config, args)
        config.setdefault("theme_fusion", {})["enabled"] = True
        config.setdefault("theme_fusion", {})["theme_encoder_path"] = str(theme_encoder_path)
        theme_fusion_dir = model_dir / "fold_1_theme"
        return LatentTransformerTrainingPipeline(config, overrides=overrides).run(latent_dir, theme_fusion_dir)

    def _apply_common_config_overrides(self, config: dict, args: argparse.Namespace) -> None:
        """Apply shared CLI overrides to the loaded config."""
        if args.pi_entropy_weight is not None:
            config.setdefault("mdn_head", {})["pi_entropy_weight"] = float(args.pi_entropy_weight)
        if args.context_bars is not None:
            config.setdefault("latent_transformer", {})["context_bars"] = int(args.context_bars)
        if args.theme_fusion_mode is not None:
            config.setdefault("theme_fusion", {})["mode"] = str(args.theme_fusion_mode)
        if args.theme_fusion_target is not None:
            config.setdefault("theme_fusion", {})["target"] = str(args.theme_fusion_target)
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

    def _apply_anchor_motion_overrides(self, config: dict, args: argparse.Namespace) -> None:
        """Apply Anchor/Motion Composer-specific CLI overrides."""
        if args.context_bars is not None:
            config.setdefault("anchor_motion_composer", {})["context_bars"] = int(args.context_bars)
        if args.composer_model_hidden_dim is not None:
            config.setdefault("anchor_motion_composer", {})["hidden_dim"] = int(args.composer_model_hidden_dim)
        if args.composer_model_layers is not None:
            config.setdefault("anchor_motion_composer", {})["n_layers"] = int(args.composer_model_layers)
        if args.composer_dropout is not None:
            config.setdefault("anchor_motion_composer", {})["dropout"] = float(args.composer_dropout)
        if args.composer_hidden_dim is not None:
            config.setdefault("anchor_motion_composer", {})["composer_hidden_dim"] = int(args.composer_hidden_dim)
        if args.composer_layers is not None:
            config.setdefault("anchor_motion_composer", {})["composer_layers"] = int(args.composer_layers)

    @staticmethod
    def _training_mode(args: argparse.Namespace) -> str:
        """Resolve training mode from the new option and the compatibility flag."""
        if bool(args.enable_theme_fusion):
            return "theme_fusion"
        return str(args.training_mode).strip().lower().replace("-", "_")


def main() -> None:
    """Run CLI."""
    LatentTransformerTrainingCLI().run()


if __name__ == "__main__":
    main()
