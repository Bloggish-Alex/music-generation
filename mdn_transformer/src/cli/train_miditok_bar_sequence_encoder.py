#!/usr/bin/env python3
"""CLI for training MidiTok-style bar sequence encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.miditok_bar_sequence_training_pipeline import MidiTokBarSequenceTrainingPipeline


class MidiTokBarSequenceTrainingCLI:
    """Command-line adapter."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build parser."""
        parser = argparse.ArgumentParser(description="Train MidiTok-style bar event sequence encoder.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--latent-dir", type=Path, default=None)
        parser.add_argument("--encoded-dir", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--validation-ratio", type=float, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--max-events", type=int, default=None)
        parser.add_argument("--d-model", type=int, default=None)
        parser.add_argument("--n-layers", type=int, default=None)
        parser.add_argument("--n-heads", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument("--cosine-loss-weight", type=float, default=None)
        parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row limit. Do not use for final training.")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run training."""
        args = self.build_parser().parse_args(argv)
        config = self._load_config(args.config)
        self._apply_model_overrides(config, args)
        overrides = {
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "validation_ratio": args.validation_ratio,
            "random_seed": args.seed,
            "max_rows": args.max_rows,
        }
        diagnostics = MidiTokBarSequenceTrainingPipeline(config, overrides=overrides).run(
            model_dir=args.model_dir,
            latent_dir=args.latent_dir,
            encoded_dir=args.encoded_dir,
        )
        print(f"MidiTok sequence encoder -> {diagnostics['checkpoint']}")
        print(f"Diagnostics -> {Path(args.model_dir) / 'miditok_bar_sequence_encoder_diagnostics.json'}")
        print(f"Embeddings -> {diagnostics['embedding_export'].get('embedding_path')}")

    def _load_config(self, path: Optional[Path]) -> Dict[str, Any]:
        """Load YAML config when available."""
        if path is None:
            return {}
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _apply_model_overrides(self, config: Dict[str, Any], args: argparse.Namespace) -> None:
        """Apply model-only CLI overrides into config dict."""
        section = config.setdefault("miditok_bar_sequence_encoder", {})
        mapping = {
            "max_events": args.max_events,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "dropout": args.dropout,
            "cosine_loss_weight": args.cosine_loss_weight,
        }
        for key, value in mapping.items():
            if value is not None:
                section[key] = value


def main() -> None:
    """Run CLI."""
    MidiTokBarSequenceTrainingCLI().run()


if __name__ == "__main__":
    main()
