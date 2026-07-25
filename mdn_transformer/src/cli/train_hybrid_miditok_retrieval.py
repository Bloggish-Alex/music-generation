#!/usr/bin/env python3
"""CLI for training hybrid latent + MidiTok retrieval."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader
from model.hybrid_miditok_retrieval import HybridMidiTokRetrievalConfig
from model.miditok_bar_sequence_encoder import MidiTokBarSequenceEncoderConfig
from pipeline.hybrid_miditok_retrieval_pipeline import (
    HybridMidiTokTrainingConfig,
    HybridMidiTokTrainingPipeline,
)


class HybridMidiTokRetrievalTrainingCLI:
    """Train hybrid retrieval from exported latent and encoded bar tensors."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI parser."""
        parser = argparse.ArgumentParser(description="Train hybrid latent + MidiTok next-bar retrieval.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--latent-dir", type=Path, default=None)
        parser.add_argument("--encoded-dir", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--context-bars", type=int, default=None)
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--weight-decay", type=float, default=None)
        parser.add_argument("--validation-ratio", type=float, default=None)
        parser.add_argument("--early-stopping-patience", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--max-rows", type=int, default=None, help="Debug only. Limit latent rows for a smoke run.")
        parser.add_argument("--d-model", type=int, default=None)
        parser.add_argument("--n-layers", type=int, default=None)
        parser.add_argument("--n-heads", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument("--retrieval-dim", type=int, default=None)
        parser.add_argument("--temperature", type=float, default=None)
        parser.add_argument("--max-events", type=int, default=None)
        parser.add_argument("--event-d-model", type=int, default=None)
        parser.add_argument("--event-layers", type=int, default=None)
        parser.add_argument("--event-heads", type=int, default=None)
        parser.add_argument("--field-embedding-dim", type=int, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run training."""
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        model_config = self._model_config(config, args)
        event_config = self._event_config(config, args, latent_dim=int(model_config.latent_dim))
        training_config = self._training_config(args)
        result = HybridMidiTokTrainingPipeline(model_config, event_config, training_config).run(
            model_dir=args.model_dir,
            latent_dir=args.latent_dir,
            encoded_dir=args.encoded_dir,
        )
        print(f"Saved hybrid MidiTok retrieval model -> {result.model_path}")
        print(f"Diagnostics -> {result.diagnostics_path}")
        print(f"Summary -> {result.summary_path}")

    def _model_config(self, config: dict, args: argparse.Namespace) -> HybridMidiTokRetrievalConfig:
        """Build model config with CLI overrides."""
        value = HybridMidiTokRetrievalConfig.from_config(config)
        updates = {}
        for arg_name, field_name in (
            ("context_bars", "context_bars"),
            ("d_model", "d_model"),
            ("n_layers", "n_layers"),
            ("n_heads", "n_heads"),
            ("dropout", "dropout"),
            ("retrieval_dim", "retrieval_dim"),
            ("temperature", "temperature"),
        ):
            override = getattr(args, arg_name)
            if override is not None:
                updates[field_name] = override
        return replace(value, **updates)

    def _event_config(self, config: dict, args: argparse.Namespace, latent_dim: int) -> MidiTokBarSequenceEncoderConfig:
        """Build event encoder config with CLI overrides."""
        value = MidiTokBarSequenceEncoderConfig.from_config(config)
        updates = {"latent_dim": int(latent_dim)}
        for arg_name, field_name in (
            ("max_events", "max_events"),
            ("event_d_model", "d_model"),
            ("event_layers", "n_layers"),
            ("event_heads", "n_heads"),
            ("dropout", "dropout"),
            ("field_embedding_dim", "field_embedding_dim"),
        ):
            override = getattr(args, arg_name)
            if override is not None:
                updates[field_name] = override
        return replace(value, **updates)

    def _training_config(self, args: argparse.Namespace) -> HybridMidiTokTrainingConfig:
        """Build training config."""
        value = HybridMidiTokTrainingConfig()
        updates = {}
        for arg_name, field_name in (
            ("epochs", "epochs"),
            ("batch_size", "batch_size"),
            ("learning_rate", "learning_rate"),
            ("weight_decay", "weight_decay"),
            ("validation_ratio", "validation_ratio"),
            ("early_stopping_patience", "early_stopping_patience"),
            ("device", "device"),
            ("seed", "random_seed"),
            ("max_rows", "max_rows"),
        ):
            override = getattr(args, arg_name)
            if override is not None:
                updates[field_name] = override
        return replace(value, **updates)


def main() -> None:
    """Run CLI."""
    HybridMidiTokRetrievalTrainingCLI().run()


if __name__ == "__main__":
    main()
