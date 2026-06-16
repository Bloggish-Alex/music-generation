#!/usr/bin/env python3
"""Train the standalone LSTM next-token decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from common.config_loader import ConfigLoader
from common.model_store import ModelBundle
from decoder.lstm_token_model import LSTMDecoderConfig, LSTMDecoderTrainer
from decoder.sequence_dataset import DecoderSequenceDatasetBuilder, WindowTensorBuilder


class LSTMDecoderTrainingCLI:
    """CLI for training the LSTM decoder without changing generation."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Train LSTM decoder from an existing model_bundle.json.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--context-size", type=int, default=None)
        parser.add_argument("--hidden-dim", type=int, default=None)
        parser.add_argument("--num-layers", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--weight-decay", type=float, default=None)
        parser.add_argument("--validation-ratio", type=float, default=None)
        parser.add_argument("--random-seed", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--condition-on-hidden-state", action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument("--state-embedding-dim", type=int, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        decoder_config = self._config_from_args(config, args)
        bundle = ModelBundle.load(args.model_dir)
        _, windows, summary = DecoderSequenceDatasetBuilder(
            decoder_config.context_size,
            condition_on_hidden_state=decoder_config.condition_on_hidden_state,
        ).build(bundle)
        inputs, targets = WindowTensorBuilder().build(
            windows,
            hidden_state_count=summary.hidden_state_count,
            condition_on_hidden_state=summary.condition_on_hidden_state,
        )
        target_states = WindowTensorBuilder().target_states(windows)
        trainer = LSTMDecoderTrainer(decoder_config)
        metadata = trainer.fit(
            inputs,
            targets,
            vocab_size=len(bundle.symbol_vocabulary.symbol_to_descriptor),
            dataset_summary=summary.to_dict(),
            target_states=target_states,
        )
        trainer.save(args.output_dir, metadata)
        (args.output_dir / "lstm_decoder_report.json").write_text(
            json.dumps({
                "dataset": summary.to_dict(),
                "validation_metrics": metadata.validation_metrics,
                "training_log": metadata.training_log,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"Trained LSTM decoder -> {args.output_dir}")
        print(f"Validation top1={metadata.validation_metrics.get('top1_accuracy')} top5={metadata.validation_metrics.get('top5_accuracy')}")

    def _config_from_args(self, config: dict, args: argparse.Namespace) -> LSTMDecoderConfig:
        base = LSTMDecoderConfig.from_config(config)
        values = {
            "context_size": args.context_size,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "validation_ratio": args.validation_ratio,
            "random_seed": args.random_seed,
            "device": args.device,
            "condition_on_hidden_state": args.condition_on_hidden_state,
            "state_embedding_dim": args.state_embedding_dim,
        }
        payload = {
            key: (value if value is not None else getattr(base, key))
            for key, value in values.items()
        }
        return LSTMDecoderConfig(**payload)


def main() -> None:
    LSTMDecoderTrainingCLI().run()


if __name__ == "__main__":
    main()
