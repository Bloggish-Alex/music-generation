#!/usr/bin/env python3
"""Evaluate a trained LSTM decoder on a model bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from common.model_store import ModelBundle
from decoder.lstm_token_model import LSTMDecoderTrainer, LSTMTokenModel
from decoder.sequence_dataset import DecoderSequenceDatasetBuilder, WindowTensorBuilder


class LSTMDecoderAnalysisCLI:
    """CLI for compact LSTM decoder diagnostics."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Analyze a trained LSTM decoder.")
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--device", type=str, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        bundle = ModelBundle.load(args.model_dir)
        model = LSTMTokenModel.load(args.model_dir, device=args.device)
        _, windows, summary = DecoderSequenceDatasetBuilder(
            model.metadata.config.context_size,
            condition_on_hidden_state=model.metadata.config.condition_on_hidden_state,
        ).build(bundle)
        inputs, targets = WindowTensorBuilder().build(
            windows,
            hidden_state_count=summary.hidden_state_count,
            condition_on_hidden_state=summary.condition_on_hidden_state,
        )
        target_states = WindowTensorBuilder().target_states(windows)
        trainer = LSTMDecoderTrainer(model.metadata.config)
        metrics = trainer.evaluate(inputs, targets, model=model.model, target_states=target_states)
        report = {
            "dataset": summary.to_dict(),
            "model_metadata": model.metadata.to_dict(),
            "evaluation_metrics": metrics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"LSTM decoder analysis -> {args.output}")
        print(f"top1={metrics.get('top1_accuracy')} top5={metrics.get('top5_accuracy')} perplexity={metrics.get('perplexity')}")


def main() -> None:
    LSTMDecoderAnalysisCLI().run()


if __name__ == "__main__":
    main()
