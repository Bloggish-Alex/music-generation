#!/usr/bin/env python3
"""CLI entrypoint for DVAE training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader
from pipeline.dvae_training_pipeline import DVAETrainingPipeline


class TrainingCLI:
    """Command-line adapter for the DVAE training pipeline."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI arguments."""
        parser = argparse.ArgumentParser(description="Train DVAE from symbolic music files.")
        parser.add_argument("--music-dir", type=Path, required=True)
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument(
            "--transpose-semitones",
            type=str,
            default=None,
            help="Comma-separated semitone shifts, for example: 0 or 0,1,2,3,4,5,6,7,8,9,10,11",
        )
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """Run the training pipeline."""
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        overrides = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "transpose_semitones": args.transpose_semitones,
        }
        result = DVAETrainingPipeline(config, overrides=overrides).run(args.music_dir, args.model_dir)
        print(f"Saved DVAE model -> {result.model_path}")
        print(f"Diagnostics -> {result.diagnostics_path}")
        print(f"Summary -> {result.summary_path}")


def main() -> None:
    """Run the training CLI."""
    TrainingCLI().run()


if __name__ == "__main__":
    main()
