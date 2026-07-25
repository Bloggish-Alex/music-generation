#!/usr/bin/env python3
"""CLI entrypoint for standard training stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config_loader import ConfigLoader
from pipeline.training_orchestrator import TrainingOrchestrator, TrainingOrchestratorConfig


class TrainingCLI:
    """Command-line adapter for standard training stages."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Build CLI arguments."""
        parser = argparse.ArgumentParser(description="Train MDN Transformer experiment stages.")
        parser.add_argument("--music-dir", type=Path, required=True)
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument(
            "--stage",
            type=str,
            default="dvae",
            help="dvae / latent / miditok / retrieval / base_pitch / remi_motion / remi_direct / all / best / comma-separated stages. Default keeps old DVAE-only behavior.",
        )
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--dvae-epochs", type=int, default=None)
        parser.add_argument("--miditok-epochs", type=int, default=None)
        parser.add_argument("--retrieval-epochs", type=int, default=None)
        parser.add_argument("--base-pitch-epochs", type=int, default=None)
        parser.add_argument("--remi-motion-epochs", type=int, default=None)
        parser.add_argument("--remi-motion-batch-size", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--max-rows", type=int, default=None, help="Debug only. Limit sequence-model rows.")
        parser.add_argument("--save-z", action="store_true", help="Also save sampled latent z during latent export.")
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
        options = TrainingOrchestratorConfig(
            stage=str(args.stage),
            device=args.device,
            max_rows=args.max_rows,
            dvae_epochs=args.dvae_epochs if args.dvae_epochs is not None else args.epochs,
            miditok_epochs=args.miditok_epochs,
            retrieval_epochs=args.retrieval_epochs,
            base_pitch_epochs=args.base_pitch_epochs,
            remi_motion_epochs=args.remi_motion_epochs,
            remi_motion_batch_size=args.remi_motion_batch_size,
            batch_size=args.batch_size,
            random_seed=args.seed,
            transpose_semitones=args.transpose_semitones,
            dvae_learning_rate=args.learning_rate,
            save_z=bool(args.save_z),
        )
        result = TrainingOrchestrator(config, options).run(args.music_dir, args.model_dir)
        print(f"Training manifest -> {result.manifest_path}")
        for stage in result.stages:
            print(f"Stage {stage.name}: {stage.status}")


def main() -> None:
    """Run the training CLI."""
    TrainingCLI().run()


if __name__ == "__main__":
    main()
