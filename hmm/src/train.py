#!/usr/bin/env python3
"""Full training CLI for the DFA/HMM music engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from config_loader import ConfigLoader
from diagnostics import TrainingDiagnostics
from encoder import SymbolEncoderFactory
from form_hmm import FormHMMTrainer, FormTemplateLibrary
from model_store import ModelBundle, ObservationBarPoolBuilder
from music_input import InputParser


class TrainingPipeline:
    """End-to-end training pipeline with stage diagnostics."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.diagnostics = TrainingDiagnostics()

    def run(self, music_dir: Path, model_dir: Path) -> ModelBundle:
        parser = InputParser.from_style_config(self.config)
        songs = parser.parse_directory(music_dir)
        bars = [bar for song in songs for bar in song.bars]
        self.diagnostics.record_input_summary(len(songs), parser.failed_files, len(bars))

        encoding = SymbolEncoderFactory().from_config(self.config).fit(songs)
        vocab = encoding.observation_vocab

        self.diagnostics.record_clustering(encoding.diagnostics)
        self.diagnostics.record_observation_vocab(encoding.diagnostics.get("observation_vocab", {}))

        hmm_trainer = FormHMMTrainer(self.config)
        form_models = hmm_trainer.train(songs, vocab)

        for form_name, diag in hmm_trainer.diagnostics.items():
            self.diagnostics.record_hmm(form_name, diag)

        pool_builder = ObservationBarPoolBuilder()
        pool_diagnostics = pool_builder.diagnostics(pool_builder.build(songs), vocab)
        self.diagnostics.record_observation_bar_pools(pool_diagnostics)

        summary = {
            "music_dir": str(music_dir),
            "model_dir": str(model_dir),
            "song_count": len(songs),
            "bar_count": len(bars),
            "codebook_count": len(encoding.global_codebook),
            "observation_count": len(vocab.composite_to_observation),
            "symbol_count": len(vocab.composite_to_observation),
            "observation_bar_pools": pool_diagnostics,
            "forms": sorted(form_models),
        }
        templates = {
            name: {
                "sections": [
                    {
                        "name": section.name,
                        "length": section.length,
                        "source": section.source,
                        "pitch_offset": section.pitch_offset,
                        "cadence": section.cadence,
                        "start_degree": section.start_degree,
                    }
                    for section in template.sections
                ]
            }
            for name, template in FormTemplateLibrary.from_style_config(self.config).templates.items()
        }
        bundle = ModelBundle.from_training(
            self.config,
            songs,
            vocab,
            form_models,
            summary,
            form_templates=templates,
            global_codebook=encoding.global_codebook,
            encoder_model=encoding.encoder_model,
        )
        self.diagnostics.record_stage("encoder_model", {
            "codebook_count": len(bundle.encoder_model.codebook.entries) if bundle.encoder_model else 0,
            "symbol_count": (
                len(bundle.encoder_model.vocabulary.symbol_to_descriptor)
                if bundle.encoder_model else len(vocab.composite_to_observation)
            ),
            "symbol_id_field": "observation_id",
            "codebook_id_field": "codebook_id",
            "vocabulary_interface": "SymbolVocabulary",
        })
        return bundle


class TrainingCLI:
    """CLI for model training."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Train DFA/HMM music generation model.")
        parser.add_argument("--music-dir", type=Path, required=True)
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--config", type=Path, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        pipeline = TrainingPipeline(config)
        bundle = pipeline.run(args.music_dir, args.model_dir)
        bundle.save(args.model_dir)
        diagnostics_path = args.model_dir / "training_diagnostics.json"
        pipeline.diagnostics.write(diagnostics_path)
        summary_path = args.model_dir / "training_summary.json"
        summary_path.write_text(json.dumps(bundle.training_summary, indent=2), encoding="utf-8")
        print(f"Trained model -> {args.model_dir}")
        print(f"Diagnostics -> {diagnostics_path}")


def main() -> None:
    TrainingCLI().run()


if __name__ == "__main__":
    main()
