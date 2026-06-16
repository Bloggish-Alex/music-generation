#!/usr/bin/env python3
"""Full training CLI for the symbolic music engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from common.config_loader import ConfigLoader
from decoder.candidate_selector import CandidateSelectorTrainer
from diagnostics.diagnostics import TrainingDiagnostics
from encoder.encoder import SymbolEncoderFactory
from decoder.lstm_token_model import LSTMDecoderConfig, LSTMDecoderTrainer
from decoder.sequence_dataset import DecoderSequenceDatasetBuilder, WindowTensorBuilder
from decoder.form_template import FormTemplateLibrary, TemplateFormModel
from common.model_store import ModelBundle, ObservationBarPoolBuilder
from data.music_input import InputParser


class TrainingPipeline:
    """End-to-end training pipeline with stage diagnostics."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.diagnostics = TrainingDiagnostics()

    def run(self, music_dir: Path, model_dir: Path) -> ModelBundle:
        model_dir.mkdir(parents=True, exist_ok=True)
        parser = InputParser.from_style_config(self.config)
        songs = parser.parse_directory(music_dir)
        bars = [bar for song in songs for bar in song.bars]
        self.diagnostics.record_input_summary(len(songs), parser.failed_files, len(bars))

        encoder = SymbolEncoderFactory().from_config(self.config)
        encoding = encoder.fit(songs)
        vocab = encoding.observation_vocab
        artifact_diagnostics = encoder.save_artifacts(model_dir)

        self.diagnostics.record_encoder(encoding.diagnostics)
        self.diagnostics.record_stage("encoder_artifacts", artifact_diagnostics)
        self.diagnostics.record_observation_vocab(encoding.diagnostics.get("observation_vocab", {}))

        selector_trainer = CandidateSelectorTrainer(
            self.config,
            mode=str(self.config.get("harmonic_engine", {}).get("mode", "major")),
        )
        candidate_selector_model = selector_trainer.fit(encoding.global_codebook)
        self.diagnostics.record_stage("candidate_selector", selector_trainer.diagnostics)

        template_library = FormTemplateLibrary.from_style_config(self.config)
        form_models = {
            name: TemplateFormModel.from_template(template, len(vocab.composite_to_observation))
            for name, template in template_library.templates.items()
        }
        self.diagnostics.record_stage("form_templates", {
            "model_type": "template_form_model",
            "learned_sequence_model": False,
            "forms": sorted(form_models),
        })

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
            "encoder_artifacts": artifact_diagnostics,
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
            for name, template in template_library.templates.items()
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
            candidate_selector_model=candidate_selector_model,
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
        self._train_lstm_decoder(bundle, model_dir)
        return bundle

    def _train_lstm_decoder(self, bundle: ModelBundle, model_dir: Path) -> None:
        decoder_config = LSTMDecoderConfig.from_config(self.config)
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
        trainer.save(model_dir, metadata)
        report = {
            "dataset": summary.to_dict(),
            "validation_metrics": metadata.validation_metrics,
            "training_log": metadata.training_log,
        }
        (model_dir / "lstm_decoder_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        self.diagnostics.record_stage("lstm_decoder", report)


class TrainingCLI:
    """CLI for model training."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Train symbolic music generation model.")
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
