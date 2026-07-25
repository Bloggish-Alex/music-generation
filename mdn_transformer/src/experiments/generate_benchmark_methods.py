#!/usr/bin/env python3
"""Generate MIDI continuations from representation benchmark methods."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.dvae_midi_render import DVAEMidiRenderConfig
from experiments.representation_benchmark import (
    BenchmarkConfig,
    RepresentationBenchmark,
    SequenceDataset,
    TransformerAnchorMotionComposer,
    TransformerTransitionComposer,
)
from model.dvae import DVAEMusicConfig, DenoisingMusicVAE
from pipeline.latent_generation_pipeline import SequenceTensorMidiRenderer


@dataclass(frozen=True)
class BenchmarkGenerationConfig:
    """Configuration for benchmark-method MIDI generation."""

    model_dir: Path
    latent_dir: Path
    output_dir: Path
    methods: tuple[str, ...] = ("hybrid_transition_composer", "hybrid_anchor_motion_composer")
    bars: int = 32
    primer_bars: int = 8
    context_bars: int = 8
    epochs: int = 60
    batch_size: int = 256
    hidden_dim: int = 256
    composer_hidden_dim: Optional[int] = None
    composer_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    validation_fold_count: int = 5
    validation_fold_index: int = 0
    validation_every: int = 0
    max_songs: Optional[int] = None
    max_rows: Optional[int] = None
    seed_song_id: Optional[str] = None
    random_seed: int = 42
    device: str = "cpu"
    tempo_bpm: int = 120
    base_pitch: int = 60


class BenchmarkMethodMidiGenerator:
    """Train selected benchmark methods and render generated latent continuations."""

    def __init__(self, config: BenchmarkGenerationConfig) -> None:
        self.config = config
        self._set_seed()

    def run(self) -> Dict[str, Any]:
        """Generate all requested methods and write MIDI, tensors, and diagnostics."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        bench_config = self._benchmark_config()
        benchmark = RepresentationBenchmark(bench_config)
        mu, rows, latent_summary = benchmark._load_latent()
        selected_indices = benchmark._select_rows(rows)
        selected_rows = [rows[index] for index in selected_indices]
        groups = benchmark._group_selected_rows(selected_rows)
        samples = benchmark._build_samples(groups, selected_rows)
        train_samples, val_samples, split_diag = benchmark._split_samples(samples)
        seed_song_id = self._select_seed_song(groups)
        dvae = self._load_dvae(self.config.model_dir / "dvae.pt")

        report: Dict[str, Any] = {
            "config": self._config_dict(),
            "latent_summary": latent_summary,
            "split": split_diag,
            "seed_song_id": seed_song_id,
            "methods": {},
        }
        for method in self.config.methods:
            method_report = self._run_method(
                benchmark=benchmark,
                method=method,
                mu=mu,
                rows=rows,
                selected_indices=selected_indices,
                selected_rows=selected_rows,
                groups=groups,
                train_samples=train_samples,
                val_samples=val_samples,
                seed_song_id=seed_song_id,
                dvae=dvae,
            )
            report["methods"][method] = method_report
        report_path = self.config.output_dir / "benchmark_method_generation_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def _run_method(
        self,
        benchmark: RepresentationBenchmark,
        method: str,
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        selected_indices: Sequence[int],
        selected_rows: Sequence[Dict[str, Any]],
        groups: Dict[str, List[int]],
        train_samples: Sequence[Any],
        val_samples: Sequence[Any],
        seed_song_id: str,
        dvae: DenoisingMusicVAE,
    ) -> Dict[str, Any]:
        """Train one benchmark method and render a continuation."""
        dataset = benchmark._build_representation(method, mu, rows, selected_indices, selected_rows, groups)
        values = np.asarray(dataset.values, dtype=np.float32)
        train_targets = np.asarray([sample.target_index for sample in train_samples], dtype=np.int64)
        mean = values[train_targets].mean(axis=0).astype(np.float32)
        std = values[train_targets].std(axis=0).astype(np.float32)
        std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
        train_dataset = SequenceDataset(values, train_samples, self.config.context_bars, mean, std, context_mode=dataset.context_mode)
        val_dataset = SequenceDataset(values, val_samples, self.config.context_bars, mean, std, context_mode=dataset.context_mode)
        model = self._build_model(method, int(values.shape[1]), dataset.context_mode)
        setattr(model, "benchmark_task_mode", str(dataset.task_mode))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        loss_fn = nn.MSELoss()
        history: List[Dict[str, float]] = []
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val = math.inf
        for epoch in range(int(self.config.epochs)):
            train_metrics = benchmark._run_epoch(model, train_dataset, optimizer, loss_fn)
            row = {
                "epoch": float(epoch + 1),
                "train_loss": float(train_metrics["mse"]),
            }
            run_validation = int(self.config.validation_every) > 0 and (
                (epoch + 1) % int(self.config.validation_every) == 0
                or (epoch + 1) == int(self.config.epochs)
            )
            if run_validation:
                val_metrics = benchmark._run_epoch(model, val_dataset, None, loss_fn)
                row["val_loss"] = float(val_metrics["mse"])
                if row["val_loss"] < best_val:
                    best_val = row["val_loss"]
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(row)
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        val_eval = benchmark._evaluate(model, val_dataset, values, mean, std, split_name="validation")
        generated_values = self._rollout(
            model=model,
            dataset=dataset,
            values=values,
            mean=mean,
            std=std,
            groups=groups,
            seed_song_id=seed_song_id,
        )
        latent_dim = int(mu.shape[1])
        generated_latent = generated_values[:, :latent_dim].astype(np.float32)
        tensors = self._decode_latents(dvae, generated_latent)
        midi_path = self.config.output_dir / f"{method}.mid"
        tensor_path = self.config.output_dir / f"{method}.bar_tensors.npz"
        np.savez_compressed(tensor_path, bars=tensors, latent_mu=generated_latent, representation_values=generated_values)
        midi_diag = SequenceTensorMidiRenderer(DVAEMidiRenderConfig(
            tempo_bpm=int(self.config.tempo_bpm),
            default_base_pitch=int(self.config.base_pitch),
        )).render(tensors, midi_path, base_pitch=int(self.config.base_pitch))
        method_report = {
            "method": str(method),
            "context_mode": str(dataset.context_mode),
            "task_mode": str(dataset.task_mode),
            "best_val_loss": float(best_val) if math.isfinite(best_val) else None,
            "history_tail": history[-5:],
            "val_eval": val_eval,
            "midi_path": str(midi_path),
            "tensor_path": str(tensor_path),
            "midi": midi_diag,
        }
        (self.config.output_dir / f"{method}.generation_diagnostics.json").write_text(
            json.dumps(method_report, indent=2),
            encoding="utf-8",
        )
        return method_report

    def _rollout(
        self,
        model: nn.Module,
        dataset: Any,
        values: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        groups: Dict[str, List[int]],
        seed_song_id: str,
    ) -> np.ndarray:
        """Roll out a representation sequence using a fixed primer from one song."""
        ordered = groups[seed_song_id]
        primer_count = min(int(self.config.primer_bars), len(ordered), int(self.config.bars))
        if primer_count < 1:
            raise ValueError("Need at least one primer bar to generate.")
        generated = [np.asarray(values[index], dtype=np.float32).copy() for index in ordered[:primer_count]]
        while len(generated) < int(self.config.bars):
            features = self._context_features(generated, mean, std, str(dataset.context_mode))
            context = torch.from_numpy(features[None, :]).float().to(self.config.device)
            current = torch.from_numpy(((generated[-1] - mean) / std)[None, :]).float().to(self.config.device)
            with torch.no_grad():
                if isinstance(model, (TransformerTransitionComposer, TransformerAnchorMotionComposer)):
                    output = model(context, current)
                else:
                    output = model(context)
                if isinstance(output, dict):
                    prediction = output["composed"]
                else:
                    prediction = output
            next_value = prediction.detach().cpu().numpy()[0].astype(np.float32) * std + mean
            generated.append(next_value.astype(np.float32))
        return np.stack(generated, axis=0).astype(np.float32)

    def _context_features(self, values: Sequence[np.ndarray], mean: np.ndarray, std: np.ndarray, context_mode: str) -> np.ndarray:
        """Build one normalized context feature vector for benchmark models."""
        dim = int(len(values[-1]))
        context = np.zeros((int(self.config.context_bars), dim), dtype=np.float32)
        mask = np.ones((int(self.config.context_bars), 1), dtype=np.float32)
        valid = np.zeros((int(self.config.context_bars),), dtype=bool)
        recent = list(values)[-int(self.config.context_bars):]
        offset = int(self.config.context_bars) - len(recent)
        for local, value in enumerate(recent):
            slot = offset + local
            context[slot] = (np.asarray(value, dtype=np.float32) - mean) / std
            mask[slot, 0] = 0.0
            valid[slot] = True
        if context_mode == "state":
            return np.concatenate([context.reshape(-1), mask.reshape(-1)], axis=0)
        delta = np.zeros_like(context, dtype=np.float32)
        delta_mask = np.ones((int(self.config.context_bars), 1), dtype=np.float32)
        for slot in range(1, int(self.config.context_bars)):
            if bool(valid[slot]) and bool(valid[slot - 1]):
                delta[slot] = context[slot] - context[slot - 1]
                delta_mask[slot, 0] = 0.0
        if context_mode == "state_delta_steps":
            return np.concatenate([context, delta, mask, delta_mask], axis=1).reshape(-1).astype(np.float32)
        return np.concatenate([context.reshape(-1), delta.reshape(-1), mask.reshape(-1), delta_mask.reshape(-1)], axis=0).astype(np.float32)

    def _decode_latents(self, model: DenoisingMusicVAE, latents: np.ndarray) -> np.ndarray:
        """Decode latent means into renderable bar tensors."""
        tensors: List[np.ndarray] = []
        batch_size = int(self.config.batch_size)
        with torch.no_grad():
            for start in range(0, int(latents.shape[0]), batch_size):
                batch = torch.from_numpy(latents[start:start + batch_size]).float().to(self.config.device)
                pitch, state_logits, velocity, chord = model.decoder(batch)
                state = torch.argmax(state_logits, dim=-1)
                state_one_hot = torch.nn.functional.one_hot(state, num_classes=3).float()
                tensor = torch.zeros(
                    (batch.shape[0], int(model.config.tracks), int(model.config.steps_per_bar), int(model.config.feature_dim)),
                    device=batch.device,
                    dtype=torch.float32,
                )
                tensor[..., 0] = pitch
                tensor[..., 1:4] = state_one_hot
                tensor[..., 4] = velocity
                tensor[..., 5:5 + chord.shape[-1]] = chord
                tensors.append(tensor.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(tensors, axis=0).astype(np.float32)

    def _build_model(self, method: str, representation_dim: int, context_mode: str) -> nn.Module:
        """Create the predictor model for a supported benchmark method."""
        if method == "hybrid_transition_composer":
            return TransformerTransitionComposer(
                representation_dim=representation_dim,
                context_bars=int(self.config.context_bars),
                hidden_dim=int(self.config.hidden_dim),
                composer_hidden_dim=self.config.composer_hidden_dim,
                composer_layers=int(self.config.composer_layers),
                dropout=float(self.config.dropout),
            ).to(self.config.device)
        if method == "hybrid_anchor_motion_composer":
            return TransformerAnchorMotionComposer(
                representation_dim=representation_dim,
                context_bars=int(self.config.context_bars),
                hidden_dim=int(self.config.hidden_dim),
                composer_hidden_dim=self.config.composer_hidden_dim,
                composer_layers=int(self.config.composer_layers),
                dropout=float(self.config.dropout),
            ).to(self.config.device)
        raise ValueError(f"Unsupported generation benchmark method: {method}")

    def _load_dvae(self, path: Path) -> DenoisingMusicVAE:
        """Load the trained DVAE decoder."""
        checkpoint = torch.load(path, map_location=self.config.device)
        config = DVAEMusicConfig(**checkpoint["config"])
        model = DenoisingMusicVAE(config).to(self.config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def _select_seed_song(self, groups: Dict[str, List[int]]) -> str:
        """Choose a deterministic seed song with enough bars."""
        if self.config.seed_song_id:
            seed = str(self.config.seed_song_id)
            if seed not in groups:
                raise ValueError(f"Unknown seed_song_id: {seed}")
            return seed
        candidates = [song_id for song_id, indices in groups.items() if len(indices) >= int(self.config.primer_bars)]
        if not candidates:
            raise ValueError("No song has enough bars for the requested primer.")
        return sorted(candidates)[0]

    def _benchmark_config(self) -> BenchmarkConfig:
        """Return the matching benchmark config used for data prep and training."""
        return BenchmarkConfig(
            model_dir=self.config.model_dir,
            latent_dir=self.config.latent_dir,
            output_dir=self.config.output_dir,
            representations=self.config.methods,
            context_bars=int(self.config.context_bars),
            epochs=int(self.config.epochs),
            batch_size=int(self.config.batch_size),
            hidden_dim=int(self.config.hidden_dim),
            composer_hidden_dim=self.config.composer_hidden_dim,
            composer_layers=int(self.config.composer_layers),
            dropout=float(self.config.dropout),
            learning_rate=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
            validation_fold_count=int(self.config.validation_fold_count),
            validation_fold_index=int(self.config.validation_fold_index),
            max_songs=self.config.max_songs,
            max_rows=self.config.max_rows,
            random_seed=int(self.config.random_seed),
            device=str(self.config.device),
        )

    def _set_seed(self) -> None:
        """Seed Python, numpy, and torch."""
        seed = int(self.config.random_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _config_dict(self) -> Dict[str, Any]:
        """Return JSON-safe config."""
        result = dict(self.config.__dict__)
        for key in ("model_dir", "latent_dir", "output_dir"):
            result[key] = str(result[key])
        result["methods"] = list(self.config.methods)
        return result


def parse_args(argv: Optional[Sequence[str]] = None) -> BenchmarkGenerationConfig:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate MIDI from benchmark transition-composer methods.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="hybrid_transition_composer,hybrid_anchor_motion_composer")
    parser.add_argument("--bars", type=int, default=32)
    parser.add_argument("--primer-bars", type=int, default=8)
    parser.add_argument("--context-bars", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--composer-hidden-dim", type=int, default=None, help="Defaults to --hidden-dim.")
    parser.add_argument("--composer-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-fold-count", type=int, default=5)
    parser.add_argument("--validation-fold-index", type=int, default=0)
    parser.add_argument("--validation-every", type=int, default=0, help="Run validation every N epochs during training. Default 0 only evaluates after training.")
    parser.add_argument("--max-songs", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed-song-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tempo", type=int, default=120)
    parser.add_argument("--base-pitch", type=int, default=60)
    args = parser.parse_args(argv)
    model_dir = Path(args.model_dir)
    return BenchmarkGenerationConfig(
        model_dir=model_dir,
        latent_dir=Path(args.latent_dir) if args.latent_dir else model_dir / "latent",
        output_dir=Path(args.output_dir),
        methods=tuple(item.strip() for item in str(args.methods).split(",") if item.strip()),
        bars=int(args.bars),
        primer_bars=int(args.primer_bars),
        context_bars=int(args.context_bars),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        composer_hidden_dim=args.composer_hidden_dim,
        composer_layers=int(args.composer_layers),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        validation_fold_count=int(args.validation_fold_count),
        validation_fold_index=int(args.validation_fold_index),
        validation_every=int(args.validation_every),
        max_songs=args.max_songs,
        max_rows=args.max_rows,
        seed_song_id=args.seed_song_id,
        random_seed=int(args.seed),
        device=str(args.device),
        tempo_bpm=int(args.tempo),
        base_pitch=int(args.base_pitch),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run benchmark-method MIDI generation."""
    config = parse_args(argv)
    report = BenchmarkMethodMidiGenerator(config).run()
    print(f"Benchmark method generation complete -> {config.output_dir}")
    for method, info in report["methods"].items():
        print(f"{method}: {info['midi_path']}")


if __name__ == "__main__":
    main()
