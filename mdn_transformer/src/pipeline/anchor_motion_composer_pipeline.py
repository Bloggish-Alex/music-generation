#!/usr/bin/env python3
"""Training and generation pipeline for the Anchor/Motion Composer backend."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from codec.bar_feature_extractor import BAR_FEATURE_NAMES, BarFeatureExtractor, EncodedBarFeatureStore
from common.config_loader import ConfigView
from diagnostics.diagnostics import DiagnosticsBase
from diagnostics.dvae_midi_render import DVAEMidiRenderConfig
from model.dvae import DVAEMusicConfig, DenoisingMusicVAE
from model.latent_composer import AnchorMotionComposer, AnchorMotionComposerConfig
from pipeline.latent_generation_pipeline import GenerationActionPlanner, LatentGenerationConfig, LatentGenerationResult, SequenceTensorMidiRenderer
from pipeline.latent_transformer_training_pipeline import LatentDatasetReader


@dataclass(frozen=True)
class AnchorMotionComposerTrainingConfig:
    """Training hyperparameters for Anchor/Motion Composer."""

    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    validation_fold_count: int = 5
    validation_fold_index: int = 0
    random_seed: int = 42
    device: str = "cpu"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AnchorMotionComposerTrainingConfig":
        """Build training config from style config."""
        section = ConfigView(config).section("anchor_motion_composer_training")
        fallback = ConfigView(config).section("latent_transformer_training")
        return cls(
            epochs=int(section.get("epochs", fallback.get("epochs", 80))),
            batch_size=int(section.get("batch_size", 128)),
            learning_rate=float(section.get("learning_rate", fallback.get("learning_rate", 5.0e-4))),
            weight_decay=float(section.get("weight_decay", fallback.get("weight_decay", 1.0e-4))),
            validation_fold_count=int(section.get("validation_fold_count", fallback.get("validation_fold_count", 5))),
            validation_fold_index=int(section.get("validation_fold_index", fallback.get("validation_fold_index", 0))),
            random_seed=int(section.get("random_seed", fallback.get("random_seed", 42))),
            device=str(section.get("device", fallback.get("device", "cpu"))),
            early_stopping_patience=int(section.get("early_stopping_patience", fallback.get("early_stopping_patience", 5))),
            early_stopping_min_delta=float(section.get("early_stopping_min_delta", fallback.get("early_stopping_min_delta", 0.0))),
        )


@dataclass
class AnchorMotionSample:
    """One next-representation training sample."""

    context_indices: List[int]
    current_index: int
    target_index: int
    song_id: str
    base_song_id: str
    target_bar_index: int
    form_id: int = 0
    action_id: int = 0
    composer_id: int = 0
    position_id: int = 0


@dataclass
class AnchorMotionTrainingResult:
    """Paths produced by composer training."""

    model_path: Path
    diagnostics_path: Path
    summary_path: Path
    diagnostics: Dict[str, Any]


class AnchorMotionConditionVocab:
    """Stable categorical condition vocabularies for late fusion."""

    UNKNOWN = "UNKNOWN"

    def __init__(
        self,
        form_to_id: Dict[str, int],
        action_to_id: Dict[str, int],
        composer_to_id: Dict[str, int],
        position_vocab_size: int,
    ) -> None:
        self.form_to_id = dict(form_to_id)
        self.action_to_id = dict(action_to_id)
        self.composer_to_id = dict(composer_to_id)
        self.position_vocab_size = int(position_vocab_size)

    @classmethod
    def from_rows(cls, rows: Sequence[Dict[str, Any]], position_vocab_size: int) -> "AnchorMotionConditionVocab":
        """Build vocabularies from latent row metadata."""
        return cls(
            form_to_id=cls._vocab(cls._form_name(row) for row in rows),
            action_to_id=cls._vocab(cls._action_name(row) for row in rows),
            composer_to_id=cls._vocab(cls._composer_name(row) for row in rows),
            position_vocab_size=int(position_vocab_size),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict[str, Any]) -> "AnchorMotionConditionVocab":
        """Load vocabularies from checkpoint."""
        data = checkpoint.get("condition_vocab", {})
        return cls(
            form_to_id={str(k): int(v) for k, v in data.get("form_to_id", {cls.UNKNOWN: 0}).items()},
            action_to_id={str(k): int(v) for k, v in data.get("action_to_id", {cls.UNKNOWN: 0}).items()},
            composer_to_id={str(k): int(v) for k, v in data.get("composer_to_id", {cls.UNKNOWN: 0}).items()},
            position_vocab_size=int(data.get("position_vocab_size", 8)),
        )

    def ids_for_row(self, row: Dict[str, Any]) -> tuple[int, int, int, int]:
        """Return [form, action, composer, position] ids for a row."""
        return (
            self.form_id(self._form_name(row)),
            self.action_id(self._action_name(row)),
            self.composer_id(self._composer_name(row)),
            self.position_id(int(row.get("bar_index", 0))),
        )

    def ids_for_generation(self, form: str, action: str, composer: str, bar_index: int) -> tuple[int, int, int, int]:
        """Return ids for generation-time planned conditions."""
        return (
            self.form_id(form),
            self.action_id(action),
            self.composer_id(composer),
            self.position_id(int(bar_index)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe vocabularies."""
        return {
            "form_to_id": dict(self.form_to_id),
            "action_to_id": dict(self.action_to_id),
            "composer_to_id": dict(self.composer_to_id),
            "position_vocab_size": int(self.position_vocab_size),
        }

    def form_id(self, value: str) -> int:
        """Return form id."""
        return int(self.form_to_id.get(str(value), self.form_to_id.get(self.UNKNOWN, 0)))

    def action_id(self, value: str) -> int:
        """Return action id."""
        return int(self.action_to_id.get(str(value), self.action_to_id.get(self.UNKNOWN, 0)))

    def composer_id(self, value: str) -> int:
        """Return composer id."""
        return int(self.composer_to_id.get(str(value), self.composer_to_id.get(self.UNKNOWN, 0)))

    def position_id(self, bar_index: int) -> int:
        """Return period-local position id."""
        return int(bar_index) % max(1, int(self.position_vocab_size))

    @classmethod
    def _vocab(cls, values: Sequence[str]) -> Dict[str, int]:
        """Build deterministic vocabulary with UNKNOWN at 0."""
        ordered = [cls.UNKNOWN]
        seen = {cls.UNKNOWN}
        for value in sorted({str(item or cls.UNKNOWN) for item in values}):
            if value not in seen:
                ordered.append(value)
                seen.add(value)
        return {value: index for index, value in enumerate(ordered)}

    @classmethod
    def _form_name(cls, row: Dict[str, Any]) -> str:
        """Return normalized form name."""
        return str(row.get("form") or cls.UNKNOWN)

    @classmethod
    def _action_name(cls, row: Dict[str, Any]) -> str:
        """Return normalized action name."""
        return str(row.get("action") or cls.UNKNOWN)

    @classmethod
    def _composer_name(cls, row: Dict[str, Any]) -> str:
        """Return normalized composer name when metadata exists."""
        return str(row.get("composer") or row.get("composer_id") or row.get("artist") or cls.UNKNOWN)


class AnchorMotionDataset(Dataset):
    """Fixed-context dataset for composer training."""

    def __init__(
        self,
        representation: np.ndarray,
        samples: Sequence[AnchorMotionSample],
        context_bars: int,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.representation = np.asarray(representation, dtype=np.float32)
        self.samples = list(samples)
        self.context_bars = int(context_bars)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.dim = int(self.representation.shape[1])

    def __len__(self) -> int:
        """Return sample count."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return context steps, target, current representation, and condition IDs."""
        sample = self.samples[index]
        context_steps = self.context_steps(sample.context_indices)
        target = self.normalize(self.representation[sample.target_index])
        current = self.normalize(self.representation[sample.current_index])
        condition = np.asarray([sample.form_id, sample.action_id, sample.composer_id, sample.position_id], dtype=np.int64)
        return (
            torch.from_numpy(context_steps).float(),
            torch.from_numpy(target).float(),
            torch.from_numpy(current).float(),
            torch.from_numpy(condition).long(),
        )

    def normalize(self, value: np.ndarray) -> np.ndarray:
        """Normalize one representation vector."""
        return (np.asarray(value, dtype=np.float32) - self.mean) / self.std

    def context_steps(self, context_indices: Sequence[int]) -> np.ndarray:
        """Build [context_bars, 2 * dim + 2] state+delta context steps."""
        context = np.zeros((self.context_bars, self.dim), dtype=np.float32)
        mask = np.ones((self.context_bars, 1), dtype=np.float32)
        valid = np.zeros((self.context_bars,), dtype=bool)
        recent = list(context_indices)[-self.context_bars:]
        offset = self.context_bars - len(recent)
        for local, row_index in enumerate(recent):
            slot = offset + local
            context[slot] = self.normalize(self.representation[row_index])
            mask[slot, 0] = 0.0
            valid[slot] = True
        delta = np.zeros_like(context, dtype=np.float32)
        delta_mask = np.ones((self.context_bars, 1), dtype=np.float32)
        for slot in range(1, self.context_bars):
            if bool(valid[slot]) and bool(valid[slot - 1]):
                delta[slot] = context[slot] - context[slot - 1]
                delta_mask[slot, 0] = 0.0
        return np.concatenate([context, delta, mask, delta_mask], axis=1).astype(np.float32)


class AnchorMotionComposerTrainingPipeline:
    """Train the formal Anchor/Motion Composer checkpoint."""

    def __init__(self, config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> None:
        self.config = config
        self.training_config = self._with_overrides(AnchorMotionComposerTrainingConfig.from_config(config), overrides or {})
        self.model_config = AnchorMotionComposerConfig.from_config(config)
        self.reader = LatentDatasetReader()
        self.diagnostics = DiagnosticsBase("anchor_motion_composer_training")

    def run(self, latent_dir: str | Path, model_dir: str | Path) -> AnchorMotionTrainingResult:
        """Train and save composer checkpoints."""
        self._set_seed()
        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mu, rows, latent_summary = self.reader.load(latent_dir)
        features, feature_source = EncodedBarFeatureStore(output_dir / "encoded").matrix_for_rows(rows)
        condition_vocab = AnchorMotionConditionVocab.from_rows(rows, int(self.model_config.position_vocab_size))
        self.model_config = self._resolved_model_config(mu, features, condition_vocab)
        representation = np.concatenate([mu.astype(np.float32), features.astype(np.float32)], axis=1).astype(np.float32)
        samples = self._build_samples(rows, condition_vocab)
        train_samples, val_samples, split = self._split_samples(samples)
        if not train_samples or not val_samples:
            raise ValueError("Anchor/Motion Composer needs non-empty train and validation samples.")
        train_targets = np.asarray([sample.target_index for sample in train_samples], dtype=np.int64)
        mean = representation[train_targets].mean(axis=0).astype(np.float32)
        std = representation[train_targets].std(axis=0).astype(np.float32)
        std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
        train_dataset = AnchorMotionDataset(representation, train_samples, self.model_config.context_bars, mean, std)
        val_dataset = AnchorMotionDataset(representation, val_samples, self.model_config.context_bars, mean, std)
        model = AnchorMotionComposer(self.model_config).to(self.training_config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.training_config.learning_rate),
            weight_decay=float(self.training_config.weight_decay),
        )
        history, best_state, best_epoch, best_val = self._train(model, optimizer, train_dataset, val_dataset)
        final_path = output_dir / "anchor_motion_composer.final.pt"
        self._save_model(final_path, model, mean, std, condition_vocab, checkpoint_role="final")
        if best_state is not None:
            model.load_state_dict(best_state)
        model_path = output_dir / "anchor_motion_composer.pt"
        self._save_model(model_path, model, mean, std, condition_vocab, checkpoint_role="best")
        train_eval = self._evaluate(model, train_dataset)
        val_eval = self._evaluate(model, val_dataset)
        self.diagnostics.record_stage("input", {
            "latent_dir": str(latent_dir),
            "latent_summary": latent_summary,
            "row_count": int(len(rows)),
            "latent_shape": [int(value) for value in mu.shape],
            "feature_shape": [int(value) for value in features.shape],
            "representation_shape": [int(value) for value in representation.shape],
            "feature_names": list(BAR_FEATURE_NAMES),
            "feature_source": feature_source,
            "normalization": {
                "continuous_representation": "mean/std from training target rows",
                "condition_ids": "categorical ids consumed by learned embeddings",
            },
            "condition_vocab": condition_vocab.to_dict(),
        })
        self.diagnostics.record_stage("training", {
            "config": self.training_config.__dict__,
            "model_config": self.model_config.to_dict(),
            "split": split,
            "history": history,
            "model_selection": {
                "monitor": "val_mse",
                "best_epoch": int(best_epoch),
                "best_val_mse": float(best_val),
            },
            "saved_checkpoints": {
                "best": str(model_path),
                "final": str(final_path),
            },
        })
        self.diagnostics.record_stage("train_eval", train_eval)
        self.diagnostics.record_stage("val_eval", val_eval)
        diagnostics_path = output_dir / "anchor_motion_composer_training_diagnostics.json"
        summary_path = output_dir / "anchor_motion_composer_training_summary.json"
        summary = {
            "model_path": str(model_path),
            "diagnostics_path": str(diagnostics_path),
            "summary_path": str(summary_path),
            "best_epoch": int(best_epoch),
            "best_val_mse": float(best_val),
            "train_eval": train_eval,
            "val_eval": val_eval,
        }
        self.diagnostics.write(diagnostics_path)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return AnchorMotionTrainingResult(model_path=model_path, diagnostics_path=diagnostics_path, summary_path=summary_path, diagnostics=self.diagnostics.to_dict())

    def _train(
        self,
        model: AnchorMotionComposer,
        optimizer: torch.optim.Optimizer,
        train_dataset: AnchorMotionDataset,
        val_dataset: AnchorMotionDataset,
    ) -> tuple[List[Dict[str, float]], Optional[Dict[str, torch.Tensor]], int, float]:
        """Train with best-checkpoint selection on validation MSE."""
        history: List[Dict[str, float]] = []
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val = float("inf")
        best_epoch = 0
        patience_used = 0
        min_delta = float(self.training_config.early_stopping_min_delta)
        for epoch in range(int(self.training_config.epochs)):
            train = self._run_epoch(model, train_dataset, optimizer)
            val = self._run_epoch(model, val_dataset, None)
            row = {"epoch": float(epoch + 1), "train_mse": float(train["mse"]), "val_mse": float(val["mse"])}
            history.append(row)
            if float(val["mse"]) < best_val - min_delta:
                best_val = float(val["mse"])
                best_epoch = int(epoch + 1)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience_used = 0
            else:
                patience_used += 1
                if patience_used >= int(self.training_config.early_stopping_patience):
                    row["early_stopped"] = 1.0
                    break
        return history, best_state, best_epoch, best_val

    def _run_epoch(
        self,
        model: AnchorMotionComposer,
        dataset: AnchorMotionDataset,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Dict[str, float]:
        """Run one training or evaluation epoch."""
        is_train = optimizer is not None
        model.train(is_train)
        loader = DataLoader(dataset, batch_size=int(self.training_config.batch_size), shuffle=is_train)
        total = 0.0
        count = 0
        loss_fn = nn.MSELoss(reduction="sum")
        for context_steps, target, current, condition_ids in loader:
            context_steps = context_steps.to(self.training_config.device)
            target = target.to(self.training_config.device)
            current = current.to(self.training_config.device)
            condition_ids = condition_ids.to(self.training_config.device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            output = model(context_steps, current, condition_ids)
            loss = loss_fn(output["composed"], target)
            if is_train:
                loss.backward()
                optimizer.step()
            total += float(loss.detach().cpu())
            count += int(target.numel())
        return {"mse": float(total / max(1, count))}

    def _evaluate(self, model: AnchorMotionComposer, dataset: AnchorMotionDataset) -> Dict[str, Any]:
        """Return compact validation metrics."""
        model.eval()
        predictions: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        currents: List[np.ndarray] = []
        loader = DataLoader(dataset, batch_size=int(self.training_config.batch_size), shuffle=False)
        with torch.no_grad():
            for context_steps, target, current, condition_ids in loader:
                output = model(
                    context_steps.to(self.training_config.device),
                    current.to(self.training_config.device),
                    condition_ids.to(self.training_config.device),
                )
                predictions.append(output["composed"].detach().cpu().numpy())
                targets.append(target.numpy())
                currents.append(current.numpy())
        pred = np.concatenate(predictions, axis=0)
        target = np.concatenate(targets, axis=0)
        current = np.concatenate(currents, axis=0)
        mse = np.mean((pred - target) ** 2, axis=1)
        pred_step = np.linalg.norm(pred - current, axis=1)
        true_step = np.linalg.norm(target - current, axis=1)
        cos = self._direction_cosine(pred - current, target - current)
        return {
            "sample_count": int(len(mse)),
            "mse": self._numeric_summary(mse),
            "direction_cosine": self._numeric_summary(cos),
            "pred_to_current_distance": self._numeric_summary(pred_step),
            "true_to_current_distance": self._numeric_summary(true_step),
            "under_moving_rate": float(np.mean(pred_step < true_step * 0.5)),
        }

    def _save_model(
        self,
        path: Path,
        model: AnchorMotionComposer,
        mean: np.ndarray,
        std: np.ndarray,
        condition_vocab: AnchorMotionConditionVocab,
        checkpoint_role: str,
    ) -> None:
        """Write model checkpoint."""
        torch.save({
            "model_type": "AnchorMotionComposer",
            "checkpoint_role": str(checkpoint_role),
            "model_config": self.model_config.to_dict(),
            "training_config": self.training_config.__dict__,
            "state_dict": model.state_dict(),
            "representation_mean": np.asarray(mean, dtype=np.float32),
            "representation_std": np.asarray(std, dtype=np.float32),
            "feature_names": list(BAR_FEATURE_NAMES),
            "condition_vocab": condition_vocab.to_dict(),
        }, path)

    def _build_samples(self, rows: Sequence[Dict[str, Any]], condition_vocab: AnchorMotionConditionVocab) -> List[AnchorMotionSample]:
        """Build song-local autoregressive samples."""
        grouped = self._group_rows(rows)
        samples: List[AnchorMotionSample] = []
        for song_id, indices in grouped.items():
            for local_position in range(1, len(indices)):
                target_index = int(indices[local_position])
                current_index = int(indices[local_position - 1])
                context = indices[max(0, local_position - int(self.model_config.context_bars)):local_position]
                form_id, action_id, composer_id, position_id = condition_vocab.ids_for_row(rows[target_index])
                samples.append(AnchorMotionSample(
                    context_indices=[int(item) for item in context],
                    current_index=current_index,
                    target_index=target_index,
                    song_id=str(song_id),
                    base_song_id=self._base_song_id(str(song_id)),
                    target_bar_index=int(rows[target_index].get("bar_index", local_position)),
                    form_id=int(form_id),
                    action_id=int(action_id),
                    composer_id=int(composer_id),
                    position_id=int(position_id),
                ))
        if not samples:
            raise ValueError("No Anchor/Motion samples were built.")
        return samples

    def _split_samples(self, samples: Sequence[AnchorMotionSample]) -> tuple[List[AnchorMotionSample], List[AnchorMotionSample], Dict[str, Any]]:
        """Split by base_song_id."""
        by_base: Dict[str, List[AnchorMotionSample]] = {}
        for sample in samples:
            by_base.setdefault(sample.base_song_id, []).append(sample)
        base_ids = sorted(by_base)
        fold_count = max(1, int(self.training_config.validation_fold_count))
        fold_index = int(self.training_config.validation_fold_index) % fold_count
        val_ids = [base_id for index, base_id in enumerate(base_ids) if index % fold_count == fold_index]
        if len(val_ids) >= len(base_ids):
            val_ids = base_ids[-1:]
        val_set = set(val_ids)
        train = [sample for base_id, group in by_base.items() if base_id not in val_set for sample in group]
        val = [sample for base_id, group in by_base.items() if base_id in val_set for sample in group]
        return train, val, {
            "split_unit": "base_song_id",
            "validation_fold_count": int(fold_count),
            "validation_fold_index": int(fold_index),
            "base_song_count": int(len(base_ids)),
            "train_sample_count": int(len(train)),
            "validation_sample_count": int(len(val)),
            "validation_base_song_ids": sorted(val_set),
        }

    def _group_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Group rows by song_id and sort by bar_index."""
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        return {song_id: sorted(indices, key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx)))) for song_id, indices in grouped.items()}

    def _resolved_model_config(
        self,
        mu: np.ndarray,
        features: np.ndarray,
        condition_vocab: AnchorMotionConditionVocab,
    ) -> AnchorMotionComposerConfig:
        """Resolve latent and feature dimensions from data."""
        values = dict(self.model_config.__dict__)
        values["latent_dim"] = int(mu.shape[1])
        values["feature_dim"] = int(features.shape[1])
        values["form_vocab_size"] = max(int(values.get("form_vocab_size", 2)), len(condition_vocab.form_to_id))
        values["action_vocab_size"] = max(int(values.get("action_vocab_size", 2)), len(condition_vocab.action_to_id))
        values["composer_vocab_size"] = max(int(values.get("composer_vocab_size", 2)), len(condition_vocab.composer_to_id))
        values["position_vocab_size"] = max(int(values.get("position_vocab_size", 8)), int(condition_vocab.position_vocab_size))
        return AnchorMotionComposerConfig(**values)

    def _with_overrides(self, base: AnchorMotionComposerTrainingConfig, overrides: Dict[str, Any]) -> AnchorMotionComposerTrainingConfig:
        """Apply CLI overrides."""
        values = dict(base.__dict__)
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        values["device"] = self._resolve_device(str(values.get("device", "cpu")))
        return AnchorMotionComposerTrainingConfig(**values)

    def _resolve_device(self, requested: str) -> str:
        """Return a valid device string."""
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested

    def _set_seed(self) -> None:
        """Seed RNGs."""
        seed = int(self.training_config.random_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _base_song_id(self, song_id: str) -> str:
        """Collapse transposition suffix."""
        return re.sub(r"_T[+-]?\d+$", "", str(song_id))

    def _direction_cosine(self, pred_delta: np.ndarray, true_delta: np.ndarray) -> np.ndarray:
        """Return cosine between predicted and true movement vectors."""
        numerator = np.sum(pred_delta * true_delta, axis=1)
        denominator = np.linalg.norm(pred_delta, axis=1) * np.linalg.norm(true_delta, axis=1)
        return numerator / np.clip(denominator, 1.0e-8, None)

    def _numeric_summary(self, values: Sequence[float] | np.ndarray) -> Dict[str, Any]:
        """Summarize numeric values."""
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "mean": float(np.mean(array)) if array.size else 0.0,
            "median": float(np.median(array)) if array.size else 0.0,
            "min": float(np.min(array)) if array.size else 0.0,
            "max": float(np.max(array)) if array.size else 0.0,
        }


class AnchorMotionComposerGenerationPipeline:
    """Generate latents with a trained Anchor/Motion Composer and decode with DVAE."""

    def __init__(self, config: LatentGenerationConfig) -> None:
        self.config = config
        self.reader = LatentDatasetReader()

    def run(
        self,
        model_dir: str | Path,
        latent_dir: str | Path,
        output_json: str | Path,
        output_midi: str | Path,
        seed_song_id: Optional[str] = None,
        composer_path: Optional[str | Path] = None,
        dvae_path: Optional[str | Path] = None,
    ) -> LatentGenerationResult:
        """Run generation from a trained composer checkpoint."""
        self._set_seed()
        model_directory = Path(model_dir)
        checkpoint_path = Path(composer_path) if composer_path else model_directory / "anchor_motion_composer.pt"
        dvae_checkpoint_path = Path(dvae_path) if dvae_path else model_directory / "dvae.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        model_config = AnchorMotionComposerConfig(**checkpoint["model_config"])
        condition_vocab = AnchorMotionConditionVocab.from_checkpoint(checkpoint)
        model = AnchorMotionComposer(model_config).to(self.config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        mean = np.asarray(checkpoint["representation_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["representation_std"], dtype=np.float32)
        mu, rows, latent_summary = self.reader.load(latent_dir)
        grouped = self._group_rows(rows)
        selected_song_id = self._select_song_id(grouped, seed_song_id)
        ordered = grouped[selected_song_id]
        target_bars = max(1, int(self.config.bars))
        primer_count = max(1, min(int(self.config.primer_bars), len(ordered), target_bars))
        primer_indices = [int(index) for index in ordered[:primer_count]]
        primer_features, feature_source = EncodedBarFeatureStore(model_directory / "encoded").matrix_for_rows([rows[index] for index in primer_indices])
        primer_mu = np.asarray([mu[index] for index in primer_indices], dtype=np.float32)
        primer_values = np.concatenate([primer_mu, primer_features.astype(np.float32)], axis=1).astype(np.float32)
        dvae = self._load_dvae(dvae_checkpoint_path)
        generated_values, steps, selected_row_indices = self._rollout(
            model=model,
            dvae=dvae,
            primer_values=primer_values,
            rows=rows,
            ordered=ordered,
            mean=mean,
            std=std,
            condition_vocab=condition_vocab,
        )
        generated_mu = generated_values[:, : int(model_config.latent_dim)].astype(np.float32)
        tensors = self._decode_tensors(dvae, generated_mu)
        sequence_diagnostics = self._sequence_diagnostics(tensors, steps)
        output_json_path = Path(output_json)
        output_midi_path = Path(output_midi)
        tensor_path = output_json_path.with_suffix(".bar_tensors.npz")
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tensor_path,
            bars=tensors.astype(np.float32),
            latent_mu=generated_mu.astype(np.float32),
            representation_values=generated_values.astype(np.float32),
            selected_row_indices=np.asarray(selected_row_indices, dtype=np.int64),
        )
        midi_diag = SequenceTensorMidiRenderer(DVAEMidiRenderConfig(
            tempo_bpm=int(self.config.tempo_bpm),
            default_base_pitch=int(self.config.base_pitch),
        )).render(tensors, output_midi_path, base_pitch=int(self.config.base_pitch))
        diagnostics = {
            "model_dir": str(model_directory),
            "latent_dir": str(latent_dir),
            "composer_checkpoint": str(checkpoint_path),
            "dvae_checkpoint": str(dvae_checkpoint_path),
            "checkpoint_role": checkpoint.get("checkpoint_role"),
            "generation_backend": "anchor_motion_composer",
            "feature_feedback_mode": "decoded_tensor_closed_loop",
            "latent_summary": latent_summary,
            "config": self.config.__dict__,
            "model_config": model_config.to_dict(),
            "condition_vocab": condition_vocab.to_dict(),
            "feature_source": feature_source,
            "selected_song_id": selected_song_id,
            "seed_song_id": seed_song_id,
            "primer_bars": int(min(self.config.primer_bars, len(ordered), self.config.bars)),
            "generated_bar_count": int(generated_mu.shape[0]),
            "sequence_diagnostics": sequence_diagnostics,
            "steps": steps,
            "midi": midi_diag,
            "tensor_path": str(tensor_path),
            "json_path": str(output_json_path),
            "midi_path": str(output_midi_path),
        }
        output_json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return LatentGenerationResult(json_path=output_json_path, midi_path=output_midi_path, tensor_path=tensor_path, diagnostics=diagnostics)

    def _rollout(
        self,
        model: AnchorMotionComposer,
        dvae: DenoisingMusicVAE,
        primer_values: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        ordered: Sequence[int],
        mean: np.ndarray,
        std: np.ndarray,
        condition_vocab: AnchorMotionConditionVocab,
    ) -> tuple[np.ndarray, List[Dict[str, Any]], List[int]]:
        """Autoregressively roll out hybrid representations.

        The model predicts a full hybrid vector, but generation only trusts the
        latent part. The explicit features are recomputed from the decoded bar
        tensor before the next step sees them, keeping rollout feedback aligned
        with the physical output that will become MIDI.
        """
        target_bars = max(1, int(self.config.bars))
        primer_count = max(1, min(int(self.config.primer_bars), len(ordered), target_bars))
        latent_dim = int(model.config.latent_dim)
        feature_dim = int(model.config.feature_dim)
        feature_extractor = BarFeatureExtractor()
        generated = [np.asarray(primer_values[index], dtype=np.float32).copy() for index in range(primer_count)]
        selected_row_indices = [int(index) for index in ordered[:primer_count]]
        steps: List[Dict[str, Any]] = [
            {
                "bar_index": int(index),
                "source": "primer",
                "selected_row_index": int(ordered[index]),
                "source_song_id": str(rows[ordered[index]].get("song_id", "UNKNOWN")),
                "source_bar_index": int(rows[ordered[index]].get("bar_index", index)),
            }
            for index in range(primer_count)
        ]
        dataset_proxy = AnchorMotionDataset(primer_values, [], model.config.context_bars, mean, std)
        action_planner = GenerationActionPlanner(self.config, condition_vocab.action_to_id)
        seed_row = rows[ordered[0]] if ordered else {}
        seed_form = AnchorMotionConditionVocab._form_name(seed_row)
        seed_composer = AnchorMotionConditionVocab._composer_name(seed_row)
        while len(generated) < target_bars:
            bar_index = int(len(generated))
            source_row = rows[ordered[bar_index]] if bar_index < len(ordered) else {}
            target_action = action_planner.action_name(bar_index=bar_index, source_row=source_row)
            target_form = str(source_row.get("form") or seed_form)
            target_composer = str(
                source_row.get("composer")
                or source_row.get("composer_id")
                or source_row.get("artist")
                or seed_composer
            )
            condition = np.asarray(
                condition_vocab.ids_for_generation(target_form, target_action, target_composer, bar_index),
                dtype=np.int64,
            )
            context_steps = self._context_steps_for_values(dataset_proxy, generated)
            current = dataset_proxy.normalize(generated[-1])
            with torch.no_grad():
                output = model(
                    torch.from_numpy(context_steps).unsqueeze(0).float().to(self.config.device),
                    torch.from_numpy(current).unsqueeze(0).float().to(self.config.device),
                    torch.from_numpy(condition).unsqueeze(0).long().to(self.config.device),
                )
            predicted_norm = output["composed"].detach().cpu().numpy()[0].astype(np.float32)
            predicted_value = (predicted_norm * std + mean).astype(np.float32)
            next_latent = predicted_value[:latent_dim].astype(np.float32)
            predicted_features = predicted_value[latent_dim: latent_dim + feature_dim].astype(np.float32)
            decoded_tensor = self._decode_tensors(dvae, next_latent.reshape(1, latent_dim))[0]
            decoded_features = feature_extractor.features(decoded_tensor).astype(np.float32)
            next_value = np.concatenate([next_latent, decoded_features], axis=0).astype(np.float32)
            generated.append(next_value)
            selected_row_indices.append(-1)
            steps.append({
                "bar_index": int(bar_index),
                "source": "generated",
                "feature_feedback_mode": "decoded_tensor_closed_loop",
                "form": str(target_form),
                "action": str(target_action),
                "composer": str(target_composer),
                "condition_ids": [int(value) for value in condition.tolist()],
                "anchor_norm": float(torch.linalg.norm(output["anchor"]).detach().cpu()),
                "motion_norm": float(torch.linalg.norm(output["motion"]).detach().cpu()),
                "composed_norm": float(np.linalg.norm(predicted_norm)),
                "latent_step_l2": float(np.linalg.norm(next_latent - generated[-2][:latent_dim])),
                "predicted_feature_l2": float(np.linalg.norm(predicted_features)),
                "decoded_feature_l2": float(np.linalg.norm(decoded_features)),
                "predicted_vs_decoded_feature_l2": float(np.linalg.norm(predicted_features - decoded_features)),
                "decoded_note_density": float(decoded_features[0]) if decoded_features.shape[0] > 0 else 0.0,
                "decoded_active_density": float(decoded_features[1]) if decoded_features.shape[0] > 1 else 0.0,
                "decoded_hold_density": float(decoded_features[3]) if decoded_features.shape[0] > 3 else 0.0,
            })
        return np.stack(generated, axis=0).astype(np.float32), steps, selected_row_indices

    def _context_steps_for_values(self, dataset: AnchorMotionDataset, values: Sequence[np.ndarray]) -> np.ndarray:
        """Build context steps from generated raw representation values."""
        dim = int(dataset.dim)
        context = np.zeros((dataset.context_bars, dim), dtype=np.float32)
        mask = np.ones((dataset.context_bars, 1), dtype=np.float32)
        valid = np.zeros((dataset.context_bars,), dtype=bool)
        recent = list(values)[-dataset.context_bars:]
        offset = dataset.context_bars - len(recent)
        for local, value in enumerate(recent):
            slot = offset + local
            context[slot] = dataset.normalize(np.asarray(value, dtype=np.float32))
            mask[slot, 0] = 0.0
            valid[slot] = True
        delta = np.zeros_like(context, dtype=np.float32)
        delta_mask = np.ones((dataset.context_bars, 1), dtype=np.float32)
        for slot in range(1, dataset.context_bars):
            if bool(valid[slot]) and bool(valid[slot - 1]):
                delta[slot] = context[slot] - context[slot - 1]
                delta_mask[slot, 0] = 0.0
        return np.concatenate([context, delta, mask, delta_mask], axis=1).astype(np.float32)

    def _load_dvae(self, path: Path) -> DenoisingMusicVAE:
        """Load trained DVAE."""
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        config = DVAEMusicConfig(**checkpoint["config"])
        model = DenoisingMusicVAE(config).to(self.config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def _decode_tensors(self, dvae: DenoisingMusicVAE, latent_mu: np.ndarray) -> np.ndarray:
        """Decode latent sequence into bar tensors."""
        with torch.no_grad():
            z = torch.from_numpy(latent_mu.astype(np.float32)).to(self.config.device)
            pitch, state_logits, velocity, chord = dvae.decoder(z)
            state = torch.argmax(state_logits, dim=-1)
            state_one_hot = torch.nn.functional.one_hot(state, num_classes=3).float()
            tensor = torch.zeros(
                (z.shape[0], int(dvae.config.tracks), int(dvae.config.steps_per_bar), int(dvae.config.feature_dim)),
                dtype=torch.float32,
                device=z.device,
            )
            tensor[..., 0] = pitch
            tensor[..., 1:4] = state_one_hot
            tensor[..., 4] = velocity
            tensor[..., 5:5 + chord.shape[-1]] = chord
        return tensor.detach().cpu().numpy().astype(np.float32)

    def _sequence_diagnostics(self, tensors: np.ndarray, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize generated tensors."""
        bars = np.asarray(tensors, dtype=np.float32)
        note_counts: List[int] = []
        active_counts: List[int] = []
        pitch_ranges: List[float] = []
        first_pitches: List[Optional[float]] = []
        last_pitches: List[Optional[float]] = []
        for bar in bars:
            note_mask = bar[..., 2] > 0.5
            active_mask = (bar[..., 2] > 0.5) | (bar[..., 3] > 0.5)
            pitches = bar[..., 0][note_mask] * 24.0 + float(self.config.base_pitch)
            note_counts.append(int(note_mask.sum()))
            active_counts.append(int(active_mask.sum()))
            pitch_ranges.append(float(np.max(pitches) - np.min(pitches)) if len(pitches) else 0.0)
            first_pitch, last_pitch = self._bar_boundary_pitches(bar)
            first_pitches.append(first_pitch)
            last_pitches.append(last_pitch)
        boundary_jumps: List[float] = []
        adjacent_bar_l2: List[float] = []
        for index in range(1, len(bars)):
            adjacent_bar_l2.append(float(np.linalg.norm(bars[index] - bars[index - 1])))
            if last_pitches[index - 1] is not None and first_pitches[index] is not None:
                jump = float(abs(float(first_pitches[index]) - float(last_pitches[index - 1])))
                boundary_jumps.append(jump)
                if index < len(steps):
                    steps[index]["boundary_jump_from_previous"] = jump
        return {
            "note_count": self._numeric_summary(note_counts),
            "active_slot_count": self._numeric_summary(active_counts),
            "pitch_range": self._numeric_summary(pitch_ranges),
            "adjacent_bar_l2": self._numeric_summary(adjacent_bar_l2),
            "boundary_jump_abs": self._numeric_summary(boundary_jumps),
            "boundary_jump_gt12_count": int(sum(1 for value in boundary_jumps if value > 12.0)),
            "boundary_jump_gt24_count": int(sum(1 for value in boundary_jumps if value > 24.0)),
        }

    def _bar_boundary_pitches(self, bar: np.ndarray) -> tuple[Optional[float], Optional[float]]:
        """Return first and last note-on pitch."""
        events: List[tuple[int, int, float]] = []
        for slot_index in range(bar.shape[1]):
            for track_index in range(bar.shape[0]):
                if float(bar[track_index, slot_index, 2]) > 0.5:
                    events.append((int(slot_index), int(track_index), float(bar[track_index, slot_index, 0] * 24.0 + float(self.config.base_pitch))))
        if not events:
            return None, None
        return float(events[0][2]), float(events[-1][2])

    def _group_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Group row indices by song_id."""
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        return {song_id: sorted(indices, key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx)))) for song_id, indices in grouped.items()}

    def _select_song_id(self, grouped: Dict[str, List[int]], seed_song_id: Optional[str]) -> str:
        """Select primer song."""
        if seed_song_id:
            if seed_song_id in grouped:
                return seed_song_id
            pattern = re.compile(str(seed_song_id))
            matches = [song_id for song_id in grouped if pattern.search(song_id)]
            if matches:
                return sorted(matches)[0]
            raise ValueError(f"seed_song_id not found: {seed_song_id}")
        rng = random.Random(int(self.config.seed))
        return rng.choice(sorted(grouped.keys()))

    def _set_seed(self) -> None:
        """Seed RNG."""
        random.seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))
        torch.manual_seed(int(self.config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.config.seed))

    def _numeric_summary(self, values: Sequence[float | int]) -> Dict[str, Any]:
        """Return numeric summary."""
        if not values:
            return {"n": 0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }
