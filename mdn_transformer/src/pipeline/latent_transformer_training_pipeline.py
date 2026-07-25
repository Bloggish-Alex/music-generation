#!/usr/bin/env python3
"""Training pipeline for Latent-Transformer + MDN next-bar prediction."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common.config_loader import ConfigView
from diagnostics.diagnostics import DiagnosticsBase
from model.latent_transformer import (
    LatentTransformerConfig,
    LatentTransformerMDN,
    MDNConfig,
    MDNDiagnostics,
    MDNLoss,
)
from pipeline.latent_augmentation import LatentTrainingAugmenter
from pipeline.theme_embedding_provider import FrozenThemeEmbeddingProvider, ThemeFusionConfig


@dataclass(frozen=True)
class LatentTransformerTrainingConfig:
    """Configuration for latent sequence training."""

    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    validation_ratio: float = 0.1
    validation_split_unit: str = "base_song_id"
    validation_fold_count: int = 5
    validation_fold_index: int = 0
    random_seed: int = 42
    device: str = "cpu"
    diagnostics_top_k: int = 20
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LatentTransformerTrainingConfig":
        """Build training config from style config."""
        section = ConfigView(config).section("latent_transformer_training")
        return cls(
            epochs=int(section.get("epochs", 80)),
            batch_size=int(section.get("batch_size", 64)),
            learning_rate=float(section.get("learning_rate", 5.0e-4)),
            weight_decay=float(section.get("weight_decay", 1.0e-4)),
            validation_ratio=float(section.get("validation_ratio", 0.1)),
            validation_split_unit=str(section.get("validation_split_unit", "base_song_id")),
            validation_fold_count=int(section.get("validation_fold_count", 5)),
            validation_fold_index=int(section.get("validation_fold_index", 0)),
            random_seed=int(section.get("random_seed", 42)),
            device=str(section.get("device", "cpu")),
            diagnostics_top_k=int(section.get("diagnostics_top_k", 20)),
            early_stopping_patience=int(section.get("early_stopping_patience", 5)),
            early_stopping_min_delta=float(section.get("early_stopping_min_delta", 0.0)),
        )


@dataclass
class LatentSequenceSample:
    """One autoregressive next-latent training sample."""

    context_mu: np.ndarray
    context_action_ids: np.ndarray
    context_position_ids: np.ndarray
    padding_mask: np.ndarray
    target_mu: np.ndarray
    target_action_id: int
    target_position_id: int
    song_id: str
    base_song_id: str
    target_bar_index: int
    target_row_index: int = -1
    theme_embedding: Optional[np.ndarray] = None
    theme_tokens: Optional[np.ndarray] = None


@dataclass
class LatentSequenceBuildResult:
    """In-memory sequence dataset plus metadata."""

    samples: List[LatentSequenceSample]
    action_to_id: Dict[str, int]
    source_summary: Dict[str, Any]


@dataclass
class LatentTransformerTrainingResult:
    """Paths and diagnostics produced by training."""

    model_path: Path
    diagnostics_path: Path
    summary_path: Path
    diagnostics: Dict[str, Any]


@dataclass
class LatentTransformerFitResult:
    """Training history and best-metric metadata."""

    history: List[Dict[str, float]]
    best_epoch: int
    best_metric: float
    monitor_metric: str = "val_nll"
    early_stopped: bool = False
    stopped_epoch: int = 0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None


class LatentSequenceDataset(Dataset):
    """Torch dataset for fixed-context latent sequence samples."""

    def __init__(self, samples: Sequence[LatentSequenceSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Return sample tensors."""
        sample = self.samples[index]
        return (
            torch.from_numpy(sample.context_mu).float(),
            torch.from_numpy(sample.context_action_ids).long(),
            torch.from_numpy(sample.context_position_ids).long(),
            torch.from_numpy(sample.padding_mask).bool(),
            torch.from_numpy(sample.target_mu).float(),
            torch.tensor(sample.target_action_id, dtype=torch.long),
            torch.tensor(sample.target_position_id, dtype=torch.long),
            torch.from_numpy(self._theme_embedding(sample)).float(),
            torch.from_numpy(self._theme_tokens(sample)).float(),
        )

    def _theme_embedding(self, sample: LatentSequenceSample) -> np.ndarray:
        """Return sample theme embedding or an empty vector when disabled."""
        if sample.theme_embedding is None:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(sample.theme_embedding, dtype=np.float32)

    def _theme_tokens(self, sample: LatentSequenceSample) -> np.ndarray:
        """Return sample theme token sequence or an empty sequence when disabled."""
        if sample.theme_tokens is None:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(sample.theme_tokens, dtype=np.float32)


class LatentDatasetReader:
    """Read exported latent arrays and row metadata."""

    def load(self, latent_dir: str | Path) -> tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """Load latent_mu.npy, latent_index.json, and optional summary."""
        directory = Path(latent_dir)
        mu_path = directory / "latent_mu.npy"
        index_path = directory / "latent_index.json"
        summary_path = directory / "latent_summary.json"
        if not mu_path.exists():
            raise FileNotFoundError(f"Missing latent_mu.npy: {mu_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Missing latent_index.json: {index_path}")
        mu = np.load(mu_path).astype(np.float32)
        rows = json.loads(index_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        if len(rows) != int(mu.shape[0]):
            raise ValueError("latent_index.json row count must match latent_mu.npy rows.")
        return mu, rows, summary


class LatentSequenceBuilder:
    """Build fixed-length autoregressive windows inside each song only."""

    PAD_ACTION = "PAD"
    UNKNOWN_ACTION = "UNKNOWN"

    def __init__(self, context_bars: int, position_vocab_size: int) -> None:
        self.context_bars = int(context_bars)
        self.position_vocab_size = int(position_vocab_size)

    def build(self, mu: np.ndarray, rows: Sequence[Dict[str, Any]]) -> LatentSequenceBuildResult:
        """Return training samples grouped by song_id and sorted by bar_index."""
        action_to_id = self._action_vocab(rows)
        grouped = self._group_rows(rows)
        samples: List[LatentSequenceSample] = []
        skipped_short_songs = 0
        song_lengths: Dict[str, int] = {}
        for song_id, indices in grouped.items():
            ordered = sorted(indices, key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx))))
            song_lengths[song_id] = int(len(ordered))
            if len(ordered) < 2:
                skipped_short_songs += 1
                continue
            for local_target_pos in range(1, len(ordered)):
                target_row_index = ordered[local_target_pos]
                context_indices = ordered[max(0, local_target_pos - self.context_bars):local_target_pos]
                samples.append(self._make_sample(
                    mu=mu,
                    rows=rows,
                    context_indices=context_indices,
                    target_row_index=target_row_index,
                    action_to_id=action_to_id,
                ))
        source_summary = {
            "song_count": int(len(grouped)),
            "sample_count": int(len(samples)),
            "skipped_short_song_count": int(skipped_short_songs),
            "min_song_bars": int(min(song_lengths.values())) if song_lengths else 0,
            "max_song_bars": int(max(song_lengths.values())) if song_lengths else 0,
            "mean_song_bars": float(np.mean(list(song_lengths.values()))) if song_lengths else 0.0,
        }
        if not samples:
            raise ValueError("No latent sequence samples were built. Need at least two bars in one song.")
        return LatentSequenceBuildResult(samples=samples, action_to_id=action_to_id, source_summary=source_summary)

    def _make_sample(
        self,
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        context_indices: Sequence[int],
        target_row_index: int,
        action_to_id: Dict[str, int],
    ) -> LatentSequenceSample:
        """Create one left-padded fixed-context sample."""
        latent_dim = int(mu.shape[1])
        context_mu = np.zeros((self.context_bars, latent_dim), dtype=np.float32)
        context_action_ids = np.zeros((self.context_bars,), dtype=np.int64)
        context_position_ids = np.zeros((self.context_bars,), dtype=np.int64)
        padding_mask = np.ones((self.context_bars,), dtype=bool)
        for offset, row_index in enumerate(context_indices):
            slot = offset
            row = rows[row_index]
            context_mu[slot] = mu[row_index]
            context_action_ids[slot] = action_to_id.get(self._action_name(row), action_to_id[self.UNKNOWN_ACTION])
            context_position_ids[slot] = self._position_id(row)
            padding_mask[slot] = False
        target_row = rows[target_row_index]
        return LatentSequenceSample(
            context_mu=context_mu,
            context_action_ids=context_action_ids,
            context_position_ids=context_position_ids,
            padding_mask=padding_mask,
            target_mu=mu[target_row_index].astype(np.float32),
            target_action_id=action_to_id.get(self._action_name(target_row), action_to_id[self.UNKNOWN_ACTION]),
            target_position_id=self._position_id(target_row),
            song_id=str(target_row.get("song_id", "UNKNOWN")),
            base_song_id=self._base_song_id(str(target_row.get("song_id", "UNKNOWN"))),
            target_bar_index=int(target_row.get("bar_index", 0)),
            target_row_index=int(target_row_index),
        )

    def _action_vocab(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        """Create a stable action vocabulary from latent metadata."""
        actions = {self._action_name(row) for row in rows}
        ordered = [self.PAD_ACTION, self.UNKNOWN_ACTION]
        ordered.extend(action for action in sorted(actions) if action not in set(ordered))
        return {action: index for index, action in enumerate(ordered)}

    def _action_name(self, row: Dict[str, Any]) -> str:
        """Return normalized action name."""
        value = row.get("action") or self.UNKNOWN_ACTION
        return str(value)

    def _position_id(self, row: Dict[str, Any]) -> int:
        """Return period-local position id."""
        return int(row.get("bar_index", 0)) % max(1, self.position_vocab_size)

    def _group_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Group row indices by song_id."""
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        return grouped

    def _base_song_id(self, song_id: str) -> str:
        """Remove transposition suffix from song_id for leakage-safe splits."""
        return re.sub(r"_T[+-]\d+$", "", str(song_id))


class LatentTransformerTrainingPipeline:
    """Train Latent-Transformer + MDN from exported latent datasets."""

    def __init__(self, config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> None:
        self.config = config
        self.training_config = self._with_overrides(LatentTransformerTrainingConfig.from_config(config), overrides or {})
        self.transformer_config = LatentTransformerConfig.from_config(config)
        self.mdn_config = MDNConfig.from_config(config)
        self.theme_fusion_config = ThemeFusionConfig.from_config(config)
        self.reader = LatentDatasetReader()
        self.diagnostics = DiagnosticsBase("latent_transformer_training")

    def run(self, latent_dir: str | Path, model_dir: str | Path) -> LatentTransformerTrainingResult:
        """Read latent dataset, train model, and write diagnostics."""
        self._set_seed()
        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mu, rows, latent_summary = self.reader.load(latent_dir)
        self.transformer_config = self._resolved_transformer_config(mu, rows)
        sequence_data = LatentSequenceBuilder(
            context_bars=int(self.transformer_config.context_bars),
            position_vocab_size=int(self.transformer_config.position_vocab_size),
        ).build(mu, rows)
        self._attach_theme_embeddings(sequence_data.samples, mu, rows)
        train_dataset, val_dataset = self._split_samples(sequence_data.samples)
        train_dataset = self._augment_train_dataset(train_dataset)
        model = LatentTransformerMDN(self.transformer_config, self.mdn_config).to(self.training_config.device)
        loss_fn = MDNLoss(self.mdn_config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.training_config.learning_rate),
            weight_decay=float(self.training_config.weight_decay),
        )
        fit = self._train(model, loss_fn, optimizer, train_dataset, val_dataset)
        model_path = output_dir / "latent_transformer_mdn.pt"
        final_model_path = output_dir / "latent_transformer_mdn.final.pt"
        self._save_model(final_model_path, model, sequence_data.action_to_id, checkpoint_role="final")
        if fit.best_state_dict is not None:
            model.load_state_dict(fit.best_state_dict)
        train_eval = self._evaluate(model, loss_fn, train_dataset)
        val_eval = self._evaluate(model, loss_fn, val_dataset)
        self._save_model(model_path, model, sequence_data.action_to_id, checkpoint_role="best")
        self.diagnostics.record_stage("input", {
            "latent_dir": str(latent_dir),
            "latent_summary": latent_summary,
            "sequence_summary": sequence_data.source_summary,
            "action_to_id": sequence_data.action_to_id,
        })
        self.diagnostics.record_stage("training", {
            "config": self.training_config.__dict__,
            "transformer_config": self.transformer_config.to_dict(),
            "mdn_config": self.mdn_config.to_dict(),
            "theme_fusion_config": self.theme_fusion_config.to_dict(),
            "theme_fusion_runtime": self._theme_fusion_runtime(model),
            "history": fit.history,
            "model_selection": self._model_selection_summary(fit),
            "saved_checkpoints": {
                "best": str(model_path),
                "final": str(final_model_path),
            },
        })
        self.diagnostics.record_stage("train_eval", train_eval)
        self.diagnostics.record_stage("val_eval", val_eval)
        summary = self._summary(model_path, latent_dir, sequence_data, fit, train_eval, val_eval)
        diagnostics_path = output_dir / "latent_transformer_training_diagnostics.json"
        summary_path = output_dir / "latent_transformer_training_summary.json"
        self.diagnostics.write(diagnostics_path)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return LatentTransformerTrainingResult(
            model_path=model_path,
            diagnostics_path=diagnostics_path,
            summary_path=summary_path,
            diagnostics=self.diagnostics.to_dict(),
        )

    def _resolved_transformer_config(self, mu: np.ndarray, rows: Sequence[Dict[str, Any]]) -> LatentTransformerConfig:
        """Resolve latent/action vocabulary sizes from data."""
        action_to_id = LatentSequenceBuilder(
            context_bars=int(self.transformer_config.context_bars),
            position_vocab_size=int(self.transformer_config.position_vocab_size),
        )._action_vocab(rows)
        values = dict(self.transformer_config.__dict__)
        values["latent_dim"] = int(mu.shape[1])
        values["action_vocab_size"] = max(int(values["action_vocab_size"]), len(action_to_id))
        if bool(self.theme_fusion_config.enabled):
            values["theme_fusion_enabled"] = True
            values["theme_embedding_dim"] = int(self.theme_fusion_config.embedding_dim)
            values["theme_project_dim"] = int(self.theme_fusion_config.project_dim)
            values["theme_gate_init"] = float(self.theme_fusion_config.gate_init)
            values["theme_fusion_mode"] = str(self.theme_fusion_config.mode)
            values["theme_fusion_target"] = str(self.theme_fusion_config.target)
            values["theme_dropout"] = float(self.theme_fusion_config.theme_dropout)
            values["theme_embedding_noise_std"] = float(self.theme_fusion_config.embedding_noise_std)
            values["theme_token_bars"] = int(self.theme_fusion_config.token_bars)
            values["theme_cross_attention_heads"] = int(self.theme_fusion_config.cross_attention_heads)
        return LatentTransformerConfig(**values)

    def _attach_theme_embeddings(
        self,
        samples: Sequence[LatentSequenceSample],
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
    ) -> None:
        """Attach frozen theme embeddings to samples when theme fusion is enabled."""
        provider = FrozenThemeEmbeddingProvider.from_config(self.config, device=self.training_config.device)
        by_song, token_by_song, diagnostics = provider.theme_contexts_by_song(mu, rows)
        self.diagnostics.record_stage("theme_fusion", diagnostics)
        if not self.theme_fusion_config.enabled:
            return
        if int(diagnostics.get("embedding_dim", self.theme_fusion_config.embedding_dim)) != int(self.transformer_config.theme_embedding_dim):
            values = dict(self.transformer_config.__dict__)
            values["theme_embedding_dim"] = int(diagnostics.get("embedding_dim", self.theme_fusion_config.embedding_dim))
            self.transformer_config = LatentTransformerConfig(**values)
        missing_embedding = 0
        missing_tokens = 0
        token_shape = (int(self.theme_fusion_config.token_bars), int(mu.shape[1]))
        for sample in samples:
            embedding = by_song.get(sample.song_id)
            if embedding is None:
                missing_embedding += 1
                embedding = np.zeros((int(self.theme_fusion_config.embedding_dim),), dtype=np.float32)
            sample.theme_embedding = embedding.astype(np.float32)
            tokens = token_by_song.get(sample.song_id)
            if tokens is None:
                missing_tokens += 1
                tokens = np.zeros(token_shape, dtype=np.float32)
            sample.theme_tokens = np.asarray(tokens, dtype=np.float32)
        diagnostics["missing_embedding_sample_count"] = int(missing_embedding)
        diagnostics["missing_token_sample_count"] = int(missing_tokens)

    def _augment_train_dataset(self, train_dataset: LatentSequenceDataset) -> LatentSequenceDataset:
        """Apply isolated training-only latent augmentation."""
        result = LatentTrainingAugmenter.from_config(
            self.config,
            random_seed=int(self.training_config.random_seed),
        ).augment_dataset(train_dataset)
        self.diagnostics.record_stage("latent_augmentation", result.diagnostics)
        return result.dataset

    def _with_overrides(
        self,
        base: LatentTransformerTrainingConfig,
        overrides: Dict[str, Any],
    ) -> LatentTransformerTrainingConfig:
        """Apply CLI overrides and resolve device."""
        values = dict(base.__dict__)
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        values["device"] = self._resolve_device(str(values.get("device", "cpu")))
        return LatentTransformerTrainingConfig(**values)

    def _resolve_device(self, requested: str) -> str:
        """Return an available torch device."""
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested

    def _set_seed(self) -> None:
        """Seed Python, numpy, and torch RNGs."""
        seed = int(self.training_config.random_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _split_samples(self, samples: Sequence[LatentSequenceSample]) -> tuple[LatentSequenceDataset, LatentSequenceDataset]:
        """Create deterministic train/validation sample splits."""
        if self.training_config.validation_split_unit == "base_song_id":
            return self._split_samples_by_base_song(samples)
        if self.training_config.validation_split_unit != "sample":
            raise ValueError(
                "latent_transformer_training.validation_split_unit must be 'base_song_id' or 'sample'."
            )
        return self._split_samples_by_row(samples)

    def _split_samples_by_row(
        self,
        samples: Sequence[LatentSequenceSample],
    ) -> tuple[LatentSequenceDataset, LatentSequenceDataset]:
        """Create deterministic random sample-level splits."""
        indices = np.arange(len(samples))
        rng = np.random.default_rng(int(self.training_config.random_seed))
        rng.shuffle(indices)
        val_size = int(round(len(indices) * max(0.0, min(0.9, float(self.training_config.validation_ratio)))))
        if len(indices) > 1:
            val_size = max(1, val_size)
            val_size = min(val_size, len(indices) - 1)
        val_indices = set(int(index) for index in indices[:val_size])
        train_samples = [sample for index, sample in enumerate(samples) if index not in val_indices]
        val_samples = [sample for index, sample in enumerate(samples) if index in val_indices]
        self.diagnostics.record_stage("dataset_split", {
            "total_size": int(len(samples)),
            "train_size": int(len(train_samples)),
            "validation_size": int(len(val_samples)),
            "shuffle": True,
            "split_unit": "sample",
        })
        return LatentSequenceDataset(train_samples), LatentSequenceDataset(val_samples)

    def _split_samples_by_base_song(
        self,
        samples: Sequence[LatentSequenceSample],
    ) -> tuple[LatentSequenceDataset, LatentSequenceDataset]:
        """Split samples by base_song_id so augmented variants do not leak across splits."""
        groups: Dict[str, List[LatentSequenceSample]] = {}
        for sample in samples:
            groups.setdefault(sample.base_song_id, []).append(sample)
        if len(groups) < 2:
            self.diagnostics.append_event("dataset_split_fallback", {
                "reason": "Fewer than two base_song_id groups; falling back to sample split.",
                "base_song_count": int(len(groups)),
            })
            return self._split_samples_by_row(samples)

        rng = np.random.default_rng(int(self.training_config.random_seed))
        base_ids = np.asarray(sorted(groups.keys()), dtype=object)
        rng.shuffle(base_ids)
        fold_count = int(self.training_config.validation_fold_count)
        if fold_count > 1:
            fold_count = min(fold_count, len(base_ids))
            fold_index = int(self.training_config.validation_fold_index) % fold_count
            folds = np.array_split(base_ids, fold_count)
            val_base_ids = {str(base_id) for base_id in folds[fold_index].tolist()}
            split_strategy = "group_k_fold"
            target_val_size = sum(len(groups[base_id]) for base_id in val_base_ids)
        else:
            fold_count = 1
            fold_index = 0
            target_val_size = int(round(len(samples) * max(0.0, min(0.9, float(self.training_config.validation_ratio)))))
            target_val_size = max(1, target_val_size)
            val_base_ids: set[str] = set()
            val_size = 0
            for base_id in base_ids:
                if len(val_base_ids) >= len(base_ids) - 1:
                    break
                if val_size >= target_val_size and val_base_ids:
                    break
                val_base_ids.add(str(base_id))
                val_size += len(groups[str(base_id)])
            split_strategy = "group_ratio"
        train_samples = [
            sample for base_id, group in groups.items() if base_id not in val_base_ids for sample in group
        ]
        val_samples = [
            sample for base_id, group in groups.items() if base_id in val_base_ids for sample in group
        ]
        train_base_ids = set(groups.keys()) - val_base_ids
        self.diagnostics.record_stage("dataset_split", {
            "total_size": int(len(samples)),
            "train_size": int(len(train_samples)),
            "validation_size": int(len(val_samples)),
            "validation_target_size": int(target_val_size),
            "shuffle": True,
            "split_unit": "base_song_id",
            "split_strategy": split_strategy,
            "validation_fold_count": int(fold_count),
            "validation_fold_index": int(fold_index),
            "base_song_count": int(len(groups)),
            "train_base_song_count": int(len(train_base_ids)),
            "validation_base_song_count": int(len(val_base_ids)),
            "base_song_overlap_count": int(len(train_base_ids.intersection(val_base_ids))),
            "validation_base_song_ids": sorted(val_base_ids),
        })
        return LatentSequenceDataset(train_samples), LatentSequenceDataset(val_samples)

    def _train(
        self,
        model: LatentTransformerMDN,
        loss_fn: MDNLoss,
        optimizer: torch.optim.Optimizer,
        train_dataset: LatentSequenceDataset,
        val_dataset: LatentSequenceDataset,
    ) -> LatentTransformerFitResult:
        """Train model and return epoch history."""
        generator = torch.Generator().manual_seed(int(self.training_config.random_seed))
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.training_config.batch_size),
            shuffle=True,
            generator=generator,
        )
        val_loader = DataLoader(val_dataset, batch_size=int(self.training_config.batch_size), shuffle=False)
        history: List[Dict[str, float]] = []
        best_metric = math.inf
        best_epoch = 0
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        stale_epochs = 0
        monitor_metric = "val_nll"
        early_stopped = False
        stopped_epoch = 0
        min_delta = float(self.training_config.early_stopping_min_delta)
        patience = int(self.training_config.early_stopping_patience)
        for epoch in range(int(self.training_config.epochs)):
            train_metrics = self._run_epoch(model, loss_fn, train_loader, optimizer)
            val_metrics = self._run_epoch(model, loss_fn, val_loader, None)
            row = {"epoch": float(epoch + 1)}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            metric_value = float(row.get(monitor_metric, math.inf))
            if metric_value < best_metric - min_delta:
                best_metric = metric_value
                best_epoch = int(epoch + 1)
                stale_epochs = 0
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                stale_epochs += 1
            row["best_epoch"] = float(best_epoch)
            row["best_metric"] = float(best_metric)
            row["stale_epochs"] = float(stale_epochs)
            history.append(row)
            if patience > 0 and stale_epochs >= patience:
                early_stopped = True
                stopped_epoch = int(epoch + 1)
                break
        return LatentTransformerFitResult(
            history=history,
            best_epoch=int(best_epoch),
            best_metric=float(best_metric),
            monitor_metric=monitor_metric,
            early_stopped=bool(early_stopped),
            stopped_epoch=int(stopped_epoch),
            best_state_dict=best_state_dict,
        )

    def _run_epoch(
        self,
        model: LatentTransformerMDN,
        loss_fn: MDNLoss,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Dict[str, float]:
        """Run one train or validation epoch."""
        is_train = optimizer is not None
        model.train(is_train)
        totals: Dict[str, float] = {}
        count = 0
        diag = MDNDiagnostics()
        for batch in loader:
            prepared = self._prepare_batch(batch)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(is_train):
                output = model(
                    context_mu=prepared["context_mu"],
                    context_action_ids=prepared["context_action_ids"],
                    context_position_ids=prepared["context_position_ids"],
                    target_action_ids=prepared["target_action_ids"],
                    target_position_ids=prepared["target_position_ids"],
                    padding_mask=prepared["padding_mask"],
                    theme_embedding=prepared["theme_embedding"],
                    theme_tokens=prepared["theme_tokens"],
                )
                losses = loss_fn(output, prepared["target_mu"])
                if is_train:
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            batch_size = int(prepared["target_mu"].shape[0])
            count += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
            batch_diag = diag.summarize(output, prepared["target_mu"])
            for key, value in batch_diag.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        if count == 0:
            return {"nll": 0.0}
        return {key: value / count for key, value in totals.items()}

    def _prepare_batch(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move DataLoader batch to configured device."""
        names = [
            "context_mu",
            "context_action_ids",
            "context_position_ids",
            "padding_mask",
            "target_mu",
            "target_action_ids",
            "target_position_ids",
            "theme_embedding",
            "theme_tokens",
        ]
        return {name: value.to(self.training_config.device) for name, value in zip(names, batch)}

    def _evaluate(
        self,
        model: LatentTransformerMDN,
        loss_fn: MDNLoss,
        dataset: LatentSequenceDataset,
    ) -> Dict[str, Any]:
        """Evaluate full sequence dataset and summarize per-action NLL."""
        loader = DataLoader(dataset, batch_size=int(self.training_config.batch_size), shuffle=False)
        metrics = self._run_epoch(model, loss_fn, loader, optimizer=None)
        action_totals: Dict[int, float] = {}
        action_counts: Dict[int, int] = {}
        pi_argmax_counts = np.zeros(int(self.mdn_config.n_components), dtype=np.int64)
        nearest_component_counts = np.zeros(int(self.mdn_config.n_components), dtype=np.int64)
        max_pi_values: List[float] = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                prepared = self._prepare_batch(batch)
                output = model(
                    context_mu=prepared["context_mu"],
                    context_action_ids=prepared["context_action_ids"],
                    context_position_ids=prepared["context_position_ids"],
                    target_action_ids=prepared["target_action_ids"],
                    target_position_ids=prepared["target_position_ids"],
                    padding_mask=prepared["padding_mask"],
                    theme_embedding=prepared["theme_embedding"],
                    theme_tokens=prepared["theme_tokens"],
                )
                row_nll = self._row_nll(output, prepared["target_mu"])
                probs = torch.softmax(output.pi_logits, dim=-1)
                pi_argmax = torch.argmax(probs, dim=-1).detach().cpu().numpy()
                nearest_component = torch.argmin(
                    torch.linalg.vector_norm(output.mu - prepared["target_mu"].unsqueeze(1), dim=-1),
                    dim=-1,
                ).detach().cpu().numpy()
                pi_argmax_counts += np.bincount(pi_argmax, minlength=int(self.mdn_config.n_components))
                nearest_component_counts += np.bincount(nearest_component, minlength=int(self.mdn_config.n_components))
                max_pi_values.extend(probs.max(dim=-1).values.detach().cpu().tolist())
                for action_id, nll in zip(prepared["target_action_ids"].detach().cpu().tolist(), row_nll.detach().cpu().tolist()):
                    action_totals[int(action_id)] = action_totals.get(int(action_id), 0.0) + float(nll)
                    action_counts[int(action_id)] = action_counts.get(int(action_id), 0) + 1
        total_pi = max(1, int(pi_argmax_counts.sum()))
        total_nearest = max(1, int(nearest_component_counts.sum()))
        return {
            **metrics,
            "component_usage": {
                "pi_argmax_counts": [int(value) for value in pi_argmax_counts.tolist()],
                "pi_argmax_ratios": [float(value / total_pi) for value in pi_argmax_counts.tolist()],
                "nearest_component_counts": [int(value) for value in nearest_component_counts.tolist()],
                "nearest_component_ratios": [float(value / total_nearest) for value in nearest_component_counts.tolist()],
                "avg_max_pi": float(np.mean(max_pi_values)) if max_pi_values else 0.0,
                "min_max_pi": float(np.min(max_pi_values)) if max_pi_values else 0.0,
                "max_max_pi": float(np.max(max_pi_values)) if max_pi_values else 0.0,
            },
            "per_action_nll": {
                str(action_id): float(action_totals[action_id] / action_counts[action_id])
                for action_id in sorted(action_totals)
            },
            "per_action_count": {str(key): int(value) for key, value in sorted(action_counts.items())},
        }

    def _row_nll(self, output: Any, target_mu: torch.Tensor) -> torch.Tensor:
        """Return per-row NLL without reducing across batch."""
        target = target_mu.unsqueeze(1)
        log_pi = torch.log_softmax(output.pi_logits, dim=-1)
        log_sigma = torch.log(output.sigma.clamp_min(1.0e-8))
        z = (target - output.mu) / output.sigma.clamp_min(1.0e-8)
        log_prob_dim = -0.5 * (z.pow(2) + 2.0 * log_sigma + math.log(2.0 * math.pi))
        log_prob = log_prob_dim.sum(dim=-1)
        return -torch.logsumexp(log_pi + log_prob, dim=-1)

    def _save_model(
        self,
        path: Path,
        model: LatentTransformerMDN,
        action_to_id: Dict[str, int],
        checkpoint_role: str,
    ) -> None:
        """Save model state and required vocab/config."""
        torch.save({
            "model_type": "LatentTransformerMDN",
            "checkpoint_role": checkpoint_role,
            "transformer_config": self.transformer_config.to_dict(),
            "mdn_config": self.mdn_config.to_dict(),
            "theme_fusion_config": self.theme_fusion_config.to_dict(),
            "action_to_id": action_to_id,
            "id_to_action": {str(value): key for key, value in action_to_id.items()},
            "state_dict": model.state_dict(),
        }, path)

    def _summary(
        self,
        model_path: Path,
        latent_dir: str | Path,
        sequence_data: LatentSequenceBuildResult,
        fit: LatentTransformerFitResult,
        train_eval: Dict[str, Any],
        val_eval: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build compact JSON summary."""
        return {
            "model_path": str(model_path),
            "saved_checkpoints": {
                "best": str(model_path),
                "final": str(model_path.with_name("latent_transformer_mdn.final.pt")),
            },
            "latent_dir": str(latent_dir),
            "sequence_summary": sequence_data.source_summary,
            "action_to_id": sequence_data.action_to_id,
            "dataset_split": self.diagnostics.to_dict().get("stages", {}).get("dataset_split", {}),
            "theme_fusion": self.diagnostics.to_dict().get("stages", {}).get("theme_fusion", {}),
            "theme_fusion_runtime": self._theme_fusion_runtime_from_eval(),
            "latent_augmentation": self.diagnostics.to_dict().get("stages", {}).get("latent_augmentation", {}),
            "epochs": int(self.training_config.epochs),
            "trained_epochs": int(len(fit.history)),
            "final_epoch": fit.history[-1] if fit.history else {},
            "model_selection": self._model_selection_summary(fit),
            "train_eval": train_eval,
            "val_eval": val_eval,
            "diagnostic_guidance": {
                "train_eval": self._diagnostic_guidance(train_eval),
                "val_eval": self._diagnostic_guidance(val_eval),
            },
        }

    def _theme_fusion_runtime(self, model: LatentTransformerMDN) -> Dict[str, Any]:
        """Return runtime theme fusion state for diagnostics."""
        runtime = model.theme_fusion_diagnostics()
        runtime["enabled"] = bool(self.transformer_config.theme_fusion_enabled)
        return runtime

    def _theme_fusion_runtime_from_eval(self) -> Dict[str, Any]:
        """Return recorded runtime theme fusion state from diagnostics."""
        return self.diagnostics.to_dict().get("stages", {}).get("training", {}).get("theme_fusion_runtime", {})

    def _model_selection_summary(self, fit: LatentTransformerFitResult) -> Dict[str, Any]:
        """Return JSON-safe best-metric metadata without affecting training."""
        return {
            "monitor_metric": fit.monitor_metric,
            "best_epoch": int(fit.best_epoch),
            "best_metric": float(fit.best_metric),
            "early_stopped": bool(fit.early_stopped),
            "stopped_epoch": int(fit.stopped_epoch),
            "early_stopping_patience": int(self.training_config.early_stopping_patience),
            "early_stopping_min_delta": float(self.training_config.early_stopping_min_delta),
        }

    def _diagnostic_guidance(self, evaluation: Dict[str, Any]) -> Dict[str, str]:
        """Return simple interpretation hints for first-pass analysis."""
        avg_sigma = float(evaluation.get("avg_sigma", 0.0))
        entropy = float(evaluation.get("component_entropy", 0.0))
        if math.isnan(avg_sigma):
            sigma_note = "Average sigma is NaN; evaluation is invalid and the attention/padding path should be checked."
        elif avg_sigma > 1.0:
            sigma_note = "Average sigma is high; the MDN may be uncertain or underfit."
        elif avg_sigma < 0.08:
            sigma_note = "Average sigma is very low; watch for overconfident or collapsed components."
        else:
            sigma_note = "Average sigma is in a usable first-pass range."
        if math.isnan(entropy):
            entropy_note = "Mixture entropy is NaN; evaluation is invalid and should not be interpreted."
        elif entropy < 0.5:
            entropy_note = "Mixture entropy is low; component collapse is possible."
        else:
            entropy_note = "Mixture entropy suggests multiple components are being used."
        usage = evaluation.get("component_usage", {})
        avg_max_pi = float(usage.get("avg_max_pi", evaluation.get("avg_max_pi", 0.0))) if isinstance(usage, dict) else 0.0
        if avg_max_pi > 0.95:
            usage_note = "Average max mixture weight is very high; one component probably dominates."
        elif avg_max_pi > 0.80:
            usage_note = "Average max mixture weight is high; check whether component usage is too concentrated."
        else:
            usage_note = "Average max mixture weight leaves room for multimodal sampling."
        return {"sigma": sigma_note, "component_entropy": entropy_note, "component_usage": usage_note}
