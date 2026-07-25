#!/usr/bin/env python3
"""Training pipeline for Memory Latent Transformer."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common.config_loader import ConfigView
from diagnostics.diagnostics import DiagnosticsBase
from model.memory_latent_transformer import (
    InBatchMemoryContrastiveLoss,
    MemoryLatentTransformer,
    MemoryLatentTransformerConfig,
)
from pipeline.latent_transformer_training_pipeline import (
    LatentDatasetReader,
    LatentSequenceBuilder,
    LatentSequenceSample,
)
from pipeline.theme_embedding_provider import FrozenThemeEmbeddingProvider, ThemeFusionConfig


@dataclass(frozen=True)
class MemoryLatentTrainingConfig:
    """Configuration for memory latent training."""

    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    validation_ratio: float = 0.1
    validation_split_unit: str = "base_song_id"
    validation_fold_count: int = 5
    validation_fold_index: int = 0
    random_seed: int = 42
    device: str = "cpu"
    contrastive_temperature: float = 0.1
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MemoryLatentTrainingConfig":
        """Build training config from style config."""
        section = ConfigView(config).section("memory_latent_training")
        fallback = ConfigView(config).section("latent_transformer_training")
        return cls(
            epochs=int(section.get("epochs", fallback.get("epochs", 80))),
            batch_size=int(section.get("batch_size", 128)),
            learning_rate=float(section.get("learning_rate", fallback.get("learning_rate", 5.0e-4))),
            weight_decay=float(section.get("weight_decay", fallback.get("weight_decay", 1.0e-4))),
            validation_ratio=float(section.get("validation_ratio", fallback.get("validation_ratio", 0.1))),
            validation_split_unit=str(section.get("validation_split_unit", fallback.get("validation_split_unit", "base_song_id"))),
            validation_fold_count=int(section.get("validation_fold_count", fallback.get("validation_fold_count", 5))),
            validation_fold_index=int(section.get("validation_fold_index", fallback.get("validation_fold_index", 0))),
            random_seed=int(section.get("random_seed", fallback.get("random_seed", 42))),
            device=str(section.get("device", fallback.get("device", "cpu"))),
            contrastive_temperature=float(section.get("contrastive_temperature", 0.1)),
            early_stopping_patience=int(section.get("early_stopping_patience", fallback.get("early_stopping_patience", 5))),
            early_stopping_min_delta=float(section.get("early_stopping_min_delta", fallback.get("early_stopping_min_delta", 0.0))),
        )


@dataclass
class MemoryLatentTrainingResult:
    """Paths and diagnostics produced by memory latent training."""

    model_path: Path
    diagnostics_path: Path
    summary_path: Path
    diagnostics: Dict[str, Any]


@dataclass
class MemoryFitResult:
    """Training history and best checkpoint."""

    history: List[Dict[str, float]]
    best_epoch: int
    best_metric: float
    early_stopped: bool
    stopped_epoch: int
    best_state_dict: Optional[Dict[str, torch.Tensor]]


class MemoryLatentDataset(Dataset):
    """Dataset for memory latent contrastive training."""

    def __init__(self, samples: Sequence[LatentSequenceSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sample = self.samples[index]
        return (
            torch.from_numpy(sample.context_mu).float(),
            torch.from_numpy(sample.context_action_ids).long(),
            torch.from_numpy(sample.context_position_ids).long(),
            torch.from_numpy(sample.padding_mask).bool(),
            torch.from_numpy(sample.target_mu).float(),
            torch.tensor(sample.target_action_id, dtype=torch.long),
            torch.tensor(sample.target_position_id, dtype=torch.long),
            torch.tensor(sample.target_row_index, dtype=torch.long),
            torch.from_numpy(np.asarray(sample.theme_embedding if sample.theme_embedding is not None else np.zeros((0,), dtype=np.float32), dtype=np.float32)).float(),
            torch.from_numpy(np.asarray(sample.theme_tokens if sample.theme_tokens is not None else np.zeros((0, 0), dtype=np.float32), dtype=np.float32)).float(),
        )


class MemoryLatentTrainingPipeline:
    """Train Memory Latent Transformer from exported latent datasets."""

    def __init__(self, config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> None:
        self.config = config
        self.training_config = self._with_overrides(MemoryLatentTrainingConfig.from_config(config), overrides or {})
        self.model_config = MemoryLatentTransformerConfig.from_config(config)
        self.theme_fusion_config = ThemeFusionConfig.from_config(config)
        self.reader = LatentDatasetReader()
        self.diagnostics = DiagnosticsBase("memory_latent_training")

    def run(self, latent_dir: str | Path, model_dir: str | Path) -> MemoryLatentTrainingResult:
        """Read latent dataset, train model, and write outputs."""
        self._set_seed()
        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mu, rows, latent_summary = self.reader.load(latent_dir)
        self.model_config = self._resolved_model_config(mu, rows)
        sequence_data = LatentSequenceBuilder(
            context_bars=int(self.model_config.context_bars),
            position_vocab_size=int(self.model_config.position_vocab_size),
        ).build(mu, rows)
        self._attach_theme_context(sequence_data.samples, mu, rows)
        train_dataset, val_dataset = self._split_samples(sequence_data.samples)
        model = MemoryLatentTransformer(self.model_config).to(self.training_config.device)
        loss_fn = InBatchMemoryContrastiveLoss(self.training_config.contrastive_temperature)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.training_config.learning_rate, weight_decay=self.training_config.weight_decay)
        fit = self._train(model, loss_fn, optimizer, train_dataset, val_dataset)
        model_path = output_dir / "memory_latent_transformer.pt"
        final_path = output_dir / "memory_latent_transformer.final.pt"
        self._save_model(final_path, model, sequence_data.action_to_id, checkpoint_role="final")
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
            "model_config": self.model_config.to_dict(),
            "theme_fusion_config": self.theme_fusion_config.to_dict(),
            "theme_fusion_runtime": model.theme_fusion_diagnostics(),
            "history": fit.history,
            "model_selection": self._model_selection(fit),
            "saved_checkpoints": {"best": str(model_path), "final": str(final_path)},
        })
        self.diagnostics.record_stage("train_eval", train_eval)
        self.diagnostics.record_stage("val_eval", val_eval)
        summary = {
            "model_path": str(model_path),
            "saved_checkpoints": {"best": str(model_path), "final": str(final_path)},
            "latent_dir": str(latent_dir),
            "sequence_summary": sequence_data.source_summary,
            "action_to_id": sequence_data.action_to_id,
            "dataset_split": self.diagnostics.to_dict().get("stages", {}).get("dataset_split", {}),
            "theme_fusion": self.diagnostics.to_dict().get("stages", {}).get("theme_fusion", {}),
            "epochs": int(self.training_config.epochs),
            "trained_epochs": int(len(fit.history)),
            "final_epoch": fit.history[-1] if fit.history else {},
            "model_selection": self._model_selection(fit),
            "train_eval": train_eval,
            "val_eval": val_eval,
        }
        diagnostics_path = output_dir / "memory_latent_training_diagnostics.json"
        summary_path = output_dir / "memory_latent_training_summary.json"
        self.diagnostics.write(diagnostics_path)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return MemoryLatentTrainingResult(model_path, diagnostics_path, summary_path, self.diagnostics.to_dict())

    def _resolved_model_config(self, mu: np.ndarray, rows: Sequence[Dict[str, Any]]) -> MemoryLatentTransformerConfig:
        """Resolve data-dependent model config."""
        action_to_id = LatentSequenceBuilder(self.model_config.context_bars, self.model_config.position_vocab_size)._action_vocab(rows)
        values = dict(self.model_config.__dict__)
        values["latent_dim"] = int(mu.shape[1])
        values["query_dim"] = int(mu.shape[1])
        values["action_vocab_size"] = max(int(values["action_vocab_size"]), len(action_to_id))
        if bool(self.theme_fusion_config.enabled):
            values["theme_fusion_enabled"] = True
        return MemoryLatentTransformerConfig(**values)

    def _attach_theme_context(self, samples: Sequence[LatentSequenceSample], mu: np.ndarray, rows: Sequence[Dict[str, Any]]) -> None:
        """Attach frozen theme context when enabled."""
        provider = FrozenThemeEmbeddingProvider.from_config(self.config, device=self.training_config.device)
        by_song, token_by_song, diagnostics = provider.theme_contexts_by_song(mu, rows)
        self.diagnostics.record_stage("theme_fusion", diagnostics)
        if not self.theme_fusion_config.enabled:
            return
        token_shape = (int(self.theme_fusion_config.token_bars), int(mu.shape[1]))
        missing_embedding = 0
        missing_tokens = 0
        for sample in samples:
            embedding = by_song.get(sample.song_id)
            if embedding is None:
                missing_embedding += 1
                embedding = np.zeros((int(self.theme_fusion_config.embedding_dim),), dtype=np.float32)
            tokens = token_by_song.get(sample.song_id)
            if tokens is None:
                missing_tokens += 1
                tokens = np.zeros(token_shape, dtype=np.float32)
            sample.theme_embedding = np.asarray(embedding, dtype=np.float32)
            sample.theme_tokens = np.asarray(tokens, dtype=np.float32)
        diagnostics["missing_embedding_sample_count"] = int(missing_embedding)
        diagnostics["missing_token_sample_count"] = int(missing_tokens)

    def _split_samples(self, samples: Sequence[LatentSequenceSample]) -> tuple[MemoryLatentDataset, MemoryLatentDataset]:
        """Split by base_song_id or sample."""
        if self.training_config.validation_split_unit == "base_song_id":
            groups: Dict[str, List[LatentSequenceSample]] = {}
            for sample in samples:
                groups.setdefault(sample.base_song_id, []).append(sample)
            base_ids = np.asarray(sorted(groups.keys()), dtype=object)
            rng = np.random.default_rng(int(self.training_config.random_seed))
            rng.shuffle(base_ids)
            fold_count = min(max(1, int(self.training_config.validation_fold_count)), len(base_ids))
            fold_index = int(self.training_config.validation_fold_index) % fold_count
            folds = np.array_split(base_ids, fold_count)
            val_base_ids = {str(base_id) for base_id in folds[fold_index].tolist()}
            train_samples = [sample for base_id, group in groups.items() if base_id not in val_base_ids for sample in group]
            val_samples = [sample for base_id, group in groups.items() if base_id in val_base_ids for sample in group]
            self.diagnostics.record_stage("dataset_split", {
                "total_size": int(len(samples)),
                "train_size": int(len(train_samples)),
                "validation_size": int(len(val_samples)),
                "split_unit": "base_song_id",
                "split_strategy": "group_k_fold",
                "validation_fold_count": int(fold_count),
                "validation_fold_index": int(fold_index),
                "base_song_count": int(len(groups)),
                "validation_base_song_ids": sorted(val_base_ids),
            })
            return MemoryLatentDataset(train_samples), MemoryLatentDataset(val_samples)
        indices = np.arange(len(samples))
        rng = np.random.default_rng(int(self.training_config.random_seed))
        rng.shuffle(indices)
        val_size = max(1, min(len(indices) - 1, int(round(len(indices) * self.training_config.validation_ratio))))
        val_indices = {int(index) for index in indices[:val_size]}
        train_samples = [sample for index, sample in enumerate(samples) if index not in val_indices]
        val_samples = [sample for index, sample in enumerate(samples) if index in val_indices]
        self.diagnostics.record_stage("dataset_split", {
            "total_size": int(len(samples)),
            "train_size": int(len(train_samples)),
            "validation_size": int(len(val_samples)),
            "split_unit": "sample",
        })
        return MemoryLatentDataset(train_samples), MemoryLatentDataset(val_samples)

    def _train(
        self,
        model: MemoryLatentTransformer,
        loss_fn: InBatchMemoryContrastiveLoss,
        optimizer: torch.optim.Optimizer,
        train_dataset: MemoryLatentDataset,
        val_dataset: MemoryLatentDataset,
    ) -> MemoryFitResult:
        """Train with early stopping on val loss."""
        generator = torch.Generator().manual_seed(int(self.training_config.random_seed))
        train_loader = DataLoader(train_dataset, batch_size=self.training_config.batch_size, shuffle=True, generator=generator, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.training_config.batch_size, shuffle=False, drop_last=False)
        best_metric = math.inf
        best_epoch = 0
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        stale_epochs = 0
        history: List[Dict[str, float]] = []
        early_stopped = False
        stopped_epoch = 0
        for epoch in range(int(self.training_config.epochs)):
            train_metrics = self._run_epoch(model, loss_fn, train_loader, optimizer)
            val_metrics = self._run_epoch(model, loss_fn, val_loader, None)
            row = {"epoch": float(epoch + 1)}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            metric = float(row.get("val_loss", math.inf))
            if metric < best_metric - float(self.training_config.early_stopping_min_delta):
                best_metric = metric
                best_epoch = int(epoch + 1)
                stale_epochs = 0
                best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale_epochs += 1
            row["best_epoch"] = float(best_epoch)
            row["best_metric"] = float(best_metric)
            row["stale_epochs"] = float(stale_epochs)
            history.append(row)
            if int(self.training_config.early_stopping_patience) > 0 and stale_epochs >= int(self.training_config.early_stopping_patience):
                early_stopped = True
                stopped_epoch = int(epoch + 1)
                break
        return MemoryFitResult(history, best_epoch, best_metric, early_stopped, stopped_epoch, best_state_dict)

    def _run_epoch(
        self,
        model: MemoryLatentTransformer,
        loss_fn: InBatchMemoryContrastiveLoss,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Dict[str, float]:
        """Run one train/eval epoch."""
        is_train = optimizer is not None
        model.train(is_train)
        totals: Dict[str, float] = {}
        count = 0
        for batch in loader:
            prepared = self._prepare_batch(batch)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(is_train):
                query = model(
                    context_mu=prepared["context_mu"],
                    context_action_ids=prepared["context_action_ids"],
                    context_position_ids=prepared["context_position_ids"],
                    target_action_ids=prepared["target_action_ids"],
                    target_position_ids=prepared["target_position_ids"],
                    padding_mask=prepared["padding_mask"],
                    theme_embedding=prepared["theme_embedding"],
                    theme_tokens=prepared["theme_tokens"],
                )
                losses = loss_fn(query, prepared["target_mu"])
                if is_train:
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            batch_size = int(prepared["target_mu"].shape[0])
            count += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
        if count == 0:
            return {"loss": 0.0, "top1": 0.0, "top5": 0.0, "mrr": 0.0, "positive_margin": 0.0}
        return {key: value / count for key, value in totals.items()}

    def _evaluate(self, model: MemoryLatentTransformer, loss_fn: InBatchMemoryContrastiveLoss, dataset: MemoryLatentDataset) -> Dict[str, float]:
        """Evaluate dataset."""
        return self._run_epoch(model, loss_fn, DataLoader(dataset, batch_size=self.training_config.batch_size, shuffle=False), None)

    def _prepare_batch(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        names = [
            "context_mu",
            "context_action_ids",
            "context_position_ids",
            "padding_mask",
            "target_mu",
            "target_action_ids",
            "target_position_ids",
            "target_row_indices",
            "theme_embedding",
            "theme_tokens",
        ]
        return {name: value.to(self.training_config.device) for name, value in zip(names, batch)}

    def _save_model(self, path: Path, model: MemoryLatentTransformer, action_to_id: Dict[str, int], checkpoint_role: str) -> None:
        """Save checkpoint."""
        torch.save({
            "model_type": "MemoryLatentTransformer",
            "checkpoint_role": checkpoint_role,
            "model_config": self.model_config.to_dict(),
            "training_config": self.training_config.__dict__,
            "theme_fusion_config": self.theme_fusion_config.to_dict(),
            "action_to_id": action_to_id,
            "id_to_action": {str(value): key for key, value in action_to_id.items()},
            "state_dict": model.state_dict(),
        }, path)

    def _model_selection(self, fit: MemoryFitResult) -> Dict[str, Any]:
        """Return model selection summary."""
        return {
            "monitor_metric": "val_loss",
            "best_epoch": int(fit.best_epoch),
            "best_metric": float(fit.best_metric),
            "early_stopped": bool(fit.early_stopped),
            "stopped_epoch": int(fit.stopped_epoch),
            "early_stopping_patience": int(self.training_config.early_stopping_patience),
        }

    def _with_overrides(self, base: MemoryLatentTrainingConfig, overrides: Dict[str, Any]) -> MemoryLatentTrainingConfig:
        """Apply CLI overrides."""
        values = dict(base.__dict__)
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        if str(values.get("device", "cpu")).startswith("cuda") and not torch.cuda.is_available():
            values["device"] = "cpu"
        return MemoryLatentTrainingConfig(**values)

    def _set_seed(self) -> None:
        """Seed all RNGs."""
        seed = int(self.training_config.random_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
