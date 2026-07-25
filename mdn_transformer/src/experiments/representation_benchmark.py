#!/usr/bin/env python3
"""Representation benchmark for next-bar transition learning.

This experiment is intentionally standalone. It reads exported latent data and
encoded bar tensors, trains the same lightweight next-step predictor on
different representations, and writes diagnostics without touching generation
or training pipelines.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for one benchmark run."""

    model_dir: Path
    latent_dir: Path
    output_dir: Path
    representations: tuple[str, ...] = (
        "dvae_latent",
        "bar_features",
        "hybrid_latent_features",
        "hybrid_transition_features",
        "hybrid_dual_head",
        "hybrid_transition_composer",
        "hybrid_anchor_motion_composer",
        "hybrid_anchor_motion_movement_composer",
    )
    context_bars: int = 8
    epochs: int = 20
    batch_size: int = 256
    hidden_dim: int = 256
    composer_hidden_dim: Optional[int] = None
    composer_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    validation_ratio: float = 0.1
    validation_fold_count: int = 5
    validation_fold_index: int = 0
    max_songs: Optional[int] = None
    max_rows: Optional[int] = None
    ranking_eval_samples: int = 512
    ranking_candidates: int = 5000
    delta_loss_weight: float = 0.5
    consistency_loss_weight: float = 0.25
    fusion_state_weight: float = 0.5
    composer_state_loss_weight: float = 0.3
    composer_delta_loss_weight: float = 0.3
    random_seed: int = 42
    device: str = "cpu"


@dataclass
class RepresentationDataset:
    """One representation matrix and row metadata."""

    name: str
    values: np.ndarray
    rows: List[Dict[str, Any]]
    selected_global_indices: List[int]
    groups: Dict[str, List[int]]
    context_mode: str = "state"
    task_mode: str = "single_head"


@dataclass
class SequenceSample:
    """One next-bar prediction sample."""

    context_indices: List[int]
    target_index: int
    current_index: int
    song_id: str
    base_song_id: str
    target_bar_index: int


class SequenceDataset(Dataset):
    """Torch dataset for fixed-context next-bar prediction."""

    def __init__(
        self,
        representation: np.ndarray,
        samples: Sequence[SequenceSample],
        context_bars: int,
        mean: np.ndarray,
        std: np.ndarray,
        context_mode: str = "state",
    ) -> None:
        self.representation = np.asarray(representation, dtype=np.float32)
        self.samples = list(samples)
        self.context_bars = int(context_bars)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.dim = int(self.representation.shape[1])
        self.context_mode = str(context_mode)

    def __len__(self) -> int:
        """Return sample count."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return flattened context, target, current vector, target index, and current index."""
        sample = self.samples[index]
        context = np.zeros((self.context_bars, self.dim), dtype=np.float32)
        mask = np.ones((self.context_bars, 1), dtype=np.float32)
        valid = np.zeros((self.context_bars,), dtype=bool)
        offset = self.context_bars - len(sample.context_indices)
        for local, row_index in enumerate(sample.context_indices[-self.context_bars:]):
            slot = offset + local
            context[slot] = self._normalize(self.representation[row_index])
            mask[slot, 0] = 0.0
            valid[slot] = True
        features = self._context_features(context, mask, valid)
        target = self._normalize(self.representation[sample.target_index])
        current = self._normalize(self.representation[sample.current_index])
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(target).float(),
            torch.from_numpy(current).float(),
            torch.tensor(sample.target_index, dtype=torch.long),
            torch.tensor(sample.current_index, dtype=torch.long),
        )

    def _normalize(self, value: np.ndarray) -> np.ndarray:
        """Normalize one representation vector."""
        return (np.asarray(value, dtype=np.float32) - self.mean) / self.std

    def _context_features(self, context: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Return flattened state-only or state-plus-delta context features."""
        if self.context_mode == "state":
            return np.concatenate([context.reshape(-1), mask.reshape(-1)], axis=0)
        if self.context_mode not in {"state_delta", "state_delta_steps"}:
            raise ValueError(f"Unsupported context_mode: {self.context_mode}")
        delta = np.zeros_like(context, dtype=np.float32)
        delta_mask = np.ones((self.context_bars, 1), dtype=np.float32)
        for slot in range(1, self.context_bars):
            if bool(valid[slot]) and bool(valid[slot - 1]):
                delta[slot] = context[slot] - context[slot - 1]
                delta_mask[slot, 0] = 0.0
        if self.context_mode == "state_delta_steps":
            return np.concatenate([context, delta, mask, delta_mask], axis=1).reshape(-1)
        return np.concatenate([context.reshape(-1), delta.reshape(-1), mask.reshape(-1), delta_mask.reshape(-1)], axis=0)


class NextStepMLP(nn.Module):
    """Small predictor shared by all representations."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Predict the next normalized representation."""
        return self.net(value)


class DualHeadMLP(nn.Module):
    """Predict next state and next transition as separate objectives."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.state_head = nn.Linear(hidden_dim, output_dim)
        self.delta_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, value: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return state and delta predictions."""
        hidden = self.trunk(value)
        return {
            "state": self.state_head(hidden),
            "delta": self.delta_head(hidden),
        }


def _composer_mlp(input_dim: int, output_dim: int, hidden_dim: int, layers: int, dropout: float) -> nn.Sequential:
    """Build a configurable Composer MLP."""
    layer_count = max(1, int(layers))
    modules: List[nn.Module] = []
    current_dim = int(input_dim)
    for _ in range(layer_count):
        modules.extend([
            nn.Linear(current_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        ])
        current_dim = int(hidden_dim)
    modules.append(nn.Linear(current_dim, int(output_dim)))
    return nn.Sequential(*modules)


class TransformerTransitionComposer(nn.Module):
    """Predict state and motion, then compose them into the next state."""

    def __init__(
        self,
        representation_dim: int,
        context_bars: int,
        hidden_dim: int,
        composer_hidden_dim: Optional[int],
        composer_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.representation_dim = int(representation_dim)
        self.context_bars = int(context_bars)
        self.step_dim = int(2 * representation_dim + 2)
        nhead = self._attention_heads(int(hidden_dim))
        self.input_proj = nn.Linear(self.step_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, self.context_bars, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=int(hidden_dim * 2),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=2)
        self.state_head = nn.Linear(hidden_dim, representation_dim)
        self.delta_head = nn.Linear(hidden_dim, representation_dim)
        self.composer = _composer_mlp(
            input_dim=int(3 * representation_dim),
            output_dim=int(representation_dim),
            hidden_dim=int(composer_hidden_dim or hidden_dim),
            layers=int(composer_layers),
            dropout=float(dropout),
        )

    def forward(self, value: torch.Tensor, current: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return state anchor, delta, and composed next state."""
        batch_size = int(value.shape[0])
        sequence = value.reshape(batch_size, self.context_bars, self.step_dim)
        hidden = self.input_proj(sequence) + self.position
        causal_mask = torch.triu(
            torch.full((self.context_bars, self.context_bars), float("-inf"), device=value.device),
            diagonal=1,
        )
        encoded = self.trunk(hidden, mask=causal_mask)
        pooled = encoded[:, -1, :]
        state_anchor = self.state_head(pooled)
        delta = self.delta_head(pooled)
        base = current + delta
        correction = self.composer(torch.cat([current, state_anchor, delta], dim=1))
        return {
            "state": state_anchor,
            "delta": delta,
            "composed": base + correction,
        }

    def _attention_heads(self, hidden_dim: int) -> int:
        """Choose a valid small attention head count."""
        for candidate in (8, 4, 2):
            if hidden_dim % candidate == 0:
                return candidate
        return 1


class TransformerAnchorMotionComposer(nn.Module):
    """Learn hidden anchor and motion variables, supervised only by next-state loss."""

    def __init__(
        self,
        representation_dim: int,
        context_bars: int,
        hidden_dim: int,
        composer_hidden_dim: Optional[int],
        composer_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.representation_dim = int(representation_dim)
        self.context_bars = int(context_bars)
        self.step_dim = int(2 * representation_dim + 2)
        nhead = self._attention_heads(int(hidden_dim))
        self.input_proj = nn.Linear(self.step_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, self.context_bars, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=int(hidden_dim * 2),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=2)
        self.anchor_head = nn.Linear(hidden_dim, representation_dim)
        self.motion_head = nn.Linear(hidden_dim, representation_dim)
        self.composer = _composer_mlp(
            input_dim=int(3 * representation_dim),
            output_dim=int(representation_dim),
            hidden_dim=int(composer_hidden_dim or hidden_dim),
            layers=int(composer_layers),
            dropout=float(dropout),
        )

    def forward(self, value: torch.Tensor, current: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return hidden anchor, hidden motion, and composed next state."""
        batch_size = int(value.shape[0])
        sequence = value.reshape(batch_size, self.context_bars, self.step_dim)
        hidden = self.input_proj(sequence) + self.position
        causal_mask = torch.triu(
            torch.full((self.context_bars, self.context_bars), float("-inf"), device=value.device),
            diagonal=1,
        )
        encoded = self.trunk(hidden, mask=causal_mask)
        pooled = encoded[:, -1, :]
        anchor = self.anchor_head(pooled)
        motion = self.motion_head(pooled)
        composed = self.composer(torch.cat([current, anchor, motion], dim=1))
        return {
            "anchor": anchor,
            "motion": motion,
            "composed": composed,
        }

    def _attention_heads(self, hidden_dim: int) -> int:
        """Choose a valid small attention head count."""
        for candidate in (8, 4, 2):
            if hidden_dim % candidate == 0:
                return candidate
        return 1


class RepresentationBenchmark:
    """Run representation comparison for transition learning."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.rng = random.Random(int(config.random_seed))
        self.np_rng = np.random.default_rng(int(config.random_seed))
        self._bar_feature_cache: Optional[np.ndarray] = None

    def run(self) -> Dict[str, Any]:
        """Run all requested representations and write reports."""
        self._set_seed()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        mu, rows, summary = self._load_latent()
        selected_indices = self._select_rows(rows)
        selected_rows = [rows[index] for index in selected_indices]
        groups = self._group_selected_rows(selected_rows)
        samples = self._build_samples(groups, selected_rows)
        train_samples, val_samples, split_diag = self._split_samples(samples)
        if not train_samples or not val_samples:
            raise ValueError("Benchmark needs non-empty train and validation samples.")

        results: Dict[str, Any] = {
            "config": self._config_dict(),
            "input": {
                "latent_summary": summary,
                "selected_row_count": int(len(selected_rows)),
                "selected_song_count": int(len(groups)),
                "sample_count": int(len(samples)),
            },
            "split": split_diag,
            "representations": {},
        }
        for representation_name in self.config.representations:
            dataset = self._build_representation(representation_name, mu, rows, selected_indices, selected_rows, groups)
            results["representations"][representation_name] = self._run_representation(
                dataset=dataset,
                train_samples=train_samples,
                val_samples=val_samples,
            )
        self._write_reports(results)
        return results

    def _run_representation(
        self,
        dataset: RepresentationDataset,
        train_samples: Sequence[SequenceSample],
        val_samples: Sequence[SequenceSample],
    ) -> Dict[str, Any]:
        """Train and evaluate one representation."""
        values = np.asarray(dataset.values, dtype=np.float32)
        train_targets = np.asarray([sample.target_index for sample in train_samples], dtype=np.int64)
        mean = values[train_targets].mean(axis=0).astype(np.float32)
        std = values[train_targets].std(axis=0).astype(np.float32)
        std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
        train_dataset = SequenceDataset(values, train_samples, self.config.context_bars, mean, std, context_mode=dataset.context_mode)
        val_dataset = SequenceDataset(values, val_samples, self.config.context_bars, mean, std, context_mode=dataset.context_mode)
        input_dim = self._input_dim(int(values.shape[1]), dataset.context_mode)
        if dataset.task_mode in {"anchor_motion_composer", "anchor_motion_movement_composer"}:
            model: nn.Module = TransformerAnchorMotionComposer(
                representation_dim=int(values.shape[1]),
                context_bars=int(self.config.context_bars),
                hidden_dim=int(self.config.hidden_dim),
                composer_hidden_dim=self.config.composer_hidden_dim,
                composer_layers=int(self.config.composer_layers),
                dropout=float(self.config.dropout),
            ).to(self.config.device)
        elif dataset.task_mode == "transition_composer":
            model: nn.Module = TransformerTransitionComposer(
                representation_dim=int(values.shape[1]),
                context_bars=int(self.config.context_bars),
                hidden_dim=int(self.config.hidden_dim),
                composer_hidden_dim=self.config.composer_hidden_dim,
                composer_layers=int(self.config.composer_layers),
                dropout=float(self.config.dropout),
            ).to(self.config.device)
        elif dataset.task_mode == "dual_head":
            model: nn.Module = DualHeadMLP(
                input_dim=input_dim,
                output_dim=int(values.shape[1]),
                hidden_dim=int(self.config.hidden_dim),
                dropout=float(self.config.dropout),
            ).to(self.config.device)
        else:
            model = NextStepMLP(
                input_dim=input_dim,
                output_dim=int(values.shape[1]),
                hidden_dim=int(self.config.hidden_dim),
                dropout=float(self.config.dropout),
            ).to(self.config.device)
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
            train_metrics = self._run_epoch(model, train_dataset, optimizer, loss_fn)
            val_metrics = self._run_epoch(model, val_dataset, None, loss_fn)
            row = {
                "epoch": float(epoch + 1),
                "train_mse": float(train_metrics["mse"]),
                "val_mse": float(val_metrics["mse"]),
            }
            for key in (
                "state_mse",
                "delta_mse",
                "consistency_mse",
                "anchor_mse",
                "motion_reconstructed_mse",
                "composed_mse",
                "movement_position_loss",
                "movement_direction_loss",
                "movement_magnitude_loss",
            ):
                if key in train_metrics:
                    row[f"train_{key}"] = float(train_metrics[key])
                if key in val_metrics:
                    row[f"val_{key}"] = float(val_metrics[key])
            history.append(row)
            if row["val_mse"] < best_val:
                best_val = row["val_mse"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        train_eval = self._evaluate(model, train_dataset, values, mean, std, split_name="train")
        val_eval = self._evaluate(model, val_dataset, values, mean, std, split_name="validation")
        return {
            "representation_dim": int(values.shape[1]),
            "context_mode": str(dataset.context_mode),
            "task_mode": str(dataset.task_mode),
            "input_dim": int(input_dim),
            "loss_weights": {
                "delta": float(self.config.delta_loss_weight),
                "consistency": float(self.config.consistency_loss_weight),
                "fusion_state": float(self.config.fusion_state_weight),
                "composer_state": float(self.config.composer_state_loss_weight),
                "composer_delta": float(self.config.composer_delta_loss_weight),
                "composer_hidden_dim": int(self.config.composer_hidden_dim or self.config.hidden_dim),
                "composer_layers": int(self.config.composer_layers),
            },
            "history": history,
            "best_val_mse": float(best_val),
            "train_eval": train_eval,
            "val_eval": val_eval,
        }

    def _input_dim(self, representation_dim: int, context_mode: str) -> int:
        """Return model input dimension for a context mode."""
        if context_mode == "state":
            return int(self.config.context_bars * representation_dim + self.config.context_bars)
        if context_mode in {"state_delta", "state_delta_steps"}:
            return int(2 * self.config.context_bars * representation_dim + 2 * self.config.context_bars)
        raise ValueError(f"Unsupported context_mode: {context_mode}")

    def _run_epoch(
        self,
        model: nn.Module,
        dataset: SequenceDataset,
        optimizer: Optional[torch.optim.Optimizer],
        loss_fn: nn.Module,
    ) -> Dict[str, float]:
        """Run one training or evaluation epoch."""
        loader = DataLoader(
            dataset,
            batch_size=int(self.config.batch_size),
            shuffle=optimizer is not None,
        )
        model.train(optimizer is not None)
        losses: List[float] = []
        state_losses: List[float] = []
        delta_losses: List[float] = []
        consistency_losses: List[float] = []
        anchor_losses: List[float] = []
        motion_reconstructed_losses: List[float] = []
        composed_losses: List[float] = []
        movement_position_losses: List[float] = []
        movement_direction_losses: List[float] = []
        movement_magnitude_losses: List[float] = []
        for context, target, current, _target_index, _current_index in loader:
            context = context.to(self.config.device)
            target = target.to(self.config.device)
            current = current.to(self.config.device)
            if isinstance(model, (TransformerTransitionComposer, TransformerAnchorMotionComposer)):
                output = model(context, current)
            else:
                output = model(context)
            if isinstance(output, dict):
                if "anchor" in output:
                    anchor_prediction = output["anchor"]
                    motion_prediction = output["motion"]
                    composed_prediction = output["composed"]
                    motion_reconstructed = current + motion_prediction
                    anchor_loss = loss_fn(anchor_prediction, target)
                    motion_reconstructed_loss = loss_fn(motion_reconstructed, target)
                    composed_loss = loss_fn(composed_prediction, target)
                    if getattr(model, "benchmark_task_mode", "") == "anchor_motion_movement_composer":
                        loss, movement_parts = self._movement_loss(composed_prediction, target, current)
                        movement_position_losses.append(float(movement_parts["position"].detach().cpu()))
                        movement_direction_losses.append(float(movement_parts["direction"].detach().cpu()))
                        movement_magnitude_losses.append(float(movement_parts["magnitude"].detach().cpu()))
                    else:
                        loss = composed_loss
                    anchor_losses.append(float(anchor_loss.detach().cpu()))
                    motion_reconstructed_losses.append(float(motion_reconstructed_loss.detach().cpu()))
                    composed_losses.append(float(composed_loss.detach().cpu()))
                else:
                    state_prediction = output["state"]
                    delta_prediction = output["delta"]
                    delta_target = target - current
                    state_loss = loss_fn(state_prediction, target)
                    delta_loss = loss_fn(delta_prediction, delta_target)
                    if "composed" in output:
                        composed_prediction = output["composed"]
                        consistency_loss = loss_fn(composed_prediction, target)
                        loss = (
                            consistency_loss
                            + float(self.config.composer_state_loss_weight) * state_loss
                            + float(self.config.composer_delta_loss_weight) * delta_loss
                        )
                    else:
                        reconstructed_prediction = current + delta_prediction
                        consistency_loss = loss_fn(state_prediction, reconstructed_prediction)
                        loss = (
                            state_loss
                            + float(self.config.delta_loss_weight) * delta_loss
                            + float(self.config.consistency_loss_weight) * consistency_loss
                        )
                    state_losses.append(float(state_loss.detach().cpu()))
                    delta_losses.append(float(delta_loss.detach().cpu()))
                    consistency_losses.append(float(consistency_loss.detach().cpu()))
            else:
                loss = loss_fn(output, target)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
        result = {"mse": float(np.mean(losses)) if losses else math.inf}
        if state_losses:
            result["state_mse"] = float(np.mean(state_losses))
            result["delta_mse"] = float(np.mean(delta_losses))
            result["consistency_mse"] = float(np.mean(consistency_losses))
        if anchor_losses:
            result["anchor_mse"] = float(np.mean(anchor_losses))
            result["motion_reconstructed_mse"] = float(np.mean(motion_reconstructed_losses))
            result["composed_mse"] = float(np.mean(composed_losses))
        if movement_position_losses:
            result["movement_position_loss"] = float(np.mean(movement_position_losses))
            result["movement_direction_loss"] = float(np.mean(movement_direction_losses))
            result["movement_magnitude_loss"] = float(np.mean(movement_magnitude_losses))
        return result

    def _movement_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        current: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return position, direction, and magnitude movement loss without manual weights."""
        position_loss = torch.nn.functional.smooth_l1_loss(prediction, target)
        pred_delta = prediction - current
        true_delta = target - current
        direction_loss = (
            1.0
            - torch.nn.functional.cosine_similarity(pred_delta, true_delta, dim=1, eps=1.0e-8)
        ).mean()
        pred_magnitude = torch.linalg.norm(pred_delta, dim=1)
        true_magnitude = torch.linalg.norm(true_delta, dim=1)
        magnitude_loss = torch.nn.functional.smooth_l1_loss(pred_magnitude, true_magnitude)
        return position_loss + direction_loss + magnitude_loss, {
            "position": position_loss,
            "direction": direction_loss,
            "magnitude": magnitude_loss,
        }

    def _evaluate(
        self,
        model: nn.Module,
        dataset: SequenceDataset,
        all_values: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        split_name: str,
    ) -> Dict[str, Any]:
        """Evaluate prediction quality and ranking behavior."""
        loader = DataLoader(dataset, batch_size=int(self.config.batch_size), shuffle=False)
        model.eval()
        target_indices: List[int] = []
        current_indices: List[int] = []
        targets: List[np.ndarray] = []
        currents: List[np.ndarray] = []
        predictions_by_output: Dict[str, List[np.ndarray]] = {}
        with torch.no_grad():
            for context, target, current, target_index, current_index in loader:
                context_device = context.to(self.config.device)
                current_device = current.to(self.config.device)
                if isinstance(model, (TransformerTransitionComposer, TransformerAnchorMotionComposer)):
                    output = model(context_device, current_device)
                else:
                    output = model(context_device)
                target = target.float()
                current = current.float()
                if isinstance(output, dict):
                    if "anchor" in output:
                        batch_predictions = {
                            "anchor": output["anchor"].detach().cpu(),
                            "motion_reconstructed": current + output["motion"].detach().cpu(),
                            "composed": output["composed"].detach().cpu(),
                        }
                    else:
                        state_prediction = output["state"].detach().cpu()
                        delta_prediction = output["delta"].detach().cpu()
                        delta_reconstructed = current + delta_prediction
                        if "composed" in output:
                            batch_predictions = {
                                "state_head": state_prediction,
                                "delta_reconstructed": delta_reconstructed,
                                "composed": output["composed"].detach().cpu(),
                            }
                        else:
                            fusion_weight = float(self.config.fusion_state_weight)
                            fused = fusion_weight * state_prediction + (1.0 - fusion_weight) * delta_reconstructed
                            batch_predictions = {
                                "state_head": state_prediction,
                                "delta_reconstructed": delta_reconstructed,
                                "fused": fused,
                            }
                else:
                    batch_predictions = {"single": output.detach().cpu()}
                for name, prediction in batch_predictions.items():
                    predictions_by_output.setdefault(name, []).append(prediction.numpy().astype(np.float32))
                targets.append(target.numpy().astype(np.float32))
                currents.append(current.numpy().astype(np.float32))
                target_indices.extend(int(item) for item in target_index.numpy())
                current_indices.extend(int(item) for item in current_index.numpy())
        target_array = np.concatenate(targets, axis=0) if targets else np.zeros((0, all_values.shape[1]), dtype=np.float32)
        current_array = np.concatenate(currents, axis=0) if currents else np.zeros((0, all_values.shape[1]), dtype=np.float32)
        target_index_array = np.asarray(target_indices, dtype=np.int64)
        all_values_normalized = (all_values.astype(np.float32) - mean) / std
        output_eval: Dict[str, Dict[str, Any]] = {}
        for output_name, prediction_parts in predictions_by_output.items():
            prediction_array = np.concatenate(prediction_parts, axis=0)
            output_eval[output_name] = self._prediction_metrics(
                prediction=prediction_array,
                target=target_array,
                current=current_array,
                target_indices=target_index_array,
                all_values=all_values_normalized,
                split_name=split_name,
            )
        selected_output = "composed" if "composed" in output_eval else ("fused" if "fused" in output_eval else "single")
        selected = dict(output_eval.get(selected_output, self._empty_eval()))
        selected["selected_output"] = selected_output
        if len(output_eval) > 1:
            selected["output_eval"] = output_eval
        return selected

    def _prediction_metrics(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        current: np.ndarray,
        target_indices: np.ndarray,
        all_values: np.ndarray,
        split_name: str,
    ) -> Dict[str, Any]:
        """Compute metrics for one predicted next-state matrix."""
        if len(prediction) == 0:
            return self._empty_eval()
        prediction_tensor = torch.from_numpy(prediction.astype(np.float32))
        target_tensor = torch.from_numpy(target.astype(np.float32))
        current_tensor = torch.from_numpy(current.astype(np.float32))
        mse_values = torch.mean((prediction_tensor - target_tensor) ** 2, dim=1).numpy()
        pred_delta = prediction_tensor - current_tensor
        true_delta = target_tensor - current_tensor
        direction_cosines = torch.nn.functional.cosine_similarity(pred_delta, true_delta, dim=1, eps=1.0e-8).numpy()
        pred_to_current = torch.linalg.norm(prediction_tensor - current_tensor, dim=1).numpy()
        true_to_current = torch.linalg.norm(target_tensor - current_tensor, dim=1).numpy()
        pred_to_true = torch.linalg.norm(prediction_tensor - target_tensor, dim=1).numpy()
        ranking = self._ranking_metrics(
            predictions=prediction,
            target_indices=target_indices,
            all_values=all_values,
            split_name=split_name,
        )
        self_loop_like = pred_to_current < pred_to_true
        under_moving = pred_to_current < (0.5 * np.maximum(true_to_current, 1.0e-8))
        return {
            "sample_count": int(len(mse_values)),
            "mse": self._numeric_summary([float(item) for item in mse_values]),
            "direction_cosine": self._numeric_summary([float(item) for item in direction_cosines]),
            "pred_to_current_distance": self._numeric_summary([float(item) for item in pred_to_current]),
            "true_to_current_distance": self._numeric_summary([float(item) for item in true_to_current]),
            "pred_to_true_distance": self._numeric_summary([float(item) for item in pred_to_true]),
            "pred_closer_to_current_than_true_rate": float(np.mean(self_loop_like)) if len(self_loop_like) else 0.0,
            "under_moving_rate": float(np.mean(under_moving)) if len(under_moving) else 0.0,
            "ranking": ranking,
        }

    def _empty_eval(self) -> Dict[str, Any]:
        """Return an empty metric payload."""
        return {
            "sample_count": 0,
            "mse": {"n": 0},
            "direction_cosine": {"n": 0},
            "pred_to_current_distance": {"n": 0},
            "true_to_current_distance": {"n": 0},
            "pred_to_true_distance": {"n": 0},
            "pred_closer_to_current_than_true_rate": 0.0,
            "under_moving_rate": 0.0,
            "ranking": {"sample_count": 0},
        }

    def _ranking_metrics(
        self,
        predictions: np.ndarray,
        target_indices: np.ndarray,
        all_values: np.ndarray,
        split_name: str,
    ) -> Dict[str, Any]:
        """Estimate whether the true next bar is near the prediction among candidates."""
        if len(predictions) == 0:
            return {"sample_count": 0}
        eval_count = min(int(self.config.ranking_eval_samples), int(len(predictions)))
        sample_indices = self.np_rng.choice(len(predictions), size=eval_count, replace=False)
        candidate_count = min(int(self.config.ranking_candidates), int(len(all_values)))
        random_candidates = self.np_rng.choice(len(all_values), size=candidate_count, replace=False)
        ranks: List[int] = []
        top1 = top5 = top10 = 0
        for sample_index in sample_indices:
            target_index = int(target_indices[int(sample_index)])
            candidates = np.unique(np.concatenate([random_candidates, np.asarray([target_index], dtype=np.int64)]))
            values = all_values[candidates]
            distances = np.linalg.norm(values - predictions[int(sample_index)][None, :], axis=1)
            order = np.argsort(distances)
            target_local = int(np.where(candidates == target_index)[0][0])
            rank = int(np.where(order == target_local)[0][0]) + 1
            ranks.append(rank)
            top1 += int(rank <= 1)
            top5 += int(rank <= 5)
            top10 += int(rank <= 10)
        return {
            "split": split_name,
            "sample_count": int(eval_count),
            "candidate_count": int(candidate_count),
            "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
            "median_rank": float(np.median(ranks)) if ranks else 0.0,
            "top1": float(top1 / max(1, eval_count)),
            "top5": float(top5 / max(1, eval_count)),
            "top10": float(top10 / max(1, eval_count)),
        }

    def _build_representation(
        self,
        name: str,
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        selected_indices: Sequence[int],
        selected_rows: Sequence[Dict[str, Any]],
        groups: Dict[str, List[int]],
    ) -> RepresentationDataset:
        """Build one representation matrix."""
        key = str(name).strip()
        if key == "dvae_latent":
            values = np.asarray(mu[list(selected_indices)], dtype=np.float32)
            context_mode = "state"
            task_mode = "single_head"
        elif key == "bar_features":
            values = self._cached_bar_feature_matrix(selected_rows)
            context_mode = "state"
            task_mode = "single_head"
        elif key == "hybrid_latent_features":
            latent = np.asarray(mu[list(selected_indices)], dtype=np.float32)
            features = self._cached_bar_feature_matrix(selected_rows)
            values = np.concatenate([latent, features], axis=1).astype(np.float32)
            context_mode = "state"
            task_mode = "single_head"
        elif key == "hybrid_transition_features":
            latent = np.asarray(mu[list(selected_indices)], dtype=np.float32)
            features = self._cached_bar_feature_matrix(selected_rows)
            values = np.concatenate([latent, features], axis=1).astype(np.float32)
            context_mode = "state_delta"
            task_mode = "single_head"
        elif key == "hybrid_dual_head":
            latent = np.asarray(mu[list(selected_indices)], dtype=np.float32)
            features = self._cached_bar_feature_matrix(selected_rows)
            values = np.concatenate([latent, features], axis=1).astype(np.float32)
            context_mode = "state_delta"
            task_mode = "dual_head"
        elif key == "hybrid_transition_composer":
            latent = np.asarray(mu[list(selected_indices)], dtype=np.float32)
            features = self._cached_bar_feature_matrix(selected_rows)
            values = np.concatenate([latent, features], axis=1).astype(np.float32)
            context_mode = "state_delta_steps"
            task_mode = "transition_composer"
        elif key == "hybrid_anchor_motion_composer":
            latent = np.asarray(mu[list(selected_indices)], dtype=np.float32)
            features = self._cached_bar_feature_matrix(selected_rows)
            values = np.concatenate([latent, features], axis=1).astype(np.float32)
            context_mode = "state_delta_steps"
            task_mode = "anchor_motion_composer"
        elif key == "hybrid_anchor_motion_movement_composer":
            latent = np.asarray(mu[list(selected_indices)], dtype=np.float32)
            features = self._cached_bar_feature_matrix(selected_rows)
            values = np.concatenate([latent, features], axis=1).astype(np.float32)
            context_mode = "state_delta_steps"
            task_mode = "anchor_motion_movement_composer"
        else:
            raise ValueError(f"Unsupported representation: {name}")
        return RepresentationDataset(
            name=key,
            values=values,
            rows=list(selected_rows),
            selected_global_indices=list(selected_indices),
            groups=groups,
            context_mode=context_mode,
            task_mode=task_mode,
        )

    def _cached_bar_feature_matrix(self, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Return cached explicit bar feature matrix for this benchmark run."""
        if self._bar_feature_cache is None:
            self._bar_feature_cache = self._bar_feature_matrix(rows)
        return np.asarray(self._bar_feature_cache, dtype=np.float32)

    def _bar_feature_matrix(self, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Compute explicit bar feature vectors from encoded tensors."""
        tensor_path = self.config.model_dir / "encoded" / "bar_tensors.npz"
        if not tensor_path.exists():
            raise FileNotFoundError(f"Missing encoded tensor archive: {tensor_path}")
        archive = np.load(tensor_path)
        values: List[np.ndarray] = []
        for row in rows:
            key = str(row.get("tensor_key", ""))
            if key not in archive.files:
                raise KeyError(f"Missing tensor_key in archive: {key}")
            values.append(self._bar_features(np.asarray(archive[key], dtype=np.float32)))
        return np.stack(values, axis=0).astype(np.float32)

    def _bar_features(self, tensor: np.ndarray) -> np.ndarray:
        """Return explicit normalized music features for one [tracks, steps, features] bar."""
        values = np.asarray(tensor, dtype=np.float32)
        note = values[..., 2] > 0.5
        hold = values[..., 3] > 0.5
        rest = values[..., 1] > 0.5
        active = note | hold
        total_slots = max(1, int(values.shape[0] * values.shape[1]))
        note_pitches = values[..., 0][note]
        note_velocity = values[..., 4][note]
        onset_by_slot = note.sum(axis=0).astype(np.float32)
        track_note_density = note.sum(axis=1).astype(np.float32) / max(1, int(values.shape[1]))
        track_active_density = active.sum(axis=1).astype(np.float32) / max(1, int(values.shape[1]))
        first_pitch, last_pitch = self._first_last_pitch(values, note)
        if len(note_pitches):
            pitch_mean = float(np.mean(note_pitches))
            pitch_std = float(np.std(note_pitches))
            pitch_min = float(np.min(note_pitches))
            pitch_max = float(np.max(note_pitches))
            pitch_range = float(pitch_max - pitch_min)
            velocity_mean = float(np.mean(note_velocity)) if len(note_velocity) else 0.0
            velocity_std = float(np.std(note_velocity)) if len(note_velocity) else 0.0
            ordered = self._ordered_note_pitches(values, note)
            intervals = np.abs(np.diff(ordered)) if len(ordered) > 1 else np.zeros((0,), dtype=np.float32)
            interval_mean = float(np.mean(intervals)) if len(intervals) else 0.0
            interval_max = float(np.max(intervals)) if len(intervals) else 0.0
            interval_std = float(np.std(intervals)) if len(intervals) else 0.0
        else:
            pitch_mean = pitch_std = pitch_min = pitch_max = pitch_range = 0.0
            velocity_mean = velocity_std = 0.0
            interval_mean = interval_max = interval_std = 0.0
        if float(onset_by_slot.sum()) > 0.0:
            slots = np.arange(len(onset_by_slot), dtype=np.float32)
            rhythm_centroid = float(np.sum(slots * onset_by_slot) / np.sum(onset_by_slot) / max(1, len(onset_by_slot) - 1))
            rhythm_std = float(np.sqrt(np.sum(((slots / max(1, len(onset_by_slot) - 1)) - rhythm_centroid) ** 2 * onset_by_slot) / np.sum(onset_by_slot)))
            probabilities = onset_by_slot / float(onset_by_slot.sum())
            rhythm_entropy = float(-(probabilities * np.log(np.clip(probabilities, 1.0e-8, 1.0))).sum() / np.log(len(probabilities)))
        else:
            rhythm_centroid = rhythm_std = rhythm_entropy = 0.0
        features = [
            float(note.sum() / total_slots),
            float(active.sum() / total_slots),
            float(rest.sum() / total_slots),
            float(hold.sum() / total_slots),
            pitch_mean,
            pitch_std,
            pitch_min,
            pitch_max,
            pitch_range,
            first_pitch,
            last_pitch,
            float(last_pitch - first_pitch),
            velocity_mean,
            velocity_std,
            rhythm_centroid,
            rhythm_std,
            rhythm_entropy,
            float(np.count_nonzero(onset_by_slot) / max(1, len(onset_by_slot))),
            *[float(item) for item in track_note_density[:3]],
            *[float(item) for item in track_active_density[:3]],
            interval_mean,
            interval_max,
            interval_std,
        ]
        return np.asarray(features, dtype=np.float32)

    def _first_last_pitch(self, tensor: np.ndarray, note: np.ndarray) -> tuple[float, float]:
        """Return first and last normalized note-on pitch."""
        pitches = self._ordered_note_pitches(tensor, note)
        if len(pitches) == 0:
            return 0.0, 0.0
        return float(pitches[0]), float(pitches[-1])

    def _ordered_note_pitches(self, tensor: np.ndarray, note: np.ndarray) -> np.ndarray:
        """Return note-on pitches ordered by slot then track."""
        items: List[float] = []
        for slot in range(tensor.shape[1]):
            for track in range(tensor.shape[0]):
                if bool(note[track, slot]):
                    items.append(float(tensor[track, slot, 0]))
        return np.asarray(items, dtype=np.float32)

    def _load_latent(self) -> tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """Load exported latent arrays and metadata."""
        mu_path = self.config.latent_dir / "latent_mu.npy"
        index_path = self.config.latent_dir / "latent_index.json"
        summary_path = self.config.latent_dir / "latent_summary.json"
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

    def _select_rows(self, rows: Sequence[Dict[str, Any]]) -> List[int]:
        """Select a deterministic subset of rows for the benchmark."""
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        song_ids = sorted(grouped)
        self.rng.shuffle(song_ids)
        if self.config.max_songs is not None:
            song_ids = song_ids[: max(1, int(self.config.max_songs))]
        selected = [index for song_id in song_ids for index in sorted(grouped[song_id], key=lambda idx: int(rows[idx].get("bar_index", 0)))]
        if self.config.max_rows is not None:
            selected = selected[: max(2, int(self.config.max_rows))]
        return selected

    def _group_selected_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Group selected local row indices by song_id."""
        grouped: Dict[str, List[int]] = {}
        for local_index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(local_index)
        return {
            song_id: sorted(indices, key=lambda idx: int(rows[idx].get("bar_index", 0)))
            for song_id, indices in grouped.items()
        }

    def _build_samples(self, groups: Dict[str, List[int]], rows: Sequence[Dict[str, Any]]) -> List[SequenceSample]:
        """Build next-step samples from song-local ordered bars."""
        samples: List[SequenceSample] = []
        for song_id, indices in groups.items():
            if len(indices) < 2:
                continue
            for local_position in range(1, len(indices)):
                target_index = int(indices[local_position])
                current_index = int(indices[local_position - 1])
                context = indices[max(0, local_position - int(self.config.context_bars)):local_position]
                samples.append(SequenceSample(
                    context_indices=[int(item) for item in context],
                    target_index=target_index,
                    current_index=current_index,
                    song_id=song_id,
                    base_song_id=self._base_song_id(song_id),
                    target_bar_index=int(rows[target_index].get("bar_index", local_position)),
                ))
        return samples

    def _split_samples(self, samples: Sequence[SequenceSample]) -> tuple[List[SequenceSample], List[SequenceSample], Dict[str, Any]]:
        """Split by base_song_id to reduce transposition leakage."""
        by_base: Dict[str, List[SequenceSample]] = {}
        for sample in samples:
            by_base.setdefault(sample.base_song_id, []).append(sample)
        base_ids = sorted(by_base)
        if not base_ids:
            return [], [], {}
        fold_count = max(1, int(self.config.validation_fold_count))
        fold_index = int(self.config.validation_fold_index) % fold_count
        val_ids = [base_id for index, base_id in enumerate(base_ids) if index % fold_count == fold_index]
        if len(val_ids) >= len(base_ids):
            val_ids = base_ids[-1:]
        val_id_set = set(val_ids)
        train = [sample for base_id, group in by_base.items() if base_id not in val_id_set for sample in group]
        val = [sample for base_id, group in by_base.items() if base_id in val_id_set for sample in group]
        return train, val, {
            "split_unit": "base_song_id",
            "validation_fold_count": int(fold_count),
            "validation_fold_index": int(fold_index),
            "base_song_count": int(len(base_ids)),
            "train_base_song_count": int(len(base_ids) - len(val_id_set)),
            "validation_base_song_count": int(len(val_id_set)),
            "train_sample_count": int(len(train)),
            "validation_sample_count": int(len(val)),
            "validation_base_song_ids": sorted(val_id_set),
        }

    def _base_song_id(self, song_id: str) -> str:
        """Collapse transposition suffix into one base id."""
        return re.sub(r"_T[+-]?\d+$", "", str(song_id))

    def _numeric_summary(self, values: Sequence[float | int]) -> Dict[str, Any]:
        """Return compact numeric summary."""
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
        result["representations"] = list(self.config.representations)
        return result

    def _write_reports(self, results: Dict[str, Any]) -> None:
        """Write JSON and Markdown reports."""
        output_dir = self.config.output_dir
        json_path = output_dir / "representation_benchmark.json"
        md_path = output_dir / "representation_benchmark.md"
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        md_path.write_text(self._markdown_report(results), encoding="utf-8")

    def _markdown_report(self, results: Dict[str, Any]) -> str:
        """Build a concise Markdown report."""
        lines = [
            "# Representation Benchmark",
            "",
            "This report compares how easily different bar representations learn true next-bar transitions.",
            "",
            "## Dataset",
            "",
            f"- selected rows: {results['input']['selected_row_count']}",
            f"- selected songs: {results['input']['selected_song_count']}",
            f"- samples: {results['input']['sample_count']}",
            f"- train samples: {results['split'].get('train_sample_count', 0)}",
            f"- validation samples: {results['split'].get('validation_sample_count', 0)}",
            "",
            "## Metrics",
            "",
            "| Representation | Context | Task | Selected | Dim | Input Dim | Val MSE | Direction Cosine | Top1 | Top5 | Top10 | Self-loop-like | Under-moving |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, data in results["representations"].items():
            val = data["val_eval"]
            ranking = val["ranking"]
            lines.append(
                "| {name} | {context} | {task} | {selected} | {dim} | {input_dim} | {mse:.6f} | {cos:.6f} | {top1:.4f} | {top5:.4f} | {top10:.4f} | {loop:.4f} | {under:.4f} |".format(
                    name=name,
                    context=str(data.get("context_mode", "state")),
                    task=str(data.get("task_mode", "single_head")),
                    selected=str(val.get("selected_output", "single")),
                    dim=int(data["representation_dim"]),
                    input_dim=int(data.get("input_dim", 0)),
                    mse=float(val["mse"].get("mean", 0.0)),
                    cos=float(val["direction_cosine"].get("mean", 0.0)),
                    top1=float(ranking.get("top1", 0.0)),
                    top5=float(ranking.get("top5", 0.0)),
                    top10=float(ranking.get("top10", 0.0)),
                    loop=float(val.get("pred_closer_to_current_than_true_rate", 0.0)),
                    under=float(val.get("under_moving_rate", 0.0)),
                )
            )
        detail_rows: List[str] = []
        for name, data in results["representations"].items():
            output_eval = data.get("val_eval", {}).get("output_eval")
            if not isinstance(output_eval, dict):
                continue
            for output_name, val in output_eval.items():
                ranking = val.get("ranking", {})
                detail_rows.append(
                    "| {name} | {output} | {mse:.6f} | {cos:.6f} | {top1:.4f} | {top5:.4f} | {top10:.4f} | {loop:.4f} | {under:.4f} |".format(
                        name=name,
                        output=output_name,
                        mse=float(val.get("mse", {}).get("mean", 0.0)),
                        cos=float(val.get("direction_cosine", {}).get("mean", 0.0)),
                        top1=float(ranking.get("top1", 0.0)),
                        top5=float(ranking.get("top5", 0.0)),
                        top10=float(ranking.get("top10", 0.0)),
                        loop=float(val.get("pred_closer_to_current_than_true_rate", 0.0)),
                        under=float(val.get("under_moving_rate", 0.0)),
                    )
                )
        if detail_rows:
            lines.extend([
                "",
                "## Multi-Output Detail",
                "",
                "| Representation | Output | Val MSE | Direction Cosine | Top1 | Top5 | Top10 | Self-loop-like | Under-moving |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *detail_rows,
            ])
        lines.extend([
            "",
            "## Interpretation",
            "",
            "- `state_delta` context adds adjacent context deltas as input while keeping the target as next state.",
            "- `hybrid_dual_head` separates next-state prediction from next-delta prediction, then reports a fused output.",
            "- `hybrid_transition_composer` uses a causal Transformer trunk plus a Composer MLP to compute next state from current state, state anchor, and predicted delta.",
            "- `hybrid_anchor_motion_composer` learns hidden anchor and motion variables using only next-state loss.",
            "- `hybrid_anchor_motion_movement_composer` uses SmoothL1 position loss plus direction and magnitude movement losses.",
            "- Higher direction cosine means the representation makes transition direction easier to learn.",
            "- Higher top-k hit rate means the predicted next vector lands near the real next bar among candidates.",
            "- High self-loop-like or under-moving rates indicate the model tends to stay near the current bar.",
        ])
        return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> BenchmarkConfig:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Benchmark bar representations for next-bar transition learning.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, default=None, help="Defaults to --model-dir/latent.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to --model-dir/representation_benchmark.")
    parser.add_argument("--representations", type=str, default="dvae_latent,bar_features,hybrid_latent_features,hybrid_transition_features,hybrid_dual_head,hybrid_transition_composer,hybrid_anchor_motion_composer,hybrid_anchor_motion_movement_composer")
    parser.add_argument("--context-bars", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--composer-hidden-dim", type=int, default=None, help="Defaults to --hidden-dim.")
    parser.add_argument("--composer-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-fold-count", type=int, default=5)
    parser.add_argument("--validation-fold-index", type=int, default=0)
    parser.add_argument("--max-songs", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--ranking-eval-samples", type=int, default=512)
    parser.add_argument("--ranking-candidates", type=int, default=5000)
    parser.add_argument("--delta-loss-weight", type=float, default=0.5)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.25)
    parser.add_argument("--fusion-state-weight", type=float, default=0.5)
    parser.add_argument("--composer-state-loss-weight", type=float, default=0.3)
    parser.add_argument("--composer-delta-loss-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)
    model_dir = Path(args.model_dir)
    return BenchmarkConfig(
        model_dir=model_dir,
        latent_dir=Path(args.latent_dir) if args.latent_dir else model_dir / "latent",
        output_dir=Path(args.output_dir) if args.output_dir else model_dir / "representation_benchmark",
        representations=tuple(item.strip() for item in str(args.representations).split(",") if item.strip()),
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
        max_songs=args.max_songs,
        max_rows=args.max_rows,
        ranking_eval_samples=int(args.ranking_eval_samples),
        ranking_candidates=int(args.ranking_candidates),
        delta_loss_weight=float(args.delta_loss_weight),
        consistency_loss_weight=float(args.consistency_loss_weight),
        fusion_state_weight=float(args.fusion_state_weight),
        composer_state_loss_weight=float(args.composer_state_loss_weight),
        composer_delta_loss_weight=float(args.composer_delta_loss_weight),
        random_seed=int(args.seed),
        device=str(args.device),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run representation benchmark."""
    config = parse_args(argv)
    result = RepresentationBenchmark(config).run()
    print(f"Representation benchmark complete -> {config.output_dir}")
    for name, data in result["representations"].items():
        val = data["val_eval"]
        print(
            f"{name}: val_mse={val['mse'].get('mean', 0.0):.6f} "
            f"dir_cos={val['direction_cosine'].get('mean', 0.0):.6f} "
            f"top5={val['ranking'].get('top5', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
