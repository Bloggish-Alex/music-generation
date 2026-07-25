#!/usr/bin/env python3
"""Training pipeline for learned base-pitch motion."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model.base_pitch_motion import BasePitchMotionConfig, BasePitchMotionModel
from model.hybrid_miditok_retrieval import HybridMidiTokRetrievalConfig
from model.miditok_bar_sequence_encoder import MidiTokBarSequenceEncoderConfig
from pipeline.hybrid_miditok_retrieval_pipeline import HybridMidiTokDataBuilder, RetrievalSample


@dataclass(frozen=True)
class BasePitchMotionTrainingConfig:
    """Training config for base-pitch motion model."""

    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    validation_ratio: float = 0.2
    random_seed: int = 42
    device: str = "cpu"
    early_stopping_patience: int = 5
    max_rows: Optional[int] = None
    transpose_mode: str = "canonical_only"


@dataclass(frozen=True)
class BasePitchMotionTrainingResult:
    """Paths produced by base-pitch motion training."""

    model_path: Path
    diagnostics_path: Path
    summary_path: Path


class BasePitchMotionDataset(Dataset):
    """Samples for base-pitch delta classification."""

    def __init__(
        self,
        samples: Sequence[RetrievalSample],
        mu: np.ndarray,
        tokens: np.ndarray,
        token_mask: np.ndarray,
        base_pitches: np.ndarray,
        context_bars: int,
        model_config: BasePitchMotionConfig,
    ) -> None:
        self.samples = list(samples)
        self.mu = np.asarray(mu, dtype=np.float32)
        self.tokens = np.asarray(tokens, dtype=np.int64)
        self.token_mask = np.asarray(token_mask, dtype=bool)
        self.base_pitches = np.asarray(base_pitches, dtype=np.int64)
        self.context_bars = int(context_bars)
        self.model_config = model_config

    def __len__(self) -> int:
        """Return sample count."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return one left-padded context and target delta class."""
        sample = self.samples[index]
        latent_dim = int(self.mu.shape[1])
        max_events = int(self.tokens.shape[1])
        context_mu = np.zeros((self.context_bars, latent_dim), dtype=np.float32)
        context_tokens = np.zeros((self.context_bars, max_events, 5), dtype=np.int64)
        context_token_mask = np.ones((self.context_bars, max_events), dtype=bool)
        context_padding_mask = np.ones((self.context_bars,), dtype=bool)
        context_base_pitch = np.zeros((self.context_bars,), dtype=np.float32)
        recent = sample.context_indices[-self.context_bars:]
        offset = self.context_bars - len(recent)
        for local_index, row_index in enumerate(recent):
            slot = offset + local_index
            context_mu[slot] = self.mu[row_index]
            context_tokens[slot] = self.tokens[row_index]
            context_token_mask[slot] = self.token_mask[row_index]
            context_padding_mask[slot] = False
            context_base_pitch[slot] = float(self.base_pitches[row_index])
        previous_index = int(sample.context_indices[-1])
        delta = int(self.base_pitches[int(sample.target_index)] - self.base_pitches[previous_index])
        target_class = int(self.model_config.delta_to_class(delta))
        return {
            "context_mu": torch.from_numpy(context_mu).float(),
            "context_tokens": torch.from_numpy(context_tokens).long(),
            "context_token_mask": torch.from_numpy(context_token_mask).bool(),
            "context_padding_mask": torch.from_numpy(context_padding_mask).bool(),
            "context_base_pitch": torch.from_numpy(context_base_pitch).float(),
            "target_delta_class": torch.tensor(target_class, dtype=torch.long),
            "target_delta": torch.tensor(delta, dtype=torch.long),
        }


class BasePitchMotionTrainingPipeline:
    """Train a model that predicts base-pitch movement."""

    def __init__(
        self,
        model_config: BasePitchMotionConfig,
        event_config: MidiTokBarSequenceEncoderConfig,
        training_config: BasePitchMotionTrainingConfig,
    ) -> None:
        self.model_config = model_config
        self.event_config = event_config
        self.training_config = training_config
        retrieval_config = HybridMidiTokRetrievalConfig(
            latent_dim=int(model_config.latent_dim),
            context_bars=int(model_config.context_bars),
            d_model=int(model_config.d_model),
            n_layers=int(model_config.n_layers),
            n_heads=int(model_config.n_heads),
            dropout=float(model_config.dropout),
        )
        self.builder = HybridMidiTokDataBuilder(retrieval_config, event_config)

    def run(self, model_dir: str | Path, latent_dir: Optional[str | Path] = None, encoded_dir: Optional[str | Path] = None) -> BasePitchMotionTrainingResult:
        """Train and write model artifacts."""
        self._set_seed()
        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        latent_path = Path(latent_dir) if latent_dir else output_dir / "latent"
        encoded_path = Path(encoded_dir) if encoded_dir else output_dir / "encoded"
        mu, rows, tokens, token_mask, source_summary = self.builder.load(
            latent_path,
            encoded_path,
            max_rows=self.training_config.max_rows,
            transpose_mode=str(self.training_config.transpose_mode),
        )
        base_pitches = self._base_pitch_array(rows, encoded_path)
        samples = self.builder.build_samples(rows)
        train_samples, val_samples, split = self._split_samples(samples)
        train_dataset = BasePitchMotionDataset(train_samples, mu, tokens, token_mask, base_pitches, int(self.model_config.context_bars), self.model_config)
        val_dataset = BasePitchMotionDataset(val_samples, mu, tokens, token_mask, base_pitches, int(self.model_config.context_bars), self.model_config)
        model = BasePitchMotionModel(self.model_config, self.event_config).to(self.training_config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.training_config.learning_rate),
            weight_decay=float(self.training_config.weight_decay),
        )
        fit = self._fit(model, optimizer, train_dataset, val_dataset)
        if fit["best_state_dict"] is not None:
            model.load_state_dict(fit["best_state_dict"])
        train_eval = self._evaluate(model, train_dataset)
        val_eval = self._evaluate(model, val_dataset)
        model_path = output_dir / "base_pitch_motion.pt"
        torch.save({
            "model_type": "BasePitchMotionModel",
            "checkpoint_role": "best",
            "model_config": self.model_config.to_dict(),
            "event_config": self.event_config.to_dict(),
            "training_config": self.training_config.__dict__,
            "state_dict": model.state_dict(),
        }, model_path)
        diagnostics = {
            "model_path": str(model_path),
            "latent_dir": str(latent_path),
            "encoded_dir": str(encoded_path),
            "model_config": self.model_config.to_dict(),
            "event_config": self.event_config.to_dict(),
            "training_config": self.training_config.__dict__,
            "source_summary": source_summary,
            "base_pitch_distribution": self._base_pitch_distribution(base_pitches, samples),
            "sample_count": int(len(samples)),
            "split": split,
            "history": fit["history"],
            "model_selection": {
                "best_epoch": int(fit["best_epoch"]),
                "best_val_accuracy": float(fit["best_val_accuracy"]),
                "early_stopped": bool(fit["early_stopped"]),
            },
            "train_eval": train_eval,
            "val_eval": val_eval,
        }
        diagnostics_path = output_dir / "base_pitch_motion_diagnostics.json"
        summary_path = output_dir / "base_pitch_motion_summary.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        summary_path.write_text(json.dumps(self._summary(diagnostics), indent=2), encoding="utf-8")
        return BasePitchMotionTrainingResult(model_path=model_path, diagnostics_path=diagnostics_path, summary_path=summary_path)

    def _fit(self, model: BasePitchMotionModel, optimizer: torch.optim.Optimizer, train_dataset: BasePitchMotionDataset, val_dataset: BasePitchMotionDataset) -> Dict[str, Any]:
        """Train with cross-entropy over clipped delta classes."""
        train_loader = DataLoader(train_dataset, batch_size=int(self.training_config.batch_size), shuffle=True, drop_last=False)
        history: List[Dict[str, float]] = []
        best_state = None
        best_accuracy = -1.0
        best_epoch = 0
        stale = 0
        early_stopped = False
        for epoch in range(1, int(self.training_config.epochs) + 1):
            model.train()
            losses: List[float] = []
            accuracies: List[float] = []
            for batch in train_loader:
                prepared = self._batch_to_device(batch)
                optimizer.zero_grad(set_to_none=True)
                output = model.loss(prepared)
                output["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(output["loss"].detach().cpu()))
                accuracies.append(float(output["accuracy"].detach().cpu()))
            val_eval = self._evaluate(model, val_dataset)
            row = {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "train_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
                **{f"val_{key}": float(value) for key, value in val_eval.items()},
            }
            history.append(row)
            if float(val_eval["accuracy"]) > best_accuracy:
                best_accuracy = float(val_eval["accuracy"])
                best_epoch = int(epoch)
                stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if int(self.training_config.early_stopping_patience) > 0 and stale >= int(self.training_config.early_stopping_patience):
                early_stopped = True
                break
        return {
            "history": history,
            "best_state_dict": best_state,
            "best_epoch": int(best_epoch),
            "best_val_accuracy": float(best_accuracy),
            "early_stopped": bool(early_stopped),
        }

    def _evaluate(self, model: BasePitchMotionModel, dataset: BasePitchMotionDataset) -> Dict[str, float]:
        """Evaluate delta-class prediction."""
        loader = DataLoader(dataset, batch_size=int(self.training_config.batch_size), shuffle=False, drop_last=False)
        model.eval()
        losses: List[float] = []
        correct = 0
        total = 0
        abs_errors: List[float] = []
        semitone_errors: List[float] = []
        with torch.no_grad():
            for batch in loader:
                prepared = self._batch_to_device(batch)
                output = model.loss(prepared)
                logits = model(prepared)
                pred_class = torch.argmax(logits, dim=-1)
                target_class = prepared["target_delta_class"].long()
                losses.append(float(output["loss"].detach().cpu()))
                correct += int((pred_class == target_class).sum().detach().cpu())
                total += int(target_class.numel())
                abs_errors.extend(torch.abs(pred_class.float() - target_class.float()).detach().cpu().numpy().tolist())
                pred_delta = pred_class.detach().cpu().numpy().astype(np.int64) + int(self.model_config.delta_min)
                true_delta = np.clip(prepared["target_delta"].detach().cpu().numpy().astype(np.int64), int(self.model_config.delta_min), int(self.model_config.delta_max))
                semitone_errors.extend(np.abs(pred_delta - true_delta).astype(np.float32).tolist())
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": float(correct / max(1, total)),
            "class_abs_error": float(np.mean(abs_errors)) if abs_errors else 0.0,
            "semitone_abs_error": float(np.mean(semitone_errors)) if semitone_errors else 0.0,
        }

    def _base_pitch_array(self, rows: Sequence[Dict[str, Any]], encoded_dir: str | Path) -> np.ndarray:
        """Return base pitch for each latent row."""
        lookup = self._base_pitch_lookup(encoded_dir)
        values = [int(lookup.get(str(row.get("tensor_key", "")), 60)) for row in rows]
        return np.asarray(values, dtype=np.int64)

    def _base_pitch_lookup(self, encoded_dir: str | Path) -> Dict[str, int]:
        """Load source tensor base pitches from encoded diagnostics."""
        index_path = Path(encoded_dir) / "bar_tensor_index.json"
        if not index_path.exists():
            return {}
        rows = json.loads(index_path.read_text(encoding="utf-8"))
        lookup: Dict[str, int] = {}
        for row in rows:
            key = str(row.get("tensor_key", ""))
            diagnostics = row.get("diagnostics", {}) if isinstance(row.get("diagnostics", {}), dict) else {}
            value = diagnostics.get("base_pitch")
            if value is not None:
                lookup[key] = int(value)
        return lookup

    def _split_samples(self, samples: Sequence[RetrievalSample]) -> Tuple[List[RetrievalSample], List[RetrievalSample], Dict[str, Any]]:
        """Split by base_song_id."""
        base_ids = sorted({sample.base_song_id for sample in samples})
        rng = np.random.default_rng(int(self.training_config.random_seed))
        shuffled = list(base_ids)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * float(self.training_config.validation_ratio)))) if len(shuffled) > 1 else 1
        val_bases = set(shuffled[:val_count])
        train = [sample for sample in samples if sample.base_song_id not in val_bases]
        val = [sample for sample in samples if sample.base_song_id in val_bases]
        if not train:
            train = val[:]
        return train, val, {
            "train_samples": int(len(train)),
            "validation_samples": int(len(val)),
            "base_song_count": int(len(base_ids)),
            "validation_base_song_ids": sorted(val_bases),
        }

    def _base_pitch_distribution(self, base_pitches: np.ndarray, samples: Sequence[RetrievalSample]) -> Dict[str, Any]:
        """Summarize real base-pitch motion in the training corpus."""
        deltas: List[int] = []
        for sample in samples:
            previous = int(sample.context_indices[-1])
            delta = int(base_pitches[int(sample.target_index)] - base_pitches[previous])
            deltas.append(delta)
        arr = np.asarray(deltas, dtype=np.float32)
        if arr.size == 0:
            return {}
        clipped = np.clip(arr, int(self.model_config.delta_min), int(self.model_config.delta_max)).astype(np.int64)
        counts: Dict[str, int] = {}
        for value in clipped.tolist():
            counts[str(int(value))] = counts.get(str(int(value)), 0) + 1
        return {
            "base_pitch_min": int(np.min(base_pitches)) if len(base_pitches) else None,
            "base_pitch_max": int(np.max(base_pitches)) if len(base_pitches) else None,
            "delta_mean": float(np.mean(arr)),
            "delta_abs_mean": float(np.mean(np.abs(arr))),
            "delta_abs_p90": float(np.percentile(np.abs(arr), 90)),
            "delta_abs_gt7_ratio": float(np.mean(np.abs(arr) > 7)),
            "delta_abs_gt12_ratio": float(np.mean(np.abs(arr) > 12)),
            "clipped_delta_counts": counts,
        }

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move a batch to the configured device."""
        return {key: value.to(self.training_config.device) for key, value in batch.items()}

    def _summary(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Return compact summary."""
        return {
            "model_path": diagnostics["model_path"],
            "sample_count": diagnostics["sample_count"],
            "base_pitch_distribution": diagnostics["base_pitch_distribution"],
            "model_selection": diagnostics["model_selection"],
            "train_eval": diagnostics["train_eval"],
            "val_eval": diagnostics["val_eval"],
        }

    def _set_seed(self) -> None:
        """Set random seeds."""
        seed = int(self.training_config.random_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
