#!/usr/bin/env python3
"""Training pipeline for MidiTok-style bar event sequence encoder."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from codec.miditok_style_bar_encoder import MidiTokStyleBarEventEncoder
from common.config_loader import ConfigView
from model.miditok_bar_sequence_encoder import (
    MidiTokBarSequenceEncoder,
    MidiTokBarSequenceEncoderConfig,
    MidiTokBarSequenceLoss,
)
from pipeline.latent_transformer_training_pipeline import LatentDatasetReader


@dataclass(frozen=True)
class MidiTokBarSequenceTrainingConfig:
    """Training hyperparameters."""

    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    validation_ratio: float = 0.2
    random_seed: int = 42
    device: str = "cpu"
    early_stopping_patience: int = 5
    export_embeddings: bool = True
    max_rows: Optional[int] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MidiTokBarSequenceTrainingConfig":
        """Build config from style defaults."""
        section = ConfigView(config).section("miditok_bar_sequence_training")
        return cls(
            epochs=int(section.get("epochs", 40)),
            batch_size=int(section.get("batch_size", 256)),
            learning_rate=float(section.get("learning_rate", 5.0e-4)),
            weight_decay=float(section.get("weight_decay", 1.0e-4)),
            validation_ratio=float(section.get("validation_ratio", 0.2)),
            random_seed=int(section.get("random_seed", 42)),
            device=str(section.get("device", "cpu")),
            early_stopping_patience=int(section.get("early_stopping_patience", 5)),
            export_embeddings=bool(section.get("export_embeddings", True)),
            max_rows=None if section.get("max_rows", None) is None else int(section.get("max_rows")),
        )


class MidiTokTokenizedBarDataset(Dataset):
    """Tokenized bar event sequence dataset."""

    def __init__(self, tokens: np.ndarray, padding_mask: np.ndarray, target_mu: np.ndarray, row_indices: Sequence[int]) -> None:
        self.tokens = tokens.astype(np.int64)
        self.padding_mask = padding_mask.astype(bool)
        self.target_mu = target_mu.astype(np.float32)
        self.row_indices = np.asarray(row_indices, dtype=np.int64)

    def __len__(self) -> int:
        """Return row count."""
        return int(self.tokens.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one training sample."""
        return (
            torch.from_numpy(self.tokens[index]).long(),
            torch.from_numpy(self.padding_mask[index]).bool(),
            torch.from_numpy(self.target_mu[index]).float(),
            torch.tensor(int(self.row_indices[index]), dtype=torch.long),
        )


class MidiTokBarTokenBuilder:
    """Convert bar tensors into discrete MidiTok-style event token fields."""

    def __init__(self, config: MidiTokBarSequenceEncoderConfig) -> None:
        self.config = config
        self.event_encoder = MidiTokStyleBarEventEncoder()

    def build(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Return token fields and padding mask for one bar."""
        tokens = np.zeros((int(self.config.max_events), 5), dtype=np.int64)
        padding_mask = np.ones((int(self.config.max_events),), dtype=bool)
        events = self.event_encoder.events(tensor)
        truncated = max(0, len(events) - int(self.config.max_events))
        for index, event in enumerate(events[: int(self.config.max_events)]):
            tokens[index, 0] = int(np.clip(event.position + 1, 1, int(self.config.position_vocab_size) - 1))
            tokens[index, 1] = int(np.clip(event.track + 1, 1, int(self.config.track_vocab_size) - 1))
            pitch_semitone = int(round(float(event.pitch) * 24.0))
            pitch_semitone = int(np.clip(pitch_semitone, int(self.config.pitch_min), int(self.config.pitch_max)))
            tokens[index, 2] = int(pitch_semitone - int(self.config.pitch_min) + 1)
            tokens[index, 3] = int(np.clip(event.duration, 1, int(self.config.duration_vocab_size) - 1))
            velocity_id = int(np.floor(np.clip(float(event.velocity), 0.0, 0.999999) * int(self.config.velocity_bins))) + 1
            tokens[index, 4] = int(np.clip(velocity_id, 1, int(self.config.velocity_bins)))
            padding_mask[index] = False
        diagnostics = {
            "event_count": int(len(events)),
            "used_event_count": int(min(len(events), int(self.config.max_events))),
            "truncated_event_count": int(truncated),
        }
        return tokens, padding_mask, diagnostics


class MidiTokBarSequenceTrainingPipeline:
    """Train and export MidiTok-style bar sequence embeddings."""

    def __init__(self, config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> None:
        self.config = config
        self.model_config = MidiTokBarSequenceEncoderConfig.from_config(config)
        self.training_config = self._with_overrides(MidiTokBarSequenceTrainingConfig.from_config(config), overrides or {})

    def run(self, model_dir: str | Path, latent_dir: Optional[str | Path] = None, encoded_dir: Optional[str | Path] = None) -> Dict[str, Any]:
        """Train sequence encoder and write artifacts."""
        self._set_seed()
        output_dir = Path(model_dir)
        latent_path = Path(latent_dir) if latent_dir else output_dir / "latent"
        encoded_path = Path(encoded_dir) if encoded_dir else output_dir / "encoded"
        output_dir.mkdir(parents=True, exist_ok=True)
        mu, rows, latent_summary = LatentDatasetReader().load(latent_path)
        if self.training_config.max_rows is not None:
            limit = max(2, int(self.training_config.max_rows))
            mu = mu[:limit]
            rows = rows[:limit]
        self.model_config = self._resolved_model_config(mu)
        dataset, token_summary = self._build_dataset(mu, rows, encoded_path)
        train_dataset, val_dataset, split_summary = self._split_dataset(dataset, rows)
        model = MidiTokBarSequenceEncoder(self.model_config).to(self.training_config.device)
        loss_fn = MidiTokBarSequenceLoss(self.model_config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.training_config.learning_rate),
            weight_decay=float(self.training_config.weight_decay),
        )
        fit = self._train(model, loss_fn, optimizer, train_dataset, val_dataset)
        if fit["best_state_dict"] is not None:
            model.load_state_dict(fit["best_state_dict"])
        train_eval = self._evaluate(model, loss_fn, train_dataset)
        val_eval = self._evaluate(model, loss_fn, val_dataset)
        checkpoint_path = output_dir / "miditok_bar_sequence_encoder.pt"
        torch.save({
            "model_type": "MidiTokBarSequenceEncoder",
            "model_config": self.model_config.to_dict(),
            "training_config": self.training_config.__dict__,
            "state_dict": model.state_dict(),
            "checkpoint_role": "best",
        }, checkpoint_path)
        embedding_summary = {}
        if bool(self.training_config.export_embeddings):
            embedding_summary = self._export_embeddings(model, dataset, rows, encoded_path)
        diagnostics = {
            "model_dir": str(output_dir),
            "latent_dir": str(latent_path),
            "encoded_dir": str(encoded_path),
            "latent_summary": latent_summary,
            "model_config": self.model_config.to_dict(),
            "training_config": self.training_config.__dict__,
            "token_summary": token_summary,
            "split": split_summary,
            "history": fit["history"],
            "model_selection": {
                "best_epoch": int(fit["best_epoch"]),
                "best_val_mse": float(fit["best_val_mse"]),
            },
            "train_eval": train_eval,
            "val_eval": val_eval,
            "embedding_export": embedding_summary,
            "checkpoint": str(checkpoint_path),
        }
        diagnostics_path = output_dir / "miditok_bar_sequence_encoder_diagnostics.json"
        summary_path = output_dir / "miditok_bar_sequence_encoder_summary.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        summary_path.write_text(json.dumps(self._summary(diagnostics), indent=2), encoding="utf-8")
        return diagnostics

    def _build_dataset(
        self,
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        encoded_dir: Path,
    ) -> tuple[MidiTokTokenizedBarDataset, Dict[str, Any]]:
        """Build token arrays from bar_tensors.npz."""
        tensor_path = encoded_dir / "bar_tensors.npz"
        if not tensor_path.exists():
            raise FileNotFoundError(f"Missing bar_tensors.npz: {tensor_path}")
        builder = MidiTokBarTokenBuilder(self.model_config)
        archive = np.load(tensor_path)
        tokens: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        row_indices: List[int] = []
        event_counts: List[int] = []
        truncated_total = 0
        try:
            for row_index, row in enumerate(rows):
                key = str(row.get("tensor_key", ""))
                if key not in archive.files:
                    raise KeyError(f"Missing tensor_key in archive: {key}")
                token, mask, diag = builder.build(np.asarray(archive[key], dtype=np.float32))
                tokens.append(token)
                masks.append(mask)
                row_indices.append(int(row_index))
                event_counts.append(int(diag["event_count"]))
                truncated_total += int(diag["truncated_event_count"])
        finally:
            archive.close()
        dataset = MidiTokTokenizedBarDataset(
            tokens=np.stack(tokens, axis=0),
            padding_mask=np.stack(masks, axis=0),
            target_mu=mu.astype(np.float32),
            row_indices=row_indices,
        )
        summary = {
            "row_count": int(len(dataset)),
            "max_events": int(self.model_config.max_events),
            "event_count": self._numeric_summary(event_counts),
            "truncated_event_total": int(truncated_total),
            "truncated_bar_count": int(sum(1 for value in event_counts if value > int(self.model_config.max_events))),
        }
        return dataset, summary

    def _split_dataset(
        self,
        dataset: MidiTokTokenizedBarDataset,
        rows: Sequence[Dict[str, Any]],
    ) -> tuple[MidiTokTokenizedBarDataset, MidiTokTokenizedBarDataset, Dict[str, Any]]:
        """Split by base_song_id."""
        base_ids = sorted({self._base_song_id(str(row.get("song_id", "UNKNOWN"))) for row in rows})
        rng = np.random.default_rng(int(self.training_config.random_seed))
        shuffled = list(base_ids)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * float(self.training_config.validation_ratio)))) if len(shuffled) > 1 else 1
        val_bases = set(shuffled[:val_count])
        train_indices = []
        val_indices = []
        for local_index, row_index in enumerate(dataset.row_indices.tolist()):
            base_id = self._base_song_id(str(rows[int(row_index)].get("song_id", "UNKNOWN")))
            if base_id in val_bases:
                val_indices.append(local_index)
            else:
                train_indices.append(local_index)
        if not train_indices:
            train_indices = val_indices[:]
        return (
            self._subset(dataset, train_indices),
            self._subset(dataset, val_indices),
            {
                "train_rows": int(len(train_indices)),
                "validation_rows": int(len(val_indices)),
                "train_base_song_count": int(len(base_ids) - len(val_bases)),
                "validation_base_song_count": int(len(val_bases)),
                "validation_base_song_ids": sorted(val_bases),
            },
        )

    def _subset(self, dataset: MidiTokTokenizedBarDataset, indices: Sequence[int]) -> MidiTokTokenizedBarDataset:
        """Return dataset subset."""
        idx = np.asarray(indices, dtype=np.int64)
        return MidiTokTokenizedBarDataset(
            tokens=dataset.tokens[idx],
            padding_mask=dataset.padding_mask[idx],
            target_mu=dataset.target_mu[idx],
            row_indices=dataset.row_indices[idx],
        )

    def _train(
        self,
        model: MidiTokBarSequenceEncoder,
        loss_fn: MidiTokBarSequenceLoss,
        optimizer: torch.optim.Optimizer,
        train_dataset: MidiTokTokenizedBarDataset,
        val_dataset: MidiTokTokenizedBarDataset,
    ) -> Dict[str, Any]:
        """Train model."""
        train_loader = DataLoader(train_dataset, batch_size=int(self.training_config.batch_size), shuffle=True)
        history: List[Dict[str, Any]] = []
        best_val = float("inf")
        best_state = None
        best_epoch = -1
        patience = 0
        for epoch in range(int(self.training_config.epochs)):
            model.train()
            losses = []
            for tokens, mask, target, _row_indices in train_loader:
                tokens = tokens.to(self.training_config.device)
                mask = mask.to(self.training_config.device)
                target = target.to(self.training_config.device)
                optimizer.zero_grad(set_to_none=True)
                output = model(tokens, mask)
                loss_parts = loss_fn(output, target)
                loss_parts["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(loss_parts["loss"].detach().cpu()))
            val_eval = self._evaluate(model, loss_fn, val_dataset)
            train_loss = float(np.mean(losses)) if losses else 0.0
            row = {"epoch": int(epoch), "train_loss": train_loss, **{f"val_{k}": v for k, v in val_eval.items()}}
            history.append(row)
            if float(val_eval["mse"]) < best_val:
                best_val = float(val_eval["mse"])
                best_epoch = int(epoch)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= int(self.training_config.early_stopping_patience):
                    break
        return {"history": history, "best_state_dict": best_state, "best_val_mse": best_val, "best_epoch": best_epoch}

    def _evaluate(self, model: MidiTokBarSequenceEncoder, loss_fn: MidiTokBarSequenceLoss, dataset: MidiTokTokenizedBarDataset) -> Dict[str, float]:
        """Evaluate dataset."""
        loader = DataLoader(dataset, batch_size=int(self.training_config.batch_size), shuffle=False)
        model.eval()
        mse_values = []
        cosine_values = []
        loss_values = []
        with torch.no_grad():
            for tokens, mask, target, _row_indices in loader:
                tokens = tokens.to(self.training_config.device)
                mask = mask.to(self.training_config.device)
                target = target.to(self.training_config.device)
                output = model(tokens, mask)
                loss_parts = loss_fn(output, target)
                pred = output["latent_mu"]
                mse_values.append(torch.mean((pred - target) ** 2, dim=1).detach().cpu().numpy())
                cosine_values.append(torch.nn.functional.cosine_similarity(pred, target, dim=1).detach().cpu().numpy())
                loss_values.append(float(loss_parts["loss"].detach().cpu()))
        mse = np.concatenate(mse_values, axis=0) if mse_values else np.zeros((0,), dtype=np.float32)
        cosine = np.concatenate(cosine_values, axis=0) if cosine_values else np.zeros((0,), dtype=np.float32)
        return {
            "loss": float(np.mean(loss_values)) if loss_values else 0.0,
            "mse": float(np.mean(mse)) if mse.size else 0.0,
            "mse_median": float(np.median(mse)) if mse.size else 0.0,
            "cosine_mean": float(np.mean(cosine)) if cosine.size else 0.0,
            "cosine_median": float(np.median(cosine)) if cosine.size else 0.0,
        }

    def _export_embeddings(
        self,
        model: MidiTokBarSequenceEncoder,
        dataset: MidiTokTokenizedBarDataset,
        rows: Sequence[Dict[str, Any]],
        encoded_dir: Path,
    ) -> Dict[str, Any]:
        """Export CLS embeddings by tensor key."""
        loader = DataLoader(dataset, batch_size=int(self.training_config.batch_size), shuffle=False)
        model.eval()
        embedding_by_key: Dict[str, np.ndarray] = {}
        pred_by_key: Dict[str, np.ndarray] = {}
        with torch.no_grad():
            for tokens, mask, _target, row_indices in loader:
                output = model(tokens.to(self.training_config.device), mask.to(self.training_config.device))
                embeddings = output["embedding"].detach().cpu().numpy().astype(np.float32)
                pred_mu = output["latent_mu"].detach().cpu().numpy().astype(np.float32)
                for local, row_index in enumerate(row_indices.numpy().tolist()):
                    key = str(rows[int(row_index)].get("tensor_key", ""))
                    embedding_by_key[key] = embeddings[local]
                    pred_by_key[key] = pred_mu[local]
        encoded_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = encoded_dir / "miditok_sequence_embeddings.npz"
        pred_path = encoded_dir / "miditok_sequence_predicted_mu.npz"
        np.savez_compressed(embedding_path, **embedding_by_key)
        np.savez_compressed(pred_path, **pred_by_key)
        matrix = np.stack(list(embedding_by_key.values()), axis=0).astype(np.float32) if embedding_by_key else np.zeros((0, int(self.model_config.d_model)), dtype=np.float32)
        summary = {
            "embedding_path": str(embedding_path),
            "predicted_mu_path": str(pred_path),
            "row_count": int(matrix.shape[0]),
            "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            "embedding_mean_abs": float(np.mean(np.abs(matrix))) if matrix.size else 0.0,
            "embedding_std_mean": float(np.mean(np.std(matrix, axis=0))) if matrix.size else 0.0,
        }
        (encoded_dir / "miditok_sequence_embedding_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _resolved_model_config(self, mu: np.ndarray) -> MidiTokBarSequenceEncoderConfig:
        """Resolve latent dim from data."""
        values = dict(self.model_config.__dict__)
        values["latent_dim"] = int(mu.shape[1])
        return MidiTokBarSequenceEncoderConfig(**values)

    def _with_overrides(self, base: MidiTokBarSequenceTrainingConfig, overrides: Dict[str, Any]) -> MidiTokBarSequenceTrainingConfig:
        """Apply CLI overrides."""
        values = dict(base.__dict__)
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        values["device"] = self._resolve_device(str(values.get("device", "cpu")))
        return MidiTokBarSequenceTrainingConfig(**values)

    def _resolve_device(self, requested: str) -> str:
        """Resolve unavailable CUDA to CPU."""
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested

    def _set_seed(self) -> None:
        """Set RNG seeds."""
        seed = int(self.training_config.random_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _base_song_id(self, song_id: str) -> str:
        """Remove transposition suffix."""
        return re.sub(r"_T[+-]?\d+$", "", str(song_id))

    def _numeric_summary(self, values: Sequence[float | int]) -> Dict[str, Any]:
        """Return numeric summary."""
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "mean": float(np.mean(array)) if array.size else 0.0,
            "median": float(np.median(array)) if array.size else 0.0,
            "min": float(np.min(array)) if array.size else 0.0,
            "max": float(np.max(array)) if array.size else 0.0,
        }

    def _summary(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Return compact summary."""
        return {
            "checkpoint": diagnostics["checkpoint"],
            "model_config": diagnostics["model_config"],
            "token_summary": diagnostics["token_summary"],
            "model_selection": diagnostics["model_selection"],
            "train_eval": diagnostics["train_eval"],
            "val_eval": diagnostics["val_eval"],
            "embedding_export": diagnostics["embedding_export"],
        }
