#!/usr/bin/env python3
"""Training pipeline for the denoising music VAE."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from codec.action_labeler import ActionLabeler
from codec.bar_tensor_codec_factory import BarTensorCodecFactory
from common.config_loader import ConfigView
from data.core import BarTensorRecord, SongRecord
from data.music_parser import MusicDirectoryParser
from diagnostics.diagnostics import DiagnosticsBase
from model.dvae import DVAELoss, DVAEMusicConfig, DenoisingMusicVAE, DVAEOptimizerFactory


@dataclass(frozen=True)
class DVAETrainingConfig:
    """Configuration for the DVAE training pipeline."""

    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    validation_ratio: float = 0.1
    random_seed: int = 42
    device: str = "cpu"
    transpose_semitones: tuple[int, ...] = (0,)
    save_encoded_artifacts: bool = True
    diagnostics_top_k: int = 20

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DVAETrainingConfig":
        """Build training config from style config."""
        section = ConfigView(config).section("dvae_training")
        return cls(
            epochs=int(section.get("epochs", 80)),
            batch_size=int(section.get("batch_size", 128)),
            learning_rate=float(section.get("learning_rate", 1.0e-3)),
            validation_ratio=float(section.get("validation_ratio", 0.1)),
            random_seed=int(section.get("random_seed", 42)),
            device=str(section.get("device", "cpu")),
            transpose_semitones=tuple(int(x) for x in section.get("transpose_semitones", [0])),
            save_encoded_artifacts=bool(section.get("save_encoded_artifacts", True)),
            diagnostics_top_k=int(section.get("diagnostics_top_k", 20)),
        )


@dataclass
class DVAETrainingResult:
    """Result object produced by the DVAE training pipeline."""

    model_path: Path
    diagnostics_path: Path
    summary_path: Path
    diagnostics: Dict[str, Any]


class AdjacentBarPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Expose real contiguous bar pairs without materializing duplicate tensors."""

    def __init__(self, values: torch.Tensor, pairs: Sequence[tuple[int, int, str]]) -> None:
        self.values = values
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        current, following, _song_id = self.pairs[index]
        return self.values[current], self.values[following]


class DVAETrainingPipeline:
    """Assemble encoding and DVAE training into one reproducible pipeline."""

    def __init__(self, config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> None:
        self.config = config
        self.training_config = self._with_overrides(DVAETrainingConfig.from_config(config), overrides or {})
        self.dvae_config = DVAEMusicConfig.from_config(config)
        self.diagnostics = DiagnosticsBase("dvae_training")

    def run(self, music_dir: str | Path, model_dir: str | Path) -> DVAETrainingResult:
        """Encode a music directory, train DVAE, and write diagnostics."""
        self._set_seed()
        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        songs, tensors = self._build_dataset(music_dir)
        dataset = self._tensor_dataset(tensors)
        pair_dataset = self._adjacent_pair_dataset(dataset, tensors)
        train_dataset, val_dataset = self._split_pair_dataset(pair_dataset)
        model = DenoisingMusicVAE(self.dvae_config).to(self.training_config.device)
        loss_fn = DVAELoss(self.dvae_config)
        optimizer = DVAEOptimizerFactory(self.dvae_config).adamw(model, self.training_config.learning_rate)
        history = self._train(model, loss_fn, optimizer, train_dataset, val_dataset)
        reconstruction = self._evaluate_reconstruction(model, loss_fn, dataset)
        latent = self._latent_diagnostics(model, dataset)
        model_path = output_dir / "dvae.pt"
        self._save_model(model_path, model)
        if self.training_config.save_encoded_artifacts:
            self._write_encoded_artifacts(output_dir / "encoded", songs, tensors)
        self.diagnostics.record_stage("training", {
            "config": self.training_config.__dict__,
            "dvae_config": self.dvae_config.to_dict(),
            "history": history,
        })
        self.diagnostics.record_stage("reconstruction", reconstruction)
        self.diagnostics.record_stage("latent", latent)
        summary = self._summary(songs, tensors, history, reconstruction, latent, model_path)
        diagnostics_path = output_dir / "dvae_training_diagnostics.json"
        summary_path = output_dir / "dvae_training_summary.json"
        self.diagnostics.write(diagnostics_path)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return DVAETrainingResult(
            model_path=model_path,
            diagnostics_path=diagnostics_path,
            summary_path=summary_path,
            diagnostics=self.diagnostics.to_dict(),
        )

    def _with_overrides(self, base: DVAETrainingConfig, overrides: Dict[str, Any]) -> DVAETrainingConfig:
        """Apply CLI overrides to training config."""
        values = dict(base.__dict__)
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        if isinstance(values.get("transpose_semitones"), str):
            values["transpose_semitones"] = tuple(
                int(item.strip()) for item in str(values["transpose_semitones"]).split(",") if item.strip()
            )
        values["device"] = self._resolve_device(str(values.get("device", "cpu")))
        return DVAETrainingConfig(**values)

    def _resolve_device(self, requested: str) -> str:
        """Return a torch device string that is available on this machine."""
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

    def _build_dataset(self, music_dir: str | Path) -> tuple[List[SongRecord], List[BarTensorRecord]]:
        """Parse, label, and tensorize all requested transpositions."""
        all_songs: List[SongRecord] = []
        all_tensors: List[BarTensorRecord] = []
        parse_failures: List[Dict[str, str]] = []
        for semitone in self.training_config.transpose_semitones:
            parser = MusicDirectoryParser.from_config(self.config)
            songs = parser.parse_directory(music_dir, transpose_semitones=int(semitone))
            parse_failures.extend(parser.failed_files)
            labeler = ActionLabeler.from_config(self.config)
            action_diagnostics = [labeler.label_song(song) for song in songs]
            codec = BarTensorCodecFactory.create(self.config)
            tensors = [codec.encode(bar) for song in songs for bar in song.bars]
            all_songs.extend(songs)
            all_tensors.extend(tensors)
            self.diagnostics.append_event("encoding_pass", {
                "transpose_semitones": int(semitone),
                "song_count": int(len(songs)),
                "bar_count": int(len(tensors)),
                "failed_file_count": int(len(parser.failed_files)),
                "action_counts": self._global_action_counts(action_diagnostics),
            })
        self.diagnostics.record_stage("input", {
            "music_dir": str(music_dir),
            "transpose_semitones": [int(x) for x in self.training_config.transpose_semitones],
            "song_count": int(len(all_songs)),
            "bar_count": int(len(all_tensors)),
            "failed_file_count": int(len(parse_failures)),
            "failed_files": parse_failures,
        })
        if not all_tensors:
            raise ValueError("No bar tensors were produced for DVAE training.")
        return all_songs, all_tensors

    def _tensor_dataset(self, tensors: Sequence[BarTensorRecord]) -> TensorDataset:
        """Create a torch TensorDataset from bar tensor records."""
        array = np.stack([np.asarray(record.tensor, dtype=np.float32) for record in tensors], axis=0)
        return TensorDataset(torch.from_numpy(array).float())

    def _adjacent_pair_dataset(self, dataset: TensorDataset, tensors: Sequence[BarTensorRecord]) -> AdjacentBarPairDataset:
        """Build only bar_t -> bar_t+1 pairs from the same encoded song."""
        grouped: Dict[str, List[tuple[int, int]]] = {}
        for index, record in enumerate(tensors):
            grouped.setdefault(str(record.song_id), []).append((int(record.bar_index), int(index)))
        pairs: List[tuple[int, int, str]] = []
        for song_id, entries in grouped.items():
            ordered = sorted(entries)
            for (left_bar, left_index), (right_bar, right_index) in zip(ordered, ordered[1:]):
                if right_bar == left_bar + 1:
                    pairs.append((left_index, right_index, song_id))
        if not pairs:
            raise ValueError("No contiguous bar pairs were available for DVAE Chroma-delta training.")
        self.diagnostics.record_stage("adjacent_pair_dataset", {
            "pair_count": int(len(pairs)),
            "song_count": int(len(grouped)),
            "base_song_count": int(len({self._base_song_id(song_id) for song_id in grouped})),
        })
        return AdjacentBarPairDataset(dataset.tensors[0], pairs)

    def _split_pair_dataset(self, dataset: AdjacentBarPairDataset) -> tuple[Subset, Subset]:
        """Split contiguous pairs by base song so no held-out song enters training."""
        base_song_ids = sorted({self._base_song_id(song_id) for _, _, song_id in dataset.pairs})
        if len(base_song_ids) < 2:
            raise ValueError("At least two base songs are required for train/validation pair splitting.")
        generator = torch.Generator().manual_seed(int(self.training_config.random_seed))
        order = torch.randperm(len(base_song_ids), generator=generator).tolist()
        val_size = max(1, int(round(len(base_song_ids) * max(0.0, min(0.9, float(self.training_config.validation_ratio))))))
        val_base_ids = {base_song_ids[index] for index in order[:val_size]}
        train_indices = [index for index, (_, _, song_id) in enumerate(dataset.pairs) if self._base_song_id(song_id) not in val_base_ids]
        val_indices = [index for index, (_, _, song_id) in enumerate(dataset.pairs) if self._base_song_id(song_id) in val_base_ids]
        if not train_indices or not val_indices:
            raise ValueError("Base-song train/validation split produced an empty pair partition.")
        self.diagnostics.record_stage("dataset_split", {
            "split_unit": "base_song_id",
            "base_song_count": int(len(base_song_ids)),
            "validation_base_song_count": int(len(val_base_ids)),
            "total_size": int(len(dataset)),
            "train_size": int(len(train_indices)),
            "validation_size": int(len(val_indices)),
        })
        return Subset(dataset, train_indices), Subset(dataset, val_indices)

    def _base_song_id(self, song_id: str) -> str:
        """Keep all transpositions of one source piece in one split partition."""
        return str(song_id).split("_T+", maxsplit=1)[0]

    def _train(
        self,
        model: DenoisingMusicVAE,
        loss_fn: DVAELoss,
        optimizer: torch.optim.Optimizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> List[Dict[str, float]]:
        """Train the DVAE and return epoch-level metrics."""
        train_loader = DataLoader(train_dataset, batch_size=int(self.training_config.batch_size), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=int(self.training_config.batch_size), shuffle=False)
        history: List[Dict[str, float]] = []
        for epoch in range(int(self.training_config.epochs)):
            train_metrics = self._run_epoch(model, loss_fn, train_loader, optimizer=optimizer)
            val_metrics = self._run_epoch(model, loss_fn, val_loader, optimizer=None)
            row = {"epoch": float(epoch + 1)}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            history.append(row)
        return history

    def _run_epoch(
        self,
        model: DenoisingMusicVAE,
        loss_fn: DVAELoss,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Dict[str, float]:
        """Run one train or validation epoch."""
        is_train = optimizer is not None
        model.train(is_train)
        totals: Dict[str, float] = {}
        count = 0
        for batch_values in loader:
            if len(batch_values) == 1:
                current = batch_values[0].to(self.training_config.device)
                following = None
            else:
                current = batch_values[0].to(self.training_config.device)
                following = batch_values[1].to(self.training_config.device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(is_train):
                current_output = model(current, add_noise=is_train)
                if following is None:
                    losses = loss_fn(current_output, current)
                else:
                    following_output = model(following, add_noise=is_train)
                    losses = loss_fn.pair(current_output, current, following_output, following)
                if is_train:
                    losses["total_loss"].backward()
                    optimizer.step()
            batch_size = int(current.shape[0])
            count += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
        if count == 0:
            return {key: 0.0 for key in [
                "total_loss", "pitch_loss", "state_loss", "velocity_loss", "chord_loss",
                "physical_chroma_loss", "chroma_delta_loss", "kl_loss",
            ]}
        return {key: value / count for key, value in totals.items()}

    def _evaluate_reconstruction(
        self,
        model: DenoisingMusicVAE,
        loss_fn: DVAELoss,
        dataset: TensorDataset,
    ) -> Dict[str, Any]:
        """Calculate full-dataset reconstruction metrics."""
        loader = DataLoader(dataset, batch_size=int(self.training_config.batch_size), shuffle=False)
        model.eval()
        metrics = self._run_epoch(model, loss_fn, loader, optimizer=None)
        state_correct = 0
        state_total = 0
        note_on_correct = 0
        note_on_total = 0
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.training_config.device)
                output = model(batch, add_noise=False)
                pred_state = torch.argmax(output.state_logits, dim=-1)
                target_state = torch.argmax(batch[..., 1:4], dim=-1)
                state_correct += int((pred_state == target_state).sum().detach().cpu())
                state_total += int(target_state.numel())
                note_mask = target_state == 1
                note_on_correct += int(((pred_state == target_state) & note_mask).sum().detach().cpu())
                note_on_total += int(note_mask.sum().detach().cpu())
        return {
            **metrics,
            "state_accuracy": float(state_correct / state_total) if state_total else 0.0,
            "note_on_state_accuracy": float(note_on_correct / note_on_total) if note_on_total else 0.0,
        }

    def _latent_diagnostics(self, model: DenoisingMusicVAE, dataset: TensorDataset) -> Dict[str, Any]:
        """Summarize latent mean and log-variance distributions."""
        loader = DataLoader(dataset, batch_size=int(self.training_config.batch_size), shuffle=False)
        mus: List[np.ndarray] = []
        log_vars: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.training_config.device)
                mu, log_var = model.encoder(batch)
                mus.append(mu.detach().cpu().numpy())
                log_vars.append(log_var.detach().cpu().numpy())
        mu_array = np.concatenate(mus, axis=0)
        log_var_array = np.concatenate(log_vars, axis=0)
        return {
            "sample_count": int(mu_array.shape[0]),
            "latent_dim": int(mu_array.shape[1]),
            "mu_mean_abs": float(np.mean(np.abs(mu_array))),
            "mu_std_mean": float(np.mean(np.std(mu_array, axis=0))),
            "log_var_mean": float(np.mean(log_var_array)),
            "log_var_std": float(np.std(log_var_array)),
            "active_dim_count_std_gt_0_05": int(np.sum(np.std(mu_array, axis=0) > 0.05)),
        }

    def _save_model(self, path: Path, model: DenoisingMusicVAE) -> None:
        """Save model state and config."""
        payload = {
            "model_type": "DenoisingMusicVAE",
            "config": self.dvae_config.to_dict(),
            "state_dict": model.state_dict(),
        }
        torch.save(payload, path)

    def _write_encoded_artifacts(self, output_dir: Path, songs: List[SongRecord], tensors: List[BarTensorRecord]) -> None:
        """Persist encoded training data for inspection."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "songs.json").write_text(
            json.dumps([song.to_dict() for song in songs], indent=2),
            encoding="utf-8",
        )
        arrays = {
            f"{record.song_id}__bar_{record.bar_index:04d}": record.tensor
            for record in tensors
        }
        np.savez_compressed(output_dir / "bar_tensors.npz", **arrays)
        index_rows = [
            {
                "tensor_key": f"{record.song_id}__bar_{record.bar_index:04d}",
                "song_id": record.song_id,
                "bar_index": int(record.bar_index),
                "tensor_shape": record.tensor_shape,
                "diagnostics": record.diagnostics,
            }
            for record in tensors
        ]
        (output_dir / "bar_tensor_index.json").write_text(json.dumps(index_rows, indent=2), encoding="utf-8")

    def _summary(
        self,
        songs: List[SongRecord],
        tensors: List[BarTensorRecord],
        history: List[Dict[str, float]],
        reconstruction: Dict[str, Any],
        latent: Dict[str, Any],
        model_path: Path,
    ) -> Dict[str, Any]:
        """Build a compact training summary."""
        return {
            "model_path": str(model_path),
            "song_count": int(len(songs)),
            "bar_count": int(len(tensors)),
            "transpose_semitones": [int(x) for x in self.training_config.transpose_semitones],
            "epochs": int(self.training_config.epochs),
            "final_epoch": history[-1] if history else {},
            "reconstruction": reconstruction,
            "latent": latent,
            "data_size_warning": self._data_size_warning(len(tensors)),
        }

    def _data_size_warning(self, bar_count: int) -> Dict[str, Any]:
        """Provide simple guidance on whether augmentation may be needed."""
        if bar_count < 1000:
            level = "high"
            message = "Very small bar count; consider 12-key transposition augmentation."
        elif bar_count < 5000:
            level = "medium"
            message = "Moderate bar count; compare baseline against transposition augmentation."
        else:
            level = "low"
            message = "Bar count is likely sufficient for the tiny DVAE baseline."
        return {"level": level, "message": message}

    def _global_action_counts(self, action_diagnostics: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate action counts across encoded songs."""
        counts: Dict[str, int] = {}
        for song_diag in action_diagnostics:
            for action, count in song_diag.get("action_counts", {}).items():
                counts[str(action)] = counts.get(str(action), 0) + int(count)
        return counts
