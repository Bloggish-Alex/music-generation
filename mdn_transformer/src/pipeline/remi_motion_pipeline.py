#!/usr/bin/env python3
"""Training and generation pipelines for REMI-motion DVAE generation."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common.config_loader import ConfigView
from diagnostics.dvae_midi_render import DVAEMidiRenderConfig
from model.dvae import DVAEMusicConfig, DenoisingMusicVAE
from model.remi_motion import (
    AlignedRemiMotionModelConfig,
    AlignedRemiMotionPredictor,
    RemiBasePitchMotionConfig,
    RemiBasePitchMotionPredictor,
    RemiMotionLatentPredictor,
    RemiMotionModelConfig,
)
from motion.remi_adapter import RemiBarTokenCache, RemiTokenizerFactory, RemiTokenizerSettings
from pipeline.latent_generation_pipeline import SequenceTensorMidiRenderer
from pipeline.latent_transformer_training_pipeline import LatentDatasetReader


@dataclass(frozen=True)
class RemiMotionConfig:
    """Runtime and model settings for REMI-motion experiments."""

    architecture: str = "aligned_tristream"
    context_bars: int = 8
    max_context_tokens: int = 1024
    max_bar_tokens: int = 192
    latent_dim: int = 32
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    predictor_hidden_dim: int = 512
    context_pooling: str = "attention"
    vocab_size: int = 30000
    base_pitch: int = 60
    tempo_bpm: int = 100
    register_offset_min: int = -24
    register_offset_max: int = 24
    register_offset_scale: float = 24.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RemiMotionConfig":
        section = ConfigView(config).section("remi_motion")
        return cls(
            architecture=str(section.get("architecture", "aligned_tristream")),
            context_bars=int(section.get("context_bars", 8)),
            max_context_tokens=int(section.get("max_context_tokens", 1024)),
            max_bar_tokens=int(section.get("max_bar_tokens", 192)),
            latent_dim=int(section.get("latent_dim", 32)),
            d_model=int(section.get("d_model", 256)),
            n_layers=int(section.get("n_layers", 4)),
            n_heads=int(section.get("n_heads", 4)),
            dropout=float(section.get("dropout", 0.1)),
            predictor_hidden_dim=int(section.get("predictor_hidden_dim", 512)),
            context_pooling=str(section.get("context_pooling", "attention")),
            vocab_size=int(section.get("vocab_size", 30000)),
            base_pitch=int(section.get("base_pitch", 60)),
            tempo_bpm=int(section.get("tempo_bpm", 100)),
            register_offset_min=int(section.get("register_offset_min", -24)),
            register_offset_max=int(section.get("register_offset_max", 24)),
            register_offset_scale=float(section.get("register_offset_scale", 24.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RemiMotionTrainingConfig:
    """Training settings for REMI-motion latent prediction."""

    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    validation_ratio: float = 0.1
    validation_split_unit: str = "base_song_id"
    random_seed: int = 42
    device: str = "cpu"
    max_songs: Optional[int] = None
    force_rebuild_tokens: bool = False
    decode_aware_state_loss: bool = False
    decode_aware_density_loss: bool = False
    latent_loss_weight: float = 1.0
    state_loss_weight: float = 1.0
    density_loss_weight: float = 1.0
    latent_std_floor: float = 1.0e-3
    density_std_floor: float = 1.0e-3

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RemiMotionTrainingConfig":
        section = ConfigView(config).section("remi_motion_training")
        max_songs = section.get("max_songs", None)
        return cls(
            epochs=int(section.get("epochs", 40)),
            batch_size=int(section.get("batch_size", 64)),
            learning_rate=float(section.get("learning_rate", 5.0e-4)),
            weight_decay=float(section.get("weight_decay", 1.0e-4)),
            validation_ratio=float(section.get("validation_ratio", 0.1)),
            validation_split_unit=str(section.get("validation_split_unit", "base_song_id")),
            random_seed=int(section.get("random_seed", 42)),
            device=str(section.get("device", "cpu")),
            max_songs=None if max_songs is None else int(max_songs),
            force_rebuild_tokens=bool(section.get("force_rebuild_tokens", False)),
            decode_aware_state_loss=bool(section.get("decode_aware_state_loss", False)),
            decode_aware_density_loss=bool(section.get("decode_aware_density_loss", False)),
            latent_loss_weight=float(section.get("latent_loss_weight", 1.0)),
            state_loss_weight=float(section.get("state_loss_weight", 1.0)),
            density_loss_weight=float(section.get("density_loss_weight", 1.0)),
            latent_std_floor=float(section.get("latent_std_floor", 1.0e-3)),
            density_std_floor=float(section.get("density_std_floor", 1.0e-3)),
        )


@dataclass(frozen=True)
class RemiMotionGenerationConfig:
    """Generation settings for REMI-motion DVAE generation."""

    bars: int = 32
    primer_bars: int = 4
    seed: int = 42
    device: str = "cpu"
    base_pitch: int = 60
    tempo_bpm: int = 100
    base_pitch_mode: str = "learned"
    base_pitch_delta_min: Optional[int] = -7
    base_pitch_delta_max: Optional[int] = 7
    render_base_pitch_min: int = 36
    render_base_pitch_max: int = 84
    seed_song_id: Optional[str] = None
    audio_quality_enabled: bool = True
    feedback_tokenization_enabled: bool = True

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RemiMotionGenerationConfig":
        section = ConfigView(config).section("remi_motion_generation")
        fallback = ConfigView(config).section("latent_generation")
        return cls(
            bars=int(section.get("bars", fallback.get("bars", 32))),
            primer_bars=int(section.get("primer_bars", fallback.get("primer_bars", 4))),
            seed=int(section.get("seed", fallback.get("seed", 42))),
            device=str(section.get("device", fallback.get("device", "cpu"))),
            base_pitch=int(section.get("base_pitch", fallback.get("base_pitch", 60))),
            tempo_bpm=int(section.get("tempo_bpm", fallback.get("tempo_bpm", 100))),
            base_pitch_mode=str(section.get("base_pitch_mode", fallback.get("base_pitch_mode", "learned"))),
            base_pitch_delta_min=(
                None
                if section.get("base_pitch_delta_min", fallback.get("base_pitch_delta_min", -7)) is None
                else int(section.get("base_pitch_delta_min", fallback.get("base_pitch_delta_min", -7)))
            ),
            base_pitch_delta_max=(
                None
                if section.get("base_pitch_delta_max", fallback.get("base_pitch_delta_max", 7)) is None
                else int(section.get("base_pitch_delta_max", fallback.get("base_pitch_delta_max", 7)))
            ),
            render_base_pitch_min=int(section.get("render_base_pitch_min", fallback.get("render_base_pitch_min", 36))),
            render_base_pitch_max=int(section.get("render_base_pitch_max", fallback.get("render_base_pitch_max", 84))),
            seed_song_id=section.get("seed_song_id", None),
            audio_quality_enabled=bool(section.get("audio_quality_enabled", True)),
            feedback_tokenization_enabled=bool(section.get("feedback_tokenization_enabled", True)),
        )


@dataclass
class RemiMotionSample:
    """One supervised sequence sample."""

    input_ids: np.ndarray
    context_input_ids: List[np.ndarray]
    context_latents: np.ndarray
    context_register_offsets: np.ndarray
    prev_latent: np.ndarray
    target_latent: np.ndarray
    song_anchor: int
    target_register_offset_class: int
    target_register_offset: int
    song_id: str
    target_bar_index: int
    target_row_index: int


class RemiMotionDataset(Dataset):
    """Torch dataset for REMI motion samples."""

    def __init__(self, samples: Sequence[RemiMotionSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> RemiMotionSample:
        return self.samples[index]


class RemiMotionCollator:
    """Pad variable-length REMI token contexts."""

    def __init__(self, pad_token_id: int, max_context_tokens: int, context_bars: int, max_bar_tokens: int) -> None:
        self.pad_token_id = int(pad_token_id)
        self.max_context_tokens = int(max_context_tokens)
        self.context_bars = int(context_bars)
        self.max_bar_tokens = int(max_bar_tokens)

    def __call__(self, batch: Sequence[RemiMotionSample]) -> Dict[str, torch.Tensor]:
        length = min(self.max_context_tokens, max(1, max(len(sample.input_ids) for sample in batch)))
        input_ids = np.full((len(batch), length), self.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((len(batch), length), dtype=np.int64)
        max_seen_bar_tokens = max(
            1,
            max((len(ids) for sample in batch for ids in sample.context_input_ids), default=1),
        )
        bar_token_length = min(self.max_bar_tokens, max_seen_bar_tokens)
        context_input_ids = np.full(
            (len(batch), self.context_bars, bar_token_length),
            self.pad_token_id,
            dtype=np.int64,
        )
        context_attention_mask = np.zeros((len(batch), self.context_bars, bar_token_length), dtype=np.int64)
        context_bar_mask = np.zeros((len(batch), self.context_bars), dtype=np.int64)
        latent_dim = int(batch[0].context_latents.shape[-1])
        context_latents = np.zeros((len(batch), self.context_bars, latent_dim), dtype=np.float32)
        context_register_offsets = np.zeros((len(batch), self.context_bars), dtype=np.float32)
        prev_latent = np.stack([sample.prev_latent for sample in batch], axis=0).astype(np.float32)
        target_latent = np.stack([sample.target_latent for sample in batch], axis=0).astype(np.float32)
        song_anchor = np.asarray([sample.song_anchor for sample in batch], dtype=np.float32)
        target_register_offset_class = np.asarray(
            [sample.target_register_offset_class for sample in batch],
            dtype=np.int64,
        )
        target_register_offset = np.asarray([sample.target_register_offset for sample in batch], dtype=np.int64)
        for row, sample in enumerate(batch):
            ids = np.asarray(sample.input_ids[-length:], dtype=np.int64)
            input_ids[row, : len(ids)] = ids
            attention_mask[row, : len(ids)] = 1
            actual_bars = min(self.context_bars, len(sample.context_input_ids))
            start = self.context_bars - actual_bars
            for local_index in range(actual_bars):
                source_index = len(sample.context_input_ids) - actual_bars + local_index
                target_index = start + local_index
                bar_ids = np.asarray(sample.context_input_ids[source_index][:bar_token_length], dtype=np.int64)
                context_input_ids[row, target_index, : len(bar_ids)] = bar_ids
                context_attention_mask[row, target_index, : len(bar_ids)] = 1
                context_bar_mask[row, target_index] = 1
            context_latents[row, start:, :] = sample.context_latents[-actual_bars:].astype(np.float32)
            context_register_offsets[row, start:] = sample.context_register_offsets[-actual_bars:].astype(np.float32)
        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "context_input_ids": torch.from_numpy(context_input_ids),
            "context_attention_mask": torch.from_numpy(context_attention_mask),
            "context_bar_mask": torch.from_numpy(context_bar_mask),
            "context_latents": torch.from_numpy(context_latents),
            "context_register_offsets": torch.from_numpy(context_register_offsets),
            "prev_latent": torch.from_numpy(prev_latent),
            "target_latent": torch.from_numpy(target_latent),
            "song_anchor": torch.from_numpy(song_anchor),
            "target_register_offset_class": torch.from_numpy(target_register_offset_class),
            "target_register_offset": torch.from_numpy(target_register_offset),
        }


class RemiMotionLossCalculator:
    """Compute scale-normalized latent and decoded music losses."""

    def __init__(
        self,
        latent_weight: float,
        state_weight: float,
        density_weight: float,
        dvae: Optional[DenoisingMusicVAE] = None,
        latent_std_floor: float = 1.0e-3,
        density_std_floor: float = 1.0e-3,
    ) -> None:
        self.latent_weight = float(latent_weight)
        self.state_weight = float(state_weight)
        self.density_weight = float(density_weight)
        self.dvae = dvae
        self.latent_std_floor = float(latent_std_floor)
        self.density_std_floor = float(density_std_floor)
        self.latent_scale: Optional[np.ndarray] = None
        self.density_scale: Optional[np.ndarray] = None

    def fit_normalization(self, target_latents: np.ndarray, batch_size: int) -> None:
        """Fit data-derived loss scales from training targets only."""
        targets = np.asarray(target_latents, dtype=np.float32)
        if targets.ndim != 2 or len(targets) < 2:
            raise ValueError("Need at least two target latent vectors to fit loss normalization.")
        self.latent_scale = np.maximum(
            np.std(targets, axis=0).astype(np.float32),
            float(self.latent_std_floor),
        )
        if self.dvae is None or self.density_weight <= 0.0:
            return
        device = next(self.dvae.parameters()).device
        density_rows: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(targets), max(1, int(batch_size))):
                batch = torch.from_numpy(targets[start:start + max(1, int(batch_size))]).to(device)
                _, state_logits, _, _ = self.dvae.decoder(batch)
                target_state = torch.argmax(state_logits, dim=-1)
                one_hot = torch.nn.functional.one_hot(
                    target_state,
                    num_classes=int(state_logits.shape[-1]),
                ).to(dtype=torch.float32)
                density_rows.append(self._state_density(one_hot).cpu().numpy())
        density = np.concatenate(density_rows, axis=0)
        self.density_scale = np.maximum(
            np.std(density, axis=0).astype(np.float32),
            float(self.density_std_floor),
        )

    def __call__(self, pred_latent: torch.Tensor, target_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent_raw_mse = torch.nn.functional.mse_loss(pred_latent, target_latent)
        latent_scale = self._scale_tensor(self.latent_scale, pred_latent, self.latent_std_floor)
        latent_loss = torch.mean(torch.square((pred_latent - target_latent) / latent_scale))
        state_loss = pred_latent.new_tensor(0.0)
        density_loss = pred_latent.new_tensor(0.0)
        state_raw_ce = pred_latent.new_tensor(0.0)
        density_raw_mse = pred_latent.new_tensor(0.0)
        if self.dvae is not None and (self.state_weight > 0.0 or self.density_weight > 0.0):
            state_loss, density_loss, state_raw_ce, density_raw_mse = self._decoded_tensor_losses(
                pred_latent,
                target_latent,
            )
        total = (
            float(self.latent_weight) * latent_loss
            + float(self.state_weight) * state_loss
            + float(self.density_weight) * density_loss
        )
        return {
            "loss": total,
            "latent_loss": latent_loss,
            "latent_raw_mse": latent_raw_mse,
            "state_loss": state_loss,
            "state_raw_ce": state_raw_ce,
            "density_loss": density_loss,
            "density_raw_mse": density_raw_mse,
        }

    def _decoded_tensor_losses(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compare decoded state classes and density statistics through a frozen decoder."""
        if self.dvae is None:
            zero = pred_latent.new_tensor(0.0)
            return zero, zero, zero, zero
        _, pred_state_logits, _, _ = self.dvae.decoder(pred_latent)
        pred_state_probs = torch.softmax(pred_state_logits, dim=-1)
        with torch.no_grad():
            _, target_state_logits, _, _ = self.dvae.decoder(target_latent)
            target_state = torch.argmax(target_state_logits, dim=-1)
            target_state_one_hot = torch.nn.functional.one_hot(
                target_state,
                num_classes=int(pred_state_logits.shape[-1]),
            ).to(dtype=pred_state_probs.dtype, device=pred_state_probs.device)
        state_raw_ce = torch.nn.functional.cross_entropy(
            pred_state_logits.reshape(-1, pred_state_logits.shape[-1]),
            target_state.reshape(-1),
        )
        pred_density = self._state_density(pred_state_probs)
        target_density = self._state_density(target_state_one_hot)
        density_raw_mse = torch.nn.functional.mse_loss(pred_density, target_density)
        state_loss = state_raw_ce / math.log(max(2, int(pred_state_logits.shape[-1])))
        density_scale = self._scale_tensor(self.density_scale, pred_density, self.density_std_floor)
        density_loss = torch.mean(torch.square((pred_density - target_density) / density_scale))
        return state_loss, density_loss, state_raw_ce, density_raw_mse

    def _scale_tensor(
        self,
        values: Optional[np.ndarray],
        reference: torch.Tensor,
        floor: float,
    ) -> torch.Tensor:
        if values is None:
            return reference.new_tensor(1.0)
        return torch.from_numpy(values).to(device=reference.device, dtype=reference.dtype)

    def _state_density(self, state_probs: torch.Tensor) -> torch.Tensor:
        """Return per-sample decoded rest/note-on/hold/active densities."""
        rest = state_probs[..., 0].mean(dim=(1, 2))
        note_on = state_probs[..., 1].mean(dim=(1, 2))
        hold = state_probs[..., 2].mean(dim=(1, 2))
        active = note_on + hold
        return torch.stack([rest, note_on, hold, active], dim=-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latent_weight": float(self.latent_weight),
            "state_weight": float(self.state_weight),
            "density_weight": float(self.density_weight),
            "normalization": {
                "latent": "training_target_std",
                "state": "cross_entropy_divided_by_log_class_count",
                "density": "training_target_density_std",
                "latent_std_floor": float(self.latent_std_floor),
                "density_std_floor": float(self.density_std_floor),
                "latent_scale": None if self.latent_scale is None else self.latent_scale.tolist(),
                "density_scale": None if self.density_scale is None else self.density_scale.tolist(),
            },
            "decode_aware_state_loss": bool(self.dvae is not None and self.state_weight > 0.0),
            "decode_aware_density_loss": bool(self.dvae is not None and self.density_weight > 0.0),
        }


class RemiMotionSampleBuilder:
    """Build next-latent samples from REMI bar tokens and latent metadata."""

    def __init__(self, context_bars: int, max_context_tokens: int) -> None:
        self.context_bars = int(context_bars)
        self.max_context_tokens = int(max_context_tokens)

    def build(
        self,
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        token_by_key: Dict[str, List[int]],
        register_offsets: np.ndarray,
        song_anchors: np.ndarray,
        register_config: AlignedRemiMotionModelConfig,
    ) -> List[RemiMotionSample]:
        grouped: Dict[str, List[int]] = {}
        for row_index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(row_index)
        samples: List[RemiMotionSample] = []
        for song_id, indices in grouped.items():
            ordered = sorted(indices, key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx))))
            if len(ordered) < 2:
                continue
            for local_target in range(1, len(ordered)):
                context_indices = ordered[max(0, local_target - self.context_bars):local_target]
                token_ids: List[int] = []
                context_token_ids: List[np.ndarray] = []
                missing = False
                for context_index in context_indices:
                    key = str(rows[context_index].get("tensor_key", ""))
                    ids = token_by_key.get(key)
                    if not ids:
                        missing = True
                        break
                    bar_ids = np.asarray([int(item) for item in ids], dtype=np.int64)
                    context_token_ids.append(bar_ids)
                    token_ids.extend(int(item) for item in bar_ids)
                if missing or not token_ids:
                    continue
                token_array = np.asarray(token_ids[-self.max_context_tokens:], dtype=np.int64)
                target_index = ordered[local_target]
                samples.append(RemiMotionSample(
                    input_ids=token_array,
                    context_input_ids=context_token_ids,
                    context_latents=mu[context_indices].astype(np.float32),
                    context_register_offsets=register_offsets[context_indices].astype(np.float32),
                    prev_latent=mu[context_indices[-1]].astype(np.float32),
                    target_latent=mu[target_index].astype(np.float32),
                    song_anchor=int(song_anchors[target_index]),
                    target_register_offset_class=int(
                        register_config.register_offset_to_class(int(register_offsets[target_index]))
                    ),
                    target_register_offset=int(register_offsets[target_index]),
                    song_id=str(song_id),
                    target_bar_index=int(rows[target_index].get("bar_index", 0)),
                    target_row_index=int(target_index),
                ))
        if not samples:
            raise ValueError("No REMI motion samples were built. Check token cache and latent rows.")
        return samples


class RemiMotionTrainingPipeline:
    """Train the pure REMI-motion next-latent predictor."""

    def __init__(self, config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> None:
        self.config = config
        self.motion_config = self._motion_config(overrides or {})
        self.training_config = self._training_config(overrides or {})

    def run(self, model_dir: str | Path, latent_dir: Optional[str | Path] = None, encoded_dir: Optional[str | Path] = None) -> Dict[str, Any]:
        """Train and save a REMI-motion model under model_dir/remi_motion."""
        self._set_seed()
        model_path = Path(model_dir)
        latent_path = Path(latent_dir) if latent_dir else model_path / "latent"
        encoded_path = Path(encoded_dir) if encoded_dir else model_path / "encoded"
        output_dir = model_path / "remi_motion"
        output_dir.mkdir(parents=True, exist_ok=True)
        token_payload = self._token_cache(output_dir).build_or_load(
            encoded_path,
            force_rebuild=bool(self.training_config.force_rebuild_tokens),
            max_songs=self.training_config.max_songs,
        )
        tokenizer_path = self._resolve_training_artifact_path(
            token_payload.get("tokenizer_path"),
            output_dir,
            "tokenizer.json",
        )
        tokenizer = RemiTokenizerFactory(self._tokenizer_settings()).load(tokenizer_path)
        pad_token_id = int(tokenizer["PAD_None"])
        mu, rows, latent_summary = LatentDatasetReader().load(latent_path)
        source_base_pitches = self._base_pitch_array(rows, encoded_path)
        token_layers = max(1, int(self.motion_config.n_layers) // 2)
        bar_layers = max(1, int(self.motion_config.n_layers) - token_layers)
        model_config = AlignedRemiMotionModelConfig(
            vocab_size=int(len(tokenizer)),
            pad_token_id=pad_token_id,
            latent_dim=int(mu.shape[1]),
            d_model=int(self.motion_config.d_model),
            token_layers=token_layers,
            bar_layers=bar_layers,
            n_heads=int(self.motion_config.n_heads),
            dropout=float(self.motion_config.dropout),
            context_bars=int(self.motion_config.context_bars),
            max_bar_tokens=int(self.motion_config.max_bar_tokens),
            predictor_hidden_dim=int(self.motion_config.predictor_hidden_dim),
            context_pooling=str(self.motion_config.context_pooling),
            register_offset_min=int(self.motion_config.register_offset_min),
            register_offset_max=int(self.motion_config.register_offset_max),
            register_offset_scale=float(self.motion_config.register_offset_scale),
        )
        song_anchors = self._song_register_anchors(rows, source_base_pitches)
        register_offsets = source_base_pitches - song_anchors
        samples = RemiMotionSampleBuilder(
            context_bars=int(self.motion_config.context_bars),
            max_context_tokens=int(self.motion_config.max_context_tokens),
        ).build(mu, rows, token_payload["token_by_key"], register_offsets, song_anchors, model_config)
        train_samples, val_samples = self._split_samples(samples)
        collator = RemiMotionCollator(
            pad_token_id,
            int(self.motion_config.max_context_tokens),
            int(self.motion_config.context_bars),
            int(self.motion_config.max_bar_tokens),
        )
        train_loader = DataLoader(
            RemiMotionDataset(train_samples),
            batch_size=int(self.training_config.batch_size),
            shuffle=True,
            collate_fn=collator,
        )
        val_loader = DataLoader(
            RemiMotionDataset(val_samples),
            batch_size=int(self.training_config.batch_size),
            shuffle=False,
            collate_fn=collator,
        )
        model = AlignedRemiMotionPredictor(model_config).to(self.training_config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.training_config.learning_rate),
            weight_decay=float(self.training_config.weight_decay),
        )
        loss_fn = self._loss_calculator(model_path)
        loss_fn.fit_normalization(
            np.stack([sample.target_latent for sample in train_samples], axis=0),
            batch_size=int(self.training_config.batch_size),
        )
        history: List[Dict[str, float]] = []
        best_music_val = float("inf")
        best_joint_val = float("inf")
        best_music_state: Optional[Dict[str, torch.Tensor]] = None
        for epoch in range(1, int(self.training_config.epochs) + 1):
            train_metrics = self._run_epoch(model, train_loader, optimizer, loss_fn)
            val_metrics = self._evaluate(model, val_loader, loss_fn)
            history.append({
                "epoch": float(epoch),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            })
            best_joint_val = min(best_joint_val, float(val_metrics["loss"]))
            music_loss = float(val_metrics["music_loss"])
            if music_loss < best_music_val:
                best_music_val = music_loss
                best_music_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_music_state is not None:
            model.load_state_dict(best_music_state)
        checkpoint_path = output_dir / "remi_motion_predictor.pt"
        torch.save({
            "model_kind": "aligned_tristream",
            "model_config": model_config.to_dict(),
            "aligned_model_config": model_config.to_dict(),
            "motion_config": self.motion_config.to_dict(),
            "training_config": asdict(self.training_config),
            "loss_config": loss_fn.to_dict(),
            "state_dict": model.state_dict(),
            "tokenizer_path": "tokenizer.json",
            "token_cache_path": "remi_bar_tokens.json",
            "checkpoint_selection_metric": "val_music_loss",
            "best_music_val_loss": float(best_music_val),
            "best_joint_val_loss": float(best_joint_val),
        }, checkpoint_path)
        diagnostics = {
            "backend": "remi_motion",
            "model_path": str(checkpoint_path),
            "model_kind": "aligned_tristream",
            "model_config": model_config.to_dict(),
            "motion_config": self.motion_config.to_dict(),
            "training_config": asdict(self.training_config),
            "loss_config": loss_fn.to_dict(),
            "tokenizer": token_payload.get("summary", {}),
            "latent_summary": latent_summary,
            "register_distribution": self._register_distribution(
                source_base_pitches,
                song_anchors,
                register_offsets,
                samples,
                model_config,
            ),
            "sample_count": int(len(samples)),
            "train_sample_count": int(len(train_samples)),
            "validation_sample_count": int(len(val_samples)),
            "history": history,
            "checkpoint_selection_metric": "val_music_loss",
            "best_music_val_loss": float(best_music_val),
            "best_joint_val_loss": float(best_joint_val),
        }
        (output_dir / "remi_motion_training_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return diagnostics

    def _run_epoch(
        self,
        model: AlignedRemiMotionPredictor,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: "RemiMotionLossCalculator",
    ) -> Dict[str, float]:
        model.train()
        metrics: List[Dict[str, float]] = []
        for batch in loader:
            batch = self._to_device(batch)
            optimizer.zero_grad(set_to_none=True)

            pred, register_offset_logits = model(
                batch["context_input_ids"],
                batch["context_attention_mask"],
                batch["context_latents"],
                batch["context_register_offsets"],
                batch["context_bar_mask"],
            )
            result = loss_fn(pred, batch["target_latent"])
            register_offset_result = model.register_offset_loss(
                register_offset_logits,
                batch["target_register_offset_class"],
                batch["target_register_offset"],
            )
            total_loss = result["loss"] + register_offset_result["loss"]
            total_loss.backward()
            row = self._detach_loss_metrics(result)
            row["music_loss"] = float(row["loss"])
            register_offset_row = self._detach_loss_metrics(register_offset_result)
            del pred
            del register_offset_logits
            del result
            del register_offset_result

            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            row.update({
                f"register_offset_{key}": value
                for key, value in register_offset_row.items()
            })
            row["loss"] = float(row.get("loss", 0.0) + register_offset_row.get("loss", 0.0))
            metrics.append(row)
        return self._mean_metrics(metrics)

    def _evaluate(
        self,
        model: AlignedRemiMotionPredictor,
        loader: DataLoader,
        loss_fn: "RemiMotionLossCalculator",
    ) -> Dict[str, float]:
        model.eval()
        metrics: List[Dict[str, float]] = []
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                pred, register_offset_logits = model(
                    batch["context_input_ids"],
                    batch["context_attention_mask"],
                    batch["context_latents"],
                    batch["context_register_offsets"],
                    batch["context_bar_mask"],
                )
                result = loss_fn(pred, batch["target_latent"])
                register_offset_result = model.register_offset_loss(
                    register_offset_logits,
                    batch["target_register_offset_class"],
                    batch["target_register_offset"],
                )
                row = self._detach_loss_metrics(result)
                row["music_loss"] = float(row["loss"])
                row.update({
                    f"register_offset_{key}": value
                    for key, value in self._detach_loss_metrics(register_offset_result).items()
                })
                row["loss"] = float((result["loss"] + register_offset_result["loss"]).detach().cpu().item())
                metrics.append(row)
        return self._mean_metrics(metrics)

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.training_config.device) for key, value in batch.items()}

    def _loss_calculator(self, model_dir: Path) -> "RemiMotionLossCalculator":
        use_dvae = bool(self.training_config.decode_aware_state_loss) or bool(self.training_config.decode_aware_density_loss)
        state_weight = float(self.training_config.state_loss_weight) if bool(self.training_config.decode_aware_state_loss) else 0.0
        density_weight = float(self.training_config.density_loss_weight) if bool(self.training_config.decode_aware_density_loss) else 0.0
        dvae = self._load_training_dvae(model_dir / "dvae.pt") if use_dvae else None
        return RemiMotionLossCalculator(
            latent_weight=float(self.training_config.latent_loss_weight),
            state_weight=state_weight,
            density_weight=density_weight,
            dvae=dvae,
            latent_std_floor=float(self.training_config.latent_std_floor),
            density_std_floor=float(self.training_config.density_std_floor),
        )

    def _load_training_dvae(self, path: Path) -> DenoisingMusicVAE:
        checkpoint = torch.load(path, map_location=self.training_config.device, weights_only=False)
        model = DenoisingMusicVAE(DVAEMusicConfig(**checkpoint["config"])).to(self.training_config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model

    def _detach_loss_metrics(self, result: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {key: float(value.detach().cpu().item()) for key, value in result.items()}

    def _mean_metrics(self, metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
        if not metrics:
            return {"loss": 0.0, "latent_loss": 0.0, "state_loss": 0.0}
        keys = sorted({key for item in metrics for key in item})
        return {key: float(np.mean([item.get(key, 0.0) for item in metrics])) for key in keys}

    def _base_pitch_array(self, rows: Sequence[Dict[str, Any]], encoded_dir: str | Path) -> np.ndarray:
        """Return source base pitch for each latent row."""
        lookup = self._base_pitch_lookup(encoded_dir)
        values = [int(lookup.get(str(row.get("tensor_key", "")), int(self.motion_config.base_pitch))) for row in rows]
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

    def _song_register_anchors(self, rows: Sequence[Dict[str, Any]], base_pitches: np.ndarray) -> np.ndarray:
        """Return a robust median register anchor for every row in its song."""
        anchors = np.zeros(len(rows), dtype=np.int64)
        grouped: Dict[str, List[int]] = {}
        for row_index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(row_index)
        for row_indices in grouped.values():
            anchor = int(np.rint(np.median(base_pitches[row_indices])))
            anchors[row_indices] = anchor
        return anchors

    def _register_distribution(
        self,
        base_pitches: np.ndarray,
        song_anchors: np.ndarray,
        register_offsets: np.ndarray,
        samples: Sequence[RemiMotionSample],
        model_config: AlignedRemiMotionModelConfig,
    ) -> Dict[str, Any]:
        """Summarize the direct song-relative register target and its clipping coverage."""
        if register_offsets.size == 0:
            return {}
        offsets = register_offsets.astype(np.float32)
        sample_offsets = np.asarray([sample.target_register_offset for sample in samples], dtype=np.float32)
        clipped = np.clip(offsets, int(model_config.register_offset_min), int(model_config.register_offset_max))
        return {
            "definition": "bar_register_center - robust_song_median_register_center",
            "bar_register_center_min": int(np.min(base_pitches)) if len(base_pitches) else None,
            "bar_register_center_max": int(np.max(base_pitches)) if len(base_pitches) else None,
            "song_anchor_min": int(np.min(song_anchors)) if len(song_anchors) else None,
            "song_anchor_max": int(np.max(song_anchors)) if len(song_anchors) else None,
            "offset_min": int(np.min(offsets)),
            "offset_max": int(np.max(offsets)),
            "offset_mean": float(np.mean(offsets)),
            "offset_abs_mean": float(np.mean(np.abs(offsets))),
            "offset_abs_p90": float(np.percentile(np.abs(offsets), 90)),
            "offset_abs_p99": float(np.percentile(np.abs(offsets), 99)),
            "target_offset_mean": float(np.mean(sample_offsets)) if sample_offsets.size else None,
            "class_range": {
                "min": int(model_config.register_offset_min),
                "max": int(model_config.register_offset_max),
            },
            "target_clipped_ratio": float(np.mean(clipped != offsets)),
        }

    def _split_samples(self, samples: Sequence[RemiMotionSample]) -> tuple[List[RemiMotionSample], List[RemiMotionSample]]:
        rng = random.Random(int(self.training_config.random_seed))
        if self.training_config.validation_split_unit == "sample":
            shuffled = list(samples)
            rng.shuffle(shuffled)
            val_count = max(1, int(round(len(shuffled) * float(self.training_config.validation_ratio))))
            return shuffled[val_count:], shuffled[:val_count]
        groups: Dict[str, List[RemiMotionSample]] = {}
        for sample in samples:
            groups.setdefault(self._base_song_id(sample.song_id), []).append(sample)
        group_ids = sorted(groups)
        rng.shuffle(group_ids)
        val_group_count = max(1, int(round(len(group_ids) * float(self.training_config.validation_ratio))))
        val_groups = set(group_ids[:val_group_count])
        train = [sample for group_id, group_samples in groups.items() for sample in group_samples if group_id not in val_groups]
        val = [sample for group_id, group_samples in groups.items() for sample in group_samples if group_id in val_groups]
        if not train or not val:
            shuffled = list(samples)
            rng.shuffle(shuffled)
            val_count = max(1, int(round(len(shuffled) * float(self.training_config.validation_ratio))))
            return shuffled[val_count:], shuffled[:val_count]
        return train, val

    def _base_song_id(self, song_id: str) -> str:
        return re.sub(r"_T[+-]?\d+$", "", str(song_id))

    def _training_config(self, overrides: Dict[str, Any]) -> RemiMotionTrainingConfig:
        config = RemiMotionTrainingConfig.from_config(self.config)
        values = asdict(config)
        for key, value in overrides.items():
            if value is not None and key in values:
                values[key] = value
        return RemiMotionTrainingConfig(**values)

    def _motion_config(self, overrides: Dict[str, Any]) -> RemiMotionConfig:
        config = RemiMotionConfig.from_config(self.config)
        values = asdict(config)
        for key, value in overrides.items():
            if value is not None and key in values:
                values[key] = value
        return RemiMotionConfig(**values)

    def _token_cache(self, output_dir: Path) -> RemiBarTokenCache:
        return RemiBarTokenCache(output_dir, self._render_config(), self._tokenizer_settings())

    def _resolve_training_artifact_path(self, value: Any, output_dir: Path, default_name: str) -> Path:
        """Resolve tokenizer/cache paths for newly built or cached REMI artifacts."""
        default_path = output_dir / default_name
        if value in {None, ""}:
            return default_path
        candidate = Path(str(value))
        if candidate.exists():
            return candidate
        if not candidate.is_absolute():
            relative_candidate = output_dir / candidate
            if relative_candidate.exists():
                return relative_candidate
        fallback = output_dir / candidate.name
        if fallback.exists():
            return fallback
        return default_path

    def _tokenizer_settings(self) -> RemiTokenizerSettings:
        return RemiTokenizerSettings(vocab_size=int(self.motion_config.vocab_size))

    def _render_config(self) -> DVAEMidiRenderConfig:
        return DVAEMidiRenderConfig(tempo_bpm=int(self.motion_config.tempo_bpm), default_base_pitch=int(self.motion_config.base_pitch))

    def _set_seed(self) -> None:
        seed = int(self.training_config.random_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class RemiMotionGenerationPipeline:
    """Generate bar tensors by feeding REMI motion context into a DVAE latent predictor."""

    def __init__(self, config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> None:
        self.config = config
        base = RemiMotionGenerationConfig.from_config(config)
        values = asdict(base)
        for key, value in (overrides or {}).items():
            if value is not None and key in values:
                values[key] = value
        self.generation_config = RemiMotionGenerationConfig(**values)

    def run(
        self,
        model_dir: str | Path,
        output_json: str | Path,
        output_midi: str | Path,
        checkpoint_path: Optional[str | Path] = None,
        dvae_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Generate MIDI and diagnostics."""
        self._set_seed()
        model_directory = Path(model_dir)
        remi_dir = model_directory / "remi_motion"
        checkpoint_file = Path(checkpoint_path) if checkpoint_path else remi_dir / "remi_motion_predictor.pt"
        checkpoint = torch.load(checkpoint_file, map_location=self.generation_config.device, weights_only=False)
        model_kind = str(checkpoint.get("model_kind", "legacy_flat")).lower()
        is_aligned = model_kind == "aligned_tristream" or "aligned_model_config" in checkpoint
        if is_aligned:
            aligned_config = AlignedRemiMotionModelConfig(**checkpoint.get("aligned_model_config", checkpoint["model_config"]))
            model_config = aligned_config
            self.context_bars = int(aligned_config.context_bars)
            model = AlignedRemiMotionPredictor(aligned_config).to(self.generation_config.device)
        else:
            legacy_config = RemiMotionModelConfig(**checkpoint["model_config"])
            model_config = legacy_config
            self.context_bars = int(checkpoint.get("motion_config", {}).get("context_bars", 8))
            model = RemiMotionLatentPredictor(legacy_config).to(self.generation_config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        if is_aligned:
            base_pitch_model, base_pitch_model_config = None, None
        else:
            base_pitch_model, base_pitch_model_config = self._load_remi_base_pitch_motion(checkpoint)
        tokenizer_path = self._resolve_model_artifact_path(
            checkpoint.get("tokenizer_path"),
            remi_dir,
            "tokenizer.json",
        )
        tokenizer = RemiTokenizerFactory(RemiTokenizerSettings()).load(tokenizer_path)
        pad_token_id = int(tokenizer["PAD_None"])
        token_cache_path = self._resolve_model_artifact_path(
            checkpoint.get("token_cache_path"),
            remi_dir,
            "remi_bar_tokens.json",
        )
        token_payload = self._load_token_cache_payload(token_cache_path)
        token_by_key = token_payload["token_by_key"]
        mu, rows, latent_summary = LatentDatasetReader().load(model_directory / "latent")
        base_pitch_lookup = self._base_pitch_lookup(model_directory / "encoded")
        grouped = self._group_rows(rows)
        grouped = self._filter_groups_with_tokens(grouped, rows, token_by_key)
        song_id = self._select_song_id(grouped, self.generation_config.seed_song_id)
        ordered = grouped[song_id]
        primer_count = min(int(self.generation_config.primer_bars), int(self.generation_config.bars), len(ordered))
        if primer_count < 1:
            raise ValueError("Need at least one primer bar for REMI motion generation.")
        tensor_archive = np.load(model_directory / "encoded" / "bar_tensors.npz")
        generated_latents: List[np.ndarray] = [mu[index].astype(np.float32) for index in ordered[:primer_count]]
        generated_tensors: List[np.ndarray] = [
            tensor_archive[str(rows[index]["tensor_key"])].astype(np.float32)
            for index in ordered[:primer_count]
        ]
        source_base_pitches: List[int] = [
            self._row_base_pitch(rows[index], base_pitch_lookup, int(self.generation_config.base_pitch))
            for index in ordered[:primer_count]
        ]
        render_base_pitches: List[int] = self._initial_render_base_pitches(source_base_pitches)
        song_anchor = self._generation_song_anchor(render_base_pitches)
        render_register_offsets: List[int] = [int(value - song_anchor) for value in render_base_pitches]
        context_bar_tokens: List[List[int]] = []
        for index in ordered[:primer_count]:
            ids = token_by_key.get(str(rows[index]["tensor_key"]), [])
            if ids:
                context_bar_tokens.append([int(item) for item in ids])
        if not context_bar_tokens:
            raise ValueError("Primer bars have no REMI tokens.")
        dvae = self._load_dvae(Path(dvae_path) if dvae_path else model_directory / "dvae.pt")
        token_cache = RemiBarTokenCache(remi_dir, self._render_config(), RemiTokenizerSettings())
        step_diagnostics: List[Dict[str, Any]] = []
        temp_dir = Path(output_json).parent / ".remi_motion_generated_tokens"
        temp_dir.mkdir(parents=True, exist_ok=True)
        while len(generated_latents) < int(self.generation_config.bars):
            prev_latent = generated_latents[-1]
            if is_aligned:
                context_arrays = self._aligned_context_arrays(
                    context_bar_tokens,
                    generated_latents,
                    render_register_offsets,
                    context_bars=int(model_config.context_bars),
                    max_bar_tokens=int(model_config.max_bar_tokens),
                    pad_token_id=pad_token_id,
                )
                with torch.no_grad():
                    next_latent_tensor, register_offset_logits = model(
                        torch.from_numpy(context_arrays["context_input_ids"]).to(self.generation_config.device),
                        torch.from_numpy(context_arrays["context_attention_mask"]).to(self.generation_config.device),
                        torch.from_numpy(context_arrays["context_latents"]).to(self.generation_config.device),
                        torch.from_numpy(context_arrays["context_register_offsets"]).to(self.generation_config.device),
                        torch.from_numpy(context_arrays["context_bar_mask"]).to(self.generation_config.device),
                    )
                next_latent = next_latent_tensor.detach().cpu().numpy()[0].astype(np.float32)
                render_base_pitch, base_pitch_diag = self._next_render_base_pitch_from_logits(
                    register_offset_logits=register_offset_logits,
                    song_anchor=song_anchor,
                    model_config=model_config,
                )
                input_ids = self._context_ids(
                    context_bar_tokens,
                    int(checkpoint.get("motion_config", {}).get("max_context_tokens", 1024)),
                )
            else:
                input_ids = self._context_ids(context_bar_tokens, int(model_config.max_context_tokens))
                with torch.no_grad():
                    batch_ids = torch.from_numpy(input_ids.reshape(1, -1).astype(np.int64)).to(self.generation_config.device)
                    mask = torch.ones_like(batch_ids, dtype=torch.long)
                    prev = torch.from_numpy(prev_latent.reshape(1, -1).astype(np.float32)).to(self.generation_config.device)
                    next_latent = model(batch_ids, mask, prev).detach().cpu().numpy()[0].astype(np.float32)
                render_base_pitch, base_pitch_diag = self._next_render_base_pitch(
                    input_ids=input_ids,
                    prev_latent=prev_latent,
                    previous_base_pitch=int(render_base_pitches[-1]) if render_base_pitches else int(self.generation_config.base_pitch),
                    source_base_pitch=int(render_base_pitches[-1]) if render_base_pitches else int(self.generation_config.base_pitch),
                    model=base_pitch_model,
                    model_config=base_pitch_model_config,
                )
            next_tensor = self._decode_tensors(dvae, next_latent.reshape(1, -1))[0]
            bar_index = len(generated_latents)
            if bool(self.generation_config.feedback_tokenization_enabled):
                next_tokens = token_cache.tokenize_tensor_bar(
                    tokenizer,
                    next_tensor,
                    temp_dir / f"generated_bar_{bar_index:04d}.mid",
                    int(render_base_pitch),
                )
                feedback_mode = "generated_bar_tokenization"
            else:
                next_tokens = list(context_bar_tokens[-1])
                feedback_mode = "disabled_reuse_last_context_bar_tokens"
            generated_latents.append(next_latent)
            generated_tensors.append(next_tensor)
            source_base_pitch = int(base_pitch_diag.get("source_base_pitch", song_anchor))
            source_base_pitches.append(source_base_pitch)
            render_base_pitches.append(int(render_base_pitch))
            render_register_offsets.append(int(render_base_pitch - song_anchor))
            context_bar_tokens.append(next_tokens)
            step_diagnostics.append({
                "bar_index": int(bar_index),
                "context_token_count": int(len(input_ids)),
                "generated_token_count": int(len(next_tokens)),
                "latent_norm": float(np.linalg.norm(next_latent)),
                "feedback_mode": feedback_mode,
                "source_base_pitch": int(source_base_pitch),
                "song_anchor": int(song_anchor),
                "render_register_offset": int(render_base_pitch - song_anchor),
                "render_base_pitch": int(render_base_pitch),
                "register_motion": base_pitch_diag,
            })
        tensor_array = np.stack(generated_tensors, axis=0).astype(np.float32)
        latent_array = np.stack(generated_latents, axis=0).astype(np.float32)
        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_path = output_json_path.with_suffix(".bar_tensors.npz")
        np.savez_compressed(
            tensor_path,
            bars=tensor_array,
            latent_mu=latent_array,
            source_base_pitches=np.asarray(source_base_pitches, dtype=np.int64),
            song_anchor=np.asarray([song_anchor], dtype=np.int64),
            render_register_offsets=np.asarray(render_register_offsets, dtype=np.int64),
            render_base_pitches=np.asarray(render_base_pitches, dtype=np.int64),
        )
        midi_diag = SequenceTensorMidiRenderer(self._render_config()).render(
            tensor_array,
            output_midi,
            base_pitch=int(self.generation_config.base_pitch),
            base_pitches=render_base_pitches,
        )
        diagnostics = {
            "backend": "remi_motion_vae_generated",
            "model_dir": str(model_directory),
            "checkpoint": str(checkpoint_file),
            "dvae_checkpoint": str(Path(dvae_path) if dvae_path else model_directory / "dvae.pt"),
            "model_config": checkpoint.get("model_config", {}),
            "motion_config": checkpoint.get("motion_config", {}),
            "training_config": checkpoint.get("training_config", {}),
            "loss_config": checkpoint.get("loss_config", {}),
            "source_song_id": str(song_id),
            "primer_bars": int(primer_count),
            "generated_bars": int(len(generated_tensors)),
            "feedback_tokenization_enabled": bool(self.generation_config.feedback_tokenization_enabled),
            "base_pitch_mode": str(self.generation_config.base_pitch_mode),
            "source_base_pitches": [int(item) for item in source_base_pitches],
            "song_anchor": int(song_anchor),
            "render_register_offsets": [int(item) for item in render_register_offsets],
            "render_base_pitches": [int(item) for item in render_base_pitches],
            "latent_summary": latent_summary,
            "steps": step_diagnostics,
            "tensor_path": str(tensor_path),
            "midi": midi_diag,
        }
        output_json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return diagnostics

    def _context_ids(self, context_bar_tokens: Sequence[Sequence[int]], max_tokens: int) -> np.ndarray:
        bars = context_bar_tokens[-max(1, int(getattr(self, "context_bars", 8))):]
        ids: List[int] = []
        for item in bars:
            ids.extend(int(token) for token in item)
        if not ids:
            ids = [0]
        return np.asarray(ids[-max_tokens:], dtype=np.int64)

    def _aligned_context_arrays(
        self,
        context_bar_tokens: Sequence[Sequence[int]],
        generated_latents: Sequence[np.ndarray],
        render_register_offsets: Sequence[int],
        context_bars: int,
        max_bar_tokens: int,
        pad_token_id: int,
    ) -> Dict[str, np.ndarray]:
        """Return bar-aligned REMI / latent / song-relative register context arrays."""
        actual_bars = min(int(context_bars), len(context_bar_tokens), len(generated_latents), len(render_register_offsets))
        if actual_bars < 1:
            raise ValueError("Aligned REMI motion generation needs at least one context bar.")
        latent_dim = int(np.asarray(generated_latents[-1]).shape[-1])
        token_length = max(
            1,
            min(
                int(max_bar_tokens),
                max(len(context_bar_tokens[-actual_bars + offset]) for offset in range(actual_bars)),
            ),
        )
        input_ids = np.full((1, int(context_bars), token_length), int(pad_token_id), dtype=np.int64)
        attention_mask = np.zeros((1, int(context_bars), token_length), dtype=np.int64)
        bar_mask = np.zeros((1, int(context_bars)), dtype=np.int64)
        latents = np.zeros((1, int(context_bars), latent_dim), dtype=np.float32)
        register_offsets = np.zeros((1, int(context_bars)), dtype=np.float32)
        start = int(context_bars) - actual_bars
        for local_index in range(actual_bars):
            source_index = len(context_bar_tokens) - actual_bars + local_index
            target_index = start + local_index
            ids = np.asarray(context_bar_tokens[source_index][:token_length], dtype=np.int64)
            input_ids[0, target_index, : len(ids)] = ids
            attention_mask[0, target_index, : len(ids)] = 1
            bar_mask[0, target_index] = 1
            latents[0, target_index, :] = np.asarray(generated_latents[source_index], dtype=np.float32)
            register_offsets[0, target_index] = float(render_register_offsets[source_index])
        return {
            "context_input_ids": input_ids,
            "context_attention_mask": attention_mask,
            "context_bar_mask": bar_mask,
            "context_latents": latents,
            "context_register_offsets": register_offsets,
        }

    def _next_render_base_pitch_from_logits(
        self,
        register_offset_logits: torch.Tensor,
        song_anchor: int,
        model_config: AlignedRemiMotionModelConfig,
    ) -> tuple[int, Dict[str, Any]]:
        """Resolve next render base pitch directly from a song-relative register class."""
        mode = str(self.generation_config.base_pitch_mode).strip().lower()
        if mode == "fixed":
            return int(self.generation_config.base_pitch), {"mode": "fixed", "register_offset": 0}
        if mode != "learned":
            return self._clip_render_base_pitch(int(song_anchor)), {"mode": "source", "register_offset": None}
        probs = torch.softmax(register_offset_logits, dim=-1).detach().cpu().numpy()[0]
        class_id = int(np.argmax(probs))
        predicted_register_offset = int(model_config.class_to_register_offset(class_id))
        predicted = self._clip_render_base_pitch(int(song_anchor) + predicted_register_offset)
        top = np.argsort(-probs)[:5]
        return predicted, {
            "mode": "aligned_tristream_register_offset",
            "song_anchor": int(song_anchor),
            "source_base_pitch": int(song_anchor),
            "predicted_register_offset": int(predicted_register_offset),
            "predicted_class": int(class_id),
            "confidence": float(probs[class_id]),
            "top_register_offsets": [
                {
                    "register_offset": int(model_config.class_to_register_offset(int(item))),
                    "probability": float(probs[int(item)]),
                }
                for item in top
            ],
        }

    def _load_remi_base_pitch_motion(
        self,
        checkpoint: Dict[str, Any],
    ) -> tuple[Optional[RemiBasePitchMotionPredictor], Optional[RemiBasePitchMotionConfig]]:
        """Load REMI-context base-pitch motion from the REMI motion checkpoint."""
        mode = str(self.generation_config.base_pitch_mode).strip().lower()
        if mode != "learned":
            return None, None
        if "base_pitch_model_config" not in checkpoint or "base_pitch_state_dict" not in checkpoint:
            raise ValueError(
                "base_pitch_mode=learned requires a REMI motion checkpoint trained with "
                "REMI base-pitch motion. Retrain with train.py --stage remi_motion."
            )
        model_config = RemiBasePitchMotionConfig(**checkpoint["base_pitch_model_config"])
        model = RemiBasePitchMotionPredictor(model_config).to(self.generation_config.device)
        model.load_state_dict(checkpoint["base_pitch_state_dict"])
        model.eval()
        return model, model_config

    def _initial_render_base_pitches(self, source_base_pitches: Sequence[int]) -> List[int]:
        """Return primer render base pitches for the configured mode."""
        mode = str(self.generation_config.base_pitch_mode).strip().lower()
        if mode == "fixed":
            return [int(self.generation_config.base_pitch) for _ in source_base_pitches]
        return [self._clip_render_base_pitch(int(value)) for value in source_base_pitches]

    def _generation_song_anchor(self, primer_base_pitches: Sequence[int]) -> int:
        """Use the primer's robust median register as the generation-time song anchor."""
        if not primer_base_pitches:
            return self._clip_render_base_pitch(int(self.generation_config.base_pitch))
        return self._clip_render_base_pitch(int(np.rint(np.median(np.asarray(primer_base_pitches, dtype=np.float32)))))

    def _next_render_base_pitch(
        self,
        input_ids: np.ndarray,
        prev_latent: np.ndarray,
        previous_base_pitch: int,
        source_base_pitch: int,
        model: Optional[RemiBasePitchMotionPredictor],
        model_config: Optional[RemiBasePitchMotionConfig],
    ) -> tuple[int, Dict[str, Any]]:
        """Predict or resolve the render base pitch for the next bar."""
        mode = str(self.generation_config.base_pitch_mode).strip().lower()
        if mode == "fixed":
            return int(self.generation_config.base_pitch), {"mode": "fixed", "delta": 0}
        if mode != "learned":
            return self._clip_render_base_pitch(int(source_base_pitch)), {"mode": "source", "delta": None}
        if model is None or model_config is None:
            raise ValueError("base_pitch_mode=learned requires a loaded RemiBasePitchMotionPredictor.")
        with torch.no_grad():
            batch_ids = torch.from_numpy(input_ids.reshape(1, -1).astype(np.int64)).to(self.generation_config.device)
            mask = torch.ones_like(batch_ids, dtype=torch.long)
            previous_latent = torch.from_numpy(prev_latent.reshape(1, -1).astype(np.float32)).to(self.generation_config.device)
            previous_base = torch.tensor([float(previous_base_pitch)], dtype=torch.float32, device=self.generation_config.device)
            logits = model(batch_ids, mask, previous_latent, previous_base)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        class_id = int(np.argmax(probs))
        raw_delta = int(model_config.class_to_delta(class_id))
        delta = self._clip_base_pitch_delta(raw_delta)
        previous = int(previous_base_pitch)
        predicted = self._clip_render_base_pitch(previous + delta)
        top = np.argsort(-probs)[:5]
        return predicted, {
            "mode": "learned",
            "previous_base_pitch": int(previous),
            "source_base_pitch": int(source_base_pitch),
            "raw_predicted_delta": int(raw_delta),
            "predicted_delta": int(delta),
            "predicted_class": int(class_id),
            "confidence": float(probs[class_id]),
            "delta_clamped": bool(delta != raw_delta),
            "delta_clip_range": self._base_pitch_delta_clip_range(),
            "top_deltas": [
                {"delta": int(model_config.class_to_delta(int(item))), "probability": float(probs[int(item)])}
                for item in top
            ],
        }

    def _base_pitch_lookup(self, encoded_dir: Path) -> Dict[str, int]:
        """Load source base pitch by tensor key from encoding diagnostics."""
        index_path = encoded_dir / "bar_tensor_index.json"
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

    def _row_base_pitch(self, row: Dict[str, Any], lookup: Dict[str, int], fallback: int) -> int:
        """Return encoded source base pitch for a latent row."""
        return int(lookup.get(str(row.get("tensor_key", "")), int(fallback)))

    def _clip_render_base_pitch(self, value: int) -> int:
        """Clip rendered base pitch to configured usable MIDI range."""
        return max(
            int(self.generation_config.render_base_pitch_min),
            min(int(self.generation_config.render_base_pitch_max), int(value)),
        )

    def _base_pitch_delta_clip_range(self) -> Optional[Dict[str, int]]:
        """Return generation-time base-pitch delta clip range if configured."""
        if self.generation_config.base_pitch_delta_min is None and self.generation_config.base_pitch_delta_max is None:
            return None
        return {
            "min": int(self.generation_config.base_pitch_delta_min)
            if self.generation_config.base_pitch_delta_min is not None
            else -999,
            "max": int(self.generation_config.base_pitch_delta_max)
            if self.generation_config.base_pitch_delta_max is not None
            else 999,
        }

    def _clip_base_pitch_delta(self, delta: int) -> int:
        """Apply optional generation-time base-pitch delta clamp."""
        low = int(self.generation_config.base_pitch_delta_min) if self.generation_config.base_pitch_delta_min is not None else -999
        high = int(self.generation_config.base_pitch_delta_max) if self.generation_config.base_pitch_delta_max is not None else 999
        return max(low, min(high, int(delta)))

    def _load_dvae(self, path: Path) -> DenoisingMusicVAE:
        checkpoint = torch.load(path, map_location=self.generation_config.device, weights_only=False)
        model = DenoisingMusicVAE(DVAEMusicConfig(**checkpoint["config"])).to(self.generation_config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def _resolve_model_artifact_path(self, value: Any, remi_dir: Path, default_name: str) -> Path:
        """Resolve checkpoint artifact paths without trusting stale absolute paths."""
        default_path = remi_dir / default_name
        if value in {None, ""}:
            return default_path
        candidate = Path(str(value))
        if candidate.exists():
            return candidate
        if not candidate.is_absolute():
            relative_candidate = remi_dir / candidate
            if relative_candidate.exists():
                return relative_candidate
        stale_absolute_fallback = remi_dir / candidate.name
        if stale_absolute_fallback.exists():
            return stale_absolute_fallback
        return default_path

    def _load_token_cache_payload(self, path: Path) -> Dict[str, Any]:
        """Load token cache while tolerating old non-portable tokenizer_path lines."""
        text = Path(path).read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = re.sub(
                r'^\s*"tokenizer_path"\s*:\s*.*?,\s*$',
                '  "tokenizer_path": "tokenizer.json",',
                text,
                count=1,
                flags=re.MULTILINE,
            )
            return json.loads(repaired)

    def _decode_tensors(self, dvae: DenoisingMusicVAE, latent_mu: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = torch.from_numpy(latent_mu.astype(np.float32)).to(self.generation_config.device)
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

    def _group_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        return {
            song_id: sorted(indices, key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx))))
            for song_id, indices in grouped.items()
        }

    def _filter_groups_with_tokens(
        self,
        grouped: Dict[str, List[int]],
        rows: Sequence[Dict[str, Any]],
        token_by_key: Dict[str, List[int]],
    ) -> Dict[str, List[int]]:
        """Keep only songs whose primer/context rows can be represented as REMI tokens."""
        filtered: Dict[str, List[int]] = {}
        for song_id, indices in grouped.items():
            available = [index for index in indices if str(rows[index].get("tensor_key", "")) in token_by_key]
            if available:
                filtered[song_id] = available
        if not filtered:
            raise ValueError("No latent songs have matching REMI bar tokens.")
        return filtered

    def _select_song_id(self, grouped: Dict[str, List[int]], seed_song_id: Optional[str]) -> str:
        if seed_song_id:
            if seed_song_id in grouped:
                return seed_song_id
            pattern = re.compile(str(seed_song_id))
            matches = [song_id for song_id in grouped if pattern.search(song_id)]
            if matches:
                return sorted(matches)[0]
            raise ValueError(f"seed_song_id not found: {seed_song_id}")
        return random.Random(int(self.generation_config.seed)).choice(sorted(grouped.keys()))

    def _render_config(self) -> DVAEMidiRenderConfig:
        return DVAEMidiRenderConfig(
            tempo_bpm=int(self.generation_config.tempo_bpm),
            default_base_pitch=int(self.generation_config.base_pitch),
            audio_quality_enabled=bool(self.generation_config.audio_quality_enabled),
        )

    def _set_seed(self) -> None:
        seed = int(self.generation_config.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
