#!/usr/bin/env python3
"""Independent training and generation route for joint bar-trajectory diffusion."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common.config_loader import ConfigView
from diagnostics.dvae_midi_render import DVAEMidiRenderConfig
from model.decoded_harmonic_trajectory import (
    DecodedHarmonicTrajectoryConfig,
    DecodedHarmonicTrajectoryObjective,
)
from model.dvae import DenoisingMusicVAE, DVAEMusicConfig
from model.recurrent_trajectory_context import RecurrentBarKVCache, RecurrentTrajectoryDiffusion
from model.trajectory_diffusion import JointTrajectoryDiffusion, TrajectoryDiffusionConfig
from model.trajectory_flow_matching import TrajectoryFlowMatcher, TrajectoryFlowMatchingModelConfig
from motion.remi_adapter import RemiBarTokenCache, RemiTokenizerFactory, RemiTokenizerSettings
from pipeline.latent_generation_pipeline import SequenceTensorMidiRenderer
from pipeline.latent_transformer_training_pipeline import LatentDatasetReader
from pipeline.remi_motion_pipeline import (
    RemiMotionConfig,
    RemiMotionGenerationPipeline,
    RemiMotionTrainingPipeline,
)


def _apply_config_overrides(
    values: Dict[str, Any],
    overrides: Dict[str, Any],
    aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Apply known CLI overrides without letting unrelated keys leak into a config."""
    resolved = dict(values)
    aliases = aliases or {}
    for key, value in overrides.items():
        target_key = aliases.get(key, key)
        if value is not None and target_key in resolved:
            resolved[target_key] = value
    return resolved


@dataclass(frozen=True)
class TrajectoryDiffusionTrainingConfig:
    """Optimization settings for the joint diffusion experiment."""

    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    validation_ratio: float = 0.1
    validation_split_unit: str = "base_song_id"
    random_seed: int = 42
    device: str = "cpu"
    max_songs: Optional[int] = None
    force_rebuild_tokens: bool = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrajectoryDiffusionTrainingConfig":
        section = ConfigView(config).section("trajectory_diffusion_training")
        max_songs = section.get("max_songs", None)
        return cls(
            epochs=int(section.get("epochs", 40)),
            batch_size=int(section.get("batch_size", 16)),
            learning_rate=float(section.get("learning_rate", 2.0e-4)),
            weight_decay=float(section.get("weight_decay", 1.0e-4)),
            validation_ratio=float(section.get("validation_ratio", 0.1)),
            validation_split_unit=str(section.get("validation_split_unit", "base_song_id")),
            random_seed=int(section.get("random_seed", 42)),
            device=str(section.get("device", "cpu")),
            max_songs=None if max_songs is None else int(max_songs),
            force_rebuild_tokens=bool(section.get("force_rebuild_tokens", False)),
        )


@dataclass(frozen=True)
class TrajectoryDiffusionGenerationConfig:
    """Generation-only diffusion controls; architecture stays in the checkpoint."""

    sampling_steps: int = 16

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrajectoryDiffusionGenerationConfig":
        section = ConfigView(config).section("trajectory_diffusion_generation")
        return cls(sampling_steps=int(section.get("sampling_steps", 16)))


@dataclass(frozen=True)
class TrajectoryFlowMatchingConfig:
    """Optional detached conditional-flow correction after diffusion sampling."""

    enabled: bool = False
    d_model: int = 256
    layers: int = 2
    n_heads: int = 4
    hidden_dim: int = 512
    dropout: float = 0.1
    epochs: int = 20
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    proposal_sampling_steps: int = 16
    integration_steps: int = 4

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrajectoryFlowMatchingConfig":
        section = ConfigView(config).section("trajectory_flow_matching")
        return cls(
            enabled=bool(section.get("enabled", False)),
            d_model=int(section.get("d_model", 256)),
            layers=int(section.get("layers", 2)),
            n_heads=int(section.get("n_heads", 4)),
            hidden_dim=int(section.get("hidden_dim", 512)),
            dropout=float(section.get("dropout", 0.1)),
            epochs=int(section.get("epochs", 20)),
            learning_rate=float(section.get("learning_rate", 2.0e-4)),
            weight_decay=float(section.get("weight_decay", 1.0e-4)),
            proposal_sampling_steps=int(section.get("proposal_sampling_steps", 16)),
            integration_steps=int(section.get("integration_steps", 4)),
        )


@dataclass(frozen=True)
class TrajectoryShortRolloutConfig:
    """Detached generated-history training controls for trajectory models."""

    enabled: bool = False
    blocks: int = 2
    base_pitch_min: int = 36
    base_pitch_max: int = 84

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrajectoryShortRolloutConfig":
        section = ConfigView(config).section("trajectory_short_rollout")
        return cls(
            enabled=bool(section.get("enabled", False)),
            blocks=max(1, int(section.get("blocks", 2))),
            base_pitch_min=int(section.get("base_pitch_min", 36)),
            base_pitch_max=int(section.get("base_pitch_max", 84)),
        )


@dataclass(frozen=True)
class TrajectoryRecurrenceConfig:
    """Bounded Transformer-XL scheduling contract for recurrent planning.

    ``context_bars`` must equal ``commit_bars``: the current input consists
    only of bars already committed to the generated history. ``memory_bars``
    must match the diffusion configuration because it defines the detached K/V
    cache capacity. A four-bar plan may be longer than one committed bar.
    """

    enabled: bool = True
    commit_bars: int = 1
    warmup_bars: int = 16
    segments_per_update: int = 4
    memory_bars: int = 32

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrajectoryRecurrenceConfig":
        section = ConfigView(config).section("trajectory_recurrence")
        return cls(
            enabled=bool(section.get("enabled", True)),
            commit_bars=max(1, int(section.get("commit_bars", 1))),
            warmup_bars=max(4, int(section.get("warmup_bars", 16))),
            segments_per_update=max(1, int(section.get("segments_per_update", 4))),
            memory_bars=max(1, int(section.get("memory_bars", 32))),
        )

    def validate(self, model_config: TrajectoryDiffusionConfig) -> None:
        """Fail early when recurrent config sections no longer describe one system."""
        problems: List[str] = []
        if int(self.commit_bars) > int(model_config.trajectory_bars):
            problems.append("trajectory_recurrence.commit_bars must be <= trajectory_diffusion.trajectory_bars")
        if int(model_config.context_bars) != int(self.commit_bars):
            problems.append("trajectory_diffusion.context_bars must equal trajectory_recurrence.commit_bars")
        if int(self.warmup_bars) % int(self.commit_bars):
            problems.append("trajectory_recurrence.warmup_bars must be divisible by commit_bars")
        if int(model_config.memory_bars) != int(self.memory_bars):
            problems.append("trajectory_diffusion.memory_bars must equal trajectory_recurrence.memory_bars")
        if problems:
            raise ValueError("Invalid recurrent trajectory configuration: " + "; ".join(problems) + ".")


@dataclass(frozen=True)
class RecurrentSongSequence:
    """One contiguous song stream with all three input modalities aligned by bar."""

    song_id: str
    input_ids: np.ndarray
    attention_mask: np.ndarray
    latents: np.ndarray
    register_offsets: np.ndarray
    physical_chroma: np.ndarray
    positions: np.ndarray
    song_anchor: int
    empty_remi_bar_count: int

    @property
    def bar_count(self) -> int:
        return int(self.latents.shape[0])


@dataclass
class TrajectorySample:
    """One history-conditioned, multi-bar continuous target trajectory."""

    context_input_ids: List[np.ndarray]
    context_latents: np.ndarray
    context_register_offsets: np.ndarray
    target_trajectory: np.ndarray
    rollout_trajectories: np.ndarray
    song_anchor: int
    song_id: str
    target_bar_index: int


class TrajectoryDataset(Dataset):
    """Dataset wrapper for trajectory samples."""

    def __init__(self, samples: Sequence[TrajectorySample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TrajectorySample:
        return self.samples[index]


class TrajectoryCollator:
    """Right-align bar history and pad each bar's REMI event sequence."""

    def __init__(self, pad_token_id: int, context_bars: int, max_bar_tokens: int) -> None:
        self.pad_token_id = int(pad_token_id)
        self.context_bars = int(context_bars)
        self.max_bar_tokens = int(max_bar_tokens)

    def __call__(self, batch: Sequence[TrajectorySample]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Cannot collate an empty trajectory batch.")
        token_length = int(self.max_bar_tokens)
        latent_dim = int(batch[0].context_latents.shape[-1])
        horizon = int(batch[0].target_trajectory.shape[0])
        state_dim = int(batch[0].target_trajectory.shape[-1])
        input_ids = np.full((len(batch), self.context_bars, token_length), self.pad_token_id, dtype=np.int64)
        attention = np.zeros_like(input_ids)
        bar_mask = np.zeros((len(batch), self.context_bars), dtype=np.int64)
        latents = np.zeros((len(batch), self.context_bars, latent_dim), dtype=np.float32)
        registers = np.zeros((len(batch), self.context_bars), dtype=np.float32)
        targets = np.zeros((len(batch), horizon, state_dim), dtype=np.float32)
        rollout_blocks = int(batch[0].rollout_trajectories.shape[0])
        rollout_targets = np.zeros((len(batch), rollout_blocks, horizon, state_dim), dtype=np.float32)
        song_anchors = np.zeros((len(batch),), dtype=np.float32)
        for row, sample in enumerate(batch):
            actual = min(self.context_bars, len(sample.context_input_ids))
            start = self.context_bars - actual
            for local in range(actual):
                source = len(sample.context_input_ids) - actual + local
                target = start + local
                ids = np.asarray(sample.context_input_ids[source][:token_length], dtype=np.int64)
                input_ids[row, target, : len(ids)] = ids
                attention[row, target, : len(ids)] = 1
                bar_mask[row, target] = 1
            latents[row, start:, :] = sample.context_latents[-actual:].astype(np.float32)
            registers[row, start:] = sample.context_register_offsets[-actual:].astype(np.float32)
            targets[row] = sample.target_trajectory.astype(np.float32)
            rollout_targets[row] = sample.rollout_trajectories.astype(np.float32)
            song_anchors[row] = float(sample.song_anchor)
        return {
            "context_input_ids": torch.from_numpy(input_ids),
            "context_attention_mask": torch.from_numpy(attention),
            "context_bar_mask": torch.from_numpy(bar_mask),
            "context_latents": torch.from_numpy(latents),
            "context_register_offsets": torch.from_numpy(registers),
            "target_trajectory": torch.from_numpy(targets),
            "rollout_trajectories": torch.from_numpy(rollout_targets),
            "song_anchors": torch.from_numpy(song_anchors),
        }


class TrajectorySampleBuilder:
    """Build one target trajectory from each valid position inside a song."""

    def __init__(self, context_bars: int, trajectory_bars: int, rollout_blocks: int = 1) -> None:
        self.context_bars = int(context_bars)
        self.trajectory_bars = int(trajectory_bars)
        self.rollout_blocks = max(1, int(rollout_blocks))

    def build(
        self,
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        token_by_key: Dict[str, List[int]],
        register_offsets: np.ndarray,
        song_anchors: np.ndarray,
    ) -> List[TrajectorySample]:
        grouped: Dict[str, List[int]] = {}
        for row_index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(row_index)
        samples: List[TrajectorySample] = []
        for song_id, indices in grouped.items():
            ordered = sorted(indices, key=lambda index: (int(rows[index].get("bar_index", 0)), int(rows[index].get("row_index", index))))
            required_future = int(self.trajectory_bars) * int(self.rollout_blocks)
            for target_local in range(1, len(ordered) - required_future + 1):
                context_indices = ordered[max(0, target_local - self.context_bars):target_local]
                context_tokens: List[np.ndarray] = []
                for index in context_indices:
                    ids = token_by_key.get(str(rows[index].get("tensor_key", "")))
                    if not ids:
                        context_tokens = []
                        break
                    context_tokens.append(np.asarray(ids, dtype=np.int64))
                if not context_tokens:
                    continue
                rollout_indices = ordered[target_local:target_local + required_future]
                rollout_targets = np.concatenate(
                    [
                        mu[rollout_indices].astype(np.float32),
                        register_offsets[rollout_indices].astype(np.float32).reshape(-1, 1),
                    ],
                    axis=1,
                ).reshape(int(self.rollout_blocks), int(self.trajectory_bars), -1)
                samples.append(TrajectorySample(
                    context_input_ids=context_tokens,
                    context_latents=mu[context_indices].astype(np.float32),
                    context_register_offsets=register_offsets[context_indices].astype(np.float32),
                    target_trajectory=rollout_targets[0],
                    rollout_trajectories=rollout_targets,
                    song_anchor=int(song_anchors[rollout_indices[0]]),
                    song_id=song_id,
                    target_bar_index=int(rows[rollout_indices[0]].get("bar_index", 0)),
                ))
        if not samples:
            raise ValueError("No trajectory samples were built. Check REMI token cache and song lengths.")
        return samples


class TrajectoryStateNormalizer:
    """Fit only training-target statistics for the joint continuous state."""

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None, floor: float = 1.0e-3) -> None:
        self.mean = mean
        self.std = std
        self.floor = float(floor)

    def fit(self, trajectories: np.ndarray) -> None:
        rows = np.asarray(trajectories, dtype=np.float32).reshape(-1, trajectories.shape[-1])
        self.mean = np.mean(rows, axis=0).astype(np.float32)
        self.std = np.maximum(np.std(rows, axis=0).astype(np.float32), float(self.floor))

    def normalize_numpy(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Trajectory normalizer has not been fitted.")
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def denormalize_numpy(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Trajectory normalizer has not been fitted.")
        return np.asarray(values, dtype=np.float32) * self.std + self.mean

    def normalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Trajectory normalizer has not been fitted.")
        mean = torch.from_numpy(self.mean).to(device=values.device, dtype=values.dtype)
        std = torch.from_numpy(self.std).to(device=values.device, dtype=values.dtype)
        return (values - mean) / std

    def denormalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Trajectory normalizer has not been fitted.")
        mean = torch.from_numpy(self.mean).to(device=values.device, dtype=values.dtype)
        std = torch.from_numpy(self.std).to(device=values.device, dtype=values.dtype)
        return values * std + mean

    def to_dict(self) -> Dict[str, Any]:
        if self.mean is None or self.std is None:
            raise RuntimeError("Trajectory normalizer has not been fitted.")
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "floor": float(self.floor)}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrajectoryStateNormalizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            floor=float(payload.get("floor", 1.0e-3)),
        )


class DetachedTrajectoryRolloutFeedback:
    """Append detached generated trajectories to aligned REMI/latent/register context."""

    def __init__(
        self,
        dvae: DenoisingMusicVAE,
        tokenizer: Any,
        render_config: DVAEMidiRenderConfig,
        pad_token_id: int,
        register_offset_min: int,
        register_offset_max: int,
        base_pitch_min: int,
        base_pitch_max: int,
        device: str,
    ) -> None:
        self.dvae = dvae
        self.tokenizer = tokenizer
        self.pad_token_id = int(pad_token_id)
        self.register_offset_min = int(register_offset_min)
        self.register_offset_max = int(register_offset_max)
        self.base_pitch_min = int(base_pitch_min)
        self.base_pitch_max = int(base_pitch_max)
        self.device = str(device)
        self.tokenizer_bridge = RemiBarTokenCache(Path("."), render_config, RemiTokenizerSettings())

    def append(
        self,
        context: Dict[str, torch.Tensor],
        normalized_trajectory: torch.Tensor,
        normalizer: TrajectoryStateNormalizer,
        song_anchors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Detach one generated block and append it across all aligned input streams."""
        raw = normalizer.denormalize_tensor(normalized_trajectory).detach()
        tensors = self._decode_tensors(raw[..., :-1])
        offsets = torch.round(raw[..., -1]).long().clamp(self.register_offset_min, self.register_offset_max)
        rendered_offsets = offsets.to(dtype=context["context_register_offsets"].dtype)
        token_ids = self._tokenize_tensors(tensors, rendered_offsets, song_anchors)
        block_bars = int(raw.shape[1])
        token_length = int(context["context_input_ids"].shape[-1])
        batch_size = int(raw.shape[0])
        generated_ids = torch.full(
            (batch_size, block_bars, token_length), self.pad_token_id,
            dtype=context["context_input_ids"].dtype, device=context["context_input_ids"].device,
        )
        generated_attention = torch.zeros_like(generated_ids)
        for row, row_tokens in enumerate(token_ids):
            for bar, ids in enumerate(row_tokens):
                actual = min(token_length, len(ids))
                if actual > 0:
                    generated_ids[row, bar, :actual] = torch.as_tensor(
                        ids[:actual], dtype=generated_ids.dtype, device=generated_ids.device
                    )
                    generated_attention[row, bar, :actual] = 1
        return {
            "context_input_ids": torch.cat([context["context_input_ids"][:, block_bars:], generated_ids], dim=1),
            "context_attention_mask": torch.cat([context["context_attention_mask"][:, block_bars:], generated_attention], dim=1),
            "context_latents": torch.cat([context["context_latents"][:, block_bars:], raw[..., :-1].to(context["context_latents"].dtype)], dim=1),
            "context_register_offsets": torch.cat([context["context_register_offsets"][:, block_bars:], rendered_offsets], dim=1),
            "context_bar_mask": torch.cat([
                context["context_bar_mask"][:, block_bars:],
                torch.ones((batch_size, block_bars), dtype=context["context_bar_mask"].dtype, device=context["context_bar_mask"].device),
            ], dim=1),
        }

    def segment_payload(
        self,
        normalized_trajectory: torch.Tensor,
        normalizer: TrajectoryStateNormalizer,
        song_anchors: torch.Tensor,
        token_length: int,
        latent_dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Decode a detached generated segment into aligned REMI, latent, and register streams."""
        raw = normalizer.denormalize_tensor(normalized_trajectory).detach()
        return self.raw_segment_payload(raw, song_anchors, token_length, latent_dtype)

    def raw_segment_payload(
        self,
        raw: torch.Tensor,
        song_anchors: torch.Tensor,
        token_length: int,
        latent_dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Build feedback streams from an already denormalized joint state trajectory."""
        tensors = self._decode_tensors(raw[..., :-1])
        offsets = torch.round(raw[..., -1]).long().clamp(self.register_offset_min, self.register_offset_max)
        token_ids = self._tokenize_tensors(tensors, offsets, song_anchors)
        batch_size, block_bars = raw.shape[:2]
        ids = torch.full(
            (batch_size, block_bars, int(token_length)), self.pad_token_id,
            dtype=torch.long, device=raw.device,
        )
        attention = torch.zeros_like(ids)
        for row, row_tokens in enumerate(token_ids):
            for bar, values in enumerate(row_tokens):
                length = min(int(token_length), len(values))
                if length:
                    ids[row, bar, :length] = torch.as_tensor(values[:length], dtype=ids.dtype, device=ids.device)
                    attention[row, bar, :length] = 1
        return {
            "input_ids": ids,
            "attention_mask": attention,
            "latents": raw[..., :-1].to(dtype=latent_dtype),
            "register_offsets": offsets.to(dtype=latent_dtype),
            "tensors": torch.from_numpy(tensors).to(device=raw.device, dtype=torch.float32),
        }

    def _decode_tensors(self, latent: torch.Tensor) -> np.ndarray:
        shape = latent.shape[:2]
        flat = latent.reshape(-1, latent.shape[-1]).to(self.device)
        with torch.no_grad():
            pitch, state_logits, velocity, chord = self.dvae.decoder(flat)
            state = torch.argmax(state_logits, dim=-1)
            state_one_hot = torch.nn.functional.one_hot(state, num_classes=3).float()
            tensor = torch.zeros(
                (flat.shape[0], int(self.dvae.config.tracks), int(self.dvae.config.steps_per_bar), int(self.dvae.config.feature_dim)),
                dtype=torch.float32, device=flat.device,
            )
            tensor[..., 0] = pitch
            tensor[..., 1:4] = state_one_hot
            tensor[..., 4] = velocity
            tensor[..., 5:5 + chord.shape[-1]] = chord
        return tensor.reshape(*shape, *tensor.shape[1:]).detach().cpu().numpy().astype(np.float32)

    def _tokenize_tensors(
        self,
        tensors: np.ndarray,
        rendered_offsets: torch.Tensor,
        song_anchors: torch.Tensor,
    ) -> List[List[List[int]]]:
        offsets = rendered_offsets.detach().cpu().numpy()
        anchors = song_anchors.detach().cpu().numpy()
        output: List[List[List[int]]] = []
        for row in range(int(tensors.shape[0])):
            row_tokens: List[List[int]] = []
            for bar in range(int(tensors.shape[1])):
                base_pitch = int(np.clip(
                    int(round(float(anchors[row]))) + int(offsets[row, bar]),
                    self.base_pitch_min,
                    self.base_pitch_max,
                ))
                row_tokens.append(self.tokenizer_bridge.tokenize_tensor_bar_in_memory(
                    self.tokenizer, tensors[row, bar], base_pitch
                ))
            output.append(row_tokens)
        return output


class TrajectoryDiffusionTrainingPipeline(RemiMotionTrainingPipeline):
    """Train independent joint trajectory diffusion without modifying direct REMI."""

    def __init__(self, config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config, overrides={})
        self.overrides = overrides or {}
        self.motion_config = RemiMotionConfig.from_config(config)
        self.diffusion_config = self._diffusion_config(self.overrides)
        self.training_config = self._training_config(self.overrides)
        self.flow_config = self._flow_config(self.overrides)
        self.short_rollout_config = self._short_rollout_config(self.overrides)
        self.recurrence_config = TrajectoryRecurrenceConfig.from_config(config)
        harmony_section = ConfigView(config).section("trajectory_decoded_harmony")
        self.decoded_harmony_config = DecodedHarmonicTrajectoryConfig(
            enabled=bool(harmony_section.get("enabled", True)),
            state_loss_weight=float(harmony_section.get("state_loss_weight", 1.0)),
            delta_loss_weight=float(harmony_section.get("delta_loss_weight", 1.0)),
            pitch_scale=float(harmony_section.get("pitch_scale", 24.0)),
            pitch_class_sigma=float(harmony_section.get("pitch_class_sigma", 0.35)),
        )

    def run(
        self,
        model_dir: str | Path,
        latent_dir: Optional[str | Path] = None,
        encoded_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        self._set_seed()
        model_path = Path(model_dir)
        latent_path = Path(latent_dir) if latent_dir else model_path / "latent"
        encoded_path = Path(encoded_dir) if encoded_dir else model_path / "encoded"
        output_dir = model_path / "trajectory_diffusion"
        output_dir.mkdir(parents=True, exist_ok=True)
        token_payload = self._token_cache(output_dir).build_or_load(
            encoded_path,
            force_rebuild=bool(self.training_config.force_rebuild_tokens),
            max_songs=self.training_config.max_songs,
        )
        tokenizer_path = self._resolve_training_artifact_path(token_payload.get("tokenizer_path"), output_dir, "tokenizer.json")
        tokenizer = RemiTokenizerFactory(self._tokenizer_settings()).load(tokenizer_path)
        pad_token_id = int(tokenizer["PAD_None"])
        mu, rows, latent_summary = LatentDatasetReader().load(latent_path)
        base_pitches = self._base_pitch_array(rows, encoded_path)
        song_anchors = self._song_register_anchors(rows, base_pitches)
        offsets = base_pitches - song_anchors
        model_config = TrajectoryDiffusionConfig(
            vocab_size=int(len(tokenizer)),
            pad_token_id=pad_token_id,
            latent_dim=int(mu.shape[1]),
            d_model=int(self.diffusion_config.d_model),
            token_layers=int(self.diffusion_config.token_layers),
            bar_layers=int(self.diffusion_config.bar_layers),
            denoiser_layers=int(self.diffusion_config.denoiser_layers),
            n_heads=int(self.diffusion_config.n_heads),
            dropout=float(self.diffusion_config.dropout),
            context_bars=int(self.diffusion_config.context_bars),
            max_bar_tokens=int(self.diffusion_config.max_bar_tokens),
            trajectory_bars=int(self.diffusion_config.trajectory_bars),
            predictor_hidden_dim=int(self.diffusion_config.predictor_hidden_dim),
            context_pooling=str(self.diffusion_config.context_pooling),
            diffusion_steps=int(self.diffusion_config.diffusion_steps),
            sampling_steps=int(self.diffusion_config.sampling_steps),
            beta_schedule=str(self.diffusion_config.beta_schedule),
            prediction_type=str(self.diffusion_config.prediction_type),
            register_offset_scale=float(self.diffusion_config.register_offset_scale),
            register_offset_min=int(self.diffusion_config.register_offset_min),
            register_offset_max=int(self.diffusion_config.register_offset_max),
            memory_bars=int(self.recurrence_config.memory_bars),
            gradient_checkpointing=bool(self.diffusion_config.gradient_checkpointing),
        )
        if self.recurrence_config.enabled:
            return self._run_recurrent(
                model_path, output_dir, tokenizer, pad_token_id, token_payload, mu, rows, latent_summary,
                offsets, song_anchors, encoded_path, model_config,
            )
        rollout_blocks = int(self.short_rollout_config.blocks) if self.short_rollout_config.enabled else 1
        samples = TrajectorySampleBuilder(model_config.context_bars, model_config.trajectory_bars, rollout_blocks).build(
            mu, rows, token_payload["token_by_key"], offsets, song_anchors
        )
        train_samples, val_samples = self._split_trajectory_samples(samples)
        normalizer = TrajectoryStateNormalizer(floor=float(self.diffusion_config.target_std_floor))
        normalizer.fit(np.stack([sample.target_trajectory for sample in train_samples], axis=0))
        collator = TrajectoryCollator(pad_token_id, model_config.context_bars, model_config.max_bar_tokens)
        train_loader = DataLoader(TrajectoryDataset(train_samples), batch_size=int(self.training_config.batch_size), shuffle=True, collate_fn=collator)
        val_loader = DataLoader(TrajectoryDataset(val_samples), batch_size=int(self.training_config.batch_size), shuffle=False, collate_fn=collator)
        model = JointTrajectoryDiffusion(model_config).to(self.training_config.device)
        rollout_feedback = self._create_short_rollout_feedback(model_path, tokenizer, pad_token_id, model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.training_config.learning_rate), weight_decay=float(self.training_config.weight_decay))
        history: List[Dict[str, float]] = []
        best_val = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        for epoch in range(1, int(self.training_config.epochs) + 1):
            train_metrics = self._run_diffusion_epoch(model, train_loader, optimizer, normalizer, rollout_feedback)
            val_metrics = self._evaluate_diffusion(model, val_loader, normalizer, rollout_feedback)
            history.append({"epoch": float(epoch), **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"val_{key}": value for key, value in val_metrics.items()}})
            if float(val_metrics["loss"]) < best_val:
                best_val = float(val_metrics["loss"])
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        flow_model: Optional[TrajectoryFlowMatcher] = None
        flow_model_config: Optional[TrajectoryFlowMatchingModelConfig] = None
        flow_history: List[Dict[str, float]] = []
        flow_best_val: Optional[float] = None
        if self.flow_config.enabled:
            flow_model_config = TrajectoryFlowMatchingModelConfig(
                state_dim=int(model_config.state_dim),
                condition_dim=int(model_config.d_model),
                trajectory_bars=int(model_config.trajectory_bars),
                d_model=int(self.flow_config.d_model),
                layers=int(self.flow_config.layers),
                n_heads=int(self.flow_config.n_heads),
                hidden_dim=int(self.flow_config.hidden_dim),
                dropout=float(self.flow_config.dropout),
            )
            flow_model, flow_history, flow_best_val = self._train_flow_matcher(
                model, flow_model_config, train_loader, val_loader, normalizer, rollout_feedback
            )
        checkpoint_path = output_dir / "trajectory_diffusion.pt"
        checkpoint = self._build_trajectory_checkpoint(
            model_kind="joint_trajectory_diffusion",
            model=model,
            model_config=model_config,
            normalizer=normalizer,
            best_val_loss=best_val,
            flow_model=flow_model,
            flow_model_config=flow_model_config,
            flow_best_val=flow_best_val,
            short_rollout_config=self.short_rollout_config,
        )
        torch.save(checkpoint, checkpoint_path)
        diagnostics = {
            "backend": "joint_trajectory_diffusion",
            "model_path": str(checkpoint_path),
            "model_config": model_config.to_dict(),
            "motion_config": self.motion_config.to_dict(),
            "diffusion_config": self.diffusion_config.to_dict(),
            "training_config": asdict(self.training_config),
            "normalizer": normalizer.to_dict(),
            "tokenizer": token_payload.get("summary", {}),
            "latent_summary": latent_summary,
            "sample_count": int(len(samples)),
            "train_sample_count": int(len(train_samples)),
            "validation_sample_count": int(len(val_samples)),
            "trajectory_bars": int(model_config.trajectory_bars),
            "history": history,
            "prediction_type": "v",
            "checkpoint_selection_metric": "val_velocity_mse",
            "best_val_loss": float(best_val),
            "flow_matching_config": asdict(self.flow_config),
            "flow_matching_model_config": None if flow_model_config is None else flow_model_config.to_dict(),
            "flow_matching_history": flow_history,
            "flow_matching_checkpoint_selection_metric": "val_corrected_mse",
            "flow_matching_best_val_corrected_mse": flow_best_val,
            "short_rollout_config": asdict(self.short_rollout_config),
        }
        (output_dir / "trajectory_diffusion_training_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return diagnostics

    def _build_trajectory_checkpoint(
        self,
        model_kind: str,
        model: torch.nn.Module,
        model_config: TrajectoryDiffusionConfig,
        normalizer: TrajectoryStateNormalizer,
        best_val_loss: float,
        flow_model: Optional[TrajectoryFlowMatcher],
        flow_model_config: Optional[TrajectoryFlowMatchingModelConfig],
        flow_best_val: Optional[float],
        recurrence_config: Optional[TrajectoryRecurrenceConfig] = None,
        short_rollout_config: Optional[TrajectoryShortRolloutConfig] = None,
    ) -> Dict[str, Any]:
        """Assemble the stable checkpoint contract shared by both diffusion routes."""
        payload: Dict[str, Any] = {
            "model_kind": str(model_kind),
            "model_config": model_config.to_dict(),
            "motion_config": self.motion_config.to_dict(),
            "diffusion_config": self.diffusion_config.to_dict(),
            "training_config": asdict(self.training_config),
            "normalizer": normalizer.to_dict(),
            "state_dict": model.state_dict(),
            "tokenizer_path": "tokenizer.json",
            "token_cache_path": "remi_bar_tokens.json",
            "prediction_type": "v",
            "checkpoint_selection_metric": "val_total_trajectory_loss" if recurrence_config else "val_velocity_mse",
            "best_val_loss": float(best_val_loss),
            "flow_matching_config": asdict(self.flow_config),
            "flow_matching_model_config": None if flow_model_config is None else flow_model_config.to_dict(),
            "flow_matching_checkpoint_selection_metric": "val_corrected_mse",
            "flow_matching_best_val_corrected_mse": flow_best_val,
        }
        if flow_model is not None:
            payload["flow_matching_state_dict"] = flow_model.state_dict()
        if recurrence_config is not None:
            payload["recurrence_config"] = asdict(recurrence_config)
            payload["decoded_harmony_config"] = asdict(self.decoded_harmony_config)
        if short_rollout_config is not None:
            payload["short_rollout_config"] = asdict(short_rollout_config)
        return payload

    def _run_recurrent(
        self,
        model_path: Path,
        output_dir: Path,
        tokenizer: Any,
        pad_token_id: int,
        token_payload: Dict[str, Any],
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        latent_summary: Dict[str, Any],
        offsets: np.ndarray,
        song_anchors: np.ndarray,
        encoded_path: Path,
        model_config: TrajectoryDiffusionConfig,
    ) -> Dict[str, Any]:
        """Train with contiguous song segments and bounded detached K/V memory."""
        recurrence = self.recurrence_config
        recurrence.validate(model_config)
        songs = self._build_recurrent_songs(
            mu, rows, token_payload["token_by_key"], offsets, song_anchors, encoded_path, model_config,
        )
        train_songs, val_songs = self._split_recurrent_songs(songs)
        normalizer = TrajectoryStateNormalizer(floor=float(self.diffusion_config.target_std_floor))
        normalizer.fit(np.concatenate([
            np.concatenate([song.latents, song.register_offsets[:, None]], axis=1)
            for song in train_songs
        ], axis=0))
        train_windows = self._recurrent_windows(train_songs)
        val_windows = self._recurrent_windows(val_songs)
        if not train_windows or not val_windows:
            raise ValueError("Recurrent trajectory split has no valid contiguous windows. Increase data or reduce warmup/segments.")
        print(
            "Recurrent schedule: "
            f"commit={recurrence.commit_bars}, plan={model_config.trajectory_bars}, "
            f"commits_per_window={recurrence.segments_per_update}, "
            f"window_stride={self._recurrent_window_stride()}, "
            f"train_windows={len(train_windows)}, val_windows={len(val_windows)}"
        )
        model = RecurrentTrajectoryDiffusion(model_config).to(self.training_config.device)
        feedback = self._create_recurrent_feedback(model_path, tokenizer, pad_token_id, model_config)
        harmony_objective = DecodedHarmonicTrajectoryObjective(feedback.dvae, self.decoded_harmony_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.training_config.learning_rate), weight_decay=float(self.training_config.weight_decay))
        history: List[Dict[str, float]] = []
        best_val = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        for epoch in range(1, int(self.training_config.epochs) + 1):
            train_metrics = self._run_recurrent_diffusion_epoch(model, train_windows, optimizer, normalizer, feedback, harmony_objective, training=True)
            val_metrics = self._run_recurrent_diffusion_epoch(model, val_windows, None, normalizer, feedback, harmony_objective, training=False)
            history.append({"epoch": float(epoch), **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"val_{key}": value for key, value in val_metrics.items()}})
            print(
                f"[recurrent diffusion] epoch {epoch}/{self.training_config.epochs}: "
                f"train_loss={train_metrics['loss']:.6f}, val_loss={val_metrics['loss']:.6f}"
            )
            if float(val_metrics["loss"]) < best_val:
                best_val = float(val_metrics["loss"])
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        flow_model: Optional[TrajectoryFlowMatcher] = None
        flow_model_config: Optional[TrajectoryFlowMatchingModelConfig] = None
        flow_history: List[Dict[str, float]] = []
        flow_best_val: Optional[float] = None
        if self.flow_config.enabled:
            flow_model_config = TrajectoryFlowMatchingModelConfig(
                state_dim=int(model_config.state_dim), condition_dim=int(model_config.d_model),
                trajectory_bars=int(model_config.trajectory_bars), d_model=int(self.flow_config.d_model),
                layers=int(self.flow_config.layers), n_heads=int(self.flow_config.n_heads),
                hidden_dim=int(self.flow_config.hidden_dim), dropout=float(self.flow_config.dropout),
            )
            flow_model, flow_history, flow_best_val = self._train_recurrent_flow_matcher(
                model, flow_model_config, train_windows, val_windows, normalizer, feedback, harmony_objective,
            )
        checkpoint_path = output_dir / "trajectory_diffusion.pt"
        checkpoint = self._build_trajectory_checkpoint(
            model_kind="recurrent_trajectory_diffusion",
            model=model,
            model_config=model_config,
            normalizer=normalizer,
            best_val_loss=best_val,
            flow_model=flow_model,
            flow_model_config=flow_model_config,
            flow_best_val=flow_best_val,
            recurrence_config=recurrence,
        )
        torch.save(checkpoint, checkpoint_path)
        diagnostics = {
            "backend": "recurrent_trajectory_diffusion", "model_path": str(checkpoint_path),
            "model_config": model_config.to_dict(), "recurrence_config": asdict(recurrence),
            "decoded_harmony_config": asdict(self.decoded_harmony_config),
            "training_config": asdict(self.training_config), "normalizer": normalizer.to_dict(),
            "tokenizer": token_payload.get("summary", {}), "latent_summary": latent_summary,
            "song_count": len(songs), "train_song_count": len(train_songs), "validation_song_count": len(val_songs),
            "recurrent_window_stride_bars": self._recurrent_window_stride(),
            "commits_per_window": int(recurrence.segments_per_update),
            "empty_remi_bar_count": int(sum(song.empty_remi_bar_count for song in songs)),
            "empty_remi_bar_ratio": float(
                sum(song.empty_remi_bar_count for song in songs) / max(1, sum(song.bar_count for song in songs))
            ),
            "empty_remi_bar_interpretation": (
                "Monitoring only: empty REMI bars still retain aligned latent and register streams "
                "and are not automatically invalid training examples."
            ),
            "train_window_count": len(train_windows), "validation_window_count": len(val_windows),
            "history": history, "best_val_loss": float(best_val), "prediction_type": "v",
            "checkpoint_selection_metric": "val_total_trajectory_loss",
            "flow_matching_config": asdict(self.flow_config), "flow_matching_history": flow_history,
            "flow_matching_checkpoint_selection_metric": "val_corrected_mse",
            "flow_matching_best_val_corrected_mse": flow_best_val,
        }
        (output_dir / "trajectory_diffusion_training_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return diagnostics

    def _build_recurrent_songs(
        self, mu: np.ndarray, rows: Sequence[Dict[str, Any]], token_by_key: Dict[str, List[int]],
        offsets: np.ndarray, song_anchors: np.ndarray, encoded_path: Path, model_config: TrajectoryDiffusionConfig,
    ) -> List[RecurrentSongSequence]:
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            if str(row.get("tensor_key", "")) in token_by_key:
                grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        archive = np.load(encoded_path / "bar_tensors.npz")
        songs: List[RecurrentSongSequence] = []
        for song_id, indices in grouped.items():
            indices = sorted(indices, key=lambda i: (int(rows[i].get("bar_index", i)), int(rows[i].get("row_index", i))))
            ids = np.full((len(indices), int(model_config.max_bar_tokens)), int(model_config.pad_token_id), dtype=np.int64)
            attention = np.zeros_like(ids)
            for local, index in enumerate(indices):
                values = np.asarray(token_by_key[str(rows[index]["tensor_key"])], dtype=np.int64)[: int(model_config.max_bar_tokens)]
                ids[local, :len(values)] = values
                attention[local, :len(values)] = 1
            positions = np.asarray([int(rows[index].get("bar_index", local)) for local, index in enumerate(indices)], dtype=np.int64)
            physical_chroma = np.stack([
                DecodedHarmonicTrajectoryObjective.source_physical_chroma(
                    archive[str(rows[index]["tensor_key"])],
                    float(song_anchors[index] + offsets[index]),
                    float(self.decoded_harmony_config.pitch_scale),
                )
                for index in indices
            ])
            songs.append(RecurrentSongSequence(
                song_id=song_id, input_ids=ids, attention_mask=attention, latents=mu[indices].astype(np.float32),
                register_offsets=offsets[indices].astype(np.float32), physical_chroma=physical_chroma.astype(np.float32), positions=positions,
                song_anchor=int(song_anchors[indices[0]]),
                empty_remi_bar_count=int(np.sum(attention.sum(axis=1) == 0)),
            ))
        if not songs:
            raise ValueError("No songs have aligned REMI token, latent, and register streams.")
        return songs

    def _split_recurrent_songs(self, songs: Sequence[RecurrentSongSequence]) -> tuple[List[RecurrentSongSequence], List[RecurrentSongSequence]]:
        rng = random.Random(int(self.training_config.random_seed))
        groups: Dict[str, List[RecurrentSongSequence]] = {}
        for song in songs:
            groups.setdefault(self._base_song_id(song.song_id), []).append(song)
        group_ids = sorted(groups)
        rng.shuffle(group_ids)
        count = max(1, int(round(len(group_ids) * float(self.training_config.validation_ratio))))
        validation = set(group_ids[:count])
        train = [song for group, values in groups.items() if group not in validation for song in values]
        val = [song for group, values in groups.items() if group in validation for song in values]
        if train and val:
            return train, val
        shuffled = list(songs)
        rng.shuffle(shuffled)
        return shuffled[1:], shuffled[:1]

    def _recurrent_windows(self, songs: Sequence[RecurrentSongSequence]) -> List[tuple[RecurrentSongSequence, int]]:
        commit = int(self.recurrence_config.commit_bars)
        plan = int(self.diffusion_config.trajectory_bars)
        required = int(self.recurrence_config.warmup_bars) + int(self.recurrence_config.segments_per_update) * commit + (plan - commit)
        # A window already executes every commit in this rollout span. Advancing
        # it by one committed bar would train the same generated-history path
        # roughly ``segments_per_update`` times per epoch.
        stride = self._recurrent_window_stride()
        windows: List[tuple[RecurrentSongSequence, int]] = []
        for song in songs:
            for start in range(0, song.bar_count - required + 1, stride):
                windows.append((song, start))
        return windows

    def _recurrent_window_stride(self) -> int:
        """Return the non-overlapping truncated-BPTT rollout span in bars."""
        return int(self.recurrence_config.commit_bars) * int(self.recurrence_config.segments_per_update)

    def _recurrent_plan_slice(self, window_start: int, block_index: int) -> slice:
        """Return the complete future-plan slice for one committed recurrent step."""
        start = (
            int(window_start)
            + int(self.recurrence_config.warmup_bars)
            + int(block_index) * int(self.recurrence_config.commit_bars)
        )
        return slice(start, start + int(self.diffusion_config.trajectory_bars))

    def _recurrent_batch(self, windows: Sequence[tuple[RecurrentSongSequence, int]]) -> Dict[str, torch.Tensor]:
        commit = int(self.recurrence_config.commit_bars)
        plan = int(self.diffusion_config.trajectory_bars)
        warmup = int(self.recurrence_config.warmup_bars)
        segments = int(self.recurrence_config.segments_per_update)
        prefix = warmup - commit
        input_ids = np.stack([song.input_ids[start:start + prefix] for song, start in windows])
        attention = np.stack([song.attention_mask[start:start + prefix] for song, start in windows])
        latents = np.stack([song.latents[start:start + prefix] for song, start in windows])
        registers = np.stack([song.register_offsets[start:start + prefix] for song, start in windows])
        positions = np.stack([song.positions[start:start + prefix] for song, start in windows])
        current_start = prefix
        current = {
            "input_ids": np.stack([song.input_ids[start + current_start:start + warmup] for song, start in windows]),
            "attention_mask": np.stack([song.attention_mask[start + current_start:start + warmup] for song, start in windows]),
            "latents": np.stack([song.latents[start + current_start:start + warmup] for song, start in windows]),
            "register_offsets": np.stack([song.register_offsets[start + current_start:start + warmup] for song, start in windows]),
            "positions": np.stack([song.positions[start + current_start:start + warmup] for song, start in windows]),
        }
        target = np.stack([
            np.stack([
                np.concatenate([
                    song.latents[self._recurrent_plan_slice(start, block)],
                    song.register_offsets[self._recurrent_plan_slice(start, block), None],
                ], axis=1)
                for block in range(segments)
            ]) for song, start in windows
        ])
        target_positions = np.stack([
            np.stack([song.positions[self._recurrent_plan_slice(start, block)] for block in range(segments)])
            for song, start in windows
        ])
        target_chromas = np.stack([
            np.stack([song.physical_chroma[self._recurrent_plan_slice(start, block)] for block in range(segments)])
            for song, start in windows
        ])
        boundary_chromas = np.stack([
            np.stack([song.physical_chroma[self._recurrent_plan_slice(start, block).start - 1] for block in range(segments)])
            for song, start in windows
        ])
        device = self.training_config.device
        batch = {
            "warm_ids": torch.from_numpy(input_ids).to(device), "warm_attention": torch.from_numpy(attention).to(device),
            "warm_latents": torch.from_numpy(latents).to(device), "warm_registers": torch.from_numpy(registers).to(device),
            "warm_positions": torch.from_numpy(positions).to(device),
            "current_ids": torch.from_numpy(current["input_ids"]).to(device), "current_attention": torch.from_numpy(current["attention_mask"]).to(device),
            "current_latents": torch.from_numpy(current["latents"]).to(device), "current_registers": torch.from_numpy(current["register_offsets"]).to(device),
            "current_positions": torch.from_numpy(current["positions"]).to(device),
            "targets": torch.from_numpy(target.astype(np.float32)).to(device), "target_positions": torch.from_numpy(target_positions).to(device),
            "target_chromas": torch.from_numpy(target_chromas.astype(np.float32)).to(device),
            "boundary_chromas": torch.from_numpy(boundary_chromas.astype(np.float32)).to(device),
            "anchors": torch.tensor([song.song_anchor for song, _ in windows], dtype=torch.float32, device=device),
        }
        self._validate_recurrent_batch(batch)
        return batch

    def _validate_recurrent_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Guard the named recurrent plan axes before they reach the model."""
        batch_size = int(batch["current_ids"].shape[0])
        commit = int(self.recurrence_config.commit_bars)
        plan = int(self.diffusion_config.trajectory_bars)
        steps = int(self.recurrence_config.segments_per_update)
        # The trained DVAE determines latent width at runtime. The style file
        # contains only a default, so it must not reject a compatible 48/64-D
        # latent dataset here.
        latent_dim = int(batch["current_latents"].shape[-1])
        state_dim = latent_dim + 1
        expected = {
            "current_latents": (batch_size, commit, latent_dim),
            "targets": (batch_size, steps, plan, state_dim),
            "target_positions": (batch_size, steps, plan),
            "target_chromas": (batch_size, steps, plan, 12),
            "boundary_chromas": (batch_size, steps, 12),
        }
        if tuple(batch["current_ids"].shape) != (batch_size, commit, int(self.diffusion_config.max_bar_tokens)):
            raise ValueError(
                f"Recurrent batch current_ids has shape {tuple(batch['current_ids'].shape)}, expected "
                f"{(batch_size, commit, int(self.diffusion_config.max_bar_tokens))}."
            )
        for name, shape in expected.items():
            if tuple(batch[name].shape) != shape:
                raise ValueError(
                    f"Recurrent batch {name} has shape {tuple(batch[name].shape)}, expected {shape}. "
                    "Check commit_bars, trajectory_bars, and the recurrent window schedule."
                )

    def _warm_recurrent_cache(self, model: RecurrentTrajectoryDiffusion, batch: Dict[str, torch.Tensor]) -> RecurrentBarKVCache:
        cache = model.empty_cache(int(batch["warm_ids"].shape[0]), batch["warm_ids"].device, model.denoiser.state_proj[0].weight.dtype)
        with torch.no_grad():
            # The bar Transformer is causal within a segment, so encoding the
            # whole real warmup prefix yields the same admissible history as
            # one-bar cache appends while avoiding 15 Python/model calls.
            _, cache = model.encode_segment(
                batch["warm_ids"], batch["warm_attention"], batch["warm_latents"], batch["warm_registers"],
                cache, batch["warm_positions"],
            )
        return cache.detach()

    def _commit_recurrent_plan(
        self,
        feedback: DetachedTrajectoryRolloutFeedback,
        normalized_plan: torch.Tensor,
        normalizer: TrajectoryStateNormalizer,
        anchors: torch.Tensor,
        target_positions: torch.Tensor,
        token_length: int,
        latent_dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Decode and feed back only the committed prefix of an unmasked plan."""
        committed = normalized_plan[:, :int(self.recurrence_config.commit_bars)].detach()
        payload = feedback.segment_payload(committed, normalizer, anchors, token_length, latent_dtype)
        return {
            "ids": payload["input_ids"],
            "attention": payload["attention_mask"],
            "latents": payload["latents"],
            "registers": payload["register_offsets"],
            "positions": target_positions[:, :int(self.recurrence_config.commit_bars)],
        }

    def _recurrent_diffusion_step(
        self,
        model: RecurrentTrajectoryDiffusion,
        current: Dict[str, torch.Tensor],
        cache: RecurrentBarKVCache,
        batch: Dict[str, torch.Tensor],
        block: int,
        normalizer: TrajectoryStateNormalizer,
        harmony_objective: DecodedHarmonicTrajectoryObjective,
    ) -> tuple[Dict[str, torch.Tensor], RecurrentBarKVCache, Dict[str, float]]:
        """Compute loss for one full future plan from one committed context bar."""
        condition, next_cache = model.encode_segment(
            current["ids"], current["attention"], current["latents"], current["registers"], cache, current["positions"],
        )
        target_raw = batch["targets"][:, block]
        result = model.diffusion_loss(condition, normalizer.normalize_tensor(target_raw))
        predicted_raw = normalizer.denormalize_tensor(result["predicted_clean"])
        harmony = harmony_objective(
            predicted_raw,
            batch["target_chromas"][:, block],
            batch["boundary_chromas"][:, block],
            batch["anchors"],
        )
        return {
            "condition": condition,
            "total_loss": result["loss"] + harmony["total_loss"],
            "rollin_plan": result["predicted_clean"],
        }, next_cache.detach(), self._diffusion_metrics(result, target_raw, normalizer, harmony)

    def _run_recurrent_diffusion_epoch(
        self, model: RecurrentTrajectoryDiffusion, windows: Sequence[tuple[RecurrentSongSequence, int]],
        optimizer: Optional[torch.optim.Optimizer], normalizer: TrajectoryStateNormalizer,
        feedback: DetachedTrajectoryRolloutFeedback, harmony_objective: DecodedHarmonicTrajectoryObjective, training: bool,
    ) -> Dict[str, float]:
        model.train(training)
        ordered = list(windows)
        if training:
            random.shuffle(ordered)
        metrics: List[Dict[str, float]] = []
        batch_size = int(self.training_config.batch_size)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for start in range(0, len(ordered), batch_size):
                batch = self._recurrent_batch(ordered[start:start + batch_size])
                cache = self._warm_recurrent_cache(model, batch)
                current = {key: batch[f"current_{key}"] for key in ("ids", "attention", "latents", "registers", "positions")}
                local: List[Dict[str, float]] = []
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                for block in range(int(self.recurrence_config.segments_per_update)):
                    step, next_cache, step_metrics = self._recurrent_diffusion_step(
                        model, current, cache, batch, block, normalizer, harmony_objective,
                    )
                    if optimizer is not None:
                        (step["total_loss"] / float(self.recurrence_config.segments_per_update)).backward()
                    local.append(step_metrics)
                    cache = next_cache
                    if block + 1 < int(self.recurrence_config.segments_per_update):
                        sampled = self._sample_diffusion_rollin(model, step["condition"])
                        current = self._commit_recurrent_plan(
                            feedback, sampled, normalizer, batch["anchors"], batch["target_positions"][:, block],
                            int(model.config.max_bar_tokens), current["latents"].dtype,
                        )
                if optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                    optimizer.step()
                metrics.append({**self._mean_metric_rows(local), "memory_bars": float(cache.valid_mask.shape[1])})
        return self._mean_metric_rows(metrics)

    def _train_recurrent_flow_matcher(
        self, diffusion: RecurrentTrajectoryDiffusion, flow_config: TrajectoryFlowMatchingModelConfig,
        train_windows: Sequence[tuple[RecurrentSongSequence, int]], val_windows: Sequence[tuple[RecurrentSongSequence, int]],
        normalizer: TrajectoryStateNormalizer, feedback: DetachedTrajectoryRolloutFeedback,
        harmony_objective: DecodedHarmonicTrajectoryObjective,
    ) -> tuple[TrajectoryFlowMatcher, List[Dict[str, float]], float]:
        diffusion.eval()
        for parameter in diffusion.parameters():
            parameter.requires_grad_(False)
        flow = TrajectoryFlowMatcher(flow_config).to(self.training_config.device)
        optimizer = torch.optim.AdamW(flow.parameters(), lr=float(self.flow_config.learning_rate), weight_decay=float(self.flow_config.weight_decay))
        history: List[Dict[str, float]] = []
        best = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        for epoch in range(1, int(self.flow_config.epochs) + 1):
            train = self._run_recurrent_flow_epoch(diffusion, flow, train_windows, optimizer, normalizer, feedback, harmony_objective, True)
            val = self._run_recurrent_flow_epoch(diffusion, flow, val_windows, None, normalizer, feedback, harmony_objective, False)
            history.append({"epoch": float(epoch), **{f"train_{key}": value for key, value in train.items()}, **{f"val_{key}": value for key, value in val.items()}})
            print(
                f"[recurrent flow] epoch {epoch}/{self.flow_config.epochs}: "
                f"train_loss={train['loss']:.6f}, val_corrected_mse={val['corrected_mse']:.6f}"
            )
            if float(val["corrected_mse"]) < best:
                best = float(val["corrected_mse"])
                best_state = {key: value.detach().cpu().clone() for key, value in flow.state_dict().items()}
        if best_state is not None:
            flow.load_state_dict(best_state)
        return flow, history, best

    def _recurrent_flow_step(
        self,
        diffusion: RecurrentTrajectoryDiffusion,
        flow: TrajectoryFlowMatcher,
        current: Dict[str, torch.Tensor],
        cache: RecurrentBarKVCache,
        batch: Dict[str, torch.Tensor],
        block: int,
        normalizer: TrajectoryStateNormalizer,
        harmony_objective: DecodedHarmonicTrajectoryObjective,
    ) -> tuple[Dict[str, torch.Tensor], RecurrentBarKVCache, Dict[str, float]]:
        """Correct one full plan while leaving only its first bar eligible for commit."""
        with torch.no_grad():
            condition, next_cache = diffusion.encode_segment(
                current["ids"], current["attention"], current["latents"], current["registers"], cache, current["positions"],
            )
            proposal = diffusion.sample(condition, int(self.flow_config.proposal_sampling_steps))
        target = normalizer.normalize_tensor(batch["targets"][:, block])
        result = flow.flow_matching_loss(proposal, target, condition)
        corrected = flow.correct(proposal, condition, int(self.flow_config.integration_steps))
        harmony = harmony_objective(
            normalizer.denormalize_tensor(corrected),
            batch["target_chromas"][:, block],
            batch["boundary_chromas"][:, block],
            batch["anchors"],
        )
        total_loss = result["loss"] + harmony["total_loss"]
        metrics = {
            "loss": float(total_loss.detach().cpu()),
            "velocity_mse": float(result["velocity_mse"].detach().cpu()),
            "proposal_mse": float(result["proposal_mse"].detach().cpu()),
            "corrected_mse": float(torch.nn.functional.mse_loss(corrected, target).detach().cpu()),
            "decoded_harmony_total_loss": float(harmony["total_loss"].detach().cpu()),
            "decoded_harmony_state_loss": float(harmony["state_loss"].detach().cpu()),
            "decoded_harmony_delta_loss": float(harmony["delta_loss"].detach().cpu()),
        }
        return {"total_loss": total_loss, "corrected_plan": corrected}, next_cache.detach(), metrics

    def _run_recurrent_flow_epoch(
        self, diffusion: RecurrentTrajectoryDiffusion, flow: TrajectoryFlowMatcher,
        windows: Sequence[tuple[RecurrentSongSequence, int]], optimizer: Optional[torch.optim.Optimizer],
        normalizer: TrajectoryStateNormalizer, feedback: DetachedTrajectoryRolloutFeedback,
        harmony_objective: DecodedHarmonicTrajectoryObjective, training: bool,
    ) -> Dict[str, float]:
        flow.train(training)
        ordered = list(windows)
        if training:
            random.shuffle(ordered)
        rows: List[Dict[str, float]] = []
        with torch.set_grad_enabled(training):
            for start in range(0, len(ordered), int(self.training_config.batch_size)):
                batch = self._recurrent_batch(ordered[start:start + int(self.training_config.batch_size)])
                cache = self._warm_recurrent_cache(diffusion, batch)
                current = {key: batch[f"current_{key}"] for key in ("ids", "attention", "latents", "registers", "positions")}
                local: List[Dict[str, float]] = []
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                for block in range(int(self.recurrence_config.segments_per_update)):
                    step, next_cache, step_metrics = self._recurrent_flow_step(
                        diffusion, flow, current, cache, batch, block, normalizer, harmony_objective,
                    )
                    if optimizer is not None:
                        (step["total_loss"] / float(self.recurrence_config.segments_per_update)).backward()
                    local.append(step_metrics)
                    cache = next_cache
                    if block + 1 < int(self.recurrence_config.segments_per_update):
                        current = self._commit_recurrent_plan(
                            feedback, step["corrected_plan"], normalizer, batch["anchors"], batch["target_positions"][:, block],
                            int(diffusion.config.max_bar_tokens), current["latents"].dtype,
                        )
                if optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), 3.0)
                    optimizer.step()
                rows.append(self._mean_metric_rows(local))
        return self._mean_metric_rows(rows)

    def _create_recurrent_feedback(self, model_path: Path, tokenizer: Any, pad_token_id: int, model_config: TrajectoryDiffusionConfig) -> DetachedTrajectoryRolloutFeedback:
        dvae_path = model_path / "dvae.pt"
        if not dvae_path.exists():
            raise FileNotFoundError(f"Recurrent training requires the frozen DVAE checkpoint: {dvae_path}")
        return DetachedTrajectoryRolloutFeedback(
            dvae=self._load_rollout_dvae(dvae_path), tokenizer=tokenizer, render_config=self._short_rollout_render_config(),
            pad_token_id=pad_token_id, register_offset_min=int(model_config.register_offset_min), register_offset_max=int(model_config.register_offset_max),
            base_pitch_min=int(self.short_rollout_config.base_pitch_min), base_pitch_max=int(self.short_rollout_config.base_pitch_max),
            device=self.training_config.device,
        )

    def _run_diffusion_epoch(
        self,
        model: JointTrajectoryDiffusion,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        normalizer: TrajectoryStateNormalizer,
        rollout_feedback: Optional[DetachedTrajectoryRolloutFeedback] = None,
    ) -> Dict[str, float]:
        if rollout_feedback is not None:
            return self._run_diffusion_rollout_epoch(model, loader, optimizer, normalizer, rollout_feedback)
        model.train()
        metrics: List[Dict[str, float]] = []
        for batch in loader:
            batch = {key: value.to(self.training_config.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            condition = model.encode_context(
                batch["context_input_ids"], batch["context_attention_mask"], batch["context_latents"],
                batch["context_register_offsets"], batch["context_bar_mask"],
            )
            target = normalizer.normalize_tensor(batch["target_trajectory"])
            result = model.diffusion_loss(condition, target)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            metrics.append(self._diffusion_metrics(result, batch["target_trajectory"], normalizer))
        return self._mean_metric_rows(metrics)

    def _evaluate_diffusion(
        self,
        model: JointTrajectoryDiffusion,
        loader: DataLoader,
        normalizer: TrajectoryStateNormalizer,
        rollout_feedback: Optional[DetachedTrajectoryRolloutFeedback] = None,
    ) -> Dict[str, float]:
        if rollout_feedback is not None:
            return self._evaluate_diffusion_rollout(model, loader, normalizer, rollout_feedback)
        model.eval()
        metrics: List[Dict[str, float]] = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.training_config.device) for key, value in batch.items()}
                condition = model.encode_context(
                    batch["context_input_ids"], batch["context_attention_mask"], batch["context_latents"],
                    batch["context_register_offsets"], batch["context_bar_mask"],
                )
                result = model.diffusion_loss(condition, normalizer.normalize_tensor(batch["target_trajectory"]))
                metrics.append(self._diffusion_metrics(result, batch["target_trajectory"], normalizer))
        return self._mean_metric_rows(metrics)

    def _run_diffusion_rollout_epoch(
        self,
        model: JointTrajectoryDiffusion,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        normalizer: TrajectoryStateNormalizer,
        feedback: DetachedTrajectoryRolloutFeedback,
    ) -> Dict[str, float]:
        """Train each local denoising loss under a detached generated history."""
        model.train()
        metrics: List[Dict[str, float]] = []
        for batch in loader:
            batch = {key: value.to(self.training_config.device) for key, value in batch.items()}
            context = self._context_from_batch(batch)
            losses: List[torch.Tensor] = []
            local_metrics: List[Dict[str, float]] = []
            for block_index in range(int(self.short_rollout_config.blocks)):
                condition = self._encode_context(model, context)
                target_raw = batch["rollout_trajectories"][:, block_index]
                result = model.diffusion_loss(condition, normalizer.normalize_tensor(target_raw))
                losses.append(result["loss"])
                local_metrics.append(self._diffusion_metrics(result, target_raw, normalizer))
                if block_index + 1 < int(self.short_rollout_config.blocks):
                    sampled = self._sample_diffusion_rollin(model, condition)
                    context = feedback.append(context, sampled, normalizer, batch["song_anchors"])
            optimizer.zero_grad(set_to_none=True)
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            metrics.append({
                **self._mean_metric_rows(local_metrics),
                "rollout_blocks": float(self.short_rollout_config.blocks),
            })
        return self._mean_metric_rows(metrics)

    def _evaluate_diffusion_rollout(
        self,
        model: JointTrajectoryDiffusion,
        loader: DataLoader,
        normalizer: TrajectoryStateNormalizer,
        feedback: DetachedTrajectoryRolloutFeedback,
    ) -> Dict[str, float]:
        model.eval()
        metrics: List[Dict[str, float]] = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.training_config.device) for key, value in batch.items()}
                context = self._context_from_batch(batch)
                local_metrics: List[Dict[str, float]] = []
                for block_index in range(int(self.short_rollout_config.blocks)):
                    condition = self._encode_context(model, context)
                    target_raw = batch["rollout_trajectories"][:, block_index]
                    result = model.diffusion_loss(condition, normalizer.normalize_tensor(target_raw))
                    local_metrics.append(self._diffusion_metrics(result, target_raw, normalizer))
                    if block_index + 1 < int(self.short_rollout_config.blocks):
                        sampled = self._sample_diffusion_rollin(model, condition)
                        context = feedback.append(context, sampled, normalizer, batch["song_anchors"])
                metrics.append({
                    **self._mean_metric_rows(local_metrics),
                    "rollout_blocks": float(self.short_rollout_config.blocks),
                })
        return self._mean_metric_rows(metrics)

    def _context_from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "context_input_ids": batch["context_input_ids"],
            "context_attention_mask": batch["context_attention_mask"],
            "context_latents": batch["context_latents"],
            "context_register_offsets": batch["context_register_offsets"],
            "context_bar_mask": batch["context_bar_mask"],
        }

    def _encode_context(self, model: JointTrajectoryDiffusion, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        return model.encode_context(
            context["context_input_ids"], context["context_attention_mask"], context["context_latents"],
            context["context_register_offsets"], context["context_bar_mask"],
        )

    def _sample_diffusion_rollin(self, model: JointTrajectoryDiffusion, condition: torch.Tensor) -> torch.Tensor:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            sampled = model.sample(condition.detach(), int(self.diffusion_config.sampling_steps))
        if was_training:
            model.train()
        return sampled.detach()

    def _train_flow_matcher(
        self,
        diffusion: JointTrajectoryDiffusion,
        flow_config: TrajectoryFlowMatchingModelConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        normalizer: TrajectoryStateNormalizer,
        rollout_feedback: Optional[DetachedTrajectoryRolloutFeedback] = None,
    ) -> tuple[TrajectoryFlowMatcher, List[Dict[str, float]], float]:
        """Fit a detached proposal-to-data transport after diffusion has converged."""
        diffusion.eval()
        for parameter in diffusion.parameters():
            parameter.requires_grad_(False)
        flow = TrajectoryFlowMatcher(flow_config).to(self.training_config.device)
        optimizer = torch.optim.AdamW(
            flow.parameters(),
            lr=float(self.flow_config.learning_rate),
            weight_decay=float(self.flow_config.weight_decay),
        )
        history: List[Dict[str, float]] = []
        best_val_corrected_mse = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        for epoch in range(1, int(self.flow_config.epochs) + 1):
            train_metrics = self._run_flow_epoch(diffusion, flow, train_loader, optimizer, normalizer, rollout_feedback)
            val_metrics = self._evaluate_flow_matcher(diffusion, flow, val_loader, normalizer, rollout_feedback)
            history.append({
                "epoch": float(epoch),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            })
            if float(val_metrics["corrected_mse"]) < best_val_corrected_mse:
                best_val_corrected_mse = float(val_metrics["corrected_mse"])
                best_state = {key: value.detach().cpu().clone() for key, value in flow.state_dict().items()}
        if best_state is not None:
            flow.load_state_dict(best_state)
        return flow, history, float(best_val_corrected_mse)

    def _run_flow_epoch(
        self,
        diffusion: JointTrajectoryDiffusion,
        flow: TrajectoryFlowMatcher,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        normalizer: TrajectoryStateNormalizer,
        rollout_feedback: Optional[DetachedTrajectoryRolloutFeedback] = None,
    ) -> Dict[str, float]:
        if rollout_feedback is not None:
            return self._run_flow_rollout_epoch(diffusion, flow, loader, optimizer, normalizer, rollout_feedback)
        flow.train()
        metrics: List[Dict[str, float]] = []
        for batch in loader:
            batch = {key: value.to(self.training_config.device) for key, value in batch.items()}
            proposal, condition = self._sample_frozen_proposal(diffusion, batch)
            target = normalizer.normalize_tensor(batch["target_trajectory"])
            optimizer.zero_grad(set_to_none=True)
            result = flow.flow_matching_loss(proposal, target, condition)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 3.0)
            optimizer.step()
            metrics.append({
                "loss": float(result["loss"].detach().cpu().item()),
                "velocity_mse": float(result["velocity_mse"].detach().cpu().item()),
                "proposal_mse": float(result["proposal_mse"].detach().cpu().item()),
            })
        return self._mean_metric_rows(metrics)

    def _evaluate_flow_matcher(
        self,
        diffusion: JointTrajectoryDiffusion,
        flow: TrajectoryFlowMatcher,
        loader: DataLoader,
        normalizer: TrajectoryStateNormalizer,
        rollout_feedback: Optional[DetachedTrajectoryRolloutFeedback] = None,
    ) -> Dict[str, float]:
        if rollout_feedback is not None:
            return self._evaluate_flow_rollout(diffusion, flow, loader, normalizer, rollout_feedback)
        flow.eval()
        metrics: List[Dict[str, float]] = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.training_config.device) for key, value in batch.items()}
                proposal, condition = self._sample_frozen_proposal(diffusion, batch)
                target = normalizer.normalize_tensor(batch["target_trajectory"])
                result = flow.flow_matching_loss(proposal, target, condition)
                corrected = flow.correct(proposal, condition, int(self.flow_config.integration_steps))
                metrics.append({
                    "loss": float(result["loss"].detach().cpu().item()),
                    "velocity_mse": float(result["velocity_mse"].detach().cpu().item()),
                    "proposal_mse": float(result["proposal_mse"].detach().cpu().item()),
                    "corrected_mse": float(torch.nn.functional.mse_loss(corrected, target).detach().cpu().item()),
                    "correction_norm": float(torch.linalg.vector_norm(corrected - proposal, dim=-1).mean().detach().cpu().item()),
                })
        return self._mean_metric_rows(metrics)

    def _run_flow_rollout_epoch(
        self,
        diffusion: JointTrajectoryDiffusion,
        flow: TrajectoryFlowMatcher,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        normalizer: TrajectoryStateNormalizer,
        feedback: DetachedTrajectoryRolloutFeedback,
    ) -> Dict[str, float]:
        flow.train()
        metrics: List[Dict[str, float]] = []
        for batch in loader:
            batch = {key: value.to(self.training_config.device) for key, value in batch.items()}
            context = self._context_from_batch(batch)
            losses: List[torch.Tensor] = []
            local_metrics: List[Dict[str, float]] = []
            for block_index in range(int(self.short_rollout_config.blocks)):
                proposal, condition = self._sample_frozen_proposal(diffusion, context)
                target = normalizer.normalize_tensor(batch["rollout_trajectories"][:, block_index])
                result = flow.flow_matching_loss(proposal, target, condition)
                losses.append(result["loss"])
                local_metrics.append({
                    "loss": float(result["loss"].detach().cpu().item()),
                    "velocity_mse": float(result["velocity_mse"].detach().cpu().item()),
                    "proposal_mse": float(result["proposal_mse"].detach().cpu().item()),
                })
                if block_index + 1 < int(self.short_rollout_config.blocks):
                    corrected = self._correct_flow_rollin(flow, proposal, condition)
                    context = feedback.append(context, corrected, normalizer, batch["song_anchors"])
            optimizer.zero_grad(set_to_none=True)
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 3.0)
            optimizer.step()
            metrics.append({
                **self._mean_metric_rows(local_metrics),
                "rollout_blocks": float(self.short_rollout_config.blocks),
            })
        return self._mean_metric_rows(metrics)

    def _evaluate_flow_rollout(
        self,
        diffusion: JointTrajectoryDiffusion,
        flow: TrajectoryFlowMatcher,
        loader: DataLoader,
        normalizer: TrajectoryStateNormalizer,
        feedback: DetachedTrajectoryRolloutFeedback,
    ) -> Dict[str, float]:
        flow.eval()
        metrics: List[Dict[str, float]] = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.training_config.device) for key, value in batch.items()}
                context = self._context_from_batch(batch)
                local_metrics: List[Dict[str, float]] = []
                for block_index in range(int(self.short_rollout_config.blocks)):
                    proposal, condition = self._sample_frozen_proposal(diffusion, context)
                    target = normalizer.normalize_tensor(batch["rollout_trajectories"][:, block_index])
                    result = flow.flow_matching_loss(proposal, target, condition)
                    corrected = self._correct_flow_rollin(flow, proposal, condition)
                    local_metrics.append({
                        "loss": float(result["loss"].detach().cpu().item()),
                        "velocity_mse": float(result["velocity_mse"].detach().cpu().item()),
                        "proposal_mse": float(result["proposal_mse"].detach().cpu().item()),
                        "corrected_mse": float(torch.nn.functional.mse_loss(corrected, target).detach().cpu().item()),
                        "correction_norm": float(torch.linalg.vector_norm(corrected - proposal, dim=-1).mean().detach().cpu().item()),
                    })
                    if block_index + 1 < int(self.short_rollout_config.blocks):
                        context = feedback.append(context, corrected, normalizer, batch["song_anchors"])
                metrics.append({
                    **self._mean_metric_rows(local_metrics),
                    "rollout_blocks": float(self.short_rollout_config.blocks),
                })
        return self._mean_metric_rows(metrics)

    def _correct_flow_rollin(
        self,
        flow: TrajectoryFlowMatcher,
        proposal: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        was_training = flow.training
        flow.eval()
        with torch.no_grad():
            corrected = flow.correct(proposal, condition, int(self.flow_config.integration_steps))
        if was_training:
            flow.train()
        return corrected.detach()

    def _sample_frozen_proposal(
        self,
        diffusion: JointTrajectoryDiffusion,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the exact detached diffusion proposal seen by the flow trainer."""
        with torch.no_grad():
            condition = diffusion.encode_context(
                batch["context_input_ids"], batch["context_attention_mask"], batch["context_latents"],
                batch["context_register_offsets"], batch["context_bar_mask"],
            )
            proposal = diffusion.sample(condition, int(self.flow_config.proposal_sampling_steps))
        return proposal.detach(), condition.detach()

    def _diffusion_metrics(
        self,
        result: Dict[str, torch.Tensor],
        target_raw: torch.Tensor,
        normalizer: TrajectoryStateNormalizer,
        harmony: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        clean = normalizer.denormalize_tensor(result["predicted_clean"].detach())
        latent_mse = torch.nn.functional.mse_loss(clean[..., :-1], target_raw[..., :-1])
        register_mae = torch.mean(torch.abs(clean[..., -1] - target_raw[..., -1]))
        output = {
            "loss": float(result["loss"].detach().cpu().item()),
            "velocity_mse": float(result["velocity_mse"].detach().cpu().item()),
            "recovered_latent_mse": float(latent_mse.detach().cpu().item()),
            "recovered_register_offset_mae": float(register_mae.detach().cpu().item()),
        }
        if harmony is not None:
            output["loss"] += float(harmony["total_loss"].detach().cpu().item())
            output["decoded_harmony_total_loss"] = float(harmony["total_loss"].detach().cpu().item())
            output["decoded_harmony_state_loss"] = float(harmony["state_loss"].detach().cpu().item())
            output["decoded_harmony_delta_loss"] = float(harmony["delta_loss"].detach().cpu().item())
        return output

    def _mean_metric_rows(self, metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
        keys = sorted({key for row in metrics for key in row})
        return {key: float(np.mean([row[key] for row in metrics])) for key in keys}

    def _split_trajectory_samples(self, samples: Sequence[TrajectorySample]) -> tuple[List[TrajectorySample], List[TrajectorySample]]:
        rng = random.Random(int(self.training_config.random_seed))
        if self.training_config.validation_split_unit == "sample":
            shuffled = list(samples)
            rng.shuffle(shuffled)
            count = max(1, int(round(len(shuffled) * float(self.training_config.validation_ratio))))
            return shuffled[count:], shuffled[:count]
        grouped: Dict[str, List[TrajectorySample]] = {}
        for sample in samples:
            grouped.setdefault(self._base_song_id(sample.song_id), []).append(sample)
        group_ids = sorted(grouped)
        rng.shuffle(group_ids)
        val_count = max(1, int(round(len(group_ids) * float(self.training_config.validation_ratio))))
        validation = set(group_ids[:val_count])
        train = [sample for key, values in grouped.items() for sample in values if key not in validation]
        val = [sample for key, values in grouped.items() for sample in values if key in validation]
        if train and val:
            return train, val
        shuffled = list(samples)
        rng.shuffle(shuffled)
        count = max(1, int(round(len(shuffled) * float(self.training_config.validation_ratio))))
        return shuffled[count:], shuffled[:count]

    def _training_config(self, overrides: Dict[str, Any]) -> TrajectoryDiffusionTrainingConfig:
        values = _apply_config_overrides(
            asdict(TrajectoryDiffusionTrainingConfig.from_config(self.config)),
            overrides,
            aliases={},
        )
        return TrajectoryDiffusionTrainingConfig(**values)

    def _diffusion_config(self, overrides: Dict[str, Any]) -> TrajectoryDiffusionConfig:
        section = ConfigView(self.config).section("trajectory_diffusion")
        values: Dict[str, Any] = {
            "vocab_size": int(section.get("vocab_size", self.motion_config.vocab_size)),
            "pad_token_id": 0,
            "latent_dim": int(section.get("latent_dim", self.motion_config.latent_dim)),
            "d_model": int(section.get("d_model", 256)),
            "token_layers": int(section.get("token_layers", 2)),
            "bar_layers": int(section.get("bar_layers", 2)),
            "denoiser_layers": int(section.get("denoiser_layers", 2)),
            "n_heads": int(section.get("n_heads", 4)),
            "dropout": float(section.get("dropout", 0.1)),
            "context_bars": int(section.get("context_bars", 16)),
            "memory_bars": int(section.get("memory_bars", 32)),
            "gradient_checkpointing": bool(section.get("gradient_checkpointing", True)),
            "max_bar_tokens": int(section.get("max_bar_tokens", self.motion_config.max_bar_tokens)),
            "trajectory_bars": int(section.get("trajectory_bars", 4)),
            "predictor_hidden_dim": int(section.get("predictor_hidden_dim", 512)),
            "context_pooling": str(section.get("context_pooling", "attention")),
            "diffusion_steps": int(section.get("diffusion_steps", 100)),
            "sampling_steps": int(section.get("sampling_steps", 16)),
            "beta_schedule": str(section.get("beta_schedule", "cosine")),
            "prediction_type": str(section.get("prediction_type", "v")),
            "register_offset_scale": float(section.get("register_offset_scale", self.motion_config.register_offset_scale)),
            "register_offset_min": int(section.get("register_offset_min", self.motion_config.register_offset_min)),
            "register_offset_max": int(section.get("register_offset_max", self.motion_config.register_offset_max)),
            "target_std_floor": float(section.get("target_std_floor", 1.0e-3)),
        }
        return _TrajectoryRuntimeConfig(**_apply_config_overrides(values, overrides, aliases={}))

    def _flow_config(self, overrides: Dict[str, Any]) -> TrajectoryFlowMatchingConfig:
        values = _apply_config_overrides(
            asdict(TrajectoryFlowMatchingConfig.from_config(self.config)),
            overrides,
            aliases={"flow_matching_enabled": "enabled"},
        )
        return TrajectoryFlowMatchingConfig(**values)

    def _short_rollout_config(self, overrides: Dict[str, Any]) -> TrajectoryShortRolloutConfig:
        values = _apply_config_overrides(
            asdict(TrajectoryShortRolloutConfig.from_config(self.config)),
            overrides,
            aliases={"short_rollout_enabled": "enabled"},
        )
        return TrajectoryShortRolloutConfig(**values)

    def _create_short_rollout_feedback(
        self,
        model_path: Path,
        tokenizer: Any,
        pad_token_id: int,
        model_config: TrajectoryDiffusionConfig,
    ) -> Optional[DetachedTrajectoryRolloutFeedback]:
        if not self.short_rollout_config.enabled:
            return None
        dvae_path = model_path / "dvae.pt"
        if not dvae_path.exists():
            raise FileNotFoundError(f"Short rollout requires the frozen DVAE checkpoint: {dvae_path}")
        dvae = self._load_rollout_dvae(dvae_path)
        return DetachedTrajectoryRolloutFeedback(
            dvae=dvae,
            tokenizer=tokenizer,
            render_config=self._short_rollout_render_config(),
            pad_token_id=pad_token_id,
            register_offset_min=int(model_config.register_offset_min),
            register_offset_max=int(model_config.register_offset_max),
            base_pitch_min=int(self.short_rollout_config.base_pitch_min),
            base_pitch_max=int(self.short_rollout_config.base_pitch_max),
            device=self.training_config.device,
        )

    def _load_rollout_dvae(self, path: Path) -> DenoisingMusicVAE:
        checkpoint = torch.load(path, map_location=self.training_config.device, weights_only=False)
        dvae = DenoisingMusicVAE(DVAEMusicConfig(**checkpoint["config"])).to(self.training_config.device)
        dvae.load_state_dict(checkpoint["state_dict"])
        dvae.eval()
        for parameter in dvae.parameters():
            parameter.requires_grad_(False)
        return dvae

    def _short_rollout_render_config(self) -> DVAEMidiRenderConfig:
        generation = ConfigView(self.config).section("remi_motion_generation")
        fallback = ConfigView(self.config).section("latent_generation")
        return DVAEMidiRenderConfig(
            tempo_bpm=int(generation.get("tempo_bpm", fallback.get("tempo_bpm", 100))),
            default_base_pitch=int(generation.get("base_pitch", fallback.get("base_pitch", 60))),
            audio_quality_enabled=False,
        )


@dataclass(frozen=True)
class _TrajectoryRuntimeConfig:
    """Pipeline configuration including one normalization-only setting."""

    vocab_size: int
    pad_token_id: int
    latent_dim: int
    d_model: int
    token_layers: int
    bar_layers: int
    denoiser_layers: int
    n_heads: int
    dropout: float
    context_bars: int
    max_bar_tokens: int
    trajectory_bars: int
    predictor_hidden_dim: int
    context_pooling: str
    diffusion_steps: int
    sampling_steps: int
    beta_schedule: str
    prediction_type: str
    register_offset_scale: float
    register_offset_min: int
    register_offset_max: int
    target_std_floor: float
    memory_bars: int
    gradient_checkpointing: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrajectoryDiffusionGenerationPipeline(RemiMotionGenerationPipeline):
    """Generate in four-bar diffusion blocks, then feed generated REMI bars back."""

    def __init__(self, config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config, overrides=overrides)
        self.overrides = overrides or {}
        values = _apply_config_overrides(asdict(TrajectoryDiffusionGenerationConfig.from_config(config)), self.overrides)
        self.diffusion_generation_config = TrajectoryDiffusionGenerationConfig(**values)
        self.flow_config = self._flow_config(self.overrides)

    def run(
        self,
        model_dir: str | Path,
        output_json: str | Path,
        output_midi: str | Path,
        checkpoint_path: Optional[str | Path] = None,
        dvae_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        self._set_seed()
        model_directory = Path(model_dir)
        output_dir = model_directory / "trajectory_diffusion"
        checkpoint_file = Path(checkpoint_path) if checkpoint_path else output_dir / "trajectory_diffusion.pt"
        checkpoint = torch.load(checkpoint_file, map_location=self.generation_config.device, weights_only=False)
        if str(checkpoint.get("prediction_type", "")).lower() != "v":
            raise ValueError(
                "This trajectory_diffusion checkpoint uses the retired epsilon-prediction format. "
                "Retrain the v-prediction model before generation."
            )
        model_config = TrajectoryDiffusionConfig(**checkpoint["model_config"])
        if checkpoint.get("model_kind") == "recurrent_trajectory_diffusion":
            return self._run_recurrent_generation(
                checkpoint, model_directory, output_dir, checkpoint_file, model_config,
                output_json, output_midi, dvae_path,
            )
        model = JointTrajectoryDiffusion(model_config).to(self.generation_config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        flow_matcher = self._load_flow_matcher(checkpoint, model_config)
        normalizer = TrajectoryStateNormalizer.from_dict(checkpoint["normalizer"])
        tokenizer_path = self._resolve_model_artifact_path(checkpoint.get("tokenizer_path"), output_dir, "tokenizer.json")
        tokenizer = RemiTokenizerFactory(RemiTokenizerSettings()).load(tokenizer_path)
        pad_token_id = int(tokenizer["PAD_None"])
        token_payload = self._load_token_cache_payload(self._resolve_model_artifact_path(checkpoint.get("token_cache_path"), output_dir, "remi_bar_tokens.json"))
        mu, rows, latent_summary = LatentDatasetReader().load(model_directory / "latent")
        base_pitch_lookup = self._base_pitch_lookup(model_directory / "encoded")
        grouped = self._filter_groups_with_tokens(self._group_rows(rows), rows, token_payload["token_by_key"])
        song_id = self._select_song_id(grouped, self.generation_config.seed_song_id)
        ordered = grouped[song_id]
        primer_count = min(int(self.generation_config.primer_bars), int(self.generation_config.bars), len(ordered))
        if primer_count < 1:
            raise ValueError("Need at least one primer bar for trajectory diffusion generation.")
        archive = np.load(model_directory / "encoded" / "bar_tensors.npz")
        generated_latents = [mu[index].astype(np.float32) for index in ordered[:primer_count]]
        generated_tensors = [archive[str(rows[index]["tensor_key"])].astype(np.float32) for index in ordered[:primer_count]]
        source_base_pitches = [self._row_base_pitch(rows[index], base_pitch_lookup, int(self.generation_config.base_pitch)) for index in ordered[:primer_count]]
        render_base_pitches = self._initial_render_base_pitches(source_base_pitches)
        song_anchor = self._generation_song_anchor(render_base_pitches)
        render_offsets = [int(value - song_anchor) for value in render_base_pitches]
        context_tokens = [list(token_payload["token_by_key"][str(rows[index]["tensor_key"])]) for index in ordered[:primer_count]]
        dvae = self._load_dvae(Path(dvae_path) if dvae_path else model_directory / "dvae.pt")
        token_cache = RemiBarTokenCache(output_dir, self._render_config(), RemiTokenizerSettings())
        temporary_tokens = Path(output_json).parent / ".trajectory_diffusion_generated_tokens"
        temporary_tokens.mkdir(parents=True, exist_ok=True)
        steps: List[Dict[str, Any]] = []
        while len(generated_latents) < int(self.generation_config.bars):
            arrays = self._aligned_context_arrays(
                context_tokens,
                generated_latents,
                render_offsets,
                context_bars=int(model_config.context_bars),
                max_bar_tokens=int(model_config.max_bar_tokens),
                pad_token_id=pad_token_id,
            )
            with torch.no_grad():
                condition = model.encode_context(
                    torch.from_numpy(arrays["context_input_ids"]).to(self.generation_config.device),
                    torch.from_numpy(arrays["context_attention_mask"]).to(self.generation_config.device),
                    torch.from_numpy(arrays["context_latents"]).to(self.generation_config.device),
                    torch.from_numpy(arrays["context_register_offsets"]).to(self.generation_config.device),
                    torch.from_numpy(arrays["context_bar_mask"]).to(self.generation_config.device),
                )
                sampled = model.sample(condition, int(self.diffusion_generation_config.sampling_steps))
                correction_norm = 0.0
                if flow_matcher is not None:
                    corrected = flow_matcher.correct(sampled, condition, int(self.flow_config.integration_steps))
                    correction_norm = float(torch.linalg.vector_norm(corrected - sampled, dim=-1).mean().item())
                    sampled = corrected
            raw_trajectory = normalizer.denormalize_numpy(sampled.detach().cpu().numpy()[0])
            remaining = int(self.generation_config.bars) - len(generated_latents)
            for future_index, state in enumerate(raw_trajectory[:remaining]):
                latent = state[:-1].astype(np.float32)
                raw_offset = float(state[-1])
                rounded_offset = int(np.rint(raw_offset))
                supported_offset = int(np.clip(
                    rounded_offset,
                    int(model_config.register_offset_min),
                    int(model_config.register_offset_max),
                ))
                base_pitch = self._clip_render_base_pitch(song_anchor + supported_offset)
                tensor = self._decode_tensors(dvae, latent.reshape(1, -1))[0]
                bar_index = len(generated_latents)
                if bool(self.generation_config.feedback_tokenization_enabled):
                    tokens = token_cache.tokenize_tensor_bar(
                        tokenizer,
                        tensor,
                        temporary_tokens / f"generated_bar_{bar_index:04d}.mid",
                        int(base_pitch),
                    )
                    feedback = "generated_bar_tokenization"
                else:
                    tokens = list(context_tokens[-1])
                    feedback = "disabled_reuse_last_context_bar_tokens"
                generated_latents.append(latent)
                generated_tensors.append(tensor)
                source_base_pitches.append(int(song_anchor))
                render_base_pitches.append(int(base_pitch))
                render_offsets.append(int(base_pitch - song_anchor))
                context_tokens.append(tokens)
                steps.append({
                    "bar_index": int(bar_index),
                    "trajectory_block_index": int((bar_index - primer_count) // int(model_config.trajectory_bars)),
                    "trajectory_position": int(future_index),
                    "latent_norm": float(np.linalg.norm(latent)),
                    "sampled_register_offset": float(raw_offset),
                    "rounded_register_offset": int(rounded_offset),
                    "render_register_offset": int(base_pitch - song_anchor),
                    "render_base_pitch": int(base_pitch),
                    "feedback_mode": feedback,
                    "generated_token_count": int(len(tokens)),
                    "flow_matching_enabled": flow_matcher is not None,
                    "flow_correction_norm": float(correction_norm),
                })
                if len(generated_latents) >= int(self.generation_config.bars):
                    break
        tensor_array = np.stack(generated_tensors).astype(np.float32)
        latent_array = np.stack(generated_latents).astype(np.float32)
        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_path = output_json_path.with_suffix(".bar_tensors.npz")
        np.savez_compressed(
            tensor_path,
            bars=tensor_array,
            latent_mu=latent_array,
            source_base_pitches=np.asarray(source_base_pitches, dtype=np.int64),
            song_anchor=np.asarray([song_anchor], dtype=np.int64),
            render_register_offsets=np.asarray(render_offsets, dtype=np.int64),
            render_base_pitches=np.asarray(render_base_pitches, dtype=np.int64),
        )
        midi = SequenceTensorMidiRenderer(self._render_config()).render(
            tensor_array, output_midi, base_pitch=int(self.generation_config.base_pitch), base_pitches=render_base_pitches
        )
        diagnostics = {
            "backend": "joint_trajectory_diffusion",
            "model_dir": str(model_directory),
            "checkpoint": str(checkpoint_file),
            "dvae_checkpoint": str(Path(dvae_path) if dvae_path else model_directory / "dvae.pt"),
            "model_config": model_config.to_dict(),
            "generation_config": asdict(self.generation_config),
            "diffusion_generation_config": asdict(self.diffusion_generation_config),
            "flow_matching_config": asdict(self.flow_config),
            "flow_matching_enabled": flow_matcher is not None,
            "source_song_id": song_id,
            "primer_bars": int(primer_count),
            "generated_bars": int(len(generated_tensors)),
            "song_anchor": int(song_anchor),
            "source_base_pitches": [int(value) for value in source_base_pitches],
            "render_register_offsets": [int(value) for value in render_offsets],
            "render_base_pitches": [int(value) for value in render_base_pitches],
            "feedback_tokenization_enabled": bool(self.generation_config.feedback_tokenization_enabled),
            "latent_summary": latent_summary,
            "steps": steps,
            "tensor_path": str(tensor_path),
            "midi": midi,
        }
        output_json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return diagnostics

    def _run_recurrent_generation(
        self,
        checkpoint: Dict[str, Any],
        model_directory: Path,
        output_dir: Path,
        checkpoint_file: Path,
        model_config: TrajectoryDiffusionConfig,
        output_json: str | Path,
        output_midi: str | Path,
        dvae_path: Optional[str | Path],
    ) -> Dict[str, Any]:
        """Generate with a four-bar plan and one-bar receding-horizon commits."""
        recurrence = TrajectoryRecurrenceConfig(**checkpoint.get("recurrence_config", {}))
        recurrence.validate(model_config)
        commit = int(recurrence.commit_bars)
        plan = int(model_config.trajectory_bars)
        model = RecurrentTrajectoryDiffusion(model_config).to(self.generation_config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        flow_matcher = self._load_flow_matcher(checkpoint, model_config)
        normalizer = TrajectoryStateNormalizer.from_dict(checkpoint["normalizer"])
        tokenizer_path = self._resolve_model_artifact_path(checkpoint.get("tokenizer_path"), output_dir, "tokenizer.json")
        tokenizer = RemiTokenizerFactory(RemiTokenizerSettings()).load(tokenizer_path)
        pad_token_id = int(tokenizer["PAD_None"])
        token_payload = self._load_token_cache_payload(self._resolve_model_artifact_path(checkpoint.get("token_cache_path"), output_dir, "remi_bar_tokens.json"))
        mu, rows, latent_summary = LatentDatasetReader().load(model_directory / "latent")
        base_pitch_lookup = self._base_pitch_lookup(model_directory / "encoded")
        grouped = self._filter_groups_with_tokens(self._group_rows(rows), rows, token_payload["token_by_key"])
        song_id = self._select_song_id(grouped, self.generation_config.seed_song_id)
        ordered = grouped[song_id]
        if int(self.generation_config.bars) < commit:
            raise ValueError("Recurrent trajectory generation requires bars >= trajectory_recurrence.commit_bars.")
        if not bool(self.generation_config.feedback_tokenization_enabled):
            raise ValueError("Recurrent trajectory generation requires generated REMI feedback tokenization.")
        requested_primer = min(int(self.generation_config.primer_bars), int(self.generation_config.bars), len(ordered))
        primer_count = (requested_primer // commit) * commit
        if primer_count < commit:
            if len(ordered) < commit:
                raise ValueError("The selected song has fewer than one complete recurrent primer bar.")
            primer_count = commit
        archive = np.load(model_directory / "encoded" / "bar_tensors.npz")
        generated_latents = [mu[index].astype(np.float32) for index in ordered[:primer_count]]
        generated_tensors = [archive[str(rows[index]["tensor_key"])].astype(np.float32) for index in ordered[:primer_count]]
        source_base_pitches = [self._row_base_pitch(rows[index], base_pitch_lookup, int(self.generation_config.base_pitch)) for index in ordered[:primer_count]]
        render_base_pitches = self._initial_render_base_pitches(source_base_pitches)
        song_anchor = self._generation_song_anchor(render_base_pitches)
        render_offsets = [int(value - song_anchor) for value in render_base_pitches]
        context_tokens = [list(token_payload["token_by_key"][str(rows[index]["tensor_key"])]) for index in ordered[:primer_count]]
        source_positions = [int(rows[index].get("bar_index", local)) for local, index in enumerate(ordered[:primer_count])]
        dvae = self._load_dvae(Path(dvae_path) if dvae_path else model_directory / "dvae.pt")
        feedback = DetachedTrajectoryRolloutFeedback(
            dvae=dvae, tokenizer=tokenizer, render_config=self._render_config(), pad_token_id=pad_token_id,
            register_offset_min=int(model_config.register_offset_min), register_offset_max=int(model_config.register_offset_max),
            base_pitch_min=int(self._clip_render_base_pitch(-10_000)), base_pitch_max=int(self._clip_render_base_pitch(10_000)),
            device=self.generation_config.device,
        )
        current_start = primer_count - commit
        current = self._generation_segment_payload(
            context_tokens[current_start:], generated_latents[current_start:], render_offsets[current_start:], source_positions[current_start:],
            int(model_config.max_bar_tokens), pad_token_id,
        )
        cache = model.empty_cache(1, torch.device(self.generation_config.device), model.denoiser.state_proj[0].weight.dtype)
        with torch.no_grad():
            for start in range(0, current_start, commit):
                warm = self._generation_segment_payload(
                    context_tokens[start:start + commit], generated_latents[start:start + commit], render_offsets[start:start + commit], source_positions[start:start + commit],
                    int(model_config.max_bar_tokens), pad_token_id,
                )
                _, cache = model.encode_segment(warm["ids"], warm["attention"], warm["latents"], warm["registers"], cache, warm["positions"])
                cache = cache.detach()
        steps: List[Dict[str, Any]] = []
        while len(generated_latents) < int(self.generation_config.bars):
            with torch.no_grad():
                condition, cache_after_current = model.encode_segment(
                    current["ids"], current["attention"], current["latents"], current["registers"], cache, current["positions"],
                )
                sampled = model.sample(condition, int(self.diffusion_generation_config.sampling_steps))
                correction_norm = 0.0
                if flow_matcher is not None:
                    corrected = flow_matcher.correct(sampled, condition, int(self.flow_config.integration_steps))
                    correction_norm = float(torch.linalg.vector_norm(corrected - sampled, dim=-1).mean().item())
                    sampled = corrected
                raw = normalizer.denormalize_tensor(sampled).detach()
                payload = feedback.raw_segment_payload(
                    raw[:, :commit], torch.tensor([song_anchor], device=raw.device, dtype=torch.float32),
                    int(model_config.max_bar_tokens), current["latents"].dtype,
                )
            raw_np = raw.detach().cpu().numpy()[0]
            remaining = int(self.generation_config.bars) - len(generated_latents)
            commit_count = min(commit, remaining)
            generated_positions = torch.arange(
                int(current["positions"][0, -1].item()) + 1,
                int(current["positions"][0, -1].item()) + 1 + commit,
                device=current["positions"].device,
                dtype=current["positions"].dtype,
            ).unsqueeze(0)
            for local, state in enumerate(raw_np[:commit_count]):
                latent = state[:-1].astype(np.float32)
                supported_offset = int(np.clip(int(np.rint(float(state[-1]))), int(model_config.register_offset_min), int(model_config.register_offset_max)))
                base_pitch = self._clip_render_base_pitch(song_anchor + supported_offset)
                generated_latents.append(latent)
                generated_tensors.append(payload["tensors"][0, local].detach().cpu().numpy().astype(np.float32))
                source_base_pitches.append(int(song_anchor))
                render_base_pitches.append(int(base_pitch))
                render_offsets.append(int(base_pitch - song_anchor))
                steps.append({
                    "bar_index": len(generated_latents) - 1, "trajectory_position": local,
                    "planner_bars": plan, "committed_bars": commit,
                    "memory_valid_bars": int(cache_after_current.valid_mask.shape[1]), "latent_norm": float(np.linalg.norm(latent)),
                    "sampled_register_offset": float(state[-1]), "render_register_offset": int(base_pitch - song_anchor),
                    "render_base_pitch": int(base_pitch), "generated_token_count": int(torch.sum(payload["attention_mask"][0, local]).item()),
                    "flow_matching_enabled": flow_matcher is not None, "flow_correction_norm": correction_norm,
                })
            cache = cache_after_current.detach()
            current = {"ids": payload["input_ids"], "attention": payload["attention_mask"], "latents": payload["latents"], "registers": payload["register_offsets"], "positions": generated_positions}
        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_path = output_json_path.with_suffix(".bar_tensors.npz")
        tensor_array = np.stack(generated_tensors).astype(np.float32)
        latent_array = np.stack(generated_latents).astype(np.float32)
        np.savez_compressed(tensor_path, bars=tensor_array, latent_mu=latent_array, source_base_pitches=np.asarray(source_base_pitches, dtype=np.int64),
                            song_anchor=np.asarray([song_anchor], dtype=np.int64), render_register_offsets=np.asarray(render_offsets, dtype=np.int64), render_base_pitches=np.asarray(render_base_pitches, dtype=np.int64))
        midi = SequenceTensorMidiRenderer(self._render_config()).render(tensor_array, output_midi, base_pitch=int(self.generation_config.base_pitch), base_pitches=render_base_pitches)
        diagnostics = {
            "backend": "recurrent_trajectory_diffusion", "model_dir": str(model_directory), "checkpoint": str(checkpoint_file),
            "dvae_checkpoint": str(Path(dvae_path) if dvae_path else model_directory / "dvae.pt"), "model_config": model_config.to_dict(),
            "recurrence_config": asdict(recurrence), "generation_config": asdict(self.generation_config),
            "diffusion_generation_config": asdict(self.diffusion_generation_config), "flow_matching_enabled": flow_matcher is not None,
            "source_song_id": song_id, "primer_bars_requested": requested_primer, "primer_bars": primer_count,
            "generated_bars": len(generated_tensors), "song_anchor": song_anchor, "render_register_offsets": render_offsets,
            "planner_bars": plan, "committed_bars": commit,
            "render_base_pitches": render_base_pitches, "feedback_tokenization_enabled": True, "latent_summary": latent_summary,
            "steps": steps, "tensor_path": str(tensor_path), "midi": midi,
        }
        output_json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return diagnostics

    def _generation_segment_payload(
        self, token_rows: Sequence[Sequence[int]], latents: Sequence[np.ndarray], offsets: Sequence[int], positions: Sequence[int], token_length: int, pad_token_id: int,
    ) -> Dict[str, torch.Tensor]:
        ids = np.full((1, len(token_rows), token_length), int(pad_token_id), dtype=np.int64)
        attention = np.zeros_like(ids)
        for index, values in enumerate(token_rows):
            length = min(token_length, len(values))
            ids[0, index, :length] = np.asarray(values[:length], dtype=np.int64)
            attention[0, index, :length] = 1
        device = self.generation_config.device
        return {
            "ids": torch.from_numpy(ids).to(device), "attention": torch.from_numpy(attention).to(device),
            "latents": torch.from_numpy(np.asarray(latents, dtype=np.float32)[None]).to(device),
            "registers": torch.from_numpy(np.asarray(offsets, dtype=np.float32)[None]).to(device),
            "positions": torch.from_numpy(np.asarray(positions, dtype=np.int64)[None]).to(device),
        }

    def _load_flow_matcher(
        self,
        checkpoint: Dict[str, Any],
        diffusion_config: TrajectoryDiffusionConfig,
    ) -> Optional[TrajectoryFlowMatcher]:
        if not self.flow_config.enabled:
            return None
        payload = checkpoint.get("flow_matching_model_config")
        state_dict = checkpoint.get("flow_matching_state_dict")
        if payload is None or state_dict is None:
            raise ValueError(
                "trajectory_flow_matching.enabled=true requires a trajectory diffusion checkpoint trained with flow matching."
            )
        model_config = TrajectoryFlowMatchingModelConfig(**payload)
        if int(model_config.state_dim) != int(diffusion_config.state_dim):
            raise ValueError("Flow matching checkpoint state_dim does not match the diffusion checkpoint.")
        if int(model_config.condition_dim) != int(diffusion_config.d_model):
            raise ValueError("Flow matching checkpoint condition_dim does not match the diffusion checkpoint.")
        matcher = TrajectoryFlowMatcher(model_config).to(self.generation_config.device)
        matcher.load_state_dict(state_dict)
        matcher.eval()
        return matcher

    def _flow_config(self, overrides: Dict[str, Any]) -> TrajectoryFlowMatchingConfig:
        values = _apply_config_overrides(
            asdict(TrajectoryFlowMatchingConfig.from_config(self.config)),
            overrides,
            aliases={"flow_matching_enabled": "enabled"},
        )
        return TrajectoryFlowMatchingConfig(**values)
