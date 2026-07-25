#!/usr/bin/env python3
"""Differentiable physical-harmony supervision for latent trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from model.dvae import DenoisingMusicVAE


@dataclass(frozen=True)
class DecodedHarmonicTrajectoryConfig:
    """Loss configuration for decoded state and harmonic movement."""

    enabled: bool = True
    state_loss_weight: float = 1.0
    delta_loss_weight: float = 1.0
    pitch_scale: float = 24.0
    pitch_class_sigma: float = 0.35


class DecodedHarmonicTrajectoryObjective:
    """Compare a predicted latent plan after decoding it into physical Chroma."""

    def __init__(self, dvae: DenoisingMusicVAE, config: DecodedHarmonicTrajectoryConfig) -> None:
        self.dvae = dvae
        self.config = config
        self.dvae.decoder.eval()
        for parameter in self.dvae.decoder.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def source_physical_chroma(bar_tensor: np.ndarray, base_pitch: float, pitch_scale: float = 24.0) -> np.ndarray:
        """Return normalized 12-bin absolute physical Chroma for one source bar."""
        tensor = np.asarray(bar_tensor, dtype=np.float32)
        states = np.argmax(tensor[..., 1:4], axis=-1)
        active = states != 0
        chroma = np.zeros(12, dtype=np.float32)
        if not np.any(active):
            return chroma
        pitches = np.rint(tensor[..., 0][active] * float(pitch_scale) + float(base_pitch)).astype(np.int64)
        np.add.at(chroma, np.remainder(pitches, 12), 1.0)
        return chroma / max(float(chroma.sum()), 1.0)

    def __call__(
        self,
        predicted_raw: Tensor,
        target_chroma: Tensor,
        boundary_chroma: Tensor,
        song_anchors: Tensor,
    ) -> Dict[str, Tensor]:
        """Return decoded Chroma state and transition losses for one future plan."""
        if not self.config.enabled:
            zero = predicted_raw.new_zeros(())
            return {"total_loss": zero, "state_loss": zero, "delta_loss": zero}
        decoded = self._decoded_physical_chroma(predicted_raw, song_anchors)
        target = target_chroma.to(device=decoded.device, dtype=decoded.dtype)
        boundary = boundary_chroma.to(device=decoded.device, dtype=decoded.dtype)
        state_loss = self._state_loss(decoded, target)
        delta_loss = self._delta_loss(decoded, target, boundary)
        return {
            "total_loss": float(self.config.state_loss_weight) * state_loss + float(self.config.delta_loss_weight) * delta_loss,
            "state_loss": state_loss,
            "delta_loss": delta_loss,
        }

    def _decoded_physical_chroma(self, raw: Tensor, song_anchors: Tensor) -> Tensor:
        batch_size, plan_bars, _ = raw.shape
        latent = raw[..., :-1].reshape(batch_size * plan_bars, -1)
        pitch, state_logits, _velocity, _chord = self.dvae.decoder(latent)
        tracks = int(self.dvae.config.tracks)
        slots = int(self.dvae.config.steps_per_bar)
        pitch = pitch.reshape(batch_size, plan_bars, tracks, slots)
        state_logits = state_logits.reshape(batch_size, plan_bars, tracks, slots, 3)
        active = torch.softmax(state_logits, dim=-1)[..., 1:].sum(dim=-1)
        offsets = raw[..., -1].reshape(batch_size, plan_bars, 1, 1)
        anchors = song_anchors.reshape(batch_size, 1, 1, 1).to(dtype=pitch.dtype)
        semitones = pitch * float(self.config.pitch_scale) + anchors + offsets
        pitch_classes = torch.arange(12, device=pitch.device, dtype=pitch.dtype)
        distance = torch.remainder(semitones.unsqueeze(-1) - pitch_classes + 6.0, 12.0) - 6.0
        membership = torch.softmax(-0.5 * (distance / float(self.config.pitch_class_sigma)).square(), dim=-1)
        chroma = torch.sum(membership * active.unsqueeze(-1), dim=(2, 3))
        return chroma / chroma.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _state_loss(self, decoded: Tensor, target: Tensor) -> Tensor:
        active = target.sum(dim=-1) > 0.0
        per_bar = F.kl_div(torch.log(decoded.clamp_min(1.0e-8)), target, reduction="none").sum(dim=-1)
        weights = active.to(dtype=per_bar.dtype)
        return torch.sum(per_bar * weights) / weights.sum().clamp_min(1.0)

    def _delta_loss(self, decoded: Tensor, target: Tensor, boundary: Tensor) -> Tensor:
        target_previous = torch.cat([boundary.unsqueeze(1), target[:, :-1]], dim=1)
        decoded_previous = torch.cat([boundary.unsqueeze(1), decoded[:, :-1]], dim=1)
        target_delta = target - target_previous
        decoded_delta = decoded - decoded_previous
        target_energy = target_delta.square().sum(dim=-1)
        active = (target.sum(dim=-1) > 0.0) & (target_previous.sum(dim=-1) > 0.0)
        moving = active & (target_energy > 1.0e-8)
        relative_error = (decoded_delta - target_delta).square().sum(dim=-1) / target_energy.clamp_min(1.0e-8)
        weights = moving.to(dtype=relative_error.dtype)
        return torch.sum(relative_error * weights) / weights.sum().clamp_min(1.0)
