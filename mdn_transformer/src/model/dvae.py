#!/usr/bin/env python3
"""Denoising variational autoencoder for [3, 16, 16] bar tensors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from common.config_loader import ConfigView


@dataclass(frozen=True)
class DVAEMusicConfig:
    """Configuration for the tiny denoising music VAE."""

    tracks: int = 3
    steps_per_bar: int = 16
    feature_dim: int = 16
    latent_dim: int = 32
    dropout: float = 0.3
    beta_kl: float = 0.1
    pitch_loss_weight: float = 1.0
    state_loss_weight: float = 1.0
    velocity_loss_weight: float = 0.5
    chord_loss_weight: float = 0.25
    physical_chroma_loss_weight: float = 1.0
    chroma_delta_loss_weight: float = 1.0
    physical_chroma_pitch_scale: float = 24.0
    physical_chroma_sigma: float = 0.35
    note_drop_prob: float = 0.15
    sustain_fill_prob: float = 0.10
    drop_to_rest_prob: float = 0.5
    ornament_pitch_noise_std: float = 0.05
    velocity_noise_std: float = 0.01
    weight_decay: float = 1.0e-4

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DVAEMusicConfig":
        """Build DVAE config from the style configuration."""
        section = ConfigView(config).section("dvae")
        bar_tensor = ConfigView(config).section("bar_tensor")
        return cls(
            tracks=int(section.get("tracks", 3)),
            steps_per_bar=int(section.get("steps_per_bar", 16)),
            feature_dim=int(section.get("feature_dim", 16)),
            latent_dim=int(section.get("latent_dim", 32)),
            dropout=float(section.get("dropout", 0.3)),
            beta_kl=float(section.get("beta_kl", 0.1)),
            pitch_loss_weight=float(section.get("pitch_loss_weight", 1.0)),
            state_loss_weight=float(section.get("state_loss_weight", 1.0)),
            velocity_loss_weight=float(section.get("velocity_loss_weight", 0.5)),
            chord_loss_weight=float(section.get("chord_loss_weight", 0.25)),
            physical_chroma_loss_weight=float(section.get("physical_chroma_loss_weight", 1.0)),
            chroma_delta_loss_weight=float(section.get("chroma_delta_loss_weight", 1.0)),
            physical_chroma_pitch_scale=float(section.get("physical_chroma_pitch_scale", bar_tensor.get("pitch_scale", 24.0))),
            physical_chroma_sigma=float(section.get("physical_chroma_sigma", 0.35)),
            note_drop_prob=float(section.get("note_drop_prob", 0.15)),
            sustain_fill_prob=float(section.get("sustain_fill_prob", 0.10)),
            drop_to_rest_prob=float(section.get("drop_to_rest_prob", 0.5)),
            ornament_pitch_noise_std=float(section.get("ornament_pitch_noise_std", 0.05)),
            velocity_noise_std=float(section.get("velocity_noise_std", 0.01)),
            weight_decay=float(section.get("weight_decay", 1.0e-4)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe config values."""
        return asdict(self)


@dataclass
class DVAEOutput:
    """Forward output of the denoising music VAE."""

    pitch: Tensor
    state_logits: Tensor
    velocity: Tensor
    chord: Tensor
    mu: Tensor
    log_var: Tensor
    z: Tensor


class VariationNoiseInjector(nn.Module):
    """Inject lightweight variation noise into bar tensors during training."""

    REST_INDEX = 1
    NOTE_ON_INDEX = 2
    HOLD_INDEX = 3

    def __init__(self, config: DVAEMusicConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, clean: Tensor) -> Tensor:
        """Return a noisy tensor copy while preserving the target tensor."""
        if not self.training:
            return clean
        noisy = clean.clone()
        self._drop_note_on_slots(noisy)
        self._fill_hold_slots(noisy, clean)
        self._jitter_velocity(noisy)
        return noisy

    def _drop_note_on_slots(self, tensor: Tensor) -> None:
        """Randomly replace note-on slots with rest or hold states."""
        note_mask = tensor[..., self.NOTE_ON_INDEX] > 0.5
        drop_mask = note_mask & (torch.rand_like(tensor[..., self.NOTE_ON_INDEX]) < self.config.note_drop_prob)
        if not bool(drop_mask.any()):
            return
        to_rest = torch.rand_like(tensor[..., self.NOTE_ON_INDEX]) < self.config.drop_to_rest_prob
        rest_mask = drop_mask & to_rest
        hold_mask = drop_mask & ~to_rest
        tensor[..., 0] = torch.where(drop_mask, torch.zeros_like(tensor[..., 0]), tensor[..., 0])
        tensor[..., 4] = torch.where(drop_mask, torch.zeros_like(tensor[..., 4]), tensor[..., 4])
        tensor[..., self.REST_INDEX] = torch.where(rest_mask, torch.ones_like(tensor[..., self.REST_INDEX]), tensor[..., self.REST_INDEX])
        tensor[..., self.NOTE_ON_INDEX] = torch.where(drop_mask, torch.zeros_like(tensor[..., self.NOTE_ON_INDEX]), tensor[..., self.NOTE_ON_INDEX])
        tensor[..., self.HOLD_INDEX] = torch.where(hold_mask, torch.ones_like(tensor[..., self.HOLD_INDEX]), tensor[..., self.HOLD_INDEX])

    def _fill_hold_slots(self, noisy: Tensor, clean: Tensor) -> None:
        """Replace some hold slots with nearby ornamental note-on values."""
        hold_mask = clean[..., self.HOLD_INDEX] > 0.5
        fill_mask = hold_mask & (torch.rand_like(clean[..., self.HOLD_INDEX]) < self.config.sustain_fill_prob)
        if not bool(fill_mask.any()):
            return
        pitch_noise = torch.randn_like(noisy[..., 0]) * float(self.config.ornament_pitch_noise_std)
        noisy[..., 0] = torch.where(fill_mask, clean[..., 0] + pitch_noise, noisy[..., 0])
        noisy[..., self.REST_INDEX] = torch.where(fill_mask, torch.zeros_like(noisy[..., self.REST_INDEX]), noisy[..., self.REST_INDEX])
        noisy[..., self.NOTE_ON_INDEX] = torch.where(fill_mask, torch.ones_like(noisy[..., self.NOTE_ON_INDEX]), noisy[..., self.NOTE_ON_INDEX])
        noisy[..., self.HOLD_INDEX] = torch.where(fill_mask, torch.zeros_like(noisy[..., self.HOLD_INDEX]), noisy[..., self.HOLD_INDEX])

    def _jitter_velocity(self, tensor: Tensor) -> None:
        """Add tiny Gaussian noise to velocity for active slots."""
        active = (tensor[..., self.NOTE_ON_INDEX] > 0.5) | (tensor[..., self.HOLD_INDEX] > 0.5)
        noise = torch.randn_like(tensor[..., 4]) * float(self.config.velocity_noise_std)
        tensor[..., 4] = torch.where(active, torch.clamp(tensor[..., 4] + noise, 0.0, 1.0), tensor[..., 4])


class TinyMusicEncoder(nn.Module):
    """Asymmetric convolutional encoder for compact music tensors."""

    def __init__(self, config: DVAEMusicConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=config.feature_dim, out_channels=16, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), padding=(0, 0)),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.flatten_dim = 32 * 1 * (config.steps_per_bar // 2)
        self.drop = nn.Dropout(float(config.dropout))
        self.fc_mu = nn.Linear(self.flatten_dim, int(config.latent_dim))
        self.fc_log_var = nn.Linear(self.flatten_dim, int(config.latent_dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a batch into latent Gaussian parameters."""
        self._validate_input(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        features = self.feature_extractor(x)
        flat = features.reshape(features.shape[0], -1)
        dropped = self.drop(flat)
        return self.fc_mu(dropped), self.fc_log_var(dropped)

    def _validate_input(self, x: Tensor) -> None:
        """Validate the expected public tensor layout [B, 3, 16, 16]."""
        if x.ndim != 4:
            raise ValueError("DVAE input must have shape [batch, tracks, slots, features].")
        expected = (self.config.tracks, self.config.steps_per_bar, self.config.feature_dim)
        actual = tuple(int(value) for value in x.shape[1:])
        if actual != expected:
            raise ValueError(f"DVAE input shape mismatch: expected [B, {expected}], got {tuple(x.shape)}.")


class TinyMusicDecoder(nn.Module):
    """Mirror decoder that reconstructs music tensor heads from latent z."""

    def __init__(self, config: DVAEMusicConfig) -> None:
        super().__init__()
        self.config = config
        self.seed_width = int(config.steps_per_bar // 2)
        self.fc = nn.Linear(int(config.latent_dim), 32 * 1 * self.seed_width)
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2), mode="nearest"),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 1), padding=(0, 0)),
            nn.GELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
        )
        self.to_pitch = nn.Conv2d(16, 1, kernel_size=1)
        self.to_state = nn.Conv2d(16, 3, kernel_size=1)
        self.to_velocity = nn.Conv2d(16, 1, kernel_size=1)
        self.tail_dim = max(1, int(config.feature_dim) - 5)
        self.to_chord = nn.Conv2d(16, self.tail_dim, kernel_size=1)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decode latent vectors into pitch/state/velocity/chord heads."""
        seed = self.fc(z).reshape(z.shape[0], 32, 1, self.seed_width)
        hidden = self.decoder_conv(seed)
        pitch = self.to_pitch(hidden).squeeze(1)
        velocity = torch.sigmoid(self.to_velocity(hidden).squeeze(1))
        state_logits = self.to_state(hidden).permute(0, 2, 3, 1).contiguous()
        chord = self.to_chord(hidden).permute(0, 2, 3, 1).contiguous()
        return pitch, state_logits, velocity, chord


class DenoisingMusicVAE(nn.Module):
    """Denoising VAE for multi-track bar tensors."""

    def __init__(self, config: DVAEMusicConfig) -> None:
        super().__init__()
        self.config = config
        self.noise = VariationNoiseInjector(config)
        self.encoder = TinyMusicEncoder(config)
        self.decoder = TinyMusicDecoder(config)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DenoisingMusicVAE":
        """Create a model from the full style configuration."""
        return cls(DVAEMusicConfig.from_config(config))

    def forward(self, clean_x: Tensor, add_noise: bool = True) -> DVAEOutput:
        """Run denoising VAE forward pass."""
        model_input = self.noise(clean_x) if add_noise else clean_x
        mu, log_var = self.encoder(model_input)
        z = self.reparameterize(mu, log_var)
        pitch, state_logits, velocity, chord = self.decoder(z)
        return DVAEOutput(
            pitch=pitch,
            state_logits=state_logits,
            velocity=velocity,
            chord=chord,
            mu=mu,
            log_var=log_var,
            z=z,
        )

    def encode_mu(self, clean_x: Tensor) -> Tensor:
        """Return deterministic latent means for downstream models."""
        mu, _log_var = self.encoder(clean_x)
        return mu

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample z from a diagonal Gaussian using the reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class DVAELoss:
    """Multi-head reconstruction loss plus KL divergence."""

    def __init__(self, config: DVAEMusicConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DVAELoss":
        """Create loss calculator from the full style configuration."""
        return cls(DVAEMusicConfig.from_config(config))

    def __call__(self, output: DVAEOutput, target: Tensor) -> Dict[str, Tensor]:
        """Calculate total and component losses."""
        target_pitch = target[..., 0]
        target_state = torch.argmax(target[..., 1:4], dim=-1)
        target_velocity = target[..., 4]
        target_chord = target[..., 5: int(self.config.feature_dim)]
        pitch_loss = F.mse_loss(output.pitch, target_pitch)
        state_loss = F.cross_entropy(
            output.state_logits.reshape(-1, 3),
            target_state.reshape(-1),
        )
        velocity_loss = F.mse_loss(output.velocity, target_velocity)
        chord_loss = F.mse_loss(output.chord, target_chord)
        physical_chroma_loss = self._physical_chroma_loss(output, target)
        kl_loss = self._kl_divergence(output.mu, output.log_var)
        total = (
            float(self.config.pitch_loss_weight) * pitch_loss
            + float(self.config.state_loss_weight) * state_loss
            + float(self.config.velocity_loss_weight) * velocity_loss
            + float(self.config.chord_loss_weight) * chord_loss
            + float(self.config.physical_chroma_loss_weight) * physical_chroma_loss
            + float(self.config.beta_kl) * kl_loss
        )
        return {
            "total_loss": total,
            "pitch_loss": pitch_loss,
            "state_loss": state_loss,
            "velocity_loss": velocity_loss,
            "chord_loss": chord_loss,
            "physical_chroma_loss": physical_chroma_loss,
            "chroma_delta_loss": total.new_zeros(()),
            "kl_loss": kl_loss,
        }

    def pair(
        self,
        current_output: DVAEOutput,
        current_target: Tensor,
        next_output: DVAEOutput,
        next_target: Tensor,
    ) -> Dict[str, Tensor]:
        """Average bar reconstruction and supervise the physical Chroma transition."""
        current = self(current_output, current_target)
        following = self(next_output, next_target)
        losses = {
            key: (current[key] + following[key]) * 0.5
            for key in current
        }
        delta_loss = self._physical_chroma_delta_loss(current_output, current_target, next_output, next_target)
        losses["chroma_delta_loss"] = delta_loss
        losses["total_loss"] = losses["total_loss"] + float(self.config.chroma_delta_loss_weight) * delta_loss
        return losses

    def _physical_chroma_loss(self, output: DVAEOutput, target: Tensor) -> Tensor:
        """Match the physical Chroma distribution without MSE-induced averaging."""
        target_chroma = self._target_physical_chroma(target)
        decoded_chroma = self._decoded_physical_chroma(output)
        active_mask = self._target_bar_active_mask(target)
        per_bar_kl = F.kl_div(
            torch.log(decoded_chroma.clamp_min(1.0e-8)),
            target_chroma,
            reduction="none",
        ).sum(dim=-1)
        weights = active_mask.to(dtype=per_bar_kl.dtype)
        return torch.sum(per_bar_kl * weights) / weights.sum().clamp_min(1.0)

    def _physical_chroma_delta_loss(
        self,
        current_output: DVAEOutput,
        current_target: Tensor,
        next_output: DVAEOutput,
        next_target: Tensor,
    ) -> Tensor:
        """Match Chroma movement relative to each real transition's energy."""
        target_delta = self._target_physical_chroma(next_target) - self._target_physical_chroma(current_target)
        decoded_delta = self._decoded_physical_chroma(next_output) - self._decoded_physical_chroma(current_output)
        adjacent_active_mask = self._target_bar_active_mask(current_target) & self._target_bar_active_mask(next_target)
        target_energy = torch.sum(target_delta.square(), dim=-1)
        moving_mask = adjacent_active_mask & (target_energy > 1.0e-8)
        relative_error = torch.sum((decoded_delta - target_delta).square(), dim=-1) / target_energy.clamp_min(1.0e-8)
        weights = moving_mask.to(dtype=relative_error.dtype)
        return torch.sum(relative_error * weights) / weights.sum().clamp_min(1.0)

    def _target_bar_active_mask(self, target: Tensor) -> Tensor:
        return (target[..., 2] + target[..., 3]).sum(dim=(1, 2)) > 0.0

    def _target_physical_chroma(self, target: Tensor) -> Tensor:
        active = target[..., 2] + target[..., 3]
        return self._physical_chroma(target[..., 0], active)

    def _decoded_physical_chroma(self, output: DVAEOutput) -> Tensor:
        state_probability = torch.softmax(output.state_logits, dim=-1)
        active_probability = state_probability[..., 1] + state_probability[..., 2]
        return self._physical_chroma(output.pitch, active_probability)

    def _physical_chroma(self, relative_pitch: Tensor, active: Tensor) -> Tensor:
        """Pool continuous relative pitches and active probabilities into 12 pitch classes."""
        semitones = relative_pitch * float(self.config.physical_chroma_pitch_scale)
        pitch_classes = torch.arange(12, dtype=relative_pitch.dtype, device=relative_pitch.device)
        distance = torch.remainder(semitones.unsqueeze(-1) - pitch_classes + 6.0, 12.0) - 6.0
        logits = -0.5 * (distance / float(self.config.physical_chroma_sigma)).square()
        membership = torch.softmax(logits, dim=-1)
        chroma = torch.sum(membership * active.unsqueeze(-1), dim=(1, 2))
        return chroma / chroma.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _kl_divergence(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Return batch-mean KL divergence against N(0, I)."""
        per_sample = -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return torch.mean(per_sample)


class DVAEOptimizerFactory:
    """Create optimizers with the model's default regularization policy."""

    def __init__(self, config: DVAEMusicConfig) -> None:
        self.config = config

    def adamw(self, model: nn.Module, learning_rate: float = 1.0e-3) -> torch.optim.Optimizer:
        """Create AdamW with configured weight decay."""
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
