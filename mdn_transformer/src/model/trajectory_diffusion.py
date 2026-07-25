#!/usr/bin/env python3
"""Joint latent/register trajectory diffusion conditioned on aligned REMI bars."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TrajectoryDiffusionConfig:
    """Architecture for four-bar joint latent/register diffusion."""

    vocab_size: int
    pad_token_id: int
    latent_dim: int = 32
    d_model: int = 256
    token_layers: int = 2
    bar_layers: int = 2
    denoiser_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    context_bars: int = 16
    memory_bars: int = 32
    gradient_checkpointing: bool = True
    max_bar_tokens: int = 192
    trajectory_bars: int = 4
    predictor_hidden_dim: int = 512
    context_pooling: str = "attention"
    diffusion_steps: int = 100
    sampling_steps: int = 16
    beta_schedule: str = "cosine"
    prediction_type: str = "v"
    register_offset_scale: float = 24.0
    register_offset_min: int = -24
    register_offset_max: int = 24

    @property
    def state_dim(self) -> int:
        """Return the continuous joint state size: latent plus register offset."""
        return int(self.latent_dim) + 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ThreeStreamContextEncoder(nn.Module):
    """Encode aligned REMI, relative latent, and register history into one state."""

    def __init__(self, config: TrajectoryDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        if str(config.prediction_type).lower() != "v":
            raise ValueError(f"JointTrajectoryDiffusion supports v-prediction only, got {config.prediction_type!r}.")
        self.token_embedding = nn.Embedding(
            int(config.vocab_size),
            int(config.d_model),
            padding_idx=int(config.pad_token_id),
        )
        self.token_position_embedding = nn.Embedding(int(config.max_bar_tokens), int(config.d_model))
        token_layer = nn.TransformerEncoderLayer(
            d_model=int(config.d_model),
            nhead=int(config.n_heads),
            dim_feedforward=int(config.d_model) * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(
            token_layer,
            num_layers=int(config.token_layers),
            enable_nested_tensor=False,
        )
        self.context_pooling = str(config.context_pooling).lower()
        if self.context_pooling not in {"last", "mean", "attention"}:
            raise ValueError(f"Unsupported REMI context_pooling: {config.context_pooling}")
        self.token_pool_score = nn.Linear(int(config.d_model), 1) if self.context_pooling == "attention" else None
        self.token_norm = nn.LayerNorm(int(config.d_model))
        self.latent_proj = nn.Sequential(
            nn.LayerNorm(int(config.latent_dim)),
            nn.Linear(int(config.latent_dim), int(config.d_model)),
            nn.GELU(),
        )
        self.register_proj = nn.Sequential(nn.Linear(1, int(config.d_model)), nn.GELU())
        self.stream_fusion = nn.Sequential(
            nn.Linear(int(config.d_model) * 3, int(config.d_model)),
            nn.GELU(),
            nn.LayerNorm(int(config.d_model)),
        )
        self.bar_position_embedding = nn.Embedding(int(config.context_bars), int(config.d_model))
        bar_layer = nn.TransformerEncoderLayer(
            d_model=int(config.d_model),
            nhead=int(config.n_heads),
            dim_feedforward=int(config.d_model) * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.bar_encoder = nn.TransformerEncoder(
            bar_layer,
            num_layers=int(config.bar_layers),
            enable_nested_tensor=False,
        )
        self.bar_norm = nn.LayerNorm(int(config.d_model))

    def forward(
        self,
        context_input_ids: Tensor,
        context_attention_mask: Tensor,
        context_latents: Tensor,
        context_register_offsets: Tensor,
        context_bar_mask: Tensor,
    ) -> Tensor:
        """Return one causal bar-history condition per batch item."""
        if context_input_ids.ndim != 3:
            raise ValueError("context_input_ids must have shape [batch, bars, tokens].")
        if context_attention_mask.shape != context_input_ids.shape:
            raise ValueError("context_attention_mask must match context_input_ids.")
        batch_size, bar_count, token_count = context_input_ids.shape
        if bar_count > int(self.config.context_bars):
            raise ValueError(f"context bars {bar_count} exceeds context_bars={self.config.context_bars}.")
        if token_count > int(self.config.max_bar_tokens):
            raise ValueError(f"bar token length {token_count} exceeds max_bar_tokens={self.config.max_bar_tokens}.")
        if context_latents.shape[:2] != (batch_size, bar_count):
            raise ValueError("context_latents must align with the bar axis.")
        if context_register_offsets.shape[:2] != (batch_size, bar_count):
            raise ValueError("context_register_offsets must align with the bar axis.")

        flat_ids = context_input_ids.reshape(batch_size * bar_count, token_count)
        flat_mask = context_attention_mask.reshape(batch_size * bar_count, token_count)
        has_tokens = torch.any(flat_mask > 0, dim=1)
        remi_flat = torch.zeros(
            (batch_size * bar_count, int(self.config.d_model)),
            dtype=self.token_embedding.weight.dtype,
            device=context_input_ids.device,
        )
        if bool(torch.any(has_tokens).item()):
            valid_ids = flat_ids[has_tokens]
            valid_mask = flat_mask[has_tokens]
            positions = torch.arange(token_count, device=context_input_ids.device).unsqueeze(0).expand_as(valid_ids)
            token_hidden = self.token_embedding(valid_ids) + self.token_position_embedding(positions)
            token_encoded = self.token_encoder(token_hidden, src_key_padding_mask=valid_mask <= 0)
            valid_remi = self.token_norm(self._pool_tokens(token_encoded, valid_mask))
            remi_flat = remi_flat.index_copy(0, torch.nonzero(has_tokens, as_tuple=False).squeeze(1), valid_remi)
        remi_bar = remi_flat.reshape(batch_size, bar_count, int(self.config.d_model))

        latent_bar = self.latent_proj(context_latents)
        offsets = context_register_offsets.float().reshape(batch_size, bar_count, 1)
        register_bar = self.register_proj(offsets / max(1.0, float(self.config.register_offset_scale)))
        fused = self.stream_fusion(torch.cat([remi_bar, latent_bar, register_bar], dim=-1))
        positions = torch.arange(bar_count, device=context_input_ids.device).unsqueeze(0).expand(batch_size, bar_count)
        return self._encode_valid_bar_context(fused + self.bar_position_embedding(positions), context_bar_mask)

    def _pool_tokens(self, encoded: Tensor, attention_mask: Tensor) -> Tensor:
        valid = (attention_mask > 0).unsqueeze(-1)
        if self.context_pooling == "last":
            lengths = torch.sum(attention_mask > 0, dim=1).long().clamp_min(1)
            return encoded[torch.arange(encoded.shape[0], device=encoded.device), lengths - 1]
        if self.context_pooling == "mean":
            masked = encoded * valid.to(encoded.dtype)
            return masked.sum(dim=1) / valid.to(encoded.dtype).sum(dim=1).clamp_min(1.0)
        if self.token_pool_score is None:
            raise RuntimeError("token_pool_score is not initialized.")
        scores = self.token_pool_score(encoded).squeeze(-1).masked_fill(attention_mask <= 0, -1.0e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1) * valid.to(encoded.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        return torch.sum(encoded * weights, dim=1)

    def _encode_valid_bar_context(self, fused: Tensor, bar_mask: Tensor) -> Tensor:
        lengths = torch.sum(bar_mask > 0, dim=1).long()
        if torch.any(lengths < 1):
            raise ValueError("Each trajectory sample needs at least one valid context bar.")
        batch_size, bar_count, d_model = fused.shape
        pooled = torch.zeros((batch_size, d_model), dtype=fused.dtype, device=fused.device)
        for length in torch.unique(lengths).tolist():
            valid_length = int(length)
            indices = torch.nonzero(lengths == valid_length, as_tuple=False).squeeze(1)
            sequence = fused[indices, bar_count - valid_length:, :]
            encoded = self.bar_encoder(sequence, mask=self._causal_mask(valid_length, fused.device))
            pooled = pooled.index_copy(0, indices, encoded[:, -1, :])
        return self.bar_norm(pooled)

    def _causal_mask(self, length: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)


class DiffusionTimeEmbedding(nn.Module):
    """Sinusoidal diffusion-time embedding followed by a learned projection."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.proj = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) * 2),
            nn.GELU(),
            nn.Linear(int(d_model) * 2, int(d_model)),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        half = max(1, self.d_model // 2)
        exponent = -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(1, half - 1)
        angles = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] < self.d_model:
            embedding = torch.nn.functional.pad(embedding, (0, self.d_model - embedding.shape[-1]))
        return self.proj(embedding)


class TrajectoryDenoiser(nn.Module):
    """Denoise an entire short future trajectory jointly under one history condition."""

    def __init__(self, config: TrajectoryDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.state_proj = nn.Sequential(nn.Linear(int(config.state_dim), int(config.d_model)), nn.GELU())
        self.condition_proj = nn.Sequential(nn.Linear(int(config.d_model), int(config.d_model)), nn.GELU())
        self.time_embedding = DiffusionTimeEmbedding(int(config.d_model))
        self.future_position_embedding = nn.Embedding(int(config.trajectory_bars), int(config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=int(config.d_model),
            nhead=int(config.n_heads),
            dim_feedforward=int(config.d_model) * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trajectory_encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(config.denoiser_layers),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(int(config.d_model))
        self.noise_head = nn.Sequential(
            nn.Linear(int(config.d_model), int(config.predictor_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.predictor_hidden_dim), int(config.state_dim)),
        )

    def forward(self, noisy_trajectory: Tensor, condition: Tensor, timesteps: Tensor) -> Tensor:
        """Predict Gaussian noise for every future bar in the trajectory."""
        if noisy_trajectory.ndim != 3:
            raise ValueError("noisy_trajectory must have shape [batch, trajectory_bars, state_dim].")
        if noisy_trajectory.shape[1:] != (int(self.config.trajectory_bars), int(self.config.state_dim)):
            raise ValueError("noisy_trajectory shape does not match trajectory configuration.")
        if condition.shape != (noisy_trajectory.shape[0], int(self.config.d_model)):
            raise ValueError("condition must have shape [batch, d_model].")
        positions = torch.arange(int(self.config.trajectory_bars), device=noisy_trajectory.device).unsqueeze(0)
        shared = self.condition_proj(condition).unsqueeze(1) + self.time_embedding(timesteps).unsqueeze(1)
        hidden = self.state_proj(noisy_trajectory) + shared + self.future_position_embedding(positions)
        return self.noise_head(self.output_norm(self.trajectory_encoder(hidden)))


class JointTrajectoryDiffusion(nn.Module):
    """Conditional DDPM/DDIM over joint latent and register-offset trajectories."""

    def __init__(self, config: TrajectoryDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = ThreeStreamContextEncoder(config)
        self.denoiser = TrajectoryDenoiser(config)
        betas = self._build_betas(int(config.diffusion_steps), str(config.beta_schedule))
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)

    def encode_context(
        self,
        context_input_ids: Tensor,
        context_attention_mask: Tensor,
        context_latents: Tensor,
        context_register_offsets: Tensor,
        context_bar_mask: Tensor,
    ) -> Tensor:
        return self.context_encoder(
            context_input_ids,
            context_attention_mask,
            context_latents,
            context_register_offsets,
            context_bar_mask,
        )

    def diffusion_loss(self, condition: Tensor, target_trajectory: Tensor) -> Dict[str, Tensor]:
        """Apply the v-prediction diffusion objective.

        At high noise levels, epsilon prediction recovers the clean state by
        dividing by sqrt(alpha_bar), which becomes numerically explosive. The
        velocity target retains the clean-state signal at low SNR instead.
        """
        batch_size = int(target_trajectory.shape[0])
        timesteps = torch.randint(0, int(self.config.diffusion_steps), (batch_size,), device=target_trajectory.device)
        noise = torch.randn_like(target_trajectory)
        alpha_bar = self.alpha_bars[timesteps].reshape(batch_size, 1, 1)
        noisy = torch.sqrt(alpha_bar) * target_trajectory + torch.sqrt(1.0 - alpha_bar) * noise
        sqrt_alpha = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar)
        target_velocity = sqrt_alpha * noise - sqrt_one_minus_alpha * target_trajectory
        predicted_velocity = self.denoiser(noisy, condition, timesteps)
        velocity_mse = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
        predicted_clean = sqrt_alpha * noisy - sqrt_one_minus_alpha * predicted_velocity
        return {
            "loss": velocity_mse,
            "velocity_mse": velocity_mse,
            # Callers may add differentiable decoded-trajectory objectives.
            "predicted_clean": predicted_clean,
        }

    @torch.no_grad()
    def sample(self, condition: Tensor, sampling_steps: int | None = None) -> Tensor:
        """DDIM-sample a full future trajectory from white noise."""
        steps = max(1, min(int(sampling_steps or self.config.sampling_steps), int(self.config.diffusion_steps)))
        timestep_values = torch.linspace(
            int(self.config.diffusion_steps) - 1,
            0,
            steps=steps,
            device=condition.device,
        ).round().long()
        timestep_values = torch.unique_consecutive(timestep_values)
        current = torch.randn(
            (condition.shape[0], int(self.config.trajectory_bars), int(self.config.state_dim)),
            device=condition.device,
            dtype=condition.dtype,
        )
        for index, timestep in enumerate(timestep_values):
            t_value = int(timestep.item())
            t_batch = torch.full((condition.shape[0],), t_value, device=condition.device, dtype=torch.long)
            alpha_bar = self.alpha_bars[t_value].to(dtype=current.dtype)
            sqrt_alpha = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar)
            predicted_velocity = self.denoiser(current, condition, t_batch)
            predicted_clean = sqrt_alpha * current - sqrt_one_minus_alpha * predicted_velocity
            predicted_noise = sqrt_one_minus_alpha * current + sqrt_alpha * predicted_velocity
            if index + 1 == len(timestep_values):
                current = predicted_clean
                continue
            next_t = int(timestep_values[index + 1].item())
            next_alpha_bar = self.alpha_bars[next_t].to(dtype=current.dtype)
            current = torch.sqrt(next_alpha_bar) * predicted_clean + torch.sqrt(1.0 - next_alpha_bar) * predicted_noise
        return current

    def _build_betas(self, steps: int, schedule: str) -> Tensor:
        if steps < 2:
            raise ValueError("diffusion_steps must be at least 2.")
        if schedule.lower() != "cosine":
            raise ValueError(f"Unsupported diffusion beta schedule: {schedule}")
        grid = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((grid / steps) + 0.008) / 1.008 * math.pi * 0.5).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(1.0e-5, 0.999).float()
