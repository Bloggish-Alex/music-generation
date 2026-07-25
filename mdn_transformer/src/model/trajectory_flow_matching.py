#!/usr/bin/env python3
"""Conditional Flow Matching correction for sampled latent trajectories."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TrajectoryFlowMatchingModelConfig:
    """Architecture of a conditional velocity field over a future trajectory."""

    state_dim: int
    condition_dim: int
    trajectory_bars: int
    d_model: int = 256
    layers: int = 2
    n_heads: int = 4
    hidden_dim: int = 512
    dropout: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContinuousTimeEmbedding(nn.Module):
    """Map flow time in [0, 1] into the velocity-field hidden space."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.d_model),
        )

    def forward(self, flow_time: Tensor) -> Tensor:
        if flow_time.ndim != 1:
            raise ValueError("flow_time must have shape [batch].")
        half = max(1, self.d_model // 2)
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=flow_time.device, dtype=flow_time.dtype)
            / max(1, half - 1)
        )
        angles = flow_time.unsqueeze(1) * frequencies.unsqueeze(0) * (2.0 * math.pi)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] < self.d_model:
            embedding = torch.nn.functional.pad(embedding, (0, self.d_model - embedding.shape[-1]))
        return self.projection(embedding)


class TrajectoryFlowMatcher(nn.Module):
    """Learn a conditional transport from a diffusion proposal to the data manifold."""

    def __init__(self, config: TrajectoryFlowMatchingModelConfig) -> None:
        super().__init__()
        if int(config.d_model) % int(config.n_heads) != 0:
            raise ValueError("Flow matching d_model must be divisible by n_heads.")
        self.config = config
        self.state_projection = nn.Sequential(
            nn.Linear(int(config.state_dim), int(config.d_model)),
            nn.GELU(),
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(int(config.condition_dim), int(config.d_model)),
            nn.GELU(),
        )
        self.time_embedding = ContinuousTimeEmbedding(int(config.d_model))
        self.position_embedding = nn.Embedding(int(config.trajectory_bars), int(config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=int(config.d_model),
            nhead=int(config.n_heads),
            dim_feedforward=int(config.d_model) * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trajectory_encoder = nn.TransformerEncoder(layer, num_layers=int(config.layers))
        self.output_norm = nn.LayerNorm(int(config.d_model))
        self.velocity_head = nn.Sequential(
            nn.Linear(int(config.d_model), int(config.hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.hidden_dim), int(config.state_dim)),
        )

    def forward(self, state: Tensor, condition: Tensor, flow_time: Tensor) -> Tensor:
        """Return d(state)/d(flow_time) for a normalized trajectory state."""
        expected_state = (int(self.config.trajectory_bars), int(self.config.state_dim))
        if state.ndim != 3 or tuple(state.shape[1:]) != expected_state:
            raise ValueError(f"state must have shape [batch, {expected_state[0]}, {expected_state[1]}].")
        if condition.shape != (state.shape[0], int(self.config.condition_dim)):
            raise ValueError("condition must align with the batch and condition_dim.")
        if flow_time.shape != (state.shape[0],):
            raise ValueError("flow_time must align with the state batch.")
        positions = torch.arange(int(self.config.trajectory_bars), device=state.device).unsqueeze(0)
        shared = self.condition_projection(condition).unsqueeze(1) + self.time_embedding(flow_time).unsqueeze(1)
        hidden = self.state_projection(state) + shared + self.position_embedding(positions)
        return self.velocity_head(self.output_norm(self.trajectory_encoder(hidden)))

    def flow_matching_loss(self, proposal: Tensor, target: Tensor, condition: Tensor) -> Dict[str, Tensor]:
        """Fit the straight conditional transport proposal -> target at random times."""
        if proposal.shape != target.shape:
            raise ValueError("proposal and target must have the same trajectory shape.")
        flow_time = torch.rand((proposal.shape[0],), device=proposal.device, dtype=proposal.dtype)
        weight = flow_time.reshape(-1, 1, 1)
        interpolated = (1.0 - weight) * proposal + weight * target
        target_velocity = target - proposal
        predicted_velocity = self(interpolated, condition, flow_time)
        velocity_mse = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
        return {
            "loss": velocity_mse,
            "velocity_mse": velocity_mse,
            "proposal_mse": torch.nn.functional.mse_loss(proposal, target).detach(),
        }

    def correct(self, proposal: Tensor, condition: Tensor, integration_steps: int) -> Tensor:
        """Integrate the learned velocity field from flow time 0 to 1 with Heun steps.

        Callers that only sample can use ``torch.no_grad()``. Keeping this
        method differentiable lets decoded trajectory losses supervise the
        correction field directly during training.
        """
        steps = max(1, int(integration_steps))
        current = proposal
        delta = 1.0 / float(steps)
        batch_size = int(proposal.shape[0])
        for index in range(steps):
            start = float(index) * delta
            end = float(index + 1) * delta
            start_time = torch.full((batch_size,), start, device=current.device, dtype=current.dtype)
            first_velocity = self(current, condition, start_time)
            predicted = current + delta * first_velocity
            end_time = torch.full((batch_size,), end, device=current.device, dtype=current.dtype)
            second_velocity = self(predicted, condition, end_time)
            current = current + 0.5 * delta * (first_velocity + second_velocity)
        return current
