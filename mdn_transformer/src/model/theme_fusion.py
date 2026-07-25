#!/usr/bin/env python3
"""Pluggable theme fusion adapters for latent sequence models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ThemeFusionAdapterConfig:
    """Runtime configuration for theme fusion modules."""

    enabled: bool = False
    mode: str = "film_pi"
    target: str = "pi"
    embedding_dim: int = 64
    latent_dim: int = 32
    d_model: int = 128
    project_dim: int = 16
    gate_init: float = 0.01
    dropout: float = 0.3
    theme_dropout: float = 0.3
    embedding_noise_std: float = 0.03
    token_bars: int = 8
    cross_attention_heads: int = 4


@dataclass
class ThemeFusionResult:
    """Hidden states after optional theme fusion."""

    hidden: torch.Tensor
    pi_hidden: torch.Tensor


class BaseThemeFusionAdapter(nn.Module):
    """Base class for swappable theme fusion adapters."""

    def forward(
        self,
        hidden: torch.Tensor,
        theme_embedding: Optional[torch.Tensor] = None,
        theme_tokens: Optional[torch.Tensor] = None,
    ) -> ThemeFusionResult:
        """Return hidden states for distribution heads and the pi head."""
        return ThemeFusionResult(hidden=hidden, pi_hidden=hidden)

    def runtime_diagnostics(self) -> Dict[str, object]:
        """Return JSON-safe runtime diagnostics."""
        return {"enabled": False, "mode": "none"}


class NoThemeFusionAdapter(BaseThemeFusionAdapter):
    """No-op adapter used when theme fusion is disabled."""


class ThemeDropoutMixin:
    """Shared helpers for theme regularization."""

    config: ThemeFusionAdapterConfig

    def _gate_value(self) -> torch.Tensor:
        return torch.sigmoid(self.theme_gate_logit)

    def _initial_gate_logit(self) -> float:
        value = min(max(float(self.config.gate_init), 1.0e-5), 1.0 - 1.0e-5)
        return math.log(value / (1.0 - value))

    def _regularize_embedding(self, theme_embedding: torch.Tensor) -> torch.Tensor:
        output = theme_embedding
        if self.training and float(self.config.theme_dropout) > 0.0:
            keep = torch.rand((output.shape[0], 1), device=output.device, dtype=output.dtype)
            keep = (keep >= float(self.config.theme_dropout)).to(dtype=output.dtype)
            output = output * keep
        if self.training and float(self.config.embedding_noise_std) > 0.0:
            output = output + torch.randn_like(output) * float(self.config.embedding_noise_std)
        return output


class FiLMPiThemeFusionAdapter(BaseThemeFusionAdapter, ThemeDropoutMixin):
    """Use a frozen theme embedding to softly modulate only the MDN pi head."""

    def __init__(self, config: ThemeFusionAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.theme_projector = nn.Sequential(
            nn.LayerNorm(int(config.embedding_dim)),
            nn.Linear(int(config.embedding_dim), int(config.project_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.project_dim), int(config.d_model) * 2),
        )
        self.theme_gate_logit = nn.Parameter(torch.tensor(self._initial_gate_logit(), dtype=torch.float32))

    def forward(
        self,
        hidden: torch.Tensor,
        theme_embedding: Optional[torch.Tensor] = None,
        theme_tokens: Optional[torch.Tensor] = None,
    ) -> ThemeFusionResult:
        """Return hidden unchanged and pi hidden modulated by theme FiLM."""
        if theme_embedding is None or int(theme_embedding.shape[-1]) == 0:
            raise ValueError("theme_embedding is required for FiLM theme fusion.")
        theme = self._regularize_embedding(theme_embedding.to(dtype=hidden.dtype, device=hidden.device))
        gamma, beta = self.theme_projector(theme).chunk(2, dim=-1)
        gate = self._gate_value().to(dtype=hidden.dtype, device=hidden.device)
        modulated = hidden * (1.0 + gate * torch.tanh(gamma)) + gate * beta
        if str(self.config.target).lower() == "all":
            return ThemeFusionResult(hidden=modulated, pi_hidden=modulated)
        return ThemeFusionResult(hidden=hidden, pi_hidden=modulated)

    def runtime_diagnostics(self) -> Dict[str, object]:
        """Return runtime gate state."""
        return {
            "enabled": True,
            "mode": str(self.config.mode),
            "target": str(self.config.target),
            "theme_gate": float(self._gate_value().detach().cpu()),
        }


class CrossAttentionPiThemeFusionAdapter(BaseThemeFusionAdapter, ThemeDropoutMixin):
    """Attend from current history state into opening theme token sequence."""

    def __init__(self, config: ThemeFusionAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.token_projection = nn.Sequential(
            nn.LayerNorm(int(config.latent_dim)),
            nn.Linear(int(config.latent_dim), int(config.d_model)),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=int(config.d_model),
            num_heads=max(1, int(config.cross_attention_heads)),
            dropout=float(config.dropout),
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(int(config.d_model))
        self.output_dropout = nn.Dropout(float(config.dropout))
        self.theme_gate_logit = nn.Parameter(torch.tensor(self._initial_gate_logit(), dtype=torch.float32))

    def forward(
        self,
        hidden: torch.Tensor,
        theme_embedding: Optional[torch.Tensor] = None,
        theme_tokens: Optional[torch.Tensor] = None,
    ) -> ThemeFusionResult:
        """Return hidden unchanged and pi hidden cross-attended to theme tokens."""
        if theme_tokens is None or theme_tokens.ndim != 3 or int(theme_tokens.shape[1]) == 0:
            raise ValueError("theme_tokens with shape [batch, bars, latent_dim] is required for cross-attention theme fusion.")
        tokens = theme_tokens.to(dtype=hidden.dtype, device=hidden.device)
        if self.training and float(self.config.theme_dropout) > 0.0:
            keep = torch.rand((tokens.shape[0], 1, 1), device=tokens.device, dtype=tokens.dtype)
            keep = (keep >= float(self.config.theme_dropout)).to(dtype=tokens.dtype)
            tokens = tokens * keep
        if self.training and float(self.config.embedding_noise_std) > 0.0:
            tokens = tokens + torch.randn_like(tokens) * float(self.config.embedding_noise_std)
        projected = self.token_projection(tokens)
        attended, _ = self.cross_attention(hidden.unsqueeze(1), projected, projected, need_weights=False)
        gate = self._gate_value().to(dtype=hidden.dtype, device=hidden.device)
        modulated = self.output_norm(hidden + gate * self.output_dropout(attended.squeeze(1)))
        if str(self.config.target).lower() == "all":
            return ThemeFusionResult(hidden=modulated, pi_hidden=modulated)
        return ThemeFusionResult(hidden=hidden, pi_hidden=modulated)

    def runtime_diagnostics(self) -> Dict[str, object]:
        """Return runtime gate state."""
        return {
            "enabled": True,
            "mode": str(self.config.mode),
            "target": str(self.config.target),
            "theme_gate": float(self._gate_value().detach().cpu()),
        }


class FiLMCrossAttentionPiThemeFusionAdapter(BaseThemeFusionAdapter):
    """Apply FiLM first, then theme-token cross attention."""

    def __init__(self, config: ThemeFusionAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.film = FiLMPiThemeFusionAdapter(config)
        self.cross_attention = CrossAttentionPiThemeFusionAdapter(config)

    def forward(
        self,
        hidden: torch.Tensor,
        theme_embedding: Optional[torch.Tensor] = None,
        theme_tokens: Optional[torch.Tensor] = None,
    ) -> ThemeFusionResult:
        """Combine song-level embedding and token-sequence attention for pi hidden."""
        film_result = self.film(hidden, theme_embedding=theme_embedding, theme_tokens=theme_tokens)
        cross_result = self.cross_attention(film_result.pi_hidden, theme_embedding=theme_embedding, theme_tokens=theme_tokens)
        if str(self.config.target).lower() == "all":
            return ThemeFusionResult(hidden=cross_result.pi_hidden, pi_hidden=cross_result.pi_hidden)
        return ThemeFusionResult(hidden=hidden, pi_hidden=cross_result.pi_hidden)

    def runtime_diagnostics(self) -> Dict[str, object]:
        """Return diagnostics for both sub-adapters."""
        return {
            "enabled": True,
            "mode": str(self.config.mode),
            "target": str(self.config.target),
            "film_gate": self.film.runtime_diagnostics().get("theme_gate"),
            "cross_attention_gate": self.cross_attention.runtime_diagnostics().get("theme_gate"),
        }


def build_theme_fusion_adapter(config: ThemeFusionAdapterConfig) -> BaseThemeFusionAdapter:
    """Construct a theme fusion adapter by mode."""
    if not bool(config.enabled):
        return NoThemeFusionAdapter()
    mode = str(config.mode).lower()
    if mode == "film_pi":
        return FiLMPiThemeFusionAdapter(config)
    if mode == "cross_attention_pi":
        return CrossAttentionPiThemeFusionAdapter(config)
    if mode == "film_cross_attention_pi":
        return FiLMCrossAttentionPiThemeFusionAdapter(config)
    raise ValueError(f"Unsupported theme_fusion.mode: {config.mode}")
