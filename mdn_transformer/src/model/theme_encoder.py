#!/usr/bin/env python3
"""BiLSTM + self-attention theme encoder for DVAE latent sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config_loader import ConfigView


@dataclass(frozen=True)
class ThemeEncoderConfig:
    """Configuration for the latent theme encoder."""

    latent_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 1
    embedding_dim: int = 64
    projection_hidden_dim: int = 128
    dropout: float = 0.1

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ThemeEncoderConfig":
        """Build model config from style config."""
        section = ConfigView(config).section("theme_encoder")
        return cls(
            latent_dim=int(section.get("latent_dim", 32)),
            hidden_dim=int(section.get("hidden_dim", 64)),
            num_layers=int(section.get("num_layers", 1)),
            embedding_dim=int(section.get("embedding_dim", 64)),
            projection_hidden_dim=int(section.get("projection_hidden_dim", 128)),
            dropout=float(section.get("dropout", 0.1)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe config values."""
        return dict(self.__dict__)


class SelfAttentionPool(nn.Module):
    """Learned attention pooling over a short latent theme sequence."""

    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Dropout(float(dropout)),
            nn.Linear(input_dim, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Pool [batch, time, dim] into [batch, dim]."""
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=-1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class BiLSTMAttentionThemeEncoder(nn.Module):
    """Encode opening DVAE latent bars into a normalized theme embedding."""

    def __init__(self, config: ThemeEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=int(config.latent_dim),
            hidden_size=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            dropout=float(config.dropout) if int(config.num_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        encoded_dim = int(config.hidden_dim) * 2
        self.pool = SelfAttentionPool(encoded_dim, float(config.dropout))
        self.projector = nn.Sequential(
            nn.LayerNorm(encoded_dim),
            nn.Linear(encoded_dim, int(config.projection_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.projection_hidden_dim), int(config.embedding_dim)),
        )

    def forward(self, theme_mu: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized theme embeddings."""
        encoded, _ = self.lstm(theme_mu)
        pooled = self.pool(encoded)
        embedding = self.projector(pooled)
        return F.normalize(embedding, dim=-1)
