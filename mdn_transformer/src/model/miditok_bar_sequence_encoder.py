#!/usr/bin/env python3
"""MidiTok-style bar event sequence encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config_loader import ConfigView


@dataclass(frozen=True)
class MidiTokBarSequenceEncoderConfig:
    """Configuration for bar-local event sequence encoding."""

    latent_dim: int = 32
    max_events: int = 64
    pitch_min: int = -48
    pitch_max: int = 48
    velocity_bins: int = 8
    position_vocab_size: int = 17
    track_vocab_size: int = 4
    duration_vocab_size: int = 17
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    field_embedding_dim: int = 32
    cosine_loss_weight: float = 0.1

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MidiTokBarSequenceEncoderConfig":
        """Build config from style defaults."""
        section = ConfigView(config).section("miditok_bar_sequence_encoder")
        return cls(
            latent_dim=int(section.get("latent_dim", 32)),
            max_events=int(section.get("max_events", 64)),
            pitch_min=int(section.get("pitch_min", -48)),
            pitch_max=int(section.get("pitch_max", 48)),
            velocity_bins=int(section.get("velocity_bins", 8)),
            position_vocab_size=int(section.get("position_vocab_size", 17)),
            track_vocab_size=int(section.get("track_vocab_size", 4)),
            duration_vocab_size=int(section.get("duration_vocab_size", 17)),
            d_model=int(section.get("d_model", 128)),
            n_layers=int(section.get("n_layers", 4)),
            n_heads=int(section.get("n_heads", 4)),
            dropout=float(section.get("dropout", 0.1)),
            field_embedding_dim=int(section.get("field_embedding_dim", 32)),
            cosine_loss_weight=float(section.get("cosine_loss_weight", 0.1)),
        )

    @property
    def pitch_vocab_size(self) -> int:
        """Return pitch bins plus pad bucket."""
        return int(self.pitch_max) - int(self.pitch_min) + 2

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe dict."""
        return dict(self.__dict__)


class MidiTokBarSequenceEncoder(nn.Module):
    """Encode a bar-local event token sequence into a continuous embedding and latent estimate."""

    PAD_ID = 0

    def __init__(self, config: MidiTokBarSequenceEncoderConfig) -> None:
        super().__init__()
        self.config = config
        emb_dim = int(config.field_embedding_dim)
        self.position_embedding = nn.Embedding(int(config.position_vocab_size), emb_dim, padding_idx=self.PAD_ID)
        self.track_embedding = nn.Embedding(int(config.track_vocab_size), emb_dim, padding_idx=self.PAD_ID)
        self.pitch_embedding = nn.Embedding(int(config.pitch_vocab_size), emb_dim, padding_idx=self.PAD_ID)
        self.duration_embedding = nn.Embedding(int(config.duration_vocab_size), emb_dim, padding_idx=self.PAD_ID)
        self.velocity_embedding = nn.Embedding(int(config.velocity_bins) + 1, emb_dim, padding_idx=self.PAD_ID)
        self.event_projection = nn.Linear(emb_dim * 5, int(config.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(config.d_model)))
        self.position_encoding = SinusoidalPositionEncoding(int(config.d_model), int(config.max_events) + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=int(config.d_model),
            nhead=int(config.n_heads),
            dim_feedforward=int(config.d_model) * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(config.n_layers))
        self.norm = nn.LayerNorm(int(config.d_model))
        self.to_latent = nn.Sequential(
            nn.Linear(int(config.d_model), int(config.d_model)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.d_model), int(config.latent_dim)),
        )

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return sequence embedding and predicted VAE latent.

        tokens shape: [B, max_events, 5] with fields:
        position_id, track_id, pitch_id, duration_id, velocity_id.
        padding_mask shape: [B, max_events], True where padded.
        """
        position = self.position_embedding(tokens[..., 0].long())
        track = self.track_embedding(tokens[..., 1].long())
        pitch = self.pitch_embedding(tokens[..., 2].long())
        duration = self.duration_embedding(tokens[..., 3].long())
        velocity = self.velocity_embedding(tokens[..., 4].long())
        events = self.event_projection(torch.cat([position, track, pitch, duration, velocity], dim=-1))
        batch_size = int(events.shape[0])
        cls = self.cls_token.expand(batch_size, 1, -1)
        sequence = torch.cat([cls, events], dim=1)
        sequence = self.position_encoding(sequence)
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=padding_mask.device)
        full_mask = torch.cat([cls_mask, padding_mask.bool()], dim=1)
        encoded = self.transformer(sequence, src_key_padding_mask=full_mask)
        embedding = self.norm(encoded[:, 0])
        latent = self.to_latent(embedding)
        return {"embedding": embedding, "latent_mu": latent}


class MidiTokBarSequenceLoss(nn.Module):
    """Latent distillation loss for MidiTok-style bar sequence encoder."""

    def __init__(self, config: MidiTokBarSequenceEncoderConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, output: Dict[str, torch.Tensor], target_mu: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return loss components."""
        pred = output["latent_mu"]
        mse = F.mse_loss(pred, target_mu)
        cosine = 1.0 - F.cosine_similarity(pred, target_mu, dim=-1).mean()
        loss = mse + float(self.config.cosine_loss_weight) * cosine
        return {"loss": loss, "mse": mse.detach(), "cosine_loss": cosine.detach()}


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal sequence position encoding."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add fixed sequence positions."""
        return x + self.pe[:, : x.shape[1], :]
