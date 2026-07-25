#!/usr/bin/env python3
"""Learn base-pitch motion for generated bar sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config_loader import ConfigView
from model.hybrid_miditok_retrieval import HybridContextEncoder
from model.miditok_bar_sequence_encoder import MidiTokBarSequenceEncoder, MidiTokBarSequenceEncoderConfig


@dataclass(frozen=True)
class BasePitchMotionConfig:
    """Configuration for base-pitch delta prediction."""

    latent_dim: int = 32
    context_bars: int = 16
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    delta_min: int = -12
    delta_max: int = 12
    base_pitch_scale: float = 127.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BasePitchMotionConfig":
        """Build config from style defaults."""
        section = ConfigView(config).section("base_pitch_motion")
        return cls(
            latent_dim=int(section.get("latent_dim", 32)),
            context_bars=int(section.get("context_bars", 16)),
            d_model=int(section.get("d_model", 128)),
            n_layers=int(section.get("n_layers", 2)),
            n_heads=int(section.get("n_heads", 4)),
            dropout=float(section.get("dropout", 0.1)),
            delta_min=int(section.get("delta_min", -12)),
            delta_max=int(section.get("delta_max", 12)),
            base_pitch_scale=float(section.get("base_pitch_scale", 127.0)),
        )

    @property
    def delta_class_count(self) -> int:
        """Return number of clipped semitone delta classes."""
        return int(self.delta_max) - int(self.delta_min) + 1

    def delta_to_class(self, delta: int) -> int:
        """Convert a semitone delta to a clipped class id."""
        clipped = max(int(self.delta_min), min(int(self.delta_max), int(delta)))
        return int(clipped - int(self.delta_min))

    def class_to_delta(self, class_id: int) -> int:
        """Convert class id back to semitone delta."""
        return int(class_id) + int(self.delta_min)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe config."""
        return dict(self.__dict__)


class BasePitchMotionModel(nn.Module):
    """Predict next base-pitch movement from recent generated bar context."""

    def __init__(self, config: BasePitchMotionConfig, event_config: MidiTokBarSequenceEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.event_config = event_config
        self.event_encoder = MidiTokBarSequenceEncoder(event_config)
        feature_dim = int(config.latent_dim) + int(event_config.d_model) + 1
        self.input_projection = nn.Linear(feature_dim, int(config.d_model))
        self.context_encoder = HybridContextEncoder(config)  # structural config fields are compatible
        self.delta_head = nn.Sequential(
            nn.Linear(int(config.d_model), int(config.d_model)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.d_model), int(config.delta_class_count)),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return logits over clipped semitone delta classes."""
        context = self.encode_context(
            context_mu=batch["context_mu"],
            context_tokens=batch["context_tokens"],
            context_token_mask=batch["context_token_mask"],
            context_padding_mask=batch["context_padding_mask"],
            context_base_pitch=batch["context_base_pitch"],
        )
        return self.delta_head(context)

    def encode_context(
        self,
        context_mu: torch.Tensor,
        context_tokens: torch.Tensor,
        context_token_mask: torch.Tensor,
        context_padding_mask: torch.Tensor,
        context_base_pitch: torch.Tensor,
    ) -> torch.Tensor:
        """Encode recent bars and their rendered base-pitch trajectory."""
        batch_size, context_bars, max_events, fields = context_tokens.shape
        flat_tokens = context_tokens.reshape(batch_size * context_bars, max_events, fields)
        flat_mask = context_token_mask.reshape(batch_size * context_bars, max_events)
        event_embedding = self.event_encoder(flat_tokens, flat_mask)["embedding"].reshape(batch_size, context_bars, -1)
        base = context_base_pitch.float().unsqueeze(-1) / max(1.0, float(self.config.base_pitch_scale))
        feature = torch.cat([context_mu, event_embedding, base], dim=-1)
        hidden = self.input_projection(feature)
        return self.context_encoder(hidden, context_padding_mask)

    def loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return cross-entropy loss and accuracy."""
        logits = self.forward(batch)
        target = batch["target_delta_class"].long()
        loss = F.cross_entropy(logits, target)
        pred = torch.argmax(logits, dim=-1)
        accuracy = (pred == target).float().mean()
        abs_error = torch.abs(pred.float() - target.float()).mean()
        return {"loss": loss, "accuracy": accuracy.detach(), "class_abs_error": abs_error.detach()}
