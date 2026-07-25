#!/usr/bin/env python3
"""Hybrid latent + MidiTok-style next-bar retrieval model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config_loader import ConfigView
from model.miditok_bar_sequence_encoder import MidiTokBarSequenceEncoder, MidiTokBarSequenceEncoderConfig


@dataclass(frozen=True)
class HybridMidiTokRetrievalConfig:
    """Configuration for latent + MidiTok retrieval."""

    latent_dim: int = 32
    context_bars: int = 16
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    retrieval_dim: int = 128
    temperature: float = 0.1

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HybridMidiTokRetrievalConfig":
        """Build config from style config."""
        section = ConfigView(config).section("hybrid_miditok_retrieval")
        return cls(
            latent_dim=int(section.get("latent_dim", 32)),
            context_bars=int(section.get("context_bars", 16)),
            d_model=int(section.get("d_model", 128)),
            n_layers=int(section.get("n_layers", 2)),
            n_heads=int(section.get("n_heads", 4)),
            dropout=float(section.get("dropout", 0.1)),
            retrieval_dim=int(section.get("retrieval_dim", 128)),
            temperature=float(section.get("temperature", 0.1)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe config."""
        return dict(self.__dict__)


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position encodings to batch-first sequence."""
        return x + self.pe[:, : x.shape[1], :]


class HybridContextEncoder(nn.Module):
    """Encode a fixed window of hybrid bar features."""

    def __init__(self, config: HybridMidiTokRetrievalConfig) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(config.d_model)))
        self.position_encoding = SinusoidalPositionEncoding(int(config.d_model), int(config.context_bars) + 1)
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

    def forward(self, context: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Return one context embedding."""
        batch_size = int(context.shape[0])
        cls = self.cls_token.expand(batch_size, 1, -1)
        sequence = self.position_encoding(torch.cat([cls, context], dim=1))
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=padding_mask.device)
        full_mask = torch.cat([cls_mask, padding_mask.bool()], dim=1)
        encoded = self.transformer(sequence, src_key_padding_mask=full_mask)
        return self.norm(encoded[:, 0])


class HybridMidiTokRetrievalModel(nn.Module):
    """Contrastive next-bar retrieval model using latent mu plus event tokens."""

    def __init__(
        self,
        config: HybridMidiTokRetrievalConfig,
        event_config: MidiTokBarSequenceEncoderConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.event_config = event_config
        self.event_encoder = MidiTokBarSequenceEncoder(event_config)
        feature_dim = int(config.latent_dim) + int(event_config.d_model)
        self.input_projection = nn.Linear(feature_dim, int(config.d_model))
        self.target_projection = nn.Linear(feature_dim, int(config.retrieval_dim))
        self.context_encoder = HybridContextEncoder(config)
        self.context_projection = nn.Linear(int(config.d_model), int(config.retrieval_dim))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalized context and target embeddings for contrastive training."""
        context = self.encode_context(
            context_mu=batch["context_mu"],
            context_tokens=batch["context_tokens"],
            context_token_mask=batch["context_token_mask"],
            context_padding_mask=batch["context_padding_mask"],
        )
        target = self.encode_targets(
            target_mu=batch["target_mu"],
            target_tokens=batch["target_tokens"],
            target_token_mask=batch["target_token_mask"],
        )
        return context, target

    def encode_context(
        self,
        context_mu: torch.Tensor,
        context_tokens: torch.Tensor,
        context_token_mask: torch.Tensor,
        context_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a context window into retrieval space."""
        batch_size, context_bars, max_events, fields = context_tokens.shape
        flat_tokens = context_tokens.reshape(batch_size * context_bars, max_events, fields)
        flat_mask = context_token_mask.reshape(batch_size * context_bars, max_events)
        event_embedding = self.event_encoder(flat_tokens, flat_mask)["embedding"].reshape(batch_size, context_bars, -1)
        feature = torch.cat([context_mu, event_embedding], dim=-1)
        hidden = self.input_projection(feature)
        context = self.context_encoder(hidden, context_padding_mask)
        return F.normalize(self.context_projection(context), dim=-1)

    def encode_targets(
        self,
        target_mu: torch.Tensor,
        target_tokens: torch.Tensor,
        target_token_mask: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode candidate target bars into retrieval space."""
        embeddings = []
        total = int(target_mu.shape[0])
        step = int(batch_size or total)
        for start in range(0, total, step):
            end = min(total, start + step)
            event_embedding = self.event_encoder(target_tokens[start:end], target_token_mask[start:end])["embedding"]
            feature = torch.cat([target_mu[start:end], event_embedding], dim=-1)
            embeddings.append(F.normalize(self.target_projection(feature), dim=-1))
        return torch.cat(embeddings, dim=0)

