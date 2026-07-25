#!/usr/bin/env python3
"""REMI-context motion model that predicts the next DVAE latent vector."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RemiMotionModelConfig:
    """Configuration for the REMI motion latent predictor."""

    vocab_size: int
    pad_token_id: int
    latent_dim: int = 32
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    max_context_tokens: int = 1024
    predictor_hidden_dim: int = 512
    context_pooling: str = "last"

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe config values."""
        return asdict(self)


class RemiMotionLatentPredictor(nn.Module):
    """Causal Transformer over REMI tokens, followed by a latent predictor head."""

    def __init__(self, config: RemiMotionModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            int(config.vocab_size),
            int(config.d_model),
            padding_idx=int(config.pad_token_id),
        )
        self.position_embedding = nn.Embedding(int(config.max_context_tokens), int(config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=int(config.d_model),
            nhead=int(config.n_heads),
            dim_feedforward=int(config.d_model) * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(config.n_layers))
        self.context_norm = nn.LayerNorm(int(config.d_model))
        self.context_pooling = str(config.context_pooling).lower()
        if self.context_pooling not in {"last", "mean", "attention"}:
            raise ValueError(f"Unsupported REMI context_pooling: {config.context_pooling}")
        if self.context_pooling == "attention":
            self.context_pool_score = nn.Linear(int(config.d_model), 1)
        else:
            self.context_pool_score = None
        self.prev_latent_proj = nn.Sequential(
            nn.LayerNorm(int(config.latent_dim)),
            nn.Linear(int(config.latent_dim), int(config.d_model)),
            nn.GELU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(int(config.d_model) * 2, int(config.predictor_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.predictor_hidden_dim), int(config.latent_dim)),
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor, prev_latent: Tensor) -> Tensor:
        """Predict the next latent vector from REMI history and previous latent."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens].")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape.")
        token_count = int(input_ids.shape[1])
        if token_count > int(self.config.max_context_tokens):
            raise ValueError(
                f"input sequence length {token_count} exceeds max_context_tokens={self.config.max_context_tokens}."
            )
        positions = torch.arange(token_count, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = self._causal_mask(token_count, input_ids.device)
        padding_mask = attention_mask <= 0
        encoded = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        pooled = self.context_norm(self._pool_context(encoded, attention_mask))
        previous = self.prev_latent_proj(prev_latent)
        return self.predictor(torch.cat([pooled, previous], dim=-1))

    def _causal_mask(self, length: int, device: torch.device) -> Tensor:
        """Return a standard causal attention mask."""
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def _last_valid_hidden(self, encoded: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool the hidden state at the last non-padding token."""
        lengths = torch.sum(attention_mask > 0, dim=1).long()
        lengths = torch.clamp(lengths, min=1)
        batch_indices = torch.arange(encoded.shape[0], device=encoded.device)
        return encoded[batch_indices, lengths - 1]

    def _pool_context(self, encoded: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool token-level hidden states into one context vector."""
        if self.context_pooling == "last":
            return self._last_valid_hidden(encoded, attention_mask)
        valid = (attention_mask > 0).unsqueeze(-1)
        if self.context_pooling == "mean":
            masked = encoded * valid.to(encoded.dtype)
            denom = torch.sum(valid.to(encoded.dtype), dim=1).clamp_min(1.0)
            return masked.sum(dim=1) / denom
        if self.context_pool_score is None:
            raise RuntimeError("context_pool_score is not initialized.")
        scores = self.context_pool_score(encoded).squeeze(-1)
        scores = scores.masked_fill(attention_mask <= 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        return torch.sum(encoded * weights, dim=1)


@dataclass(frozen=True)
class RemiBasePitchMotionConfig:
    """Configuration for REMI-context base-pitch delta prediction."""

    vocab_size: int
    pad_token_id: int
    latent_dim: int = 32
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    max_context_tokens: int = 1024
    predictor_hidden_dim: int = 512
    context_pooling: str = "attention"
    delta_min: int = -12
    delta_max: int = 12
    base_pitch_scale: float = 127.0

    @property
    def delta_class_count(self) -> int:
        """Return number of clipped semitone delta classes."""
        return int(self.delta_max) - int(self.delta_min) + 1

    def delta_to_class(self, delta: int) -> int:
        """Convert semitone delta to clipped class id."""
        clipped = max(int(self.delta_min), min(int(self.delta_max), int(delta)))
        return int(clipped - int(self.delta_min))

    def class_to_delta(self, class_id: int) -> int:
        """Convert class id back to semitone delta."""
        return int(class_id) + int(self.delta_min)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe config values."""
        return asdict(self)


class RemiBasePitchMotionPredictor(nn.Module):
    """Predict base-pitch movement from REMI history, previous latent, and previous base pitch."""

    def __init__(self, config: RemiBasePitchMotionConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            int(config.vocab_size),
            int(config.d_model),
            padding_idx=int(config.pad_token_id),
        )
        self.position_embedding = nn.Embedding(int(config.max_context_tokens), int(config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=int(config.d_model),
            nhead=int(config.n_heads),
            dim_feedforward=int(config.d_model) * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(config.n_layers))
        self.context_norm = nn.LayerNorm(int(config.d_model))
        self.context_pooling = str(config.context_pooling).lower()
        if self.context_pooling not in {"last", "mean", "attention"}:
            raise ValueError(f"Unsupported REMI context_pooling: {config.context_pooling}")
        if self.context_pooling == "attention":
            self.context_pool_score = nn.Linear(int(config.d_model), 1)
        else:
            self.context_pool_score = None
        self.prev_latent_proj = nn.Sequential(
            nn.LayerNorm(int(config.latent_dim)),
            nn.Linear(int(config.latent_dim), int(config.d_model)),
            nn.GELU(),
        )
        self.prev_base_pitch_proj = nn.Sequential(
            nn.Linear(1, int(config.d_model)),
            nn.GELU(),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(int(config.d_model) * 3, int(config.predictor_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.predictor_hidden_dim), int(config.delta_class_count)),
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        prev_latent: Tensor,
        prev_base_pitch: Tensor,
    ) -> Tensor:
        """Return logits over base-pitch delta classes."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens].")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape.")
        token_count = int(input_ids.shape[1])
        if token_count > int(self.config.max_context_tokens):
            raise ValueError(
                f"input sequence length {token_count} exceeds max_context_tokens={self.config.max_context_tokens}."
            )
        positions = torch.arange(token_count, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = self._causal_mask(token_count, input_ids.device)
        padding_mask = attention_mask <= 0
        encoded = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        pooled = self.context_norm(self._pool_context(encoded, attention_mask))
        previous_latent = self.prev_latent_proj(prev_latent)
        base = prev_base_pitch.float().reshape(-1, 1) / max(1.0, float(self.config.base_pitch_scale))
        previous_base = self.prev_base_pitch_proj(base)
        return self.delta_head(torch.cat([pooled, previous_latent, previous_base], dim=-1))

    def loss(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        prev_latent: Tensor,
        prev_base_pitch: Tensor,
        target_delta_class: Tensor,
        target_delta: Tensor,
    ) -> Dict[str, Tensor]:
        """Return cross-entropy loss and delta diagnostics."""
        logits = self(input_ids, attention_mask, prev_latent, prev_base_pitch)
        target = target_delta_class.long()
        loss = torch.nn.functional.cross_entropy(logits, target)
        pred_class = torch.argmax(logits, dim=-1)
        accuracy = (pred_class == target).float().mean()
        pred_delta = pred_class + int(self.config.delta_min)
        true_delta = torch.clamp(target_delta.long(), int(self.config.delta_min), int(self.config.delta_max))
        semitone_abs_error = torch.abs(pred_delta.float() - true_delta.float()).mean()
        return {
            "loss": loss,
            "accuracy": accuracy.detach(),
            "semitone_abs_error": semitone_abs_error.detach(),
        }

    def _causal_mask(self, length: int, device: torch.device) -> Tensor:
        """Return a standard causal attention mask."""
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def _last_valid_hidden(self, encoded: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool the hidden state at the last non-padding token."""
        lengths = torch.sum(attention_mask > 0, dim=1).long()
        lengths = torch.clamp(lengths, min=1)
        batch_indices = torch.arange(encoded.shape[0], device=encoded.device)
        return encoded[batch_indices, lengths - 1]

    def _pool_context(self, encoded: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool token-level hidden states into one context vector."""
        if self.context_pooling == "last":
            return self._last_valid_hidden(encoded, attention_mask)
        valid = (attention_mask > 0).unsqueeze(-1)
        if self.context_pooling == "mean":
            masked = encoded * valid.to(encoded.dtype)
            denom = torch.sum(valid.to(encoded.dtype), dim=1).clamp_min(1.0)
            return masked.sum(dim=1) / denom
        if self.context_pool_score is None:
            raise RuntimeError("context_pool_score is not initialized.")
        scores = self.context_pool_score(encoded).squeeze(-1)
        scores = scores.masked_fill(attention_mask <= 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        return torch.sum(encoded * weights, dim=1)


@dataclass(frozen=True)
class AlignedRemiMotionModelConfig:
    """Configuration for aligned REMI / latent / register-offset bar modeling."""

    vocab_size: int
    pad_token_id: int
    latent_dim: int = 32
    d_model: int = 256
    token_layers: int = 2
    bar_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    context_bars: int = 8
    max_bar_tokens: int = 192
    predictor_hidden_dim: int = 512
    context_pooling: str = "attention"
    register_offset_min: int = -24
    register_offset_max: int = 24
    register_offset_scale: float = 24.0

    @property
    def register_offset_class_count(self) -> int:
        """Return the number of song-anchor-relative register classes."""
        return int(self.register_offset_max) - int(self.register_offset_min) + 1

    def register_offset_to_class(self, offset: int) -> int:
        """Convert a clipped song-anchor-relative register offset to a class id."""
        clipped = max(int(self.register_offset_min), min(int(self.register_offset_max), int(offset)))
        return int(clipped - int(self.register_offset_min))

    def class_to_register_offset(self, class_id: int) -> int:
        """Convert a class id back to a song-anchor-relative register offset."""
        return int(class_id) + int(self.register_offset_min)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe config values."""
        return asdict(self)


class AlignedRemiMotionPredictor(nn.Module):
    """Bar-aligned REMI, latent, and song-relative register model with joint outputs."""

    def __init__(self, config: AlignedRemiMotionModelConfig) -> None:
        super().__init__()
        self.config = config
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
        self.token_encoder = nn.TransformerEncoder(token_layer, num_layers=int(config.token_layers))
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
        self.register_offset_proj = nn.Sequential(
            nn.Linear(1, int(config.d_model)),
            nn.GELU(),
        )
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
        self.bar_encoder = nn.TransformerEncoder(bar_layer, num_layers=int(config.bar_layers))
        self.bar_norm = nn.LayerNorm(int(config.d_model))
        self.latent_head = nn.Sequential(
            nn.Linear(int(config.d_model), int(config.predictor_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.predictor_hidden_dim), int(config.latent_dim)),
        )
        self.register_offset_head = nn.Sequential(
            nn.Linear(int(config.d_model), int(config.predictor_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.predictor_hidden_dim), int(config.register_offset_class_count)),
        )

    def forward(
        self,
        context_input_ids: Tensor,
        context_attention_mask: Tensor,
        context_latents: Tensor,
        context_register_offsets: Tensor,
        context_bar_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict next latent vector and song-anchor-relative register offset."""
        if context_input_ids.ndim != 3:
            raise ValueError("context_input_ids must have shape [batch, bars, tokens].")
        if context_attention_mask.shape != context_input_ids.shape:
            raise ValueError("context_attention_mask must match context_input_ids.")
        batch_size, bar_count, token_count = context_input_ids.shape
        if bar_count > int(self.config.context_bars):
            raise ValueError(f"context bars {bar_count} exceeds context_bars={self.config.context_bars}.")
        if token_count > int(self.config.max_bar_tokens):
            raise ValueError(f"bar token length {token_count} exceeds max_bar_tokens={self.config.max_bar_tokens}.")

        flat_ids = context_input_ids.reshape(batch_size * bar_count, token_count)
        flat_mask = context_attention_mask.reshape(batch_size * bar_count, token_count)

        # Early contexts contain padding-only bars. Passing an all-padding row to
        # MultiheadAttention produces an undefined all-masked softmax in inference.
        # Encode only real bars and represent absent history as a zero REMI vector.
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
        offset = context_register_offsets.float().reshape(batch_size, bar_count, 1)
        offset = offset / max(1.0, float(self.config.register_offset_scale))
        register_bar = self.register_offset_proj(offset)
        fused = self.stream_fusion(torch.cat([remi_bar, latent_bar, register_bar], dim=-1))
        bar_positions = torch.arange(bar_count, device=context_input_ids.device).unsqueeze(0).expand(batch_size, bar_count)
        fused = fused + self.bar_position_embedding(bar_positions)
        pooled = self._encode_valid_bar_context(fused, context_bar_mask)
        return self.latent_head(pooled), self.register_offset_head(pooled)

    def register_offset_loss(
        self,
        register_offset_logits: Tensor,
        target_register_offset_class: Tensor,
        target_register_offset: Tensor,
    ) -> Dict[str, Tensor]:
        """Return direct song-anchor-relative register classification loss."""
        target = target_register_offset_class.long()
        raw_loss = torch.nn.functional.cross_entropy(register_offset_logits, target)
        loss = raw_loss / math.log(max(2, int(self.config.register_offset_class_count)))
        pred_class = torch.argmax(register_offset_logits, dim=-1)
        accuracy = (pred_class == target).float().mean()
        predicted_offset = pred_class + int(self.config.register_offset_min)
        true_offset = torch.clamp(
            target_register_offset.long(),
            int(self.config.register_offset_min),
            int(self.config.register_offset_max),
        )
        semitone_abs_error = torch.abs(predicted_offset.float() - true_offset.float()).mean()
        return {
            "loss": loss,
            "raw_ce": raw_loss.detach(),
            "accuracy": accuracy.detach(),
            "semitone_abs_error": semitone_abs_error.detach(),
        }

    def _pool_tokens(self, encoded: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool token-level REMI states into one bar embedding."""
        valid = (attention_mask > 0).unsqueeze(-1)
        if self.context_pooling == "last":
            lengths = torch.sum(attention_mask > 0, dim=1).long().clamp_min(1)
            batch_indices = torch.arange(encoded.shape[0], device=encoded.device)
            return encoded[batch_indices, lengths - 1]
        if self.context_pooling == "mean":
            masked = encoded * valid.to(encoded.dtype)
            denom = torch.sum(valid.to(encoded.dtype), dim=1).clamp_min(1.0)
            return masked.sum(dim=1) / denom
        if self.token_pool_score is None:
            raise RuntimeError("token_pool_score is not initialized.")
        scores = self.token_pool_score(encoded).squeeze(-1)
        scores = scores.masked_fill(attention_mask <= 0, -1.0e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        return torch.sum(encoded * weights, dim=1)

    def _encode_valid_bar_context(self, fused: Tensor, bar_mask: Tensor) -> Tensor:
        """Causally encode only real, right-aligned bar histories.

        A padded query before the first real bar cannot attend to any key once the
        causal and padding masks are combined. Some inference kernels return NaN
        for that all-masked attention row. Grouping by valid history length avoids
        constructing those rows while retaining the original recency positions.
        """
        lengths = torch.sum(bar_mask > 0, dim=1).long()
        if torch.any(lengths < 1):
            raise ValueError("Each REMI motion sample needs at least one valid context bar.")

        batch_size, bar_count, d_model = fused.shape
        pooled = torch.zeros((batch_size, d_model), dtype=fused.dtype, device=fused.device)
        for length in torch.unique(lengths).tolist():
            valid_length = int(length)
            row_indices = torch.nonzero(lengths == valid_length, as_tuple=False).squeeze(1)
            sequence = fused[row_indices, bar_count - valid_length:, :]
            causal_mask = self._causal_mask(valid_length, fused.device)
            encoded = self.bar_encoder(sequence, mask=causal_mask)
            pooled = pooled.index_copy(0, row_indices, encoded[:, -1, :])
        return self.bar_norm(pooled)

    def _causal_mask(self, length: int, device: torch.device) -> Tensor:
        """Return a standard causal attention mask."""
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)
