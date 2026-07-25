#!/usr/bin/env python3
"""Transformer-XL style recurrent context for trajectory diffusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from model.trajectory_diffusion import JointTrajectoryDiffusion, TrajectoryDiffusionConfig


@dataclass
class RecurrentBarKVCache:
    """Detached per-layer causal K/V memory for one batch of song streams."""

    keys: List[Tensor]
    values: List[Tensor]
    valid_mask: Tensor

    def detach(self) -> "RecurrentBarKVCache":
        return RecurrentBarKVCache(
            keys=[item.detach() for item in self.keys],
            values=[item.detach() for item in self.values],
            valid_mask=self.valid_mask.detach(),
        )


class RotaryBarPositionEncoding(nn.Module):
    """Apply RoPE to bar-level attention queries and keys."""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        if int(head_dim) % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")
        inverse_frequency = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / float(head_dim)))
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, values: Tensor, positions: Tensor) -> Tensor:
        if values.ndim != 4:
            raise ValueError("RoPE values must have shape [batch, heads, bars, head_dim].")
        if positions.shape != (values.shape[0], values.shape[2]):
            raise ValueError("RoPE positions must align with the bar axis.")
        angles = positions.to(dtype=values.dtype).unsqueeze(-1) * self.inverse_frequency.to(dtype=values.dtype)
        cosine = torch.cos(angles).unsqueeze(1)
        sine = torch.sin(angles).unsqueeze(1)
        even = values[..., 0::2]
        odd = values[..., 1::2]
        rotated = torch.stack([even * cosine - odd * sine, even * sine + odd * cosine], dim=-1)
        return rotated.flatten(-2)


class RecurrentBarAttentionLayer(nn.Module):
    """One causal Transformer layer that accepts prior detached K/V entries."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if int(d_model) % int(n_heads) != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model) // int(n_heads)
        self.norm_attention = nn.LayerNorm(self.d_model)
        self.query = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key = nn.Linear(self.d_model, self.d_model, bias=False)
        self.value = nn.Linear(self.d_model, self.d_model, bias=False)
        self.output = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.norm_feedforward = nn.LayerNorm(self.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model * 4, self.d_model),
            nn.Dropout(float(dropout)),
        )
        self.rope = RotaryBarPositionEncoding(self.head_dim)

    def forward(
        self,
        hidden: Tensor,
        cached_keys: Tensor,
        cached_values: Tensor,
        cached_valid: Tensor,
        positions: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, bar_count, _ = hidden.shape
        normalized = self.norm_attention(hidden)
        query = self._split_heads(self.query(normalized))
        key = self._split_heads(self.key(normalized))
        value = self._split_heads(self.value(normalized))
        query = self.rope(query, positions)
        key = self.rope(key, positions)
        all_keys = torch.cat([cached_keys, key], dim=2)
        all_values = torch.cat([cached_values, value], dim=2)
        attention_mask = self._attention_mask(cached_valid, bar_count, hidden.device, hidden.dtype)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query, all_keys, all_values, attn_mask=attention_mask, dropout_p=float(self.dropout.p) if self.training else 0.0,
        )
        attended = self._merge_heads(attended)
        hidden = hidden + self.dropout(self.output(attended))
        hidden = hidden + self.feedforward(self.norm_feedforward(hidden))
        return hidden, key, value

    def _attention_mask(self, cached_valid: Tensor, bar_count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        batch_size, memory_bars = cached_valid.shape
        total = memory_bars + int(bar_count)
        mask = torch.zeros((batch_size, 1, int(bar_count), total), device=device, dtype=dtype)
        if memory_bars > 0:
            invalid_memory = ~cached_valid.bool()
            mask[:, :, :, :memory_bars] = invalid_memory[:, None, None, :].to(dtype) * torch.finfo(dtype).min
        future = torch.triu(torch.ones((bar_count, bar_count), device=device, dtype=torch.bool), diagonal=1)
        mask[:, :, :, memory_bars:] = future[None, None].to(dtype) * torch.finfo(dtype).min
        return mask

    def _split_heads(self, values: Tensor) -> Tensor:
        return values.reshape(values.shape[0], values.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, values: Tensor) -> Tensor:
        return values.transpose(1, 2).reshape(values.shape[0], values.shape[2], self.d_model)


class RecurrentBarTransformer(nn.Module):
    """Segment-level recurrent Transformer with bounded detached K/V memory."""

    def __init__(self, d_model: int, layers: int, n_heads: int, dropout: float, memory_bars: int, gradient_checkpointing: bool) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.memory_bars = int(memory_bars)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.layers = nn.ModuleList([
            RecurrentBarAttentionLayer(self.d_model, int(n_heads), float(dropout))
            for _ in range(int(layers))
        ])
        self.output_norm = nn.LayerNorm(self.d_model)

    def empty_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> RecurrentBarKVCache:
        keys = [torch.empty((batch_size, layer.n_heads, 0, layer.head_dim), device=device, dtype=dtype) for layer in self.layers]
        values = [torch.empty((batch_size, layer.n_heads, 0, layer.head_dim), device=device, dtype=dtype) for layer in self.layers]
        valid = torch.empty((batch_size, 0), device=device, dtype=torch.bool)
        return RecurrentBarKVCache(keys=keys, values=values, valid_mask=valid)

    def forward(self, hidden: Tensor, cache: RecurrentBarKVCache, positions: Tensor) -> Tuple[Tensor, RecurrentBarKVCache]:
        if len(cache.keys) != len(self.layers) or len(cache.values) != len(self.layers):
            raise ValueError("Recurrent cache layer count does not match the Transformer.")
        if cache.valid_mask.shape[0] != hidden.shape[0]:
            raise ValueError("Recurrent cache batch does not match the current segment.")
        current = hidden
        current_keys: List[Tensor] = []
        current_values: List[Tensor] = []
        for index, layer in enumerate(self.layers):
            if self.training and self.gradient_checkpointing and current.requires_grad:
                current, key, value = checkpoint(
                    lambda segment: layer(segment, cache.keys[index], cache.values[index], cache.valid_mask, positions),
                    current,
                    use_reentrant=False,
                )
            else:
                current, key, value = layer(current, cache.keys[index], cache.values[index], cache.valid_mask, positions)
            current_keys.append(key)
            current_values.append(value)
        next_valid = torch.cat([
            cache.valid_mask,
            torch.ones((hidden.shape[0], hidden.shape[1]), device=hidden.device, dtype=torch.bool),
        ], dim=1)
        next_keys = [torch.cat([cache.keys[index], key], dim=2)[:, :, -self.memory_bars:] for index, key in enumerate(current_keys)]
        next_values = [torch.cat([cache.values[index], value], dim=2)[:, :, -self.memory_bars:] for index, value in enumerate(current_values)]
        next_cache = RecurrentBarKVCache(
            keys=next_keys,
            values=next_values,
            valid_mask=next_valid[:, -self.memory_bars:],
        )
        return self.output_norm(current), next_cache


class RecurrentThreeStreamContextEncoder(nn.Module):
    """Encode current bar segments while carrying detached per-layer bar memory."""

    def __init__(self, config: TrajectoryDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(int(config.vocab_size), int(config.d_model), padding_idx=int(config.pad_token_id))
        self.token_position_embedding = nn.Embedding(int(config.max_bar_tokens), int(config.d_model))
        token_layer = nn.TransformerEncoderLayer(
            d_model=int(config.d_model), nhead=int(config.n_heads), dim_feedforward=int(config.d_model) * 4,
            dropout=float(config.dropout), activation="gelu", batch_first=True, norm_first=True,
        )
        # norm_first=True intentionally keeps the pre-norm Transformer design.
        # PyTorch cannot use its nested-tensor fast path for that combination.
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
        self.latent_projection = nn.Sequential(nn.LayerNorm(int(config.latent_dim)), nn.Linear(int(config.latent_dim), int(config.d_model)), nn.GELU())
        self.register_projection = nn.Sequential(nn.Linear(1, int(config.d_model)), nn.GELU())
        self.stream_fusion = nn.Sequential(
            nn.Linear(int(config.d_model) * 3, int(config.d_model)), nn.GELU(), nn.LayerNorm(int(config.d_model)),
        )
        self.bar_transformer = RecurrentBarTransformer(
            d_model=int(config.d_model), layers=int(config.bar_layers), n_heads=int(config.n_heads), dropout=float(config.dropout),
            memory_bars=int(config.memory_bars), gradient_checkpointing=bool(config.gradient_checkpointing),
        )

    def empty_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> RecurrentBarKVCache:
        return self.bar_transformer.empty_cache(batch_size, device, dtype)

    def forward_segment(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        latents: Tensor,
        register_offsets: Tensor,
        cache: RecurrentBarKVCache,
        positions: Tensor,
    ) -> Tuple[Tensor, RecurrentBarKVCache]:
        if input_ids.ndim != 3 or attention_mask.shape != input_ids.shape:
            raise ValueError("REMI segment tensors must have shape [batch, bars, tokens].")
        batch_size, bar_count, token_count = input_ids.shape
        if latents.shape != (batch_size, bar_count, int(self.config.latent_dim)):
            raise ValueError("Latent segment must align with REMI bars.")
        if register_offsets.shape != (batch_size, bar_count) or positions.shape != (batch_size, bar_count):
            raise ValueError("Register offsets and positions must align with segment bars.")
        flattened_ids = input_ids.reshape(batch_size * bar_count, token_count)
        flattened_mask = attention_mask.reshape(batch_size * bar_count, token_count)
        has_tokens = torch.any(flattened_mask > 0, dim=1)
        remi = torch.zeros((batch_size * bar_count, int(self.config.d_model)), dtype=self.token_embedding.weight.dtype, device=input_ids.device)
        if bool(torch.any(has_tokens).item()):
            valid_ids = flattened_ids[has_tokens]
            valid_mask = flattened_mask[has_tokens]
            token_positions = torch.arange(token_count, device=input_ids.device).unsqueeze(0).expand_as(valid_ids)
            encoded = self.token_encoder(
                self.token_embedding(valid_ids) + self.token_position_embedding(token_positions),
                src_key_padding_mask=valid_mask <= 0,
            )
            remi = remi.index_copy(0, torch.nonzero(has_tokens, as_tuple=False).squeeze(1), self.token_norm(self._pool_tokens(encoded, valid_mask)))
        remi = remi.reshape(batch_size, bar_count, int(self.config.d_model))
        latent = self.latent_projection(latents)
        register = self.register_projection(register_offsets.float().unsqueeze(-1) / max(1.0, float(self.config.register_offset_scale)))
        fused = self.stream_fusion(torch.cat([remi, latent, register], dim=-1))
        encoded, next_cache = self.bar_transformer(fused, cache, positions)
        return encoded[:, -1, :], next_cache

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


class RecurrentTrajectoryDiffusion(JointTrajectoryDiffusion):
    """Joint diffusion whose condition comes from recurrent bar-level K/V memory."""

    def __init__(self, config: TrajectoryDiffusionConfig) -> None:
        super().__init__(config)
        self.context_encoder = RecurrentThreeStreamContextEncoder(config)

    def empty_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> RecurrentBarKVCache:
        return self.context_encoder.empty_cache(batch_size, device, dtype)

    def encode_segment(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        latents: Tensor,
        register_offsets: Tensor,
        cache: RecurrentBarKVCache,
        positions: Tensor,
    ) -> Tuple[Tensor, RecurrentBarKVCache]:
        return self.context_encoder.forward_segment(input_ids, attention_mask, latents, register_offsets, cache, positions)
