#!/usr/bin/env python3
"""Memory Latent Transformer for non-parametric next-bar retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config_loader import ConfigView
from model.latent_transformer import SinusoidalPositionEncoding
from model.theme_fusion import ThemeFusionAdapterConfig, build_theme_fusion_adapter


@dataclass(frozen=True)
class MemoryLatentTransformerConfig:
    """Configuration for a Transformer that outputs memory retrieval queries."""

    latent_dim: int = 32
    context_bars: int = 32
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    action_vocab_size: int = 8
    action_embedding_dim: int = 8
    position_vocab_size: int = 8
    position_embedding_dim: int = 8
    query_dim: int = 32
    theme_fusion_enabled: bool = False
    theme_embedding_dim: int = 64
    theme_project_dim: int = 16
    theme_gate_init: float = 0.01
    theme_fusion_mode: str = "film_cross_attention_pi"
    theme_dropout: float = 0.3
    theme_embedding_noise_std: float = 0.03
    theme_token_bars: int = 8
    theme_cross_attention_heads: int = 4

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MemoryLatentTransformerConfig":
        """Build model config from style config."""
        section = ConfigView(config).section("memory_latent_transformer")
        fallback = ConfigView(config).section("latent_transformer")
        theme_fusion = ConfigView(config).section("theme_fusion")
        latent_dim = int(section.get("latent_dim", fallback.get("latent_dim", 32)))
        return cls(
            latent_dim=latent_dim,
            context_bars=int(section.get("context_bars", fallback.get("context_bars", 32))),
            d_model=int(section.get("d_model", fallback.get("d_model", 128))),
            n_layers=int(section.get("n_layers", fallback.get("n_layers", 4))),
            n_heads=int(section.get("n_heads", fallback.get("n_heads", 4))),
            dropout=float(section.get("dropout", fallback.get("dropout", 0.1))),
            action_vocab_size=int(section.get("action_vocab_size", fallback.get("action_vocab_size", 8))),
            action_embedding_dim=int(section.get("action_embedding_dim", fallback.get("action_embedding_dim", 8))),
            position_vocab_size=int(section.get("position_vocab_size", fallback.get("position_vocab_size", 8))),
            position_embedding_dim=int(section.get("position_embedding_dim", fallback.get("position_embedding_dim", 8))),
            query_dim=int(section.get("query_dim", latent_dim)),
            theme_fusion_enabled=bool(theme_fusion.get("enabled", False)),
            theme_embedding_dim=int(theme_fusion.get("embedding_dim", 64)),
            theme_project_dim=int(theme_fusion.get("project_dim", 16)),
            theme_gate_init=float(theme_fusion.get("gate_init", 0.01)),
            theme_fusion_mode=str(theme_fusion.get("mode", "film_cross_attention_pi")),
            theme_dropout=float(theme_fusion.get("theme_dropout", 0.3)),
            theme_embedding_noise_std=float(theme_fusion.get("embedding_noise_std", 0.03)),
            theme_token_bars=int(theme_fusion.get("token_bars", theme_fusion.get("theme_bars", 8))),
            theme_cross_attention_heads=int(theme_fusion.get("cross_attention_heads", 4)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe config values."""
        return dict(self.__dict__)


class MemoryLatentTransformer(nn.Module):
    """Causal Transformer that predicts a query for nearest real latent bars."""

    def __init__(self, config: MemoryLatentTransformerConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = int(config.latent_dim) + int(config.action_embedding_dim) + int(config.position_embedding_dim)
        self.action_embedding = nn.Embedding(int(config.action_vocab_size), int(config.action_embedding_dim))
        self.position_embedding = nn.Embedding(int(config.position_vocab_size), int(config.position_embedding_dim))
        self.input_projection = nn.Linear(input_dim, int(config.d_model))
        self.sequence_position = SinusoidalPositionEncoding(int(config.d_model), int(config.context_bars))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(config.d_model),
            nhead=int(config.n_heads),
            dim_feedforward=int(config.d_model) * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(config.n_layers))
        cond_dim = int(config.action_embedding_dim) + int(config.position_embedding_dim)
        self.output_condition = nn.Sequential(
            nn.Linear(int(config.d_model) + cond_dim, int(config.d_model)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
        )
        self.theme_fusion = build_theme_fusion_adapter(ThemeFusionAdapterConfig(
            enabled=bool(config.theme_fusion_enabled),
            mode=str(config.theme_fusion_mode),
            target="all",
            embedding_dim=int(config.theme_embedding_dim),
            latent_dim=int(config.latent_dim),
            d_model=int(config.d_model),
            project_dim=int(config.theme_project_dim),
            gate_init=float(config.theme_gate_init),
            dropout=float(config.dropout),
            theme_dropout=float(config.theme_dropout),
            embedding_noise_std=float(config.theme_embedding_noise_std),
            token_bars=int(config.theme_token_bars),
            cross_attention_heads=int(config.theme_cross_attention_heads),
        ))
        self.query_head = nn.Sequential(
            nn.LayerNorm(int(config.d_model)),
            nn.Linear(int(config.d_model), int(config.query_dim)),
        )

    def forward(
        self,
        context_mu: torch.Tensor,
        context_action_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        target_action_ids: torch.Tensor,
        target_position_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        theme_embedding: Optional[torch.Tensor] = None,
        theme_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return retrieval query vectors."""
        action_emb = self.action_embedding(context_action_ids)
        position_emb = self.position_embedding(context_position_ids)
        x = torch.cat([context_mu, action_emb, position_emb], dim=-1)
        x = self.sequence_position(self.input_projection(x))
        encoded = self.transformer(x, mask=self._causal_mask(x.shape[1], x.device), src_key_padding_mask=padding_mask)
        pooled = self._last_valid_state(encoded, padding_mask)
        target_cond = torch.cat([self.action_embedding(target_action_ids), self.position_embedding(target_position_ids)], dim=-1)
        hidden = self.output_condition(torch.cat([pooled, target_cond], dim=-1))
        fused = self.theme_fusion(hidden, theme_embedding=theme_embedding, theme_tokens=theme_tokens)
        return self.query_head(fused.hidden)

    def theme_fusion_diagnostics(self) -> Dict[str, Any]:
        """Return runtime diagnostics from the theme adapter."""
        return self.theme_fusion.runtime_diagnostics()

    def _last_valid_state(self, encoded: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Return the final non-padded history state for every row."""
        if padding_mask is None:
            return encoded[:, -1, :]
        positions = torch.arange(encoded.shape[1], device=encoded.device).unsqueeze(0)
        indices = torch.where(~padding_mask, positions, torch.zeros_like(positions)).max(dim=1).values
        return encoded[torch.arange(encoded.shape[0], device=encoded.device), indices, :]

    def _causal_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        """Return upper-triangular causal attention mask."""
        return torch.triu(torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=device), diagonal=1)


class InBatchMemoryContrastiveLoss(nn.Module):
    """Contrastive next-bar loss over in-batch memory candidates."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, query: torch.Tensor, target_mu: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return cross-entropy loss and retrieval metrics."""
        query_norm = F.normalize(query, dim=-1)
        target_norm = F.normalize(target_mu, dim=-1)
        logits = torch.matmul(query_norm, target_norm.transpose(0, 1)) / max(1.0e-6, float(self.temperature))
        labels = torch.arange(query.shape[0], dtype=torch.long, device=query.device)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            ranks = torch.argsort(logits, dim=-1, descending=True)
            correct = ranks == labels.unsqueeze(1)
            top1 = correct[:, :1].any(dim=1).float().mean()
            top5 = correct[:, : min(5, logits.shape[1])].any(dim=1).float().mean()
            mrr = (1.0 / (correct.float().argmax(dim=1).float() + 1.0)).mean()
            positive = logits.diag()
            masked = logits.masked_fill(torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device), -1.0e9)
            margin = positive - masked.max(dim=1).values
        return {
            "loss": loss,
            "top1": top1,
            "top5": top5,
            "mrr": mrr,
            "positive_margin": margin.mean(),
        }
