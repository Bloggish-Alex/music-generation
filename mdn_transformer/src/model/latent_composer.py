#!/usr/bin/env python3
"""Latent composer models for deterministic next-bar representation prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from common.config_loader import ConfigView


@dataclass(frozen=True)
class AnchorMotionComposerConfig:
    """Configuration for the Anchor/Motion Composer."""

    latent_dim: int = 32
    feature_dim: int = 27
    context_bars: int = 32
    hidden_dim: int = 512
    n_layers: int = 2
    dropout: float = 0.2
    composer_hidden_dim: int = 512
    composer_layers: int = 2
    condition_enabled: bool = True
    form_vocab_size: int = 2
    action_vocab_size: int = 2
    composer_vocab_size: int = 2
    position_vocab_size: int = 8
    form_embedding_dim: int = 8
    action_embedding_dim: int = 8
    composer_embedding_dim: int = 8
    position_embedding_dim: int = 8

    @property
    def representation_dim(self) -> int:
        """Return latent plus explicit-feature dimension."""
        return int(self.latent_dim) + int(self.feature_dim)

    @property
    def step_dim(self) -> int:
        """Return one context-step input width: state, delta, state mask, delta mask."""
        return int(2 * self.representation_dim + 2)

    @property
    def condition_dim(self) -> int:
        """Return late-fusion condition width."""
        if not bool(self.condition_enabled):
            return 0
        return (
            int(self.form_embedding_dim)
            + int(self.action_embedding_dim)
            + int(self.composer_embedding_dim)
            + int(self.position_embedding_dim)
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AnchorMotionComposerConfig":
        """Build config from style defaults."""
        section = ConfigView(config).section("anchor_motion_composer")
        fallback = ConfigView(config).section("latent_transformer")
        return cls(
            latent_dim=int(section.get("latent_dim", fallback.get("latent_dim", 32))),
            feature_dim=int(section.get("feature_dim", 27)),
            context_bars=int(section.get("context_bars", fallback.get("context_bars", 32))),
            hidden_dim=int(section.get("hidden_dim", 512)),
            n_layers=int(section.get("n_layers", 2)),
            dropout=float(section.get("dropout", 0.2)),
            composer_hidden_dim=int(section.get("composer_hidden_dim", 512)),
            composer_layers=int(section.get("composer_layers", 2)),
            condition_enabled=bool(section.get("condition_enabled", True)),
            form_vocab_size=int(section.get("form_vocab_size", 2)),
            action_vocab_size=int(section.get("action_vocab_size", 2)),
            composer_vocab_size=int(section.get("composer_vocab_size", 2)),
            position_vocab_size=int(section.get("position_vocab_size", 8)),
            form_embedding_dim=int(section.get("form_embedding_dim", 8)),
            action_embedding_dim=int(section.get("action_embedding_dim", 8)),
            composer_embedding_dim=int(section.get("composer_embedding_dim", 8)),
            position_embedding_dim=int(section.get("position_embedding_dim", 8)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe config."""
        return dict(self.__dict__)


def composer_mlp(input_dim: int, output_dim: int, hidden_dim: int, layers: int, dropout: float) -> nn.Sequential:
    """Build the final Composer MLP."""
    modules: list[nn.Module] = []
    current_dim = int(input_dim)
    for _ in range(max(1, int(layers))):
        modules.extend([
            nn.Linear(current_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        ])
        current_dim = int(hidden_dim)
    modules.append(nn.Linear(current_dim, int(output_dim)))
    return nn.Sequential(*modules)


class AnchorMotionComposer(nn.Module):
    """Predict hidden anchor and motion variables, then compose the next representation.

    The model receives a sequence of normalized hybrid representation steps:
    [state, adjacent_delta, state_mask, delta_mask]. Only the final composed
    representation is supervised.
    """

    def __init__(self, config: AnchorMotionComposerConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        self.input_proj = nn.Linear(int(config.step_dim), hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, int(config.context_bars), hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self._attention_heads(hidden_dim),
            dim_feedforward=hidden_dim * 2,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=int(config.n_layers))
        if bool(config.condition_enabled):
            self.form_embedding = nn.Embedding(int(config.form_vocab_size), int(config.form_embedding_dim))
            self.action_embedding = nn.Embedding(int(config.action_vocab_size), int(config.action_embedding_dim))
            self.composer_embedding = nn.Embedding(int(config.composer_vocab_size), int(config.composer_embedding_dim))
            self.position_embedding = nn.Embedding(int(config.position_vocab_size), int(config.position_embedding_dim))
        else:
            self.form_embedding = None
            self.action_embedding = None
            self.composer_embedding = None
            self.position_embedding = None
        conditioned_hidden_dim = hidden_dim + int(config.condition_dim)
        self.anchor_head = nn.Linear(conditioned_hidden_dim, int(config.representation_dim))
        self.motion_head = nn.Linear(conditioned_hidden_dim, int(config.representation_dim))
        self.composer = composer_mlp(
            input_dim=int(3 * config.representation_dim + config.condition_dim),
            output_dim=int(config.representation_dim),
            hidden_dim=int(config.composer_hidden_dim),
            layers=int(config.composer_layers),
            dropout=float(config.dropout),
        )

    def forward(
        self,
        context_steps: torch.Tensor,
        current: torch.Tensor,
        condition_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return anchor, motion, and composed normalized representation."""
        hidden = self.input_proj(context_steps) + self.position[:, : context_steps.shape[1], :]
        causal_mask = torch.triu(
            torch.full((context_steps.shape[1], context_steps.shape[1]), float("-inf"), device=context_steps.device),
            diagonal=1,
        )
        encoded = self.trunk(hidden, mask=causal_mask)
        pooled = encoded[:, -1, :]
        condition = self._condition_vector(condition_ids, pooled.device, pooled.shape[0])
        conditioned = torch.cat([pooled, condition], dim=1) if condition.shape[1] else pooled
        anchor = self.anchor_head(conditioned)
        motion = self.motion_head(conditioned)
        composed_input = torch.cat([current, anchor, motion, condition], dim=1) if condition.shape[1] else torch.cat([current, anchor, motion], dim=1)
        composed = self.composer(composed_input)
        return {
            "anchor": anchor,
            "motion": motion,
            "composed": composed,
        }

    def _condition_vector(self, condition_ids: Optional[torch.Tensor], device: torch.device, batch_size: int) -> torch.Tensor:
        """Embed [form, action, composer, position] IDs for late fusion."""
        if not bool(self.config.condition_enabled):
            return torch.zeros((batch_size, 0), dtype=torch.float32, device=device)
        if condition_ids is None:
            condition_ids = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
        condition_ids = condition_ids.to(device=device, dtype=torch.long)
        form_ids = condition_ids[:, 0].clamp(min=0, max=int(self.config.form_vocab_size) - 1)
        action_ids = condition_ids[:, 1].clamp(min=0, max=int(self.config.action_vocab_size) - 1)
        composer_ids = condition_ids[:, 2].clamp(min=0, max=int(self.config.composer_vocab_size) - 1)
        position_ids = condition_ids[:, 3].clamp(min=0, max=int(self.config.position_vocab_size) - 1)
        return torch.cat([
            self.form_embedding(form_ids),
            self.action_embedding(action_ids),
            self.composer_embedding(composer_ids),
            self.position_embedding(position_ids),
        ], dim=1)

    @staticmethod
    def _attention_heads(hidden_dim: int) -> int:
        """Choose a valid head count for small experiments."""
        for candidate in (8, 4, 2):
            if int(hidden_dim) % candidate == 0:
                return candidate
        return 1
