#!/usr/bin/env python3
"""Frozen Theme Encoder embedding provider for downstream sequence models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from common.config_loader import ConfigView
from model.theme_encoder import BiLSTMAttentionThemeEncoder, ThemeEncoderConfig


@dataclass(frozen=True)
class ThemeFusionConfig:
    """Configuration for gated theme fusion in the latent Transformer."""

    enabled: bool = False
    mode: str = "film_pi"
    target: str = "pi"
    theme_encoder_path: str = ""
    theme_bars: int = 8
    embedding_dim: int = 64
    project_dim: int = 16
    gate_init: float = 0.01
    theme_dropout: float = 0.3
    embedding_noise_std: float = 0.03
    token_bars: int = 8
    cross_attention_heads: int = 4

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ThemeFusionConfig":
        """Build theme fusion config from style config."""
        section = ConfigView(config).section("theme_fusion")
        return cls(
            enabled=bool(section.get("enabled", False)),
            mode=str(section.get("mode", "film_pi")),
            target=str(section.get("target", "pi")),
            theme_encoder_path=str(section.get("theme_encoder_path", "")),
            theme_bars=int(section.get("theme_bars", 8)),
            embedding_dim=int(section.get("embedding_dim", 64)),
            project_dim=int(section.get("project_dim", 16)),
            gate_init=float(section.get("gate_init", 0.01)),
            theme_dropout=float(section.get("theme_dropout", 0.3)),
            embedding_noise_std=float(section.get("embedding_noise_std", 0.03)),
            token_bars=int(section.get("token_bars", section.get("theme_bars", 8))),
            cross_attention_heads=int(section.get("cross_attention_heads", 4)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe config values."""
        return dict(self.__dict__)


class FrozenThemeEmbeddingProvider:
    """Compute song-level opening theme embeddings with a frozen Theme Encoder."""

    def __init__(self, config: ThemeFusionConfig, device: str) -> None:
        self.config = config
        self.device = str(device)

    @classmethod
    def from_config(cls, config: Dict[str, Any], device: str) -> "FrozenThemeEmbeddingProvider":
        """Build provider from global config."""
        return cls(ThemeFusionConfig.from_config(config), device=device)

    def embeddings_by_song(self, mu: np.ndarray, rows: Sequence[Dict[str, Any]]) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Return frozen theme embeddings by song_id and diagnostics."""
        embeddings, _tokens, diagnostics = self.theme_contexts_by_song(mu, rows)
        return embeddings, diagnostics

    def theme_contexts_by_song(
        self,
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
    ) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
        """Return song-level theme embeddings and opening latent token sequences."""
        if not self.config.enabled:
            return {}, {}, {"enabled": False}
        needs_embedding = self._needs_embedding()
        model_path = Path(self.config.theme_encoder_path)
        model = None
        checkpoint: Dict[str, Any] = {}
        if needs_embedding:
            if not model_path.exists():
                raise FileNotFoundError(f"theme_fusion.theme_encoder_path does not exist: {model_path}")
            model, checkpoint = self._load_model(model_path)
        grouped = self._group_rows(rows)
        embeddings: Dict[str, np.ndarray] = {}
        token_sequences: Dict[str, np.ndarray] = {}
        skipped_short_embedding = 0
        skipped_short_tokens = 0
        n_embedding_bars = max(1, int(self.config.theme_bars))
        n_token_bars = max(1, int(self.config.token_bars))
        if model is not None:
            model.eval()
        with torch.no_grad():
            for song_id, indices in grouped.items():
                ordered = sorted(indices, key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx))))
                if len(ordered) >= n_token_bars:
                    token_sequences[str(song_id)] = mu[ordered[:n_token_bars]].astype(np.float32)
                else:
                    skipped_short_tokens += 1
                if needs_embedding:
                    if len(ordered) < n_embedding_bars:
                        skipped_short_embedding += 1
                        continue
                    assert model is not None
                    selected = ordered[:n_embedding_bars]
                    sequence = torch.from_numpy(mu[selected].astype(np.float32)).unsqueeze(0).to(self.device)
                    embedding = model(sequence).squeeze(0).detach().cpu().numpy().astype(np.float32)
                    embeddings[str(song_id)] = embedding
        norms = [float(np.linalg.norm(value)) for value in embeddings.values()]
        diagnostics = {
            "enabled": True,
            "mode": str(self.config.mode),
            "target": str(self.config.target),
            "theme_encoder_path": str(model_path) if needs_embedding else "",
            "theme_bars": int(n_embedding_bars),
            "token_bars": int(n_token_bars),
            "embedding_song_count": int(len(embeddings)),
            "token_sequence_song_count": int(len(token_sequences)),
            "skipped_short_embedding_song_count": int(skipped_short_embedding),
            "skipped_short_token_song_count": int(skipped_short_tokens),
            "embedding_dim": int(checkpoint.get("model_config", {}).get("embedding_dim", self.config.embedding_dim)),
            "token_dim": int(mu.shape[1]) if mu.ndim == 2 else 0,
            "embedding_norm_mean": float(np.mean(norms)) if norms else 0.0,
            "embedding_norm_std": float(np.std(norms)) if norms else 0.0,
        }
        return embeddings, token_sequences, diagnostics

    def _needs_embedding(self) -> bool:
        """Return whether the selected fusion mode needs frozen theme embeddings."""
        return str(self.config.mode).lower() in {"film_pi", "film_cross_attention_pi"}

    def _load_model(self, path: Path) -> tuple[BiLSTMAttentionThemeEncoder, Dict[str, Any]]:
        """Load and freeze a trained Theme Encoder checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        config = ThemeEncoderConfig(**checkpoint["model_config"])
        model = BiLSTMAttentionThemeEncoder(config).to(self.device)
        model.load_state_dict(checkpoint["state_dict"])
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model, checkpoint

    def _group_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Group latent index rows by song_id."""
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        return grouped
