#!/usr/bin/env python3
"""Latent Transformer with a Mixture Density Network prediction head."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config_loader import ConfigView
from model.theme_fusion import ThemeFusionAdapterConfig, build_theme_fusion_adapter


@dataclass(frozen=True)
class LatentTransformerConfig:
    """Configuration for the latent autoregressive Transformer."""

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
    theme_fusion_enabled: bool = False
    theme_embedding_dim: int = 64
    theme_project_dim: int = 16
    theme_gate_init: float = 0.01
    theme_fusion_mode: str = "film_pi"
    theme_fusion_target: str = "pi"
    theme_dropout: float = 0.3
    theme_embedding_noise_std: float = 0.03
    theme_token_bars: int = 8
    theme_cross_attention_heads: int = 4

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LatentTransformerConfig":
        """Build model config from style config."""
        section = ConfigView(config).section("latent_transformer")
        theme_fusion = ConfigView(config).section("theme_fusion")
        return cls(
            latent_dim=int(section.get("latent_dim", 32)),
            context_bars=int(section.get("context_bars", 32)),
            d_model=int(section.get("d_model", 128)),
            n_layers=int(section.get("n_layers", 4)),
            n_heads=int(section.get("n_heads", 4)),
            dropout=float(section.get("dropout", 0.1)),
            action_vocab_size=int(section.get("action_vocab_size", 8)),
            action_embedding_dim=int(section.get("action_embedding_dim", 8)),
            position_vocab_size=int(section.get("position_vocab_size", 8)),
            position_embedding_dim=int(section.get("position_embedding_dim", 8)),
            theme_fusion_enabled=bool(theme_fusion.get("enabled", False)),
            theme_embedding_dim=int(theme_fusion.get("embedding_dim", 64)),
            theme_project_dim=int(theme_fusion.get("project_dim", 16)),
            theme_gate_init=float(theme_fusion.get("gate_init", 0.01)),
            theme_fusion_mode=str(theme_fusion.get("mode", "film_pi")),
            theme_fusion_target=str(theme_fusion.get("target", "pi")),
            theme_dropout=float(theme_fusion.get("theme_dropout", 0.3)),
            theme_embedding_noise_std=float(theme_fusion.get("embedding_noise_std", 0.03)),
            theme_token_bars=int(theme_fusion.get("token_bars", theme_fusion.get("theme_bars", 8))),
            theme_cross_attention_heads=int(theme_fusion.get("cross_attention_heads", 4)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe config values."""
        return dict(self.__dict__)


@dataclass(frozen=True)
class MDNConfig:
    """Configuration for a diagonal Gaussian mixture head."""

    n_components: int = 8
    min_sigma: float = 0.03
    max_sigma: Optional[float] = 2.0
    sigma_parameterization: str = "elu"
    pi_entropy_weight: float = 0.01
    mu_spread_init_enabled: bool = True
    mu_spread_weight_std: float = 0.5
    mu_spread_bias_std: float = 1.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MDNConfig":
        """Build MDN config from style config."""
        section = ConfigView(config).section("mdn_head")
        return cls(
            n_components=int(section.get("n_components", 8)),
            min_sigma=float(section.get("min_sigma", 0.03)),
            max_sigma=None if section.get("max_sigma", 2.0) is None else float(section.get("max_sigma", 2.0)),
            sigma_parameterization=str(section.get("sigma_parameterization", "elu")),
            pi_entropy_weight=float(section.get("pi_entropy_weight", 0.01)),
            mu_spread_init_enabled=bool(section.get("mu_spread_init_enabled", True)),
            mu_spread_weight_std=float(section.get("mu_spread_weight_std", 0.5)),
            mu_spread_bias_std=float(section.get("mu_spread_bias_std", 1.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe config values."""
        return dict(self.__dict__)


@dataclass
class MDNOutput:
    """Output tensors of the MDN head."""

    pi_logits: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor


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
        """Add fixed positions to a batch-first sequence."""
        return x + self.pe[:, : x.shape[1], :]


class MDNHead(nn.Module):
    """Mixture Density Network head for next-latent prediction."""

    def __init__(self, d_model: int, latent_dim: int, config: MDNConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_dim = int(latent_dim)
        self.to_pi = nn.Linear(d_model, int(config.n_components))
        self.to_mu = nn.Linear(d_model, int(config.n_components) * int(latent_dim))
        self.to_sigma_raw = nn.Linear(d_model, int(config.n_components) * int(latent_dim))
        self._initialize_mu_components()

    def forward(self, hidden: torch.Tensor, pi_hidden: Optional[torch.Tensor] = None) -> MDNOutput:
        """Predict mixture weights, means, and diagonal standard deviations."""
        batch_size = int(hidden.shape[0])
        k = int(self.config.n_components)
        pi_source = hidden if pi_hidden is None else pi_hidden
        pi_logits = self.to_pi(pi_source)
        mu = self.to_mu(hidden).view(batch_size, k, self.latent_dim)
        sigma_raw = self.to_sigma_raw(hidden).view(batch_size, k, self.latent_dim)
        sigma = self._sigma_from_raw(sigma_raw)
        return MDNOutput(pi_logits=pi_logits, mu=mu, sigma=sigma)

    def _sigma_from_raw(self, sigma_raw: torch.Tensor) -> torch.Tensor:
        """Map raw sigma logits to positive standard deviations."""
        strategy = str(self.config.sigma_parameterization).lower()
        if strategy == "bounded_sigmoid":
            max_sigma = float(self.config.max_sigma if self.config.max_sigma is not None else 2.0)
            sigma = self.config.min_sigma + (max_sigma - self.config.min_sigma) * torch.sigmoid(sigma_raw)
        elif strategy == "exp":
            sigma = torch.exp(sigma_raw.clamp(min=-8.0, max=5.0)) + float(self.config.min_sigma)
        elif strategy == "elu":
            sigma = F.elu(sigma_raw) + 1.0 + float(self.config.min_sigma)
        else:
            raise ValueError(f"Unsupported MDN sigma_parameterization: {self.config.sigma_parameterization}")
        if self.config.max_sigma is not None and strategy != "bounded_sigmoid":
            sigma = sigma.clamp(max=float(self.config.max_sigma))
        return sigma

    def _initialize_mu_components(self) -> None:
        """Spread initial component means so MDN components start separated."""
        if not bool(self.config.mu_spread_init_enabled):
            return
        with torch.no_grad():
            self.to_mu.weight.normal_(mean=0.0, std=float(self.config.mu_spread_weight_std))
            self.to_mu.bias.normal_(mean=0.0, std=float(self.config.mu_spread_bias_std))


class LatentTransformerMDN(nn.Module):
    """Causal Transformer conditioned by action/position, with an MDN output head."""

    def __init__(self, transformer_config: LatentTransformerConfig, mdn_config: MDNConfig) -> None:
        super().__init__()
        self.transformer_config = transformer_config
        self.mdn_config = mdn_config
        input_dim = (
            int(transformer_config.latent_dim)
            + int(transformer_config.action_embedding_dim)
            + int(transformer_config.position_embedding_dim)
        )
        self.action_embedding = nn.Embedding(
            int(transformer_config.action_vocab_size),
            int(transformer_config.action_embedding_dim),
        )
        self.position_embedding = nn.Embedding(
            int(transformer_config.position_vocab_size),
            int(transformer_config.position_embedding_dim),
        )
        self.input_projection = nn.Linear(input_dim, int(transformer_config.d_model))
        self.sequence_position = SinusoidalPositionEncoding(
            int(transformer_config.d_model),
            int(transformer_config.context_bars),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(transformer_config.d_model),
            nhead=int(transformer_config.n_heads),
            dim_feedforward=int(transformer_config.d_model) * 4,
            dropout=float(transformer_config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(transformer_config.n_layers))
        cond_dim = (
            int(transformer_config.action_embedding_dim)
            + int(transformer_config.position_embedding_dim)
        )
        self.theme_fusion = build_theme_fusion_adapter(ThemeFusionAdapterConfig(
            enabled=bool(transformer_config.theme_fusion_enabled),
            mode=str(transformer_config.theme_fusion_mode),
            target=str(transformer_config.theme_fusion_target),
            embedding_dim=int(transformer_config.theme_embedding_dim),
            latent_dim=int(transformer_config.latent_dim),
            d_model=int(transformer_config.d_model),
            project_dim=int(transformer_config.theme_project_dim),
            gate_init=float(transformer_config.theme_gate_init),
            dropout=float(transformer_config.dropout),
            theme_dropout=float(transformer_config.theme_dropout),
            embedding_noise_std=float(transformer_config.theme_embedding_noise_std),
            token_bars=int(transformer_config.theme_token_bars),
            cross_attention_heads=int(transformer_config.theme_cross_attention_heads),
        ))
        self.output_condition = nn.Sequential(
            nn.Linear(int(transformer_config.d_model) + cond_dim, int(transformer_config.d_model)),
            nn.GELU(),
            nn.Dropout(float(transformer_config.dropout)),
        )
        self.mdn_head = MDNHead(int(transformer_config.d_model), int(transformer_config.latent_dim), mdn_config)

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
    ) -> MDNOutput:
        """Predict a Gaussian mixture for the next latent vector.

        padding_mask is True for padded history positions.
        """
        action_emb = self.action_embedding(context_action_ids)
        position_emb = self.position_embedding(context_position_ids)
        x = torch.cat([context_mu, action_emb, position_emb], dim=-1)
        x = self.sequence_position(self.input_projection(x))
        causal_mask = self._causal_mask(x.shape[1], x.device)
        encoded = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        pooled = self._last_valid_state(encoded, padding_mask)
        target_parts = [
            self.action_embedding(target_action_ids),
            self.position_embedding(target_position_ids),
        ]
        target_cond = torch.cat(target_parts, dim=-1)
        hidden = self.output_condition(torch.cat([pooled, target_cond], dim=-1))
        fused = self.theme_fusion(hidden, theme_embedding=theme_embedding, theme_tokens=theme_tokens)
        return self.mdn_head(fused.hidden, pi_hidden=fused.pi_hidden)

    def theme_fusion_diagnostics(self) -> Dict[str, Any]:
        """Return runtime diagnostics from the active theme fusion adapter."""
        return self.theme_fusion.runtime_diagnostics()

    def _last_valid_state(self, encoded: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Return the final non-padded history state for every row."""
        if padding_mask is None:
            return encoded[:, -1, :]
        positions = torch.arange(encoded.shape[1], device=encoded.device).unsqueeze(0)
        indices = torch.where(~padding_mask, positions, torch.zeros_like(positions)).max(dim=1).values
        return encoded[torch.arange(encoded.shape[0], device=encoded.device), indices, :]

    def _causal_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        """Return an upper-triangular mask for causal self-attention."""
        return torch.triu(
            torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=device),
            diagonal=1,
        )


class MDNLoss(nn.Module):
    """Negative log-likelihood plus optional mixture entropy regularization."""

    def __init__(self, config: Optional[MDNConfig] = None) -> None:
        super().__init__()
        self.config = config or MDNConfig()

    def forward(self, output: MDNOutput, target_mu: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return loss terms over a batch."""
        nll = self.nll(output, target_mu)
        entropy = self.pi_entropy(output)
        entropy_loss = -float(self.config.pi_entropy_weight) * entropy
        return {
            "loss": nll + entropy_loss,
            "nll": nll,
            "pi_entropy": entropy,
            "pi_entropy_loss": entropy_loss,
        }

    def nll(self, output: MDNOutput, target_mu: torch.Tensor) -> torch.Tensor:
        """Return mean negative log-likelihood over a batch."""
        target = target_mu.unsqueeze(1)
        log_pi = F.log_softmax(output.pi_logits, dim=-1)
        log_sigma = torch.log(output.sigma.clamp_min(1.0e-8))
        z = (target - output.mu) / output.sigma.clamp_min(1.0e-8)
        log_prob_dim = -0.5 * (z.pow(2) + 2.0 * log_sigma + math.log(2.0 * math.pi))
        log_prob = log_prob_dim.sum(dim=-1)
        mixture_log_prob = torch.logsumexp(log_pi + log_prob, dim=-1)
        return -mixture_log_prob.mean()

    def pi_entropy(self, output: MDNOutput) -> torch.Tensor:
        """Return mean entropy of mixture weights."""
        probs = torch.softmax(output.pi_logits, dim=-1)
        return -(probs * torch.log(probs.clamp_min(1.0e-12))).sum(dim=-1).mean()


class MDNDiagnostics:
    """Compute small diagnostic summaries for MDN predictions."""

    def summarize(self, output: MDNOutput, target_mu: torch.Tensor) -> Dict[str, float]:
        """Return batch-level MDN health metrics."""
        with torch.no_grad():
            probs = torch.softmax(output.pi_logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1.0e-12))).sum(dim=-1)
            distances = torch.linalg.vector_norm(output.mu - target_mu.unsqueeze(1), dim=-1)
            best = torch.argmin(distances, dim=-1)
            usage = torch.bincount(best, minlength=output.mu.shape[1]).float()
            usage = usage / usage.sum().clamp_min(1.0)
            usage_entropy = -(usage * torch.log(usage.clamp_min(1.0e-12))).sum()
            return {
                "component_entropy": float(entropy.mean().detach().cpu()),
                "component_usage_entropy": float(usage_entropy.detach().cpu()),
                "avg_max_pi": float(probs.max(dim=-1).values.mean().detach().cpu()),
                "avg_sigma": float(output.sigma.mean().detach().cpu()),
                "min_sigma": float(output.sigma.min().detach().cpu()),
                "max_sigma": float(output.sigma.max().detach().cpu()),
                "latent_l2_to_best_component": float(distances.min(dim=-1).values.mean().detach().cpu()),
            }
