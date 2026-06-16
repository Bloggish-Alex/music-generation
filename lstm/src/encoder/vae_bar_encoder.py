#!/usr/bin/env python3
"""Denoising VAE encoder for bar-token latent-space experiments."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from data.core_data import BarRecord


REST_TOKEN = -1
SUSTAIN_TOKEN = -2
NOTE_TYPE_REST = 0
NOTE_TYPE_SUSTAIN = 1
NOTE_TYPE_ON = 2


@dataclass(frozen=True)
class DenoisingVAEConfig:
    """Config for denoising bar-token VAE experiments."""

    steps_per_bar: int = 16
    input_mode: str = "multi_channel"
    hidden_dim: int = 64
    latent_dim: int = 12
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 0.001
    beta_kl: float = 0.001
    kl_warmup_epochs: int = 10
    pitch_weight: float = 1.0
    onset_weight: float = 0.5
    sustain_weight: float = 0.5
    global_weight: float = 0.25
    note_drop_prob: float = 0.15
    sustain_fill_prob: float = 0.10
    drop_to_rest_prob: float = 0.5
    ornament_pitch_radius: int = 2
    pitch_scale: float = 24.0
    condition_on_previous_last_pitch: bool = True
    previous_last_pitch_scale: float = 24.0
    decoder_head_architecture: str = "split_rhythm_pitch"
    pitch_activation: str = "hardtanh"
    random_seed: int = 42
    device: str = "cpu"

    def input_dim(self) -> int:
        if self.input_mode == "token":
            return int(self.steps_per_bar)
        if self.input_mode == "multi_channel":
            return int(self.steps_per_bar * 4 + 8)
        raise ValueError("vae_encoder.input_mode must be 'token' or 'multi_channel'.")


@dataclass(frozen=True)
class LatentClusteringConfig:
    """Config for clustering z_mu vectors."""

    method: str = "kmeans"
    feature_mode: str = "mu"
    logvar_weight: float = 0.25
    n_clusters: int = 384
    distance_threshold: float = 1.0
    linkage_method: str = "average"
    covariance_type: str = "full"
    reg_covar: float = 1e-6
    max_iter: int = 200
    random_seed: int = 42


class BarTokenCodec:
    """Convert relative bar tokens to multi-head training targets."""

    def __init__(self, config: DenoisingVAEConfig) -> None:
        self.config = config

    def clean_tokens(self, bar: BarRecord) -> List[int]:
        tokens = [int(token) for token in bar.relative_tokens[: self.config.steps_per_bar]]
        if len(tokens) < self.config.steps_per_bar:
            tokens.extend([REST_TOKEN] * (self.config.steps_per_bar - len(tokens)))
        return tokens

    def input_vector(self, tokens: Sequence[int]) -> np.ndarray:
        if self.config.input_mode == "token":
            return np.asarray([float(token) / self.config.pitch_scale for token in tokens], dtype=np.float32)
        if self.config.input_mode != "multi_channel":
            raise ValueError("vae_encoder.input_mode must be 'token' or 'multi_channel'.")
        event = self.type_targets(tokens).astype(np.float32) / 2.0
        pitch = self.pitch_targets(tokens)
        onset = self.onset_targets(tokens)
        sustain = self.sustain_targets(tokens)
        step_features = np.stack([event, pitch, onset, sustain], axis=1).reshape(-1)
        return np.concatenate([step_features, self.global_targets(tokens)], axis=0).astype(np.float32)

    def type_targets(self, tokens: Sequence[int]) -> np.ndarray:
        result = []
        for token in tokens:
            token = int(token)
            if token == REST_TOKEN:
                result.append(NOTE_TYPE_REST)
            elif token == SUSTAIN_TOKEN:
                result.append(NOTE_TYPE_SUSTAIN)
            else:
                result.append(NOTE_TYPE_ON)
        return np.asarray(result, dtype=np.int64)

    def pitch_targets(self, tokens: Sequence[int]) -> np.ndarray:
        return np.asarray([
            float(token) / self.config.pitch_scale if int(token) >= 0 else 0.0
            for token in tokens
        ], dtype=np.float32)

    def note_mask(self, tokens: Sequence[int]) -> np.ndarray:
        return np.asarray([1.0 if int(token) >= 0 else 0.0 for token in tokens], dtype=np.float32)

    def onset_targets(self, tokens: Sequence[int]) -> np.ndarray:
        return self.note_mask(tokens)

    def sustain_targets(self, tokens: Sequence[int]) -> np.ndarray:
        return np.asarray([1.0 if int(token) == SUSTAIN_TOKEN else 0.0 for token in tokens], dtype=np.float32)

    def global_targets(self, tokens: Sequence[int]) -> np.ndarray:
        values = [int(token) for token in tokens]
        count = max(1, len(values))
        note_values = [token for token in values if token >= 0]
        note_count = len(note_values)
        rest_count = sum(1 for token in values if token == REST_TOKEN)
        sustain_count = sum(1 for token in values if token == SUSTAIN_TOKEN)
        intervals = [
            int(right) - int(left)
            for left, right in zip(note_values, note_values[1:])
        ]
        variance = float(np.var(note_values)) if note_values else 0.0
        pitch_range = float(max(note_values) - min(note_values)) if note_values else 0.0
        interval_mean = float(np.mean(intervals)) if intervals else 0.0
        interval_var = float(np.var(intervals)) if intervals else 0.0
        return np.asarray([
            float(note_count / count),
            float(rest_count / count),
            float(sustain_count / count),
            float((note_count + sustain_count) / count),
            float(min(1.0, variance / (self.config.pitch_scale ** 2))),
            float(1.0 / (1.0 + max(0.0, variance))),
            float(pitch_range / self.config.pitch_scale),
            float(interval_var / (self.config.pitch_scale ** 2)),
        ], dtype=np.float32)


class BarNoiseInjector:
    """Apply token-level variation noise to relative-token bars."""

    def __init__(self, config: DenoisingVAEConfig) -> None:
        self.config = config

    def apply(self, tokens: Sequence[int], rng: np.random.Generator) -> List[int]:
        clean = [int(token) for token in tokens]
        noisy = list(clean)
        note_indices = [index for index, token in enumerate(clean) if token >= 0]
        for index in note_indices:
            if rng.random() < self.config.note_drop_prob:
                noisy[index] = REST_TOKEN if rng.random() < self.config.drop_to_rest_prob else SUSTAIN_TOKEN
        note_values = [token for token in clean if token >= 0]
        if note_values:
            for index, token in enumerate(clean):
                if token == SUSTAIN_TOKEN and rng.random() < self.config.sustain_fill_prob:
                    base = int(rng.choice(note_values))
                    delta = int(rng.integers(-self.config.ornament_pitch_radius, self.config.ornament_pitch_radius + 1))
                    noisy[index] = max(0, base + delta)
        if note_indices and not any(token >= 0 for token in noisy):
            restore_index = int(rng.choice(note_indices))
            noisy[restore_index] = clean[restore_index]
        return noisy


class DenoisingBarDataset:
    """Small torch Dataset wrapper that creates fresh noise each epoch."""

    def __init__(self, bars: Sequence[BarRecord], config: DenoisingVAEConfig) -> None:
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __init__(self, outer: "DenoisingBarDataset") -> None:
                self.outer = outer

            def __len__(self) -> int:
                return len(self.outer.clean_tokens)

            def __getitem__(self, index: int) -> Dict[str, Any]:
                outer = self.outer
                clean = outer.clean_tokens[index]
                noisy = outer.noise.apply(clean, outer.rng)
                return {
                    "x": torch.tensor(outer.codec.input_vector(noisy), dtype=torch.float32),
                    "previous_last_pitch": torch.tensor(
                        outer.previous_last_pitch[index],
                        dtype=torch.float32,
                    ),
                    "type_target": torch.tensor(outer.codec.type_targets(clean), dtype=torch.long),
                    "pitch_target": torch.tensor(outer.codec.pitch_targets(clean), dtype=torch.float32),
                    "note_mask": torch.tensor(outer.codec.note_mask(clean), dtype=torch.float32),
                    "onset_target": torch.tensor(outer.codec.onset_targets(clean), dtype=torch.float32),
                    "sustain_target": torch.tensor(outer.codec.sustain_targets(clean), dtype=torch.float32),
                    "global_target": torch.tensor(outer.codec.global_targets(clean), dtype=torch.float32),
                }

        self.config = config
        self.codec = BarTokenCodec(config)
        self.noise = BarNoiseInjector(config)
        self.rng = np.random.default_rng(config.random_seed)
        self.clean_tokens = [self.codec.clean_tokens(bar) for bar in bars]
        self.previous_last_pitch = self._previous_last_pitch_scalars(bars)
        self.torch_dataset = _Dataset(self)

    def _previous_last_pitch_scalars(self, bars: Sequence[BarRecord]) -> List[float]:
        by_song: Dict[tuple[str, str], List[tuple[int, int, BarRecord]]] = defaultdict(list)
        for index, bar in enumerate(bars):
            by_song[(str(bar.song_id), str(bar.file_path))].append((int(bar.bar_index), index, bar))
        result = [0.0 for _ in bars]
        for _song_key, rows in by_song.items():
            previous_pitch: Optional[int] = None
            for _bar_index, index, bar in sorted(rows, key=lambda item: item[0]):
                result[index] = self._normalize_previous_pitch(previous_pitch)
                previous_pitch = self._last_note_pitch(self.codec.clean_tokens(bar))
        return result

    def _last_note_pitch(self, tokens: Sequence[int]) -> Optional[int]:
        for token in reversed(tokens):
            if int(token) >= 0:
                return int(token)
        return None

    def _normalize_previous_pitch(self, pitch: Optional[int]) -> float:
        if pitch is None:
            return 0.0
        scale = max(1.0e-6, float(self.config.previous_last_pitch_scale))
        return float(pitch) / scale


class DenoisingBarVAE:
    """Factory for the torch module to keep torch import local to training."""

    @staticmethod
    def build(config: DenoisingVAEConfig):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(config.input_dim(), config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                )
                self.mu = nn.Linear(config.hidden_dim, config.latent_dim)
                self.logvar = nn.Linear(config.hidden_dim, config.latent_dim)
                self.decoder = nn.Sequential(
                    nn.Linear(config.latent_dim + DenoisingBarVAE.condition_dim(config), config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                )
                self.rhythm_adapter = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                )
                self.pitch_adapter = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                )
                self.type_head = nn.Linear(config.hidden_dim, config.steps_per_bar * 3)
                self.pitch_head = nn.Linear(config.hidden_dim, config.steps_per_bar)
                self.pitch_activation = nn.Hardtanh(min_val=-1.0, max_val=1.0)
                self.onset_head = nn.Linear(config.hidden_dim, config.steps_per_bar)
                self.sustain_head = nn.Linear(config.hidden_dim, config.steps_per_bar)
                self.global_head = nn.Linear(config.hidden_dim, 8)

            def encode(self, x):
                hidden = self.encoder(x)
                return self.mu(hidden), self.logvar(hidden)

            def reparameterize(self, mu, logvar):
                if not self.training:
                    return mu
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decoder_input(self, z, previous_last_pitch=None):
                if not config.condition_on_previous_last_pitch:
                    return z
                if previous_last_pitch is None:
                    previous_last_pitch = torch.zeros((z.shape[0],), dtype=z.dtype, device=z.device)
                if len(previous_last_pitch.shape) == 1:
                    previous_last_pitch = previous_last_pitch[:, None]
                return torch.cat([z, previous_last_pitch], dim=-1)

            def forward(self, x, previous_last_pitch=None):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                hidden = self.decoder(self.decoder_input(z, previous_last_pitch))
                rhythm_hidden = self.rhythm_adapter(hidden)
                pitch_hidden = self.pitch_adapter(hidden)
                type_logits = self.type_head(rhythm_hidden).view(-1, config.steps_per_bar, 3)
                pitch = self.pitch_activation(self.pitch_head(pitch_hidden))
                onset = self.onset_head(rhythm_hidden)
                sustain = self.sustain_head(rhythm_hidden)
                global_features = self.global_head(hidden)
                return type_logits, pitch, onset, sustain, global_features, mu, logvar

        return _Model()

    @staticmethod
    def condition_dim(config: DenoisingVAEConfig) -> int:
        return 1 if bool(config.condition_on_previous_last_pitch) else 0


class DenoisingVAETrainer:
    """Train the denoising VAE and expose z_mu latent features."""

    def __init__(self, config: DenoisingVAEConfig) -> None:
        self.config = config
        self.training_log: List[Dict[str, float]] = []
        self.model = None

    def fit(self, bars: Sequence[BarRecord]) -> "DenoisingVAETrainer":
        import torch
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader

        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        device = torch.device(self.config.device)
        dataset = DenoisingBarDataset(bars, self.config)
        loader = DataLoader(dataset.torch_dataset, batch_size=self.config.batch_size, shuffle=True)
        model = DenoisingBarVAE.build(self.config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.epochs):
            model.train()
            totals = Counter()
            for batch in loader:
                x = batch["x"].to(device)
                previous_last_pitch = batch["previous_last_pitch"].to(device)
                type_target = batch["type_target"].to(device)
                pitch_target = batch["pitch_target"].to(device)
                note_mask = batch["note_mask"].to(device)
                onset_target = batch["onset_target"].to(device)
                sustain_target = batch["sustain_target"].to(device)
                global_target = batch["global_target"].to(device)
                type_logits, pitch, onset, sustain, global_features, mu, logvar = model(x, previous_last_pitch)
                type_loss = functional.cross_entropy(
                    type_logits.reshape(-1, 3),
                    type_target.reshape(-1),
                )
                pitch_error = (pitch - pitch_target) ** 2
                pitch_loss = (pitch_error * note_mask).sum() / torch.clamp(note_mask.sum(), min=1.0)
                onset_loss = functional.binary_cross_entropy_with_logits(onset, onset_target)
                sustain_loss = functional.binary_cross_entropy_with_logits(sustain, sustain_target)
                global_loss = functional.mse_loss(global_features, global_target)
                kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
                kl_weight = self._kl_weight(epoch)
                loss = (
                    type_loss
                    + self.config.pitch_weight * pitch_loss
                    + self.config.onset_weight * onset_loss
                    + self.config.sustain_weight * sustain_loss
                    + self.config.global_weight * global_loss
                    + kl_weight * kl_loss
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_size = int(x.shape[0])
                totals["count"] += batch_size
                totals["loss"] += float(loss.detach().cpu()) * batch_size
                totals["type_loss"] += float(type_loss.detach().cpu()) * batch_size
                totals["pitch_loss"] += float(pitch_loss.detach().cpu()) * batch_size
                totals["onset_loss"] += float(onset_loss.detach().cpu()) * batch_size
                totals["sustain_loss"] += float(sustain_loss.detach().cpu()) * batch_size
                totals["global_loss"] += float(global_loss.detach().cpu()) * batch_size
                totals["kl_loss"] += float(kl_loss.detach().cpu()) * batch_size
            count = max(1, int(totals["count"]))
            self.training_log.append({
                "epoch": float(epoch),
                "loss": float(totals["loss"] / count),
                "type_loss": float(totals["type_loss"] / count),
                "pitch_loss": float(totals["pitch_loss"] / count),
                "onset_loss": float(totals["onset_loss"] / count),
                "sustain_loss": float(totals["sustain_loss"] / count),
                "global_loss": float(totals["global_loss"] / count),
                "kl_loss": float(totals["kl_loss"] / count),
                "kl_weight": float(self._kl_weight(epoch)),
            })
        self.model = model
        return self

    def encode(self, bars: Sequence[BarRecord]) -> np.ndarray:
        mu, _ = self.encode_distribution(bars)
        return mu

    def encode_distribution(self, bars: Sequence[BarRecord]) -> tuple[np.ndarray, np.ndarray]:
        import torch

        if self.model is None:
            raise RuntimeError("VAE model has not been trained.")
        codec = BarTokenCodec(self.config)
        x = np.asarray([codec.input_vector(codec.clean_tokens(bar)) for bar in bars], dtype=np.float32)
        device = torch.device(self.config.device)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32, device=device)
            mu, logvar = self.model.encode(tensor)
        return (
            mu.detach().cpu().numpy().astype(np.float32),
            logvar.detach().cpu().numpy().astype(np.float32),
        )

    def evaluate_reconstruction(self, bars: Sequence[BarRecord]) -> Dict[str, Any]:
        import torch

        if self.model is None:
            raise RuntimeError("VAE model has not been trained.")
        codec = BarTokenCodec(self.config)
        dataset = DenoisingBarDataset(bars, self.config)
        x = np.asarray([codec.input_vector(codec.clean_tokens(bar)) for bar in bars], dtype=np.float32)
        previous_last_pitch = np.asarray(dataset.previous_last_pitch, dtype=np.float32)
        type_targets = np.asarray([codec.type_targets(codec.clean_tokens(bar)) for bar in bars], dtype=np.int64)
        pitch_targets = np.asarray([codec.pitch_targets(codec.clean_tokens(bar)) for bar in bars], dtype=np.float32)
        note_masks = np.asarray([codec.note_mask(codec.clean_tokens(bar)) for bar in bars], dtype=np.float32)
        onset_targets = np.asarray([codec.onset_targets(codec.clean_tokens(bar)) for bar in bars], dtype=np.float32)
        sustain_targets = np.asarray([codec.sustain_targets(codec.clean_tokens(bar)) for bar in bars], dtype=np.float32)
        global_targets = np.asarray([codec.global_targets(codec.clean_tokens(bar)) for bar in bars], dtype=np.float32)
        device = torch.device(self.config.device)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32, device=device)
            condition = torch.tensor(previous_last_pitch, dtype=torch.float32, device=device)
            type_logits, pitch, onset, sustain, global_features, mu, logvar = self.model(tensor, condition)
            type_pred = torch.argmax(type_logits, dim=-1).detach().cpu().numpy()
            pitch_pred = pitch.detach().cpu().numpy()
            onset_pred = (torch.sigmoid(onset).detach().cpu().numpy() >= 0.5).astype(np.float32)
            sustain_pred = (torch.sigmoid(sustain).detach().cpu().numpy() >= 0.5).astype(np.float32)
            global_pred = global_features.detach().cpu().numpy()
            kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        type_accuracy = float(np.mean(type_pred == type_targets)) if type_targets.size else 0.0
        note_positions = note_masks > 0
        note_type_accuracy = float(np.mean(type_pred[note_positions] == NOTE_TYPE_ON)) if np.any(note_positions) else 0.0
        special_positions = ~note_positions
        special_type_accuracy = float(np.mean(type_pred[special_positions] == type_targets[special_positions])) if np.any(special_positions) else 0.0
        pitch_mse = float(np.mean(((pitch_pred - pitch_targets) ** 2)[note_positions])) if np.any(note_positions) else 0.0
        onset_accuracy = float(np.mean(onset_pred == onset_targets)) if onset_targets.size else 0.0
        sustain_accuracy = float(np.mean(sustain_pred == sustain_targets)) if sustain_targets.size else 0.0
        global_mse = float(np.mean((global_pred - global_targets) ** 2)) if global_targets.size else 0.0
        return {
            "type_accuracy": type_accuracy,
            "note_on_type_accuracy": note_type_accuracy,
            "rest_sustain_type_accuracy": special_type_accuracy,
            "pitch_mse_note_on": pitch_mse,
            "onset_accuracy": onset_accuracy,
            "sustain_accuracy": sustain_accuracy,
            "global_feature_mse": global_mse,
            "kl_loss": float(kl_loss.detach().cpu()),
        }

    def save(self, path: Path) -> None:
        import torch

        if self.model is None:
            raise RuntimeError("VAE model has not been trained.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": asdict(self.config),
            "state_dict": self.model.state_dict(),
            "training_log": self.training_log,
        }, path)

    def _kl_weight(self, epoch: int) -> float:
        warmup = max(0, int(self.config.kl_warmup_epochs))
        if warmup <= 0:
            return float(self.config.beta_kl)
        return float(self.config.beta_kl * min(1.0, float(epoch + 1) / float(warmup)))


class LatentFeatureBuilder:
    """Build clustering features from VAE posterior parameters."""

    def __init__(self, config: LatentClusteringConfig) -> None:
        self.config = config

    def build(self, mu: np.ndarray, logvar: Optional[np.ndarray] = None) -> np.ndarray:
        mode = str(self.config.feature_mode)
        if mode == "mu":
            return np.asarray(mu, dtype=np.float32)
        if mode == "mu_logvar":
            if logvar is None:
                raise ValueError("latent clustering feature_mode='mu_logvar' requires VAE logvar.")
            return np.concatenate([
                np.asarray(mu, dtype=np.float32),
                float(self.config.logvar_weight) * np.asarray(logvar, dtype=np.float32),
            ], axis=1).astype(np.float32)
        raise ValueError("latent clustering feature_mode must be 'mu' or 'mu_logvar'.")

    def diagnostics(self, mu: np.ndarray, logvar: Optional[np.ndarray], features: np.ndarray) -> Dict[str, Any]:
        result = {
            "clustering_method": str(self.config.method),
            "feature_mode": str(self.config.feature_mode),
            "logvar_weight": float(self.config.logvar_weight),
            "mu_shape": [int(item) for item in mu.shape],
            "feature_shape": [int(item) for item in features.shape],
            "mu_mean_abs": float(np.mean(np.abs(mu))) if mu.size else 0.0,
            "mu_std": float(np.std(mu)) if mu.size else 0.0,
            "feature_mean_abs": float(np.mean(np.abs(features))) if features.size else 0.0,
            "feature_std": float(np.std(features)) if features.size else 0.0,
        }
        if logvar is not None and logvar.size:
            sigma = np.exp(0.5 * np.asarray(logvar, dtype=np.float64))
            result.update({
                "logvar_shape": [int(item) for item in logvar.shape],
                "logvar_mean": float(np.mean(logvar)),
                "logvar_std": float(np.std(logvar)),
                "sigma_mean": float(np.mean(sigma)),
                "sigma_std": float(np.std(sigma)),
            })
        return result


class LatentClusterAnalyzer:
    """Cluster latent vectors and summarize label distribution."""

    def __init__(self, config: LatentClusteringConfig) -> None:
        self.config = config
        self.model_diagnostics: Dict[str, Any] = {}
        self.assignment_confidence: Optional[np.ndarray] = None
        self.assignment_entropy: Optional[np.ndarray] = None

    def cluster(self, latents: np.ndarray) -> np.ndarray:
        self.model_diagnostics = {}
        self.assignment_confidence = None
        self.assignment_entropy = None
        if len(latents) == 0:
            return np.asarray([], dtype=int)
        if self.config.method == "kmeans":
            n_clusters = min(max(1, int(self.config.n_clusters)), len(latents))
            model = KMeans(
                n_clusters=n_clusters,
                random_state=self.config.random_seed,
                n_init="auto",
            )
            labels = model.fit_predict(latents).astype(int)
            self.model_diagnostics = {
                "backend": "kmeans",
                "n_clusters": int(n_clusters),
                "inertia": float(model.inertia_),
                "n_iter": int(model.n_iter_),
            }
            return labels
        if self.config.method == "gmm":
            n_components = min(max(1, int(self.config.n_clusters)), len(latents))
            model = GaussianMixture(
                n_components=n_components,
                covariance_type=str(self.config.covariance_type),
                reg_covar=float(self.config.reg_covar),
                max_iter=int(self.config.max_iter),
                random_state=int(self.config.random_seed),
            )
            labels = model.fit_predict(latents).astype(int)
            weights = np.asarray(model.weights_, dtype=np.float64)
            posterior = np.asarray(model.predict_proba(latents), dtype=np.float64)
            self.assignment_confidence = np.max(posterior, axis=1)
            self.assignment_entropy = -np.sum(
                posterior * np.log(np.maximum(posterior, 1e-12)),
                axis=1,
            )
            self.model_diagnostics = {
                "backend": "gmm",
                "n_components": int(n_components),
                "covariance_type": str(model.covariance_type),
                "reg_covar": float(self.config.reg_covar),
                "max_iter": int(self.config.max_iter),
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
                "lower_bound": float(model.lower_bound_),
                "weight_min": float(np.min(weights)) if weights.size else 0.0,
                "weight_max": float(np.max(weights)) if weights.size else 0.0,
                "weight_entropy": self._probability_entropy(weights),
                "effective_component_count": float(math.exp(self._probability_entropy(weights))) if weights.size else 0.0,
                "assignment_confidence_mean": float(np.mean(self.assignment_confidence)) if self.assignment_confidence.size else 0.0,
                "assignment_confidence_median": float(np.median(self.assignment_confidence)) if self.assignment_confidence.size else 0.0,
                "assignment_confidence_min": float(np.min(self.assignment_confidence)) if self.assignment_confidence.size else 0.0,
                "assignment_entropy_mean": float(np.mean(self.assignment_entropy)) if self.assignment_entropy.size else 0.0,
                "assignment_entropy_median": float(np.median(self.assignment_entropy)) if self.assignment_entropy.size else 0.0,
                "low_confidence_ratio_lt_0_60": float(np.mean(self.assignment_confidence < 0.60)) if self.assignment_confidence.size else 0.0,
                "low_confidence_ratio_lt_0_80": float(np.mean(self.assignment_confidence < 0.80)) if self.assignment_confidence.size else 0.0,
            }
            return labels
        if self.config.method == "agglomerative":
            if len(latents) == 1:
                return np.asarray([0], dtype=int)
            distances = pdist(latents, metric="euclidean")
            z_matrix = linkage(distances, method=self.config.linkage_method)
            raw = fcluster(z_matrix, t=float(self.config.distance_threshold), criterion="distance")
            labels = self._compact(raw)
            self.model_diagnostics = {
                "backend": "agglomerative",
                "distance_threshold": float(self.config.distance_threshold),
                "linkage_method": str(self.config.linkage_method),
                "used_label_count": int(len(set(int(label) for label in labels))),
            }
            return labels
        raise ValueError("latent clustering method must be 'kmeans', 'gmm', or 'agglomerative'.")

    def diagnostics(self, bars: Sequence[BarRecord], latents: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        counts = Counter(int(label) for label in labels)
        values = np.asarray([int(value) for value in counts.values() if int(value) > 0], dtype=np.float64)
        total = int(np.sum(values)) if values.size else 0
        probabilities = values / max(1.0, float(total)) if values.size else np.asarray([], dtype=np.float64)
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))) if values.size else 0.0
        singleton_count = int(np.sum(values == 1)) if values.size else 0
        max_label, max_count = self._max_count(counts)
        return {
            "config": asdict(self.config),
            "total_assignments": total,
            "used_label_count": int(len(values)),
            "singleton_label_count": singleton_count,
            "singleton_ratio": float(singleton_count / max(1, len(values))),
            "max_label": str(max_label) if max_label is not None else None,
            "max_label_count": int(max_count),
            "max_label_ratio": float(max_count / max(1, total)),
            "entropy": entropy,
            "normalized_entropy": float(entropy / math.log(len(values))) if len(values) > 1 else 0.0,
            "effective_label_count": float(math.exp(entropy)) if values.size else 0.0,
            "cluster_model": self.model_diagnostics,
            "assignment_confidence": self._assignment_confidence_summary(labels),
            "label_confidence": self._label_confidence_summary(labels, counts),
            "latent": self._latent_summary(latents),
            "top_labels": self._top_labels(labels, counts, total),
            "top_label_examples": self._top_label_examples(bars, labels, counts),
            "nearest_neighbors": self._nearest_neighbors(bars, latents),
        }

    def _probability_entropy(self, probabilities: np.ndarray) -> float:
        if probabilities.size == 0:
            return 0.0
        normalized = probabilities / max(float(np.sum(probabilities)), 1e-12)
        return float(-np.sum(normalized * np.log(np.maximum(normalized, 1e-12))))

    def _assignment_confidence_summary(self, labels: Sequence[int]) -> Dict[str, Any]:
        if self.assignment_confidence is None or len(self.assignment_confidence) != len(labels):
            return {}
        confidence = np.asarray(self.assignment_confidence, dtype=np.float64)
        entropy = (
            np.asarray(self.assignment_entropy, dtype=np.float64)
            if self.assignment_entropy is not None and len(self.assignment_entropy) == len(labels)
            else np.asarray([], dtype=np.float64)
        )
        summary = {
            "mean": float(np.mean(confidence)),
            "median": float(np.median(confidence)),
            "min": float(np.min(confidence)),
            "p10": float(np.percentile(confidence, 10)),
            "p25": float(np.percentile(confidence, 25)),
            "p75": float(np.percentile(confidence, 75)),
            "p90": float(np.percentile(confidence, 90)),
            "low_confidence_count_lt_0_60": int(np.sum(confidence < 0.60)),
            "low_confidence_ratio_lt_0_60": float(np.mean(confidence < 0.60)),
            "low_confidence_count_lt_0_80": int(np.sum(confidence < 0.80)),
            "low_confidence_ratio_lt_0_80": float(np.mean(confidence < 0.80)),
        }
        if entropy.size:
            summary.update({
                "entropy_mean": float(np.mean(entropy)),
                "entropy_median": float(np.median(entropy)),
                "entropy_p90": float(np.percentile(entropy, 90)),
            })
        return summary

    def _label_confidence_summary(self, labels: Sequence[int], counts: Counter, limit: int = 20) -> List[Dict[str, Any]]:
        if self.assignment_confidence is None or len(self.assignment_confidence) != len(labels):
            return []
        label_array = np.asarray([int(label) for label in labels], dtype=np.int64)
        confidence = np.asarray(self.assignment_confidence, dtype=np.float64)
        entropy = (
            np.asarray(self.assignment_entropy, dtype=np.float64)
            if self.assignment_entropy is not None and len(self.assignment_entropy) == len(labels)
            else None
        )
        result = []
        for label, count in counts.most_common(limit):
            mask = label_array == int(label)
            label_confidence = confidence[mask]
            item = {
                "label": str(label),
                "count": int(count),
                "confidence_mean": float(np.mean(label_confidence)) if label_confidence.size else 0.0,
                "confidence_min": float(np.min(label_confidence)) if label_confidence.size else 0.0,
                "low_confidence_ratio_lt_0_80": float(np.mean(label_confidence < 0.80)) if label_confidence.size else 0.0,
            }
            if entropy is not None:
                label_entropy = entropy[mask]
                item["entropy_mean"] = float(np.mean(label_entropy)) if label_entropy.size else 0.0
            result.append(item)
        return result

    def _compact(self, labels: Sequence[int]) -> np.ndarray:
        mapping: Dict[int, int] = {}
        result: List[int] = []
        for label in labels:
            value = int(label)
            if value not in mapping:
                mapping[value] = len(mapping)
            result.append(mapping[value])
        return np.asarray(result, dtype=int)

    def _max_count(self, counts: Counter) -> tuple[Optional[int], int]:
        if not counts:
            return None, 0
        label, count = max(counts.items(), key=lambda item: int(item[1]))
        return int(label), int(count)

    def _latent_summary(self, latents: np.ndarray) -> Dict[str, Any]:
        if latents.size == 0:
            return {}
        return {
            "shape": [int(item) for item in latents.shape],
            "mean_abs": float(np.mean(np.abs(latents))),
            "std": float(np.std(latents)),
            "dim_mean": [float(value) for value in np.mean(latents, axis=0)],
            "dim_std": [float(value) for value in np.std(latents, axis=0)],
        }

    def _top_labels(self, labels: Sequence[int], counts: Counter, total: int, limit: int = 20) -> List[Dict[str, Any]]:
        label_confidence = {
            item["label"]: item
            for item in self._label_confidence_summary(labels, counts, limit)
        }
        result = []
        for label, count in counts.most_common(limit):
            item = {
                "label": str(label),
                "count": int(count),
                "ratio": float(count / max(1, total)),
            }
            confidence = label_confidence.get(str(label), {})
            if confidence:
                item.update({
                    "confidence_mean": confidence.get("confidence_mean"),
                    "confidence_min": confidence.get("confidence_min"),
                    "low_confidence_ratio_lt_0_80": confidence.get("low_confidence_ratio_lt_0_80"),
                })
            result.append(item)
        return result

    def _top_label_examples(
        self,
        bars: Sequence[BarRecord],
        labels: np.ndarray,
        counts: Counter,
        limit: int = 8,
        examples_per_label: int = 4,
    ) -> List[Dict[str, Any]]:
        bars_by_label: Dict[int, List[BarRecord]] = defaultdict(list)
        confidence_by_bar: Dict[int, Dict[str, float]] = {}
        if self.assignment_confidence is not None and len(self.assignment_confidence) == len(labels):
            for index, confidence in enumerate(self.assignment_confidence):
                confidence_by_bar[int(index)] = {"assignment_confidence": round(float(confidence), 6)}
        if self.assignment_entropy is not None and len(self.assignment_entropy) == len(labels):
            for index, entropy in enumerate(self.assignment_entropy):
                confidence_by_bar.setdefault(int(index), {})["assignment_entropy"] = round(float(entropy), 6)
        index_by_bar_id = {id(bar): index for index, bar in enumerate(bars)}
        for bar, label in zip(bars, labels):
            bars_by_label[int(label)].append(bar)
        result = []
        for label, count in counts.most_common(limit):
            result.append({
                "label": str(label),
                "count": int(count),
                "examples": [
                    {
                        **self._bar_example(bar),
                        **confidence_by_bar.get(index_by_bar_id.get(id(bar), -1), {}),
                    }
                    for bar in bars_by_label[int(label)][:examples_per_label]
                ],
            })
        return result

    def _nearest_neighbors(
        self,
        bars: Sequence[BarRecord],
        latents: np.ndarray,
        anchor_count: int = 8,
        neighbor_count: int = 5,
    ) -> List[Dict[str, Any]]:
        if len(bars) <= 1:
            return []
        anchors = np.linspace(0, len(bars) - 1, num=min(anchor_count, len(bars)), dtype=int)
        result = []
        for anchor in anchors:
            distances = np.linalg.norm(latents - latents[int(anchor)], axis=1)
            nearest = [
                int(index)
                for index in np.argsort(distances)
                if int(index) != int(anchor)
            ][:neighbor_count]
            result.append({
                "anchor": self._bar_example(bars[int(anchor)]),
                "neighbors": [
                    {
                        **self._bar_example(bars[index]),
                        "latent_distance": float(distances[index]),
                    }
                    for index in nearest
                ],
            })
        return result

    def _bar_example(self, bar: BarRecord) -> Dict[str, Any]:
        return {
            "song_id": bar.song_id,
            "bar_index": int(bar.bar_index),
            "relative_tokens": list(bar.relative_tokens),
            "absolute_tokens": list(bar.absolute_tokens),
            "token_variance": round(float(bar.token_variance), 6),
            "sharing_score": round(float(bar.sharing_score), 6),
        }
