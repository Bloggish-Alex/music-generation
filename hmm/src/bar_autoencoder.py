#!/usr/bin/env python3
"""Neural bar-token autoencoder features for clustering."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core_data import BarRecord


@dataclass(frozen=True)
class BarAutoencoderConfig:
    enabled: bool = False
    token_strategy: str = "relative"
    latent_dim: int = 8
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 0.001
    random_seed: int = 42
    device: str = "cpu"
    normalize_latent: bool = True
    quantization_bins: int = 32
    quantization_clip: float = 3.0


class BarTokenAutoencoderFeatureExtractor:
    """Train a 16 -> latent -> 16 autoencoder and expose latent vectors."""

    def __init__(self, config: BarAutoencoderConfig) -> None:
        self.config = config
        self.token_to_index: Dict[int, int] = {}
        self.index_to_token: Dict[int, int] = {}
        self.sequence_length: int = 0
        self.latent_by_bar_id: Dict[int, np.ndarray] = {}
        self.latent_mean: Optional[np.ndarray] = None
        self.latent_std: Optional[np.ndarray] = None
        self.diagnostics: Dict[str, Any] = {"enabled": False}

    def fit(self, bars: Sequence[BarRecord]) -> None:
        bars = list(bars)
        if not self.config.enabled or not bars:
            self.diagnostics = {"enabled": False}
            return
        self._validate_bars(bars)
        token_rows = [self._tokens_for_bar(bar) for bar in bars]
        self._build_vocab(token_rows)
        encoded_rows = np.asarray(token_rows, dtype=np.float32)
        latents, training_diagnostics = self._train_and_encode(encoded_rows)
        if self.config.normalize_latent:
            self.latent_mean = np.mean(latents, axis=0)
            self.latent_std = np.std(latents, axis=0)
            latents = (latents - self.latent_mean) / np.maximum(self.latent_std, 1e-6)
        self.latent_by_bar_id = {
            id(bar): latents[index].astype(np.float64, copy=False)
            for index, bar in enumerate(bars)
        }
        self.diagnostics = {
            "enabled": True,
            "config": asdict(self.config),
            "bar_count": len(bars),
            "sequence_length": int(self.sequence_length),
            "token_min": min(self.token_to_index) if self.token_to_index else None,
            "token_max": max(self.token_to_index) if self.token_to_index else None,
            "latent_mean": self._rounded_vector(self.latent_mean) if self.latent_mean is not None else None,
            "latent_std": self._rounded_vector(self.latent_std) if self.latent_std is not None else None,
            "training": training_diagnostics,
        }

    def vector_for(self, bar: BarRecord) -> np.ndarray:
        vector = self.latent_by_bar_id.get(id(bar))
        if vector is not None:
            return vector
        if not self.latent_by_bar_id:
            raise RuntimeError("Bar autoencoder has not been fitted.")
        raise KeyError("Bar is not present in the fitted autoencoder corpus.")

    def tokens_for_bar(self, bar: BarRecord) -> List[int]:
        """Return quantized latent values as an edit-distance token sequence."""
        vector = self.vector_for(bar)
        bins = max(2, int(self.config.quantization_bins))
        clip = max(1e-6, float(self.config.quantization_clip))
        clipped = np.clip(vector, -clip, clip)
        scaled = (clipped + clip) / (2.0 * clip)
        quantized = np.rint(scaled * float(bins - 1)).astype(int)
        return [int(item) for item in quantized.tolist()]

    def _validate_bars(self, bars: Sequence[BarRecord]) -> None:
        lengths = {len(self._tokens_for_bar(bar)) for bar in bars}
        if not lengths:
            self.sequence_length = 0
            return
        if len(lengths) != 1:
            raise ValueError(f"Autoencoder requires fixed-length bar tokens, got lengths={sorted(lengths)}.")
        self.sequence_length = int(next(iter(lengths)))

    def _tokens_for_bar(self, bar: BarRecord) -> List[int]:
        return [int(token) for token in bar.tokens_for_edit_distance(self.config.token_strategy)]

    def _build_vocab(self, token_rows: Sequence[Sequence[int]]) -> None:
        tokens = sorted({int(token) for row in token_rows for token in row})
        self.token_to_index = {token: index for index, token in enumerate(tokens)}
        self.index_to_token = {index: token for token, index in self.token_to_index.items()}

    def _train_and_encode(self, encoded_rows: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise RuntimeError("bar_autoencoder backend requires PyTorch.") from exc

        torch.manual_seed(int(self.config.random_seed))
        random.seed(int(self.config.random_seed))
        np.random.seed(int(self.config.random_seed))

        mean = np.mean(encoded_rows, axis=0, keepdims=True)
        std = np.maximum(np.std(encoded_rows, axis=0, keepdims=True), 1e-6)
        normalized_rows = ((encoded_rows - mean) / std).astype(np.float32)

        class TokenAutoencoder(nn.Module):
            def __init__(
                self,
                sequence_length: int,
                latent_dim: int,
            ) -> None:
                super().__init__()
                self.sequence_length = sequence_length
                self.encoder = nn.Linear(sequence_length, latent_dim)
                self.decoder = nn.Linear(latent_dim, sequence_length)

            def forward(self, tokens: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
                latent = self.encoder(tokens)
                reconstruction = self.decoder(latent)
                return reconstruction, latent

            def encode(self, tokens: "torch.Tensor") -> "torch.Tensor":
                return self.encoder(tokens)

        requested_device = str(self.config.device)
        device = torch.device(requested_device if requested_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        data = torch.tensor(normalized_rows, dtype=torch.float32)
        dataset = TensorDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(self.config.batch_size)),
            shuffle=True,
        )
        model = TokenAutoencoder(
            sequence_length=int(self.sequence_length),
            latent_dim=int(self.config.latent_dim),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.config.learning_rate))
        loss_fn = nn.MSELoss()
        losses: List[float] = []
        for _epoch in range(max(1, int(self.config.epochs))):
            epoch_loss = 0.0
            model.train()
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstruction, _latent = model(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu()) * int(batch.shape[0])
            losses.append(epoch_loss / max(1, len(encoded_rows)))
        model.eval()
        with torch.no_grad():
            latent = model.encode(data.to(device)).detach().cpu().numpy().astype(np.float64)
        return latent, {
            "epochs": int(max(1, int(self.config.epochs))),
            "initial_loss": round(float(losses[0]), 6) if losses else 0.0,
            "final_loss": round(float(losses[-1]), 6) if losses else 0.0,
            "loss": "mse_on_normalized_16_token_vector",
            "device": str(device),
        }

    def _rounded_vector(self, vector: Optional[np.ndarray]) -> Optional[List[float]]:
        if vector is None:
            return None
        return [round(float(item), 6) for item in vector.tolist()]
