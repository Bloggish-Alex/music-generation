#!/usr/bin/env python3
"""Runtime contract for decoding DVAE latent prototypes to grid tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from common.architecture import SymbolID


@dataclass(frozen=True)
class DVAEDecodeRequest:
    """Request to decode one selected token into relative 16-grid tokens."""

    token_id: SymbolID
    latent_vector: List[float]
    previous_last_pitch_scalar: float = 0.0
    source: str = "medoid_mu"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DVAEDecodeResult:
    """Decoded relative token grid produced by the DVAE decoder."""

    token_id: SymbolID
    relative_tokens: List[int]
    type_probabilities: Optional[List[List[float]]] = None
    pitch_values: Optional[List[float]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DVAEDecoderRuntime(Protocol):
    """Physical-token decoder interface used before harmonic rendering."""

    def decode(self, request: DVAEDecodeRequest) -> DVAEDecodeResult:
        ...


class MissingDVAEDecoderRuntime:
    """Placeholder that documents the required model artifact."""

    def __init__(self, expected_model_path: Optional[Path] = None) -> None:
        self.expected_model_path = expected_model_path

    def decode(self, request: DVAEDecodeRequest) -> DVAEDecodeResult:
        path_hint = (
            f" Expected model artifact: {self.expected_model_path}."
            if self.expected_model_path is not None
            else ""
        )
        raise RuntimeError(
            "DVAE decoder runtime is not available yet. "
            "Persist and load the trained DenoisingBarVAE state_dict before "
            "calling decode()."
            + path_hint
        )


class TrainedDVAEDecoderRuntime:
    """Decode selected token latent prototypes through the trained VAE decoder."""

    def __init__(self, artifact_path: str | Path, device: Optional[str] = None) -> None:
        from encoder.vae_bar_encoder import DenoisingBarVAE, DenoisingVAEConfig
        import torch

        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"VAE decoder artifact not found: {self.artifact_path}")
        payload = torch.load(self.artifact_path, map_location=device or "cpu")
        self.config = DenoisingVAEConfig(**payload["config"])
        if device is not None:
            self.config = DenoisingVAEConfig(**{**asdict(self.config), "device": str(device)})
        self.device = str(device or self.config.device)
        self.model = DenoisingBarVAE.build(self.config).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def decode(self, request: DVAEDecodeRequest) -> DVAEDecodeResult:
        import torch

        with torch.no_grad():
            latent = torch.tensor([request.latent_vector], dtype=torch.float32, device=self.device)
            condition = torch.tensor(
                [float(request.previous_last_pitch_scalar)],
                dtype=torch.float32,
                device=self.device,
            )
            hidden = self.model.decoder(self.model.decoder_input(latent, condition))
            rhythm_hidden = self.model.rhythm_adapter(hidden)
            pitch_hidden = self.model.pitch_adapter(hidden)
            type_logits = self.model.type_head(rhythm_hidden).view(-1, self.config.steps_per_bar, 3)
            pitch = self.model.pitch_activation(self.model.pitch_head(pitch_hidden))
            onset = self.model.onset_head(rhythm_hidden)
            sustain = self.model.sustain_head(rhythm_hidden)
            type_prob = torch.softmax(type_logits, dim=-1).detach().cpu().numpy()[0]
            type_pred = type_prob.argmax(axis=-1)
            pitch_values = pitch.detach().cpu().numpy()[0]
            onset_prob = torch.sigmoid(onset).detach().cpu().numpy()[0]
            sustain_prob = torch.sigmoid(sustain).detach().cpu().numpy()[0]
        tokens: List[int] = []
        for token_type, pitch_value in zip(type_pred, pitch_values):
            if int(token_type) == 0:
                tokens.append(-1)
            elif int(token_type) == 1:
                tokens.append(-2)
            else:
                tokens.append(max(0, int(round(float(pitch_value) * float(self.config.pitch_scale)))))
        return DVAEDecodeResult(
            token_id=int(request.token_id),
            relative_tokens=tokens,
            type_probabilities=type_prob.tolist(),
            pitch_values=[float(value) for value in pitch_values.tolist()],
            diagnostics={
                "runtime": self.__class__.__name__,
                "artifact_path": str(self.artifact_path),
                "source": str(request.source),
                "latent_dim": int(len(request.latent_vector)),
                "previous_last_pitch_scalar": float(request.previous_last_pitch_scalar),
                "steps_per_bar": int(self.config.steps_per_bar),
                "onset_probabilities": [float(value) for value in onset_prob.tolist()],
                "sustain_probabilities": [float(value) for value in sustain_prob.tolist()],
            },
        )


class CodebookTokenFallbackDecoder:
    """Fallback adapter that returns already stored relative tokens.

    This is not the target DVAE decode path. It exists so integration can be
    tested while the VAE decoder model artifact is being persisted.
    """

    def __init__(self, token_to_relative_tokens: Dict[int, List[int]]) -> None:
        self.token_to_relative_tokens = {
            int(token_id): [int(token) for token in tokens]
            for token_id, tokens in token_to_relative_tokens.items()
        }

    def decode(self, request: DVAEDecodeRequest) -> DVAEDecodeResult:
        token_id = int(request.token_id)
        if token_id not in self.token_to_relative_tokens:
            raise KeyError(f"Token {token_id} is not present in fallback codebook tokens.")
        return DVAEDecodeResult(
            token_id=token_id,
            relative_tokens=list(self.token_to_relative_tokens[token_id]),
            diagnostics={
                "runtime": self.__class__.__name__,
                "source": "stored_codebook_relative_tokens",
                "target_runtime": "trained_dvae_decoder",
            },
        )
