#!/usr/bin/env python3
"""Render contiguous source bars through a trained DVAE reconstruction path."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from common.config_loader import ConfigView
from diagnostics.dvae_midi_render import DVAEMidiRenderConfig
from model.dvae import DVAEMusicConfig, DenoisingMusicVAE
from pipeline.latent_generation_pipeline import SequenceTensorMidiRenderer


@dataclass(frozen=True)
class DVAEReconstructionConfig:
    """Runtime settings for a continuous DVAE reconstruction listening sample."""

    bars: int = 32
    seed: int = 42
    device: str = "cpu"
    base_pitch: int = 60
    tempo_bpm: int = 120
    source_song_id: Optional[str] = None
    start_bar: Optional[int] = None
    use_source_base_pitch: bool = True
    audio_quality_enabled: bool = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DVAEReconstructionConfig":
        """Read reconstruction settings without coupling to generation settings."""
        section = ConfigView(config).section("dvae_reconstruction_generation")
        return cls(
            bars=int(section.get("bars", 32)),
            seed=int(section.get("seed", 42)),
            device=str(section.get("device", "cpu")),
            base_pitch=int(section.get("base_pitch", 60)),
            tempo_bpm=int(section.get("tempo_bpm", 120)),
            source_song_id=section.get("source_song_id", None),
            start_bar=None if section.get("start_bar", None) is None else int(section["start_bar"]),
            use_source_base_pitch=bool(section.get("use_source_base_pitch", True)),
            audio_quality_enabled=bool(section.get("audio_quality_enabled", False)),
        )


class DVAEReconstructionPipeline:
    """Encode then decode a contiguous real bar sequence for a direct DVAE test."""

    MINIMUM_BARS = 16

    def __init__(self, config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> None:
        base = DVAEReconstructionConfig.from_config(config)
        values = asdict(base)
        for key, value in (overrides or {}).items():
            if value is not None and key in values:
                values[key] = value
        self.config = DVAEReconstructionConfig(**values)

    def run(
        self,
        model_dir: str | Path,
        output_json: str | Path,
        output_midi: str | Path,
        dvae_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Write one continuous encoder-to-decoder reconstruction sample."""
        self._validate_request()
        self._set_seed()
        model_path = Path(model_dir)
        index = self._load_index(model_path / "encoded" / "bar_tensor_index.json")
        selected_rows = self._select_rows(index)
        source_tensors = self._load_tensors(model_path / "encoded" / "bar_tensors.npz", selected_rows)
        dvae_checkpoint = Path(dvae_path) if dvae_path else model_path / "dvae.pt"
        dvae = self._load_dvae(dvae_checkpoint)
        latent_mu, reconstructed = self._reconstruct(dvae, source_tensors)
        base_pitches = self._source_base_pitches(selected_rows)

        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_path = output_json_path.with_suffix(".bar_tensors.npz")
        np.savez_compressed(
            tensor_path,
            source_bars=source_tensors.astype(np.float32),
            reconstructed_bars=reconstructed.astype(np.float32),
            latent_mu=latent_mu.astype(np.float32),
            source_base_pitches=np.asarray(base_pitches, dtype=np.int64),
        )
        midi = SequenceTensorMidiRenderer(self._render_config()).render(
            reconstructed,
            output_midi,
            base_pitch=int(self.config.base_pitch),
            base_pitches=base_pitches if self.config.use_source_base_pitch else None,
        )
        diagnostics = {
            "backend": "dvae_encoder_decoder_reconstruction",
            "model_dir": str(model_path),
            "dvae_checkpoint": str(dvae_checkpoint),
            "source_song_id": str(selected_rows[0]["song_id"]),
            "start_bar": int(selected_rows[0]["bar_index"]),
            "end_bar": int(selected_rows[-1]["bar_index"]),
            "reconstructed_bars": int(reconstructed.shape[0]),
            "minimum_bars": int(self.MINIMUM_BARS),
            "use_source_base_pitch": bool(self.config.use_source_base_pitch),
            "selected_rows": [self._row_summary(row) for row in selected_rows],
            "reconstruction_metrics": self._reconstruction_metrics(source_tensors, reconstructed),
            "tensor_path": str(tensor_path),
            "midi": midi,
        }
        output_json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return diagnostics

    def _validate_request(self) -> None:
        if int(self.config.bars) < int(self.MINIMUM_BARS):
            raise ValueError(
                f"DVAE reconstruction listening samples require at least {self.MINIMUM_BARS} bars; "
                f"received bars={self.config.bars}."
            )

    def _load_index(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Missing encoded bar index: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Encoded bar index is empty or invalid: {path}")
        return [dict(row) for row in rows]

    def _select_rows(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(row)
        ordered = {
            song_id: sorted(song_rows, key=lambda row: int(row.get("bar_index", 0)))
            for song_id, song_rows in grouped.items()
        }
        song_id = self._select_song_id(ordered)
        segments = self._contiguous_segments(ordered[song_id])
        viable = [segment for segment in segments if len(segment) >= int(self.config.bars)]
        if not viable:
            longest = max((len(segment) for segment in segments), default=0)
            raise ValueError(
                f"Song '{song_id}' has no contiguous span of {self.config.bars} bars; longest span is {longest}."
            )
        starts = [(segment, offset) for segment in viable for offset in range(len(segment) - int(self.config.bars) + 1)]
        if self.config.start_bar is not None:
            matching = [(segment, offset) for segment, offset in starts if int(segment[offset].get("bar_index", -1)) == int(self.config.start_bar)]
            if not matching:
                raise ValueError(
                    f"start_bar={self.config.start_bar} cannot provide {self.config.bars} contiguous bars in '{song_id}'."
                )
            segment, offset = matching[0]
        else:
            segment, offset = random.Random(int(self.config.seed) + 7919).choice(starts)
        return [dict(row) for row in segment[offset:offset + int(self.config.bars)]]

    def _select_song_id(self, grouped: Dict[str, List[Dict[str, Any]]]) -> str:
        if self.config.source_song_id:
            requested = str(self.config.source_song_id)
            if requested in grouped:
                return requested
            pattern = re.compile(requested)
            matches = sorted(song_id for song_id in grouped if pattern.search(song_id))
            if matches:
                return matches[0]
            raise ValueError(f"source_song_id not found: {requested}")
        candidates = [song_id for song_id, rows in grouped.items() if len(rows) >= int(self.config.bars)]
        if not candidates:
            raise ValueError(f"No encoded song contains at least {self.config.bars} bars.")
        original_candidates = [song_id for song_id in candidates if not self._is_transposed_song(song_id)]
        return random.Random(int(self.config.seed)).choice(sorted(original_candidates or candidates))

    def _is_transposed_song(self, song_id: str) -> bool:
        return re.search(r"_T[+-]?\d+$", str(song_id)) is not None

    def _contiguous_segments(self, rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        segments: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        previous_index: Optional[int] = None
        for row in rows:
            current_index = int(row.get("bar_index", 0))
            if current and previous_index is not None and current_index != previous_index + 1:
                segments.append(current)
                current = []
            current.append(row)
            previous_index = current_index
        if current:
            segments.append(current)
        return segments

    def _load_tensors(self, path: Path, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Missing encoded bar tensor archive: {path}")
        with np.load(path) as archive:
            tensors = []
            for row in rows:
                key = str(row.get("tensor_key", ""))
                if key not in archive:
                    raise KeyError(f"Missing tensor_key in bar_tensors.npz: {key}")
                tensors.append(np.asarray(archive[key], dtype=np.float32))
        return np.stack(tensors, axis=0).astype(np.float32)

    def _load_dvae(self, path: Path) -> DenoisingMusicVAE:
        if not path.exists():
            raise FileNotFoundError(f"Missing DVAE checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        model = DenoisingMusicVAE(DVAEMusicConfig(**checkpoint["config"])).to(self.config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def _reconstruct(self, dvae: DenoisingMusicVAE, source_tensors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            source = torch.from_numpy(source_tensors.astype(np.float32)).to(self.config.device)
            mu = dvae.encode_mu(source)
            pitch, state_logits, velocity, chord = dvae.decoder(mu)
            states = torch.argmax(state_logits, dim=-1)
            state_one_hot = torch.nn.functional.one_hot(states, num_classes=3).float()
            reconstructed = torch.zeros_like(source)
            reconstructed[..., 0] = pitch
            reconstructed[..., 1:4] = state_one_hot
            reconstructed[..., 4] = velocity
            reconstructed[..., 5:5 + chord.shape[-1]] = chord
        return (
            mu.detach().cpu().numpy().astype(np.float32),
            reconstructed.detach().cpu().numpy().astype(np.float32),
        )

    def _source_base_pitches(self, rows: Sequence[Dict[str, Any]]) -> List[int]:
        pitches = []
        for row in rows:
            diagnostics = row.get("diagnostics", {}) if isinstance(row, dict) else {}
            value = diagnostics.get("base_pitch")
            pitches.append(int(value) if value is not None else int(self.config.base_pitch))
        return pitches

    def _reconstruction_metrics(self, source: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        target_states = np.argmax(source[..., 1:4], axis=-1)
        predicted_states = np.argmax(reconstructed[..., 1:4], axis=-1)
        note_mask = target_states == 1
        pitch_error = np.abs(source[..., 0] - reconstructed[..., 0])
        return {
            "tensor_mse": float(np.mean((source - reconstructed) ** 2)),
            "state_accuracy": float(np.mean(target_states == predicted_states)),
            "note_on_pitch_mae": float(np.mean(pitch_error[note_mask])) if np.any(note_mask) else 0.0,
            "note_on_count": int(np.sum(note_mask)),
            "reconstructed_note_on_count": int(np.sum(predicted_states == 1)),
        }

    def _row_summary(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "tensor_key": str(row.get("tensor_key", "")),
            "song_id": str(row.get("song_id", "")),
            "bar_index": int(row.get("bar_index", 0)),
        }

    def _render_config(self) -> DVAEMidiRenderConfig:
        return DVAEMidiRenderConfig(
            tempo_bpm=int(self.config.tempo_bpm),
            default_base_pitch=int(self.config.base_pitch),
            audio_quality_enabled=bool(self.config.audio_quality_enabled),
        )

    def _set_seed(self) -> None:
        random.seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))
        torch.manual_seed(int(self.config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.config.seed))
