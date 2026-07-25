#!/usr/bin/env python3
"""Decode real training latents with the DVAE decoder for isolation tests."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from common.config_loader import ConfigView
from diagnostics.dvae_midi_render import DVAEMidiRenderConfig
from model.dvae import DVAEMusicConfig, DenoisingMusicVAE
from pipeline.latent_generation_pipeline import SequenceTensorMidiRenderer
from pipeline.latent_transformer_training_pipeline import LatentDatasetReader


@dataclass(frozen=True)
class DVAELatentOracleConfig:
    """Settings for real-latent DVAE decode isolation."""

    bars: int = 32
    seed: int = 42
    device: str = "cpu"
    base_pitch: int = 60
    tempo_bpm: int = 100
    seed_song_id: Optional[str] = None
    start_bar: Optional[int] = None
    audio_quality_enabled: bool = True
    use_source_base_pitch: bool = True

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DVAELatentOracleConfig":
        section = ConfigView(config).section("dvae_latent_oracle_generation")
        remi = ConfigView(config).section("remi_motion_generation")
        latent = ConfigView(config).section("latent_generation")
        start_bar = section.get("start_bar", None)
        return cls(
            bars=int(section.get("bars", remi.get("bars", latent.get("bars", 32)))),
            seed=int(section.get("seed", remi.get("seed", latent.get("seed", 42)))),
            device=str(section.get("device", remi.get("device", latent.get("device", "cpu")))),
            base_pitch=int(section.get("base_pitch", remi.get("base_pitch", latent.get("base_pitch", 60)))),
            tempo_bpm=int(section.get("tempo_bpm", remi.get("tempo_bpm", latent.get("tempo_bpm", 100)))),
            seed_song_id=section.get("seed_song_id", None),
            start_bar=None if start_bar is None else int(start_bar),
            audio_quality_enabled=bool(section.get("audio_quality_enabled", True)),
            use_source_base_pitch=bool(section.get("use_source_base_pitch", True)),
        )


class DVAELatentOraclePipeline:
    """Render a continuous true latent sequence through only the DVAE decoder."""

    def __init__(self, config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> None:
        base = DVAELatentOracleConfig.from_config(config)
        values = asdict(base)
        for key, value in (overrides or {}).items():
            if value is not None and key in values:
                values[key] = value
        self.config = DVAELatentOracleConfig(**values)

    def run(
        self,
        model_dir: str | Path,
        output_json: str | Path,
        output_midi: str | Path,
        dvae_path: Optional[str | Path] = None,
        latent_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Decode one real latent sequence and write MIDI, tensors, and diagnostics."""
        self._set_seed()
        model_path = Path(model_dir)
        latent_path = Path(latent_dir) if latent_dir else model_path / "latent"
        mu, rows, latent_summary = LatentDatasetReader().load(latent_path)
        grouped = self._group_rows(rows)
        song_id = self._select_song_id(grouped)
        ordered = grouped[song_id]
        start = self._start_bar(ordered)
        selected_indices = ordered[start:start + int(self.config.bars)]
        if not selected_indices:
            raise ValueError("No latent rows selected for DVAE oracle decode.")

        selected_mu = mu[selected_indices].astype(np.float32)
        dvae = self._load_dvae(Path(dvae_path) if dvae_path else model_path / "dvae.pt")
        tensors = self._decode_tensors(dvae, selected_mu)
        base_pitches = self._source_base_pitches(model_path, rows, selected_indices)

        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_path = output_json_path.with_suffix(".bar_tensors.npz")
        payload: Dict[str, Any] = {
            "bars": tensors.astype(np.float32),
            "latent_mu": selected_mu.astype(np.float32),
            "source_row_indices": np.asarray(selected_indices, dtype=np.int64),
        }
        if base_pitches is not None:
            payload["source_base_pitches"] = np.asarray(base_pitches, dtype=np.int64)
        np.savez_compressed(tensor_path, **payload)

        midi_diag = SequenceTensorMidiRenderer(self._render_config()).render(
            tensors,
            output_midi,
            base_pitch=int(self.config.base_pitch),
            base_pitches=base_pitches,
        )
        diagnostics = {
            "backend": "dvae_real_latent_oracle",
            "model_dir": str(model_path),
            "dvae_checkpoint": str(Path(dvae_path) if dvae_path else model_path / "dvae.pt"),
            "latent_dir": str(latent_path),
            "source_song_id": str(song_id),
            "start_bar": int(start),
            "generated_bars": int(tensors.shape[0]),
            "use_source_base_pitch": bool(base_pitches is not None),
            "selected_rows": [self._row_summary(rows[index], index) for index in selected_indices],
            "latent_summary": latent_summary,
            "sequence_summary": self._sequence_summary(tensors, selected_mu),
            "tensor_path": str(tensor_path),
            "midi": midi_diag,
        }
        output_json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return diagnostics

    def _load_dvae(self, path: Path) -> DenoisingMusicVAE:
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        model = DenoisingMusicVAE(DVAEMusicConfig(**checkpoint["config"])).to(self.config.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def _decode_tensors(self, dvae: DenoisingMusicVAE, latent_mu: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = torch.from_numpy(latent_mu.astype(np.float32)).to(self.config.device)
            pitch, state_logits, velocity, chord = dvae.decoder(z)
            state = torch.argmax(state_logits, dim=-1)
            state_one_hot = torch.nn.functional.one_hot(state, num_classes=3).float()
            tensor = torch.zeros(
                (z.shape[0], int(dvae.config.tracks), int(dvae.config.steps_per_bar), int(dvae.config.feature_dim)),
                dtype=torch.float32,
                device=z.device,
            )
            tensor[..., 0] = pitch
            tensor[..., 1:4] = state_one_hot
            tensor[..., 4] = velocity
            tensor[..., 5:5 + chord.shape[-1]] = chord
        return tensor.detach().cpu().numpy().astype(np.float32)

    def _group_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        return {
            song_id: sorted(indices, key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx))))
            for song_id, indices in grouped.items()
        }

    def _select_song_id(self, grouped: Dict[str, List[int]]) -> str:
        if self.config.seed_song_id:
            seed_song_id = str(self.config.seed_song_id)
            if seed_song_id in grouped:
                return seed_song_id
            pattern = re.compile(seed_song_id)
            matches = [song_id for song_id in grouped if pattern.search(song_id)]
            if matches:
                return sorted(matches)[0]
            raise ValueError(f"seed_song_id not found: {seed_song_id}")
        candidates = [song_id for song_id, indices in grouped.items() if len(indices) >= int(self.config.bars)]
        if not candidates:
            candidates = sorted(grouped)
        return random.Random(int(self.config.seed)).choice(sorted(candidates))

    def _start_bar(self, ordered_indices: Sequence[int]) -> int:
        max_start = max(0, len(ordered_indices) - int(self.config.bars))
        if self.config.start_bar is not None:
            if int(self.config.start_bar) > max_start:
                raise ValueError(
                    f"start_bar={self.config.start_bar} cannot provide {self.config.bars} bars; max_start={max_start}."
                )
            return max(0, int(self.config.start_bar))
        return random.Random(int(self.config.seed) + 7919).randint(0, max_start) if max_start > 0 else 0

    def _source_base_pitches(
        self,
        model_dir: Path,
        rows: Sequence[Dict[str, Any]],
        selected_indices: Sequence[int],
    ) -> Optional[List[int]]:
        if not bool(self.config.use_source_base_pitch):
            return None
        index_path = model_dir / "encoded" / "bar_tensor_index.json"
        if not index_path.exists():
            return None
        rows_by_key = self._encoded_index_by_key(index_path)
        base_pitches: List[int] = []
        for row_index in selected_indices:
            tensor_key = str(rows[row_index].get("tensor_key", ""))
            encoded_row = rows_by_key.get(tensor_key) or rows_by_key.get(self._strip_transposition(tensor_key))
            diagnostics = encoded_row.get("diagnostics", {}) if encoded_row else {}
            value = diagnostics.get("base_pitch")
            base_pitches.append(int(value) if value is not None else int(self.config.base_pitch))
        return base_pitches

    def _encoded_index_by_key(self, index_path: Path) -> Dict[str, Dict[str, Any]]:
        rows = json.loads(index_path.read_text(encoding="utf-8"))
        return {str(row.get("tensor_key", "")): dict(row) for row in rows}

    def _strip_transposition(self, tensor_key: str) -> str:
        return re.sub(r"_T[+-]?\d+(?=__bar_)", "", str(tensor_key))

    def _sequence_summary(self, tensors: np.ndarray, latents: np.ndarray) -> Dict[str, Any]:
        note_on = tensors[..., 2] > 0.5
        rest = tensors[..., 1] > 0.5
        hold = tensors[..., 3] > 0.5
        active = note_on | hold
        note_per_bar = note_on.sum(axis=(1, 2)).astype(int)
        rest_per_bar = rest.sum(axis=(1, 2)).astype(int)
        hold_per_bar = hold.sum(axis=(1, 2)).astype(int)
        active_per_bar = active.sum(axis=(1, 2)).astype(int)
        step_distance = np.linalg.norm(np.diff(latents, axis=0), axis=1) if len(latents) > 1 else np.zeros((0,), dtype=np.float32)
        return {
            "note_on_per_bar": self._int_list_summary(note_per_bar),
            "rest_per_bar": self._int_list_summary(rest_per_bar),
            "hold_per_bar": self._int_list_summary(hold_per_bar),
            "active_slots_per_bar": self._int_list_summary(active_per_bar),
            "latent_step_distance": self._float_list_summary(step_distance),
        }

    def _int_list_summary(self, values: np.ndarray) -> Dict[str, Any]:
        if values.size == 0:
            return {"values": [], "mean": 0.0, "min": 0, "max": 0}
        return {
            "values": [int(item) for item in values.tolist()],
            "mean": float(np.mean(values)),
            "min": int(np.min(values)),
            "max": int(np.max(values)),
        }

    def _float_list_summary(self, values: np.ndarray) -> Dict[str, Any]:
        if values.size == 0:
            return {"values": [], "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "values": [float(item) for item in values.tolist()],
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    def _row_summary(self, row: Dict[str, Any], row_index: int) -> Dict[str, Any]:
        return {
            "row_index": int(row_index),
            "tensor_key": str(row.get("tensor_key", "")),
            "song_id": str(row.get("song_id", "")),
            "bar_index": int(row.get("bar_index", 0)),
            "action": str(row.get("action", "")),
            "form": str(row.get("form", "")),
            "section_label": str(row.get("section_label", "")),
        }

    def _render_config(self) -> DVAEMidiRenderConfig:
        return DVAEMidiRenderConfig(
            tempo_bpm=int(self.config.tempo_bpm),
            default_base_pitch=int(self.config.base_pitch),
            audio_quality_enabled=bool(self.config.audio_quality_enabled),
        )

    def _set_seed(self) -> None:
        seed = int(self.config.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
