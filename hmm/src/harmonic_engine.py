#!/usr/bin/env python3
"""Configuration-driven harmonic realization for generated bar sequences."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config_loader import ConfigLoader, ConfigView, ROOT_DIR
from core_data import NoteRecord
from bar_density import TokenDensityAnalyzer
from candidate_selector import (
    CandidateSelectionContext,
    CandidateSelectorModel,
    LearnedCandidateSelector,
)
from generation_data import (
    CodebookCandidate,
    CodebookEntry,
    GenerationResult,
    HarmonyBarPlan,
    RealizedBar,
    SampledBar,
    SectionPlanItem,
)


@dataclass(frozen=True)
class HarmonyEngineConfig:
    style: str = "classic"
    key: str = "C"
    mode: str = "major"
    base_octave: int = 3
    base_key_pitch: int = 48
    matrix_path: str = str(ROOT_DIR / "config" / "harmony_matrix.json")
    min_bar_base_pitch: int = 32
    max_bar_base_pitch: int = 60
    rest_token: int = -1
    sustain_token: int = -2
    default_start_degree: str = "I"
    fallback_policy: str = "raw"


@dataclass(frozen=True)
class RareBarSelectionConfig:
    enabled: bool = True
    min_candidates: int = 2
    harmony_weight: float = 1.0
    state_weight: float = 1.0
    density_weight: float = 2.0
    position_weight: float = 0.5
    density_sigma: float = 0.25
    position_sigma: float = 0.35
    top_k: int = 32
    temperature: float = 0.5
    min_score: float = 1.0e-9
    diagnostics_top_k: int = 5

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "RareBarSelectionConfig":
        section = ConfigView(config).section("harmonic_engine")
        selector = section.get("rare_bar_selector", {})
        if not isinstance(selector, dict):
            selector = {}
        return cls(
            enabled=bool(selector.get("enabled", True)),
            min_candidates=int(selector.get("min_candidates", 2)),
            harmony_weight=float(selector.get("harmony_weight", 1.0)),
            state_weight=float(selector.get("state_weight", 1.0)),
            density_weight=float(selector.get("density_weight", 2.0)),
            position_weight=float(selector.get("position_weight", 0.5)),
            density_sigma=float(selector.get("density_sigma", 0.25)),
            position_sigma=float(selector.get("position_sigma", 0.35)),
            top_k=int(selector.get("top_k", 32)),
            temperature=float(selector.get("temperature", 0.5)),
            min_score=float(selector.get("min_score", 1.0e-9)),
            diagnostics_top_k=int(selector.get("diagnostics_top_k", 5)),
        )


@dataclass(frozen=True)
class RareBarSelection:
    entry: CodebookEntry
    diagnostics: Dict[str, Any]


class KeyPitchResolver:
    """Convert user-facing key/octave settings into a MIDI root pitch."""

    SEMITONES = {
        "C": 0,
        "B#": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "FB": 4,
        "E#": 5,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
        "CB": 11,
    }
    SUPPORTED_MODES = {"major", "minor"}

    def resolve(self, key: str, mode: str, base_octave: int) -> int:
        normalized_key = self._normalize_key(key)
        normalized_mode = str(mode).lower()
        if normalized_mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Supported modes: {sorted(self.SUPPORTED_MODES)}")
        return 12 * (int(base_octave) + 1) + self.SEMITONES[normalized_key]

    def _normalize_key(self, key: str) -> str:
        normalized = str(key).strip().upper()
        if normalized not in self.SEMITONES:
            raise ValueError(f"Unsupported key '{key}'. Use names like C, F#, Bb.")
        return normalized


@dataclass(frozen=True)
class HarmonicCliOverrides:
    """Optional CLI overrides for user-facing harmonic key settings."""

    style: Optional[str] = None
    key: Optional[str] = None
    mode: Optional[str] = None
    base_octave: Optional[int] = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--style", default=None, help="Harmony matrix style name. Defaults to classic.")
        parser.add_argument("--key", default=None, help="Tonic key name, for example C, F#, or Bb.")
        parser.add_argument("--mode", default=None, choices=sorted(KeyPitchResolver.SUPPORTED_MODES))
        parser.add_argument("--base-octave", type=int, default=None)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "HarmonicCliOverrides":
        return cls(
            style=getattr(args, "style", None),
            key=getattr(args, "key", None),
            mode=getattr(args, "mode", None),
            base_octave=getattr(args, "base_octave", None),
        )

    def apply(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if self.style is None and self.key is None and self.mode is None and self.base_octave is None:
            return config
        updated = dict(config)
        section = dict(updated.get("harmonic_engine", {}))
        if self.style is not None:
            section["style"] = self.style
        if self.key is not None:
            section["key"] = self.key
        if self.mode is not None:
            section["mode"] = self.mode
        if self.base_octave is not None:
            section["base_octave"] = self.base_octave
        section.pop("base_key_pitch", None)
        updated["harmonic_engine"] = section
        return updated


@dataclass(frozen=True)
class HarmonicTrackConfig:
    name: str

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "HarmonicTrackConfig":
        return cls(name=str(payload["name"]))


@dataclass(frozen=True)
class HarmonicRenderConfig:
    tempo: int = 120
    time_signature: str = "4/4"
    velocity: int = 72
    base_key_pitch: int = 48
    track_mode: str = "single"
    tracks: tuple[HarmonicTrackConfig, ...] = (
        HarmonicTrackConfig(name="high"),
        HarmonicTrackConfig(name="middle"),
        HarmonicTrackConfig(name="low"),
    )


class TrackRenderStrategy:
    """Select MIDI output tracks for harmonized notes."""

    def tracks(self) -> tuple[HarmonicTrackConfig, ...]:
        raise NotImplementedError

    def track_for_pitch(self, pitch: int) -> str:
        raise NotImplementedError

    def diagnostics(self) -> Dict[str, Any]:
        return {}


class SingleTrackRenderStrategy(TrackRenderStrategy):
    """Render all notes into one piano track."""

    def tracks(self) -> tuple[HarmonicTrackConfig, ...]:
        return (HarmonicTrackConfig(name="piano"),)

    def track_for_pitch(self, pitch: int) -> str:
        return "piano"


class SplitByPitchTrackRenderStrategy(TrackRenderStrategy):
    """Render notes into high/middle/low tracks using dynamic key boundaries."""

    def __init__(self, config: HarmonicRenderConfig) -> None:
        self.config = config
        self.boundary = DynamicTrackBoundary.from_base_key(config.base_key_pitch)

    def tracks(self) -> tuple[HarmonicTrackConfig, ...]:
        return self.config.tracks

    def track_for_pitch(self, pitch: int) -> str:
        high_track = self._track_named("high")
        middle_track = self._track_named("middle")
        low_track = self._track_named("low")
        if pitch >= self.boundary.middle_ceil:
            return high_track.name
        if pitch > self.boundary.bass_ceil:
            return middle_track.name
        return low_track.name

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "dynamic_track_boundary": asdict(self.boundary),
        }

    def _track_named(self, name: str) -> HarmonicTrackConfig:
        for track in self.config.tracks:
            if track.name == name:
                return track
        raise ValueError(f"harmonic_render.tracks must include a '{name}' track.")


class TrackRenderStrategyFactory:
    """Build the configured track render strategy."""

    def from_config(self, config: HarmonicRenderConfig) -> TrackRenderStrategy:
        if config.track_mode == "single":
            return SingleTrackRenderStrategy()
        if config.track_mode == "split_by_pitch":
            return SplitByPitchTrackRenderStrategy(config)
        raise ValueError("harmonic_render.track_mode must be 'single' or 'split_by_pitch'.")


@dataclass(frozen=True)
class DynamicTrackBoundary:
    bass_ceil: int
    middle_ceil: int

    @classmethod
    def from_base_key(cls, base_key_pitch: int) -> "DynamicTrackBoundary":
        return cls(
            bass_ceil=int(base_key_pitch) + 8,
            middle_ceil=int(base_key_pitch) + 24,
        )


class HarmonyMatrixLibrary:
    """Load style-specific degree transition matrices."""

    def __init__(self, matrix_path: str | Path) -> None:
        self.matrix_path = Path(matrix_path)
        if not self.matrix_path.is_absolute():
            self.matrix_path = ROOT_DIR / "config" / self.matrix_path
        self.payload = json.loads(self.matrix_path.read_text(encoding="utf-8"))

    def load_style(self, style: str) -> tuple[List[str], np.ndarray]:
        value = self.payload[style]
        if isinstance(value, dict):
            degrees = [str(item) for item in value["degrees"]]
            matrix = np.asarray(value["matrix"], dtype=np.float64)
        else:
            matrix = np.asarray(value, dtype=np.float64)
            degrees = ["I", "II", "III", "IV", "V", "VI"][: matrix.shape[0]]
        return degrees, self._normalize_rows(matrix)

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        result = matrix.copy()
        for row in range(result.shape[0]):
            total = float(result[row].sum())
            if total > 0:
                result[row] /= total
        return result


class DegreeOffsetMap:
    """Map Roman-degree labels to semitone offsets for a mode."""

    MODE_OFFSETS = {
        "major": {
            "I": 0,
            "II": 2,
            "III": 4,
            "IV": 5,
            "V": 7,
            "VI": 9,
            "VII": 11,
        },
        "minor": {
            "I": 0,
            "II": 2,
            "III": 3,
            "IV": 5,
            "V": 7,
            "VI": 8,
            "VII": 10,
        },
    }
    ACCIDENTAL_OFFSETS = {
        "bII": 1,
        "bIII": 3,
        "#III": 5,
        "#IV": 6,
        "bV": 6,
        "bVI": 8,
        "bVII": 10,
    }

    def offset(self, degree: str, mode: str) -> int:
        if degree in self.ACCIDENTAL_OFFSETS:
            return int(self.ACCIDENTAL_OFFSETS[degree])
        return int(self.MODE_OFFSETS[str(mode).lower()][degree])


class BackwardGravityMarkovBridge:
    """Sample a Markov chain conditioned on a target final degree."""

    def __init__(self, matrix: np.ndarray, fallback_policy: str = "raw") -> None:
        self.matrix = matrix
        self.fallback_policy = fallback_policy
        self.diagnostics: List[Dict[str, Any]] = []

    def sample(
        self,
        length: int,
        start_state: int,
        end_state: Optional[int],
        rng: np.random.Generator,
    ) -> List[int]:
        if length <= 0:
            return []
        if end_state is None:
            return self._ordinary_walk(length, start_state, rng)
        beta = self._compute_beta(length, end_state)
        return self._bridge_walk(beta, length, start_state, rng)

    def _compute_beta(self, length: int, end_state: int) -> np.ndarray:
        beta = np.zeros((length, self.matrix.shape[0]), dtype=np.float64)
        beta[length - 1, end_state] = 1.0
        for t in range(length - 2, -1, -1):
            beta[t] = self.matrix @ beta[t + 1]
        return beta

    def _bridge_walk(
        self,
        beta: np.ndarray,
        length: int,
        start_state: int,
        rng: np.random.Generator,
    ) -> List[int]:
        current = int(start_state)
        progression = [current]
        for t in range(0, length - 1):
            raw = self.matrix[current]
            corrected = raw * beta[t + 1]
            used_fallback = False
            if float(corrected.sum()) > 0:
                probs = corrected / corrected.sum()
            else:
                used_fallback = True
                probs = self._fallback_probs(raw)
            next_state = int(rng.choice(len(probs), p=probs))
            self.diagnostics.append({
                "t": t,
                "current_state": current,
                "raw_probs": raw.tolist(),
                "beta_next": beta[t + 1].tolist(),
                "corrected_probs": probs.tolist(),
                "used_fallback": used_fallback,
                "next_state": next_state,
            })
            progression.append(next_state)
            current = next_state
        return progression

    def _ordinary_walk(self, length: int, start_state: int, rng: np.random.Generator) -> List[int]:
        current = int(start_state)
        progression = [current]
        for _ in range(length - 1):
            probs = self._fallback_probs(self.matrix[current])
            current = int(rng.choice(len(probs), p=probs))
            progression.append(current)
        return progression

    def _fallback_probs(self, row: np.ndarray) -> np.ndarray:
        if float(row.sum()) > 0:
            return row / row.sum()
        if self.fallback_policy == "uniform":
            return np.full(len(row), 1.0 / len(row), dtype=np.float64)
        probs = np.zeros(len(row), dtype=np.float64)
        probs[0] = 1.0
        return probs


class CadenceStrategy:
    """Translate section cadence names into Markov bridge constraints."""

    def plan(
        self,
        length: int,
        cadence: str,
        start_state: int,
        degree_to_index: Dict[str, int],
        bridge: BackwardGravityMarkovBridge,
        rng: np.random.Generator,
    ) -> List[int]:
        cadence = cadence or "none"
        if cadence == "perfect" and length >= 2:
            prefix = bridge.sample(length - 1, start_state, degree_to_index["V"], rng)
            return prefix + [degree_to_index["I"]]
        if cadence == "half":
            return bridge.sample(length, start_state, degree_to_index["V"], rng)
        if cadence == "open":
            return bridge.sample(length, start_state, degree_to_index["VI"], rng)
        return bridge.sample(length, start_state, None, rng)


class HarmonyProgressionPlanner:
    """Create per-section degree and base-pitch plans."""

    def __init__(self, config: HarmonyEngineConfig) -> None:
        self.config = config
        self.degrees, self.matrix = HarmonyMatrixLibrary(config.matrix_path).load_style(config.style)
        self.degree_to_index = {degree: index for index, degree in enumerate(self.degrees)}
        self.offsets = DegreeOffsetMap()
        self.diagnostics: List[Dict[str, Any]] = []

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "HarmonyProgressionPlanner":
        section = ConfigView(config).section("harmonic_engine")
        key = str(section.get("key", "C"))
        mode = str(section.get("mode", "major"))
        base_octave = int(section.get("base_octave", 3))
        if "key" in section or "mode" in section or "base_octave" in section:
            base_key_pitch = KeyPitchResolver().resolve(key, mode, base_octave)
        else:
            base_key_pitch = int(section.get("base_key_pitch", 48))
        return cls(HarmonyEngineConfig(
            style=str(section.get("style", "classic")),
            key=key,
            mode=mode,
            base_octave=base_octave,
            base_key_pitch=base_key_pitch,
            matrix_path=str(section.get("matrix_path", ROOT_DIR / "config" / "harmony_matrix.json")),
            min_bar_base_pitch=int(section.get("min_bar_base_pitch", 32)),
            max_bar_base_pitch=int(section.get("max_bar_base_pitch", 60)),
            rest_token=int(section.get("rest_token", -1)),
            sustain_token=int(section.get("sustain_token", -2)),
            default_start_degree=str(section.get("default_start_degree", "I")),
            fallback_policy=str(section.get("fallback_policy", "raw")),
        ))

    def plan(self, section_plan: Sequence[SectionPlanItem], seed: Optional[int]) -> List[HarmonyBarPlan]:
        rng = np.random.default_rng(seed)
        result: List[HarmonyBarPlan] = []
        for section in section_plan:
            bridge = BackwardGravityMarkovBridge(self.matrix, self.config.fallback_policy)
            section_base = self.config.base_key_pitch + int(section.pitch_offset)
            start_degree = str(section.start_degree or self.config.default_start_degree)
            progression = CadenceStrategy().plan(
                int(section.bars),
                str(section.cadence),
                self.degree_to_index[start_degree],
                self.degree_to_index,
                bridge,
                rng,
            )
            for local_index, degree_index in enumerate(progression):
                degree = self.degrees[degree_index]
                bar_base = self._clamp(section_base + self.offsets.offset(degree, self.config.mode))
                result.append(HarmonyBarPlan(
                    section=section.name,
                    section_local_index=local_index,
                    degree=degree,
                    degree_index=int(degree_index),
                    section_base_pitch=int(section_base),
                    bar_base_pitch=int(bar_base),
                    cadence=section.cadence,
                ))
            self.diagnostics.append({
                "section": section.name,
                "cadence": section.cadence,
                "section_base_pitch": int(section_base),
                "progression": [self.degrees[index] for index in progression],
                "bridge_steps": bridge.diagnostics,
            })
        return result

    def _clamp(self, value: int) -> int:
        return max(self.config.min_bar_base_pitch, min(self.config.max_bar_base_pitch, value))


@dataclass(frozen=True)
class VoiceSharingConfig:
    enabled: bool = True
    threshold: float = 0.8


class TokenHarmonizer:
    """Convert relative codebook tokens into pitched note spans."""

    def __init__(
        self,
        rest_token: int = -1,
        sustain_token: int = -2,
        velocity: int = 72,
        voice_sharing: VoiceSharingConfig = VoiceSharingConfig(),
    ) -> None:
        self.rest_token = rest_token
        self.sustain_token = sustain_token
        self.velocity = velocity
        self.voice_sharing = voice_sharing

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "TokenHarmonizer":
        engine = ConfigView(config).section("harmonic_engine")
        render = ConfigView(config).section("harmonic_render")
        return cls(
            rest_token=int(engine.get("rest_token", -1)),
            sustain_token=int(engine.get("sustain_token", -2)),
            velocity=int(render.get("velocity", ConfigView(config).section("midi_render").get("velocity", 72))),
            voice_sharing=VoiceSharingConfig(
                enabled=(
                    bool(render.get("voice_sharing_enabled", True))
                    and str(render.get("track_mode", "single")) == "single"
                ),
                threshold=float(render.get("voice_sharing_threshold", 0.8)),
            ),
        )

    def harmonize(
        self,
        tokens: Sequence[int],
        bar_base_pitch: int,
        bar_length_ql: float = 4.0,
        sharing_score: float = 0.0,
    ) -> List[NoteRecord]:
        step_ql = bar_length_ql / max(1, len(tokens))
        notes: List[NoteRecord] = []
        current_pitch: Optional[int] = None
        current_start = 0
        current_length = 0
        apply_voice_sharing = self.should_apply_voice_sharing(sharing_score)
        for slot, raw_token in enumerate(tokens):
            token = int(raw_token)
            if token >= 0:
                if current_pitch is not None:
                    notes.extend(self._notes(current_pitch, current_start, current_length, step_ql, apply_voice_sharing))
                current_pitch = int(bar_base_pitch + token)
                current_start = slot
                current_length = 1
            elif token == self.sustain_token and current_pitch is not None:
                current_length += 1
            else:
                if current_pitch is not None:
                    notes.extend(self._notes(current_pitch, current_start, current_length, step_ql, apply_voice_sharing))
                current_pitch = None
                current_length = 0
        if current_pitch is not None:
            notes.extend(self._notes(current_pitch, current_start, current_length, step_ql, apply_voice_sharing))
        return notes

    def should_apply_voice_sharing(self, sharing_score: float) -> bool:
        return self.voice_sharing.enabled and float(sharing_score) > self.voice_sharing.threshold

    def _notes(
        self,
        pitch: int,
        start_slot: int,
        length_slots: int,
        step_ql: float,
        apply_voice_sharing: bool,
    ) -> List[NoteRecord]:
        intervals = (0,)
        return [
            self._note(pitch + interval, start_slot, length_slots, step_ql)
            for interval in intervals
            if 0 <= pitch + interval <= 127
        ]

    def _note(self, pitch: int, start_slot: int, length_slots: int, step_ql: float) -> NoteRecord:
        return NoteRecord(
            pitch=int(pitch),
            onset_ql=float(start_slot * step_ql),
            duration_ql=float(max(1, length_slots) * step_ql),
            velocity=self.velocity,
        )


class HarmonicMidiRenderer:
    """Write harmonized notes to multi-track MIDI using pitch ranges."""

    def __init__(self, config: HarmonicRenderConfig) -> None:
        self.config = config
        self.track_strategy = TrackRenderStrategyFactory().from_config(config)

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "HarmonicMidiRenderer":
        render = ConfigView(config).section("harmonic_render")
        midi = ConfigView(config).section("midi_render")
        engine = HarmonyProgressionPlanner.from_style_config(config).config
        tracks = tuple(render.get("tracks", [
            {"name": "high"},
            {"name": "middle"},
            {"name": "low"},
        ]))
        track_configs = tuple(HarmonicTrackConfig.from_mapping(track) for track in tracks)
        return cls(HarmonicRenderConfig(
            tempo=int(render.get("tempo", midi.get("tempo", 120))),
            time_signature=str(render.get("time_signature", midi.get("time_signature", "4/4"))),
            velocity=int(render.get("velocity", midi.get("velocity", 72))),
            base_key_pitch=int(engine.base_key_pitch),
            track_mode=str(render.get("track_mode", "single")),
            tracks=track_configs,
        ))

    def write(self, bars: Sequence[RealizedBar], output_path: str | Path) -> Dict[str, Any]:
        import mido

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ts_num, ts_den = self._parse_time_signature(self.config.time_signature)
        ticks_per_beat = 480
        bar_length_ql = ts_num * (4.0 / ts_den)
        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        tracks = {track.name: mido.MidiTrack() for track in self.track_strategy.tracks()}
        for name, track in tracks.items():
            mid.tracks.append(track)
            track.append(mido.MetaMessage("track_name", name=name, time=0))
        events_by_track: Dict[str, List[tuple[int, str, int, int]]] = {name: [] for name in tracks}
        for bar_index, bar in enumerate(bars):
            for note in bar.notes:
                track_name = self.track_strategy.track_for_pitch(int(note.pitch))
                start_ql = bar_index * bar_length_ql + float(note.onset_ql)
                end_ql = start_ql + float(note.duration_ql)
                start_tick = int(round(start_ql * ticks_per_beat))
                end_tick = int(round(end_ql * ticks_per_beat))
                events_by_track[track_name].append((start_tick, "on", int(note.pitch), int(note.velocity)))
                events_by_track[track_name].append((end_tick, "off", int(note.pitch), 0))
        for name, track in tracks.items():
            self._write_track_events(track, sorted(events_by_track[name], key=lambda item: (item[0], item[1])))
        mid.save(str(output_path))
        diagnostics = {
            "output_path": str(output_path),
            "bar_count": len(bars),
            "track_mode": self.config.track_mode,
            "track_event_counts": {name: len(events) for name, events in events_by_track.items()},
        }
        diagnostics.update(self.track_strategy.diagnostics())
        return diagnostics

    def _write_track_events(self, track: Any, events: Sequence[tuple[int, str, int, int]]) -> None:
        previous_tick = 0
        for tick, kind, pitch, velocity in events:
            delta = max(0, tick - previous_tick)
            track.append(__import__("mido").Message(
                "note_on" if kind == "on" else "note_off",
                note=pitch,
                velocity=velocity,
                time=delta,
            ))
            previous_tick = tick

    def _parse_time_signature(self, value: str) -> tuple[int, int]:
        parts = value.split("/")
        return int(parts[0]), int(parts[1])


class RareBarSelector:
    """Select a concrete candidate bar from a broad codebook label distribution."""

    MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
    MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}
    MAJOR_TRIADS = {
        "I": {0, 4, 7},
        "II": {0, 3, 7},
        "III": {0, 3, 7},
        "IV": {0, 4, 7},
        "V": {0, 4, 7},
        "VI": {0, 3, 7},
        "VII": {0, 3, 6},
    }
    MINOR_TRIADS = {
        "I": {0, 3, 7},
        "II": {0, 3, 6},
        "III": {0, 4, 7},
        "IV": {0, 3, 7},
        "V": {0, 4, 7},
        "VI": {0, 4, 7},
        "VII": {0, 4, 7},
    }

    def __init__(self, config: RareBarSelectionConfig, mode: str) -> None:
        self.config = config
        self.mode = str(mode).lower()
        self.density = TokenDensityAnalyzer()

    def select(
        self,
        entry: CodebookEntry,
        sampled: SampledBar,
        harmony: HarmonyBarPlan,
        section_length: int,
        rng: np.random.Generator,
    ) -> RareBarSelection:
        if (
            not self.config.enabled
            or len(entry.candidates) < self.config.min_candidates
        ):
            return RareBarSelection(entry=entry, diagnostics={
                "used": False,
                "reason": "disabled_or_insufficient_candidates",
                "candidate_count": len(entry.candidates),
            })

        target_density = self.density.analyze(sampled.relative_tokens)
        target_position = self._target_position(sampled, section_length)
        scored = [
            self._score(candidate, sampled, harmony, target_density.note_on_ratio, target_position)
            for candidate in entry.candidates
        ]
        scores = np.array([max(self.config.min_score, item["score"]) for item in scored], dtype=np.float64)
        candidate_indices = self._candidate_indices(scores)
        probabilities = self._sampling_probabilities(scores, candidate_indices)
        if len(candidate_indices) == 0:
            selected_index = int(rng.integers(0, len(entry.candidates)))
            full_probabilities = np.full(len(entry.candidates), 1.0 / len(entry.candidates), dtype=np.float64)
        else:
            local_index = int(rng.choice(len(candidate_indices), p=probabilities))
            selected_index = int(candidate_indices[local_index])
            full_probabilities = np.zeros(len(entry.candidates), dtype=np.float64)
            for index, probability in zip(candidate_indices, probabilities):
                full_probabilities[int(index)] = float(probability)
        selected = entry.candidates[selected_index]
        selected_entry = self._entry_from_candidate(entry.codebook_id, selected, entry.candidates)
        return RareBarSelection(entry=selected_entry, diagnostics={
            "used": True,
            "codebook_id": int(entry.codebook_id),
            "candidate_count": len(entry.candidates),
            "sampling_candidate_count": int(len(candidate_indices)),
            "top_k": int(self.config.top_k),
            "temperature": float(self.config.temperature),
            "selected_index": selected_index,
            "selected_probability": round(float(full_probabilities[selected_index]), 6),
            "target_note_on_ratio": float(target_density.note_on_ratio),
            "target_position_ratio": round(float(target_position), 6),
            "selected": scored[selected_index],
            "top_candidates": self._top_candidates(scored, full_probabilities),
        })

    def _score(
        self,
        candidate: CodebookCandidate,
        sampled: SampledBar,
        harmony: HarmonyBarPlan,
        target_note_on_ratio: float,
        target_position: float,
    ) -> Dict[str, Any]:
        harmony_score = self._harmony_score(candidate.relative_tokens, harmony.degree)
        state_score = self._state_score(candidate, sampled)
        density_score = self._kernel(
            self._candidate_note_on_ratio(candidate),
            target_note_on_ratio,
            self.config.density_sigma,
        )
        position_score = self._kernel(
            float(candidate.position_ratio),
            target_position,
            self.config.position_sigma,
        )
        score = (
            harmony_score ** self.config.harmony_weight
            * state_score ** self.config.state_weight
            * density_score ** self.config.density_weight
            * position_score ** self.config.position_weight
        )
        return {
            "score": round(float(score), 12),
            "source_song": candidate.source_song,
            "source_file": candidate.source_file,
            "source_bar_index": candidate.source_bar_index,
            "kmeans_id": candidate.kmeans_id,
            "observation_id": candidate.observation_id,
            "position_ratio": round(float(candidate.position_ratio), 6),
            "note_on_ratio": self._candidate_note_on_ratio(candidate),
            "harmony_score": round(float(harmony_score), 6),
            "state_score": round(float(state_score), 6),
            "density_score": round(float(density_score), 6),
            "position_score": round(float(position_score), 6),
            "relative_tokens": list(candidate.relative_tokens),
        }

    def _harmony_score(self, tokens: Sequence[int], degree: str) -> float:
        pitch_classes = [int(token) % 12 for token in tokens if int(token) >= 0]
        if not pitch_classes:
            return 0.1
        triads = self.MINOR_TRIADS if self.mode == "minor" else self.MAJOR_TRIADS
        scale = self.MINOR_SCALE if self.mode == "minor" else self.MAJOR_SCALE
        chord = triads.get(str(degree), {0, 4, 7})
        chord_ratio = sum(1 for pc in pitch_classes if pc in chord) / len(pitch_classes)
        scale_ratio = sum(1 for pc in pitch_classes if pc in scale) / len(pitch_classes)
        return max(0.05, float(0.7 * chord_ratio + 0.3 * scale_ratio))

    def _state_score(self, candidate: CodebookCandidate, sampled: SampledBar) -> float:
        if candidate.observation_id is not None and int(candidate.observation_id) == int(sampled.observation_id):
            return 1.2
        if (
            candidate.kmeans_id is not None
            and sampled.kmeans_id is not None
            and int(candidate.kmeans_id) == int(sampled.kmeans_id)
        ):
            return 1.0
        return 0.65

    def _candidate_note_on_ratio(self, candidate: CodebookCandidate) -> float:
        if candidate.density is not None:
            return float(candidate.density.note_on_ratio)
        return float(self.density.analyze(candidate.relative_tokens).note_on_ratio)

    def _target_position(self, sampled: SampledBar, section_length: int) -> float:
        if section_length <= 1:
            return 0.0
        return float(int(sampled.section_local_index) / max(1, section_length - 1))

    def _kernel(self, value: float, target: float, sigma: float) -> float:
        sigma = max(float(sigma), 1.0e-6)
        return max(0.01, float(np.exp(-abs(float(value) - float(target)) / sigma)))

    def _candidate_indices(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return np.array([], dtype=np.int64)
        top_k = int(self.config.top_k)
        if top_k <= 0 or top_k >= len(scores):
            return np.arange(len(scores), dtype=np.int64)
        return np.array(
            sorted(
                np.argpartition(scores, -top_k)[-top_k:],
                key=lambda index: float(scores[int(index)]),
                reverse=True,
            ),
            dtype=np.int64,
        )

    def _sampling_probabilities(self, scores: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
        if len(candidate_indices) == 0:
            return np.array([], dtype=np.float64)
        selected_scores = scores[candidate_indices].astype(np.float64)
        temperature = max(float(self.config.temperature), 1.0e-6)
        adjusted = np.power(np.maximum(selected_scores, self.config.min_score), 1.0 / temperature)
        total = float(adjusted.sum())
        if total <= 0.0:
            return np.full(len(candidate_indices), 1.0 / len(candidate_indices), dtype=np.float64)
        return adjusted / total

    def _entry_from_candidate(
        self,
        codebook_id: int,
        candidate: CodebookCandidate,
        candidates: Sequence[CodebookCandidate],
    ) -> CodebookEntry:
        return CodebookEntry(
            codebook_id=int(codebook_id),
            source_song=candidate.source_song,
            source_file=candidate.source_file,
            source_bar_index=candidate.source_bar_index,
            relative_tokens=list(candidate.relative_tokens),
            absolute_tokens=list(candidate.absolute_tokens),
            density=candidate.density,
            token_variance=float(candidate.token_variance),
            sharing_score=float(candidate.sharing_score),
            candidates=list(candidates),
            latent_vector=(
                [float(value) for value in candidate.latent_vector]
                if candidate.latent_vector is not None
                else None
            ),
            position_ratio=float(candidate.position_ratio),
        )

    def _top_candidates(
        self,
        scored: Sequence[Dict[str, Any]],
        probabilities: np.ndarray,
    ) -> List[Dict[str, Any]]:
        ranked = sorted(
            enumerate(scored),
            key=lambda item: float(item[1]["score"]),
            reverse=True,
        )[: max(0, self.config.diagnostics_top_k)]
        return [
            {
                **item,
                "candidate_index": int(index),
                "probability": round(float(probabilities[index]), 6),
            }
            for index, item in ranked
        ]


class HarmonicEngine:
    """High-level facade that plans harmony and realizes generated bar IDs."""

    def __init__(
        self,
        config: Dict[str, Any],
        global_codebook: Dict[int, CodebookEntry],
        candidate_selector_model: Optional[CandidateSelectorModel] = None,
    ) -> None:
        self.config = config
        self.global_codebook = global_codebook
        self.progression = HarmonyProgressionPlanner.from_style_config(config)
        self.harmonizer = TokenHarmonizer.from_style_config(config)
        self.rare_selector = RareBarSelector(
            RareBarSelectionConfig.from_style_config(config),
            self.progression.config.mode,
        )
        self.learned_selector = LearnedCandidateSelector(
            config,
            candidate_selector_model,
            mode=self.progression.config.mode,
        )
        self.diagnostics: Dict[str, Any] = {}
        self.missing_codebook_ids: List[int] = []

    def realize(self, generation: GenerationResult) -> GenerationResult:
        harmony_plan = self.progression.plan(generation.section_plan, generation.seed)
        realized_bars = []
        rare_selection_diagnostics: List[Dict[str, Any]] = []
        section_lengths = {section.name: int(section.bars) for section in generation.section_plan}
        rng = np.random.default_rng(generation.seed)
        previous_candidate: Optional[CodebookCandidate] = None
        previous_harmony: Optional[HarmonyBarPlan] = None
        for sampled, harmony in zip(generation.sampled_bars, harmony_plan):
            codebook_id = int(sampled.codebook_id)
            codebook_entry = self._codebook_entry_for_sampled_bar(sampled)
            section_length = section_lengths.get(sampled.section, 1)
            selection = self._select_candidate(
                codebook_entry,
                sampled,
                harmony,
                section_length,
                previous_candidate,
                previous_harmony,
                rng,
            )
            codebook_entry = selection.entry
            rare_selection_diagnostics.append({
                "output_bar_index": int(sampled.output_bar_index),
                "section": sampled.section,
                "section_local_index": int(sampled.section_local_index),
                "hidden_state": int(sampled.hidden_state),
                "observation_id": int(sampled.observation_id),
                "codebook_id": codebook_id,
                "harmony_degree": harmony.degree,
                **selection.diagnostics,
            })
            tokens = [int(token) for token in codebook_entry.relative_tokens]
            sharing_score = float(codebook_entry.sharing_score)
            voice_sharing_applied = self.harmonizer.should_apply_voice_sharing(sharing_score)
            notes = self.harmonizer.harmonize(
                tokens,
                int(harmony.bar_base_pitch),
                sharing_score=sharing_score,
            )
            realized_bars.append(RealizedBar(
                sampled=sampled,
                harmony=harmony,
                codebook_source=codebook_entry.source(),
                codebook_density=codebook_entry.density,
                relative_tokens=tokens,
                token_variance=float(codebook_entry.token_variance),
                sharing_score=sharing_score,
                voice_sharing_applied=voice_sharing_applied,
                notes=notes,
            ))
            previous_candidate = self._candidate_from_entry(codebook_entry)
            previous_harmony = harmony
        self.diagnostics = {
            "harmonic_engine": asdict(self.progression.config),
            "progression": self.progression.diagnostics,
            "missing_codebook_ids": sorted(set(self.missing_codebook_ids)),
            "rare_bar_selection": {
                "config": asdict(self.rare_selector.config),
                "used_count": sum(1 for item in rare_selection_diagnostics if item.get("used")),
                "skipped_count": sum(1 for item in rare_selection_diagnostics if not item.get("used")),
                "backend_counts": self._backend_counts(rare_selection_diagnostics),
                "candidate_count_distribution": self._candidate_count_distribution(rare_selection_diagnostics),
                "events": rare_selection_diagnostics,
            },
            "candidate_selector": {
                "backend": str(ConfigView(self.config).section("candidate_selector").get("backend", "none")),
                "enabled": bool(ConfigView(self.config).section("candidate_selector").get("enabled", False)),
                "used_count": sum(
                    1 for item in rare_selection_diagnostics
                    if item.get("backend") == "learned_ranker" and item.get("used")
                ),
                "skipped_count": sum(
                    1 for item in rare_selection_diagnostics
                    if item.get("backend") == "learned_ranker" and not item.get("used")
                ),
                "events": [
                    item for item in rare_selection_diagnostics
                    if item.get("backend") == "learned_ranker"
                ],
            },
            "bars": [
                {
                    **bar.to_dict(),
                    "rare_bar_selection": rare_selection_diagnostics[index],
                }
                for index, bar in enumerate(realized_bars)
            ],
        }
        return GenerationResult(
            form=generation.form,
            seed=generation.seed,
            section_plan=generation.section_plan,
            sampled_bars=generation.sampled_bars,
            harmonic_bars=realized_bars,
        )

    def _select_candidate(
        self,
        entry: CodebookEntry,
        sampled: SampledBar,
        harmony: HarmonyBarPlan,
        section_length: int,
        previous_candidate: Optional[CodebookCandidate],
        previous_harmony: Optional[HarmonyBarPlan],
        rng: np.random.Generator,
    ) -> RareBarSelection:
        if str(getattr(sampled, "selection_mode", "")) == "joint_observation_candidate":
            selected = self._entry_from_sampled_bar(sampled, entry.candidates)
            return RareBarSelection(entry=selected, diagnostics={
                "used": True,
                "backend": "joint_observation_candidate",
                "reason": "concrete_candidate_already_selected_by_decoder",
                "candidate_count": len(entry.candidates),
                "selection_mode": sampled.selection_mode,
            })
        learned = self.learned_selector.select(
            entry,
            CandidateSelectionContext(
                sampled=sampled,
                harmony=harmony,
                section_length=section_length,
                previous_candidate=previous_candidate,
                previous_harmony=previous_harmony,
            ),
            rng,
        )
        if learned.diagnostics.get("used"):
            return RareBarSelection(entry=learned.entry, diagnostics=learned.diagnostics)
        rare = self.rare_selector.select(entry, sampled, harmony, section_length, rng)
        return RareBarSelection(entry=rare.entry, diagnostics={
            "backend": "heuristic",
            "learned_candidate_selector": learned.diagnostics,
            **rare.diagnostics,
        })

    def _entry_from_sampled_bar(
        self,
        sampled: SampledBar,
        candidates: Sequence[CodebookCandidate],
    ) -> CodebookEntry:
        relative_tokens = [int(token) for token in sampled.relative_tokens]
        matching = [
            candidate for candidate in candidates
            if (
                str(candidate.source_file) == str(sampled.source_file)
                and candidate.source_bar_index is not None
                and int(candidate.source_bar_index) == int(sampled.source_bar_index)
            )
        ]
        if matching:
            candidate = matching[0]
            return CodebookEntry(
                codebook_id=int(sampled.codebook_id),
                source_song=candidate.source_song,
                source_file=candidate.source_file,
                source_bar_index=candidate.source_bar_index,
                relative_tokens=list(candidate.relative_tokens),
                absolute_tokens=list(candidate.absolute_tokens),
                density=candidate.density,
                token_variance=float(candidate.token_variance),
                sharing_score=float(candidate.sharing_score),
                candidates=list(candidates),
                latent_vector=(
                    [float(value) for value in candidate.latent_vector]
                    if candidate.latent_vector is not None
                    else None
                ),
                position_ratio=float(candidate.position_ratio),
            )
        token_variance = self._token_variance(relative_tokens)
        return CodebookEntry(
            codebook_id=int(sampled.codebook_id),
            source_song=None,
            source_file=sampled.source_file,
            source_bar_index=int(sampled.source_bar_index),
            relative_tokens=relative_tokens,
            absolute_tokens=[int(token) for token in sampled.absolute_tokens],
            density=TokenDensityAnalyzer().analyze(relative_tokens),
            token_variance=token_variance,
            sharing_score=self._sharing_score(token_variance),
            candidates=list(candidates),
        )

    def _candidate_from_entry(self, entry: CodebookEntry) -> CodebookCandidate:
        return CodebookCandidate(
            source_song=entry.source_song,
            source_file=entry.source_file,
            source_bar_index=entry.source_bar_index,
            relative_tokens=list(entry.relative_tokens),
            absolute_tokens=list(entry.absolute_tokens),
            density=entry.density,
            token_variance=float(entry.token_variance),
            sharing_score=float(entry.sharing_score),
            kmeans_id=None,
            observation_id=None,
            latent_vector=(
                [float(value) for value in entry.latent_vector]
                if entry.latent_vector is not None
                else None
            ),
            position_ratio=float(entry.position_ratio),
        )

    def _candidate_count_distribution(self, events: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for event in events:
            key = str(int(event.get("candidate_count", 0)))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _backend_counts(self, events: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for event in events:
            key = str(event.get("backend", "unknown"))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _codebook_entry_for_sampled_bar(self, sampled: SampledBar) -> CodebookEntry:
        codebook_id = int(sampled.codebook_id)
        codebook_entry = self.global_codebook.get(codebook_id)
        if codebook_entry is not None:
            return codebook_entry
        self.missing_codebook_ids.append(codebook_id)
        relative_tokens = [int(token) for token in sampled.relative_tokens]
        token_variance = self._token_variance(relative_tokens)
        return CodebookEntry(
            codebook_id=codebook_id,
            source_song=None,
            source_file=sampled.source_file,
            source_bar_index=int(sampled.source_bar_index),
            relative_tokens=relative_tokens,
            absolute_tokens=[int(token) for token in sampled.absolute_tokens],
            density=TokenDensityAnalyzer().analyze(relative_tokens),
            token_variance=token_variance,
            sharing_score=self._sharing_score(token_variance),
        )

    def _token_variance(self, tokens: Sequence[int]) -> float:
        values = [float(token) for token in tokens if int(token) >= 0]
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return float(sum((value - mean) ** 2 for value in values) / len(values))

    def _sharing_score(self, variance: float) -> float:
        return float(1.0 / (1.0 + max(0.0, variance)))


class HarmonicEngineCLI:
    """Standalone CLI for realizing a generated JSON file into harmonic MIDI."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Realize generated bar IDs through HarmonicEngine.")
        parser.add_argument("--generation-json", type=Path, required=True)
        parser.add_argument("--codebook-json", type=Path, required=True)
        parser.add_argument("--output-json", type=Path, required=True)
        parser.add_argument("--output-midi", type=Path, required=True)
        parser.add_argument("--diagnostics-output", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        HarmonicCliOverrides.add_arguments(parser)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = HarmonicCliOverrides.from_args(args).apply(ConfigLoader().load(args.config))
        generation = GenerationResult.from_dict(json.loads(args.generation_json.read_text(encoding="utf-8")))
        codebook = {
            int(key): CodebookEntry.from_dict({**value, "codebook_id": int(key)})
            for key, value in json.loads(args.codebook_json.read_text(encoding="utf-8")).items()
        }
        engine = HarmonicEngine(config, codebook)
        realized = engine.realize(generation)
        render_diag = HarmonicMidiRenderer.from_style_config(config).write(
            realized.harmonic_bars,
            args.output_midi,
        )
        engine.diagnostics["render"] = render_diag
        args.output_json.write_text(json.dumps(realized.to_dict(), indent=2), encoding="utf-8")
        if args.diagnostics_output:
            args.diagnostics_output.write_text(json.dumps(engine.diagnostics, indent=2), encoding="utf-8")
        print(f"Wrote harmonic MIDI -> {args.output_midi}")


def main() -> None:
    HarmonicEngineCLI().run()


if __name__ == "__main__":
    main()

