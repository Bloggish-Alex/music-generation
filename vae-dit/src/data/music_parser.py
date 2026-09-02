#!/usr/bin/env python3
"""Music file parsing into clean song/bar/track records."""

from __future__ import annotations

import json
import hashlib
import unicodedata
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from common.config_loader import ConfigView
from data.core import BarRecord, MeasureSpan, NoteEvent, SongRecord, TrackRecord
from data.measure_map import extract_measure_spans, split_tunes
from data.performance_controls import collect_controls


MUSIC_SUFFIXES = {".mid", ".midi", ".abc", ".krn"}


@dataclass(frozen=True)
class MusicParserConfig:
    """Configuration for symbolic music parsing."""

    quantize_input: bool = True
    quantize_divisors: tuple[int, ...] = (4, 3)
    quantize_offsets: bool = True
    quantize_durations: bool = True
    hard_safety_limit: int = 48
    track_retention_policy: str = "error"
    register_split_low_max: int = 55
    register_split_mid_max: int = 72
    default_velocity: int = 64

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MusicParserConfig":
        """Build parser configuration from the style configuration."""
        section = ConfigView(config).section("music_parser")
        return cls(
            quantize_input=bool(section.get("quantize_input", True)),
            quantize_divisors=tuple(int(x) for x in section.get("quantize_divisors", [4, 3])),
            quantize_offsets=bool(section.get("quantize_offsets", True)),
            quantize_durations=bool(section.get("quantize_durations", True)),
            hard_safety_limit=int(section.get("hard_safety_limit", 48)),
            track_retention_policy=str(section.get("track_retention_policy", "error")),
            register_split_low_max=int(section.get("register_split_low_max", 55)),
            register_split_mid_max=int(section.get("register_split_mid_max", 72)),
            default_velocity=int(section.get("default_velocity", 64)),
        )


class FormMetadataLoader:
    """Load optional form metadata from music-dir/form.json."""

    def load(self, music_dir: str | Path) -> Dict[str, Dict[str, Any]]:
        """Read form metadata keyed by file name if form.json exists."""
        path = Path(music_dir) / "form.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        return {str(key): dict(value) for key, value in payload.items() if isinstance(value, dict)}


class MusicDirectoryParser:
    """Parse supported symbolic music files under a directory."""

    def __init__(self, config: MusicParserConfig) -> None:
        """Store parser policy and initialize the recoverable failure list."""
        self.config = config
        self.failed_files: List[Dict[str, str]] = []
        self.track_retention_events: List[Dict[str, Any]] = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MusicDirectoryParser":
        """Create a parser from the full style configuration."""
        return cls(MusicParserConfig.from_config(config))

    def parse_directory(self, music_dir: str | Path, transpose_semitones: int = 0) -> List[SongRecord]:
        """Parse all supported files and continue after single-file failures."""
        root = Path(music_dir)
        form_map = FormMetadataLoader().load(root)
        songs: List[SongRecord] = []
        for file_path in self.discover_files(root):
            try:
                songs.extend(self.parse_file(
                    file_path,
                    form_map.get(file_path.name, {}),
                    transpose_semitones=transpose_semitones,
                    dataset_root=root,
                ))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"Skipping {file_path}: {message}")
                self.failed_files.append({"file_path": str(file_path), "error": message})
        if self.failed_files:
            raise ValueError(f"parser_failures: {len(self.failed_files)}")
        return songs

    def discover_files(self, music_dir: str | Path) -> List[Path]:
        """Find supported music files recursively."""
        root = Path(music_dir)
        files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in MUSIC_SUFFIXES]
        return sorted(files)

    def parse_file(
        self,
        file_path: str | Path,
        metadata: Dict[str, Any],
        transpose_semitones: int = 0,
        dataset_root: str | Path | None = None,
    ) -> List[SongRecord]:
        """Parse one file into one independent SongRecord per Opus tune."""
        from music21 import converter

        path = Path(file_path)
        parsed = converter.parse(str(path))
        records: List[SongRecord] = []
        tunes = split_tunes(parsed)
        for tune_index, tune in enumerate(tunes):
            score = tune.transpose(int(transpose_semitones), inPlace=False) if int(transpose_semitones) != 0 else tune
            controls = collect_controls(score, path)
            self._tag_source_events(score)
            source_events = self._all_event_boundaries(score)
            score = self._quantize_score(score)
            quantization_audit = self._quantization_audit(source_events, self._all_event_boundaries(score))
            spans = extract_measure_spans(score)
            raw_tracks = self._collect_tracks(score)
            if not raw_tracks:
                raise ValueError("No note events found.")
            suffix = f"_T{int(transpose_semitones):+d}" if int(transpose_semitones) != 0 else ""
            tune_suffix = f"__tune_{tune_index:03d}" if len(tunes) > 1 else ""
            song = SongRecord(
            song_id=f"{path.stem}{tune_suffix}{suffix}",
            file_path=str(path),
            form=metadata.get("form"),
            metadata={**dict(metadata), "transpose_semitones": int(transpose_semitones), "source_file_identity": self._source_file_identity(path, dataset_root), "tune_index": tune_index, "parser_measure_count": len(spans), "track_retention": dict(self.track_retention_events[-1]), "performance_controls": {"tempo_bpm": list(controls.tempo_bpm), "key_signature": controls.key_signature, "key_confidence": controls.key_confidence, "cc64_available": controls.cc64.cc64_available, "cc64_intervals_ql": [list(interval) for interval in controls.cc64.cc64_intervals], "cc64_unavailable_reason": controls.cc64.unavailable_reason}, "quantization_audit": quantization_audit},
            )
            for bar_index, span in enumerate(spans):
                bar = self._build_bar(song, raw_tracks, bar_index, span, len(spans))
                self._assign_form_section(bar, metadata)
                song.bars.append(bar)
            records.append(song)
        return records

    def _quantize_score(self, score: Any) -> Any:
        """Quantize symbolic timing so grid encoding is stable."""
        if not self.config.quantize_input:
            return score
        # Quantizing offsets and durations prevents expressive timing drift from
        # moving equivalent notes into neighboring bar-grid slots.
        return score.quantize(
            quarterLengthDivisors=self.config.quantize_divisors,
            processOffsets=self.config.quantize_offsets,
            processDurations=self.config.quantize_durations,
            inPlace=False,
        )

    def _collect_tracks(self, score: stream.Score) -> List[tuple[int, List[tuple[float, float, int, int, int]]]]:
        """Collect note events per music21 part, with register splitting fallback."""
        parts = list(score.parts) if getattr(score, "parts", None) else []
        if len(parts) > 1:
            tracks = [(index, self._collect_events(part)) for index, part in enumerate(parts)]
            tracks = [track for track in tracks if track[1]]
            return self._select_tracks(tracks)
        events = self._collect_events(score)
        if not events:
            return []
        self.track_retention_events.append({"physical_part_count": 1, "policy": "retain_all", "retained_physical_track_indexes": [0], "dropped_part_count": 0, "dropped_note_count": 0, "dropped_note_ratio": 0.0})
        return [(0, events)]

    def _collect_events(self, container: Any) -> List[tuple[float, float, int, int, int]]:
        """Collect flat note events from one part or score."""
        events: List[tuple[float, float, int, int, int]] = []
        for element_ordinal, element in enumerate(container.flatten().notes):
            pitches = self._element_pitches(element)
            if not pitches:
                continue
            start = float(element.offset)
            duration = float(element.quarterLength)
            if duration <= 0:
                continue
            velocity = int(getattr(getattr(element, "volume", None), "velocity", None) or self.config.default_velocity)
            tagged = element.editorial.get("codec_source_event_id")
            source_ordinal = int(str(tagged).split(":")[-1]) if isinstance(tagged, str) else element_ordinal
            for pitch_index, pitch in enumerate(pitches):
                events.append((start, start + duration, int(pitch), velocity, source_ordinal * 128 + pitch_index))
        return sorted(events, key=lambda item: (item[0], item[2]))

    def _tag_source_events(self, score: Any) -> None:
        """Attach stable pre-quantization identities that music21 copies retain."""
        parts = list(score.parts) if getattr(score, "parts", None) else [score]
        for physical, part in enumerate(parts):
            for ordinal, element in enumerate(part.flatten().notes):
                element.editorial["codec_source_event_id"] = f"{physical}:{ordinal}"

    def _all_event_boundaries(self, score: Any) -> Dict[tuple[str, int], tuple[float, float]]:
        """Read source identities attached before quantization, never sort positions."""
        result = {}
        parts = list(score.parts) if getattr(score, "parts", None) else [score]
        for part in parts:
            for element in part.flatten().notes:
                event_id = element.editorial.get("codec_source_event_id")
                if not isinstance(event_id, str):
                    raise ValueError("quantization_event_identity_mismatch")
                for pitch_index, _ in enumerate(self._element_pitches(element)):
                    result[(event_id, pitch_index)] = (float(element.offset), float(element.offset + element.quarterLength))
        return result

    def _collect_tracks_for_audit(self, score: Any) -> List[tuple[int, List[tuple[float, float, int, int, int]]]]:
        parts = list(score.parts) if getattr(score, "parts", None) else []
        return [(index, self._collect_events(part)) for index, part in enumerate(parts)] if parts else [(0, self._collect_events(score))]

    @staticmethod
    def _quantization_audit(source: Dict[tuple[str, int], tuple[float, float]], quantized: Dict[tuple[str, int], tuple[float, float]]) -> Dict[str, Any]:
        if set(source) != set(quantized):
            raise ValueError("quantization_event_identity_mismatch")
        onset = [abs(quantized[key][0] - source[key][0]) for key in sorted(source)]
        end = [abs(quantized[key][1] - source[key][1]) for key in sorted(source)]
        def summary(values: List[float]) -> Dict[str, float]:
            ordered = sorted(values)
            return {"max": max(values, default=0.0), "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else 0.0}
        return {"status": "MONITOR", "quantum_ql": 0.25, "source_boundaries_retained": True, "event_count": len(onset), "nonzero_residual_count": sum(value > 1e-9 for value in onset + end), "onset_residual_ql": summary(onset), "end_residual_ql": summary(end)}

    def _element_pitches(self, element: Any) -> List[int]:
        """Extract MIDI pitches from a note or chord element."""
        from music21 import chord, note

        if isinstance(element, note.Note):
            return [int(element.pitch.midi)]
        if isinstance(element, chord.Chord):
            return [int(pitch.midi) for pitch in element.pitches]
        return []

    def _select_tracks(self, tracks: Sequence[tuple[int, List[tuple[float, float, int, int]]]]) -> List[tuple[int, List[tuple[float, float, int, int]]]]:
        """Retain every source part or fail unless explicit truncation is selected."""
        ranked = sorted(tracks, key=lambda item: (-len(item[1]), item[0]))
        if len(ranked) <= self.config.hard_safety_limit:
            self.track_retention_events.append({"physical_part_count": len(ranked), "policy": "retain_all", "retained_physical_track_indexes": [index for index, _ in ranked], "dropped_part_count": 0, "dropped_note_count": 0, "dropped_note_ratio": 0.0})
            return [(physical_index, list(track)) for physical_index, track in ranked]
        if self.config.track_retention_policy != "truncate":
            self.track_retention_events.append({"physical_part_count": len(ranked), "policy": "error", "hard_safety_limit": self.config.hard_safety_limit})
            raise ValueError("track_limit_exceeded")
        retained, dropped = ranked[: self.config.hard_safety_limit], ranked[self.config.hard_safety_limit:]
        dropped_notes = sum(len(events) for _, events in dropped)
        total_notes = sum(len(events) for _, events in ranked)
        self.track_retention_events.append({"physical_part_count": len(ranked), "policy": "truncate", "retained_physical_track_indexes": [index for index, _ in retained], "dropped_part_count": len(dropped), "dropped_note_count": dropped_notes, "dropped_note_ratio": dropped_notes / max(1, total_notes)})
        return [(physical_index, list(track)) for physical_index, track in retained]

    def _split_by_register(self, events: Sequence[tuple[float, float, int, int]]) -> List[List[tuple[float, float, int, int]]]:
        """Split a single stream into high, middle, and low register tracks."""
        high: List[tuple[float, float, int, int]] = []
        middle: List[tuple[float, float, int, int]] = []
        low: List[tuple[float, float, int, int]] = []
        for event in events:
            pitch = int(event[2])
            if pitch > self.config.register_split_mid_max:
                high.append(event)
            elif pitch > self.config.register_split_low_max:
                middle.append(event)
            else:
                low.append(event)
        return [track for track in [high, middle, low] if track]

    def _build_bar(
        self,
        song: SongRecord,
        tracks: Sequence[tuple[int, Sequence[tuple[float, float, int, int]]]],
        bar_index: int, span: MeasureSpan,
        bar_count: int,
    ) -> BarRecord:
        """Build one bar record from global note events."""
        bar_start, bar_end = span.start_ql, span.end_ql
        bar_length = bar_end - bar_start
        bar_tracks: List[TrackRecord] = []
        for track_index, (physical_track_index, track_events) in enumerate(tracks):
            notes = []
            for start, end, pitch, velocity, source_note_ordinal in track_events:
                if end <= bar_start or start >= bar_end:
                    continue
                local_start = max(0.0, float(start) - bar_start)
                local_end = min(bar_length, float(end) - bar_start)
                notes.append(NoteEvent(
                    pitch=int(pitch),
                    onset_ql=local_start,
                    duration_ql=max(0.0, local_end - local_start),
                    velocity=int(velocity),
                    source_file_identity=str(song.metadata["source_file_identity"]),
                    physical_track_index=int(physical_track_index),
                    source_note_ordinal=int(source_note_ordinal),
                    source_onset_ql=float(start),
                    continues_from_previous_bar=bool(start < bar_start),
                    continues_into_next_bar=bool(end > bar_end),
                ))
            bar_tracks.append(TrackRecord(track_index=track_index, name=f"track_{track_index}", notes=notes))
        return BarRecord(
            song_id=song.song_id,
            file_path=song.file_path,
            bar_index=int(bar_index),
            bar_length_ql=bar_length,
            time_signature=span.time_signature,
            source_measure_index=span.source_measure_index,
            meter_numerator=span.numerator,
            meter_denominator=span.denominator,
            is_pickup=span.is_pickup,
            source_bar_count=int(bar_count),
            form=song.form,
            tracks=bar_tracks,
        )

    @staticmethod
    def _source_file_identity(path: Path, dataset_root: str | Path | None) -> str:
        """Return a stable dataset-path and raw-byte identity for one source file."""
        root = Path(dataset_root) if dataset_root is not None else path.parent
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        normalized = unicodedata.normalize("NFC", relative)
        inner = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashlib.sha256(f"{normalized}\0{inner}".encode("utf-8")).hexdigest()

    def _assign_form_section(self, bar: BarRecord, metadata: Dict[str, Any]) -> None:
        """Attach section metadata from form.json without inferring it."""
        sections = metadata.get("sections") or []
        for index, section in enumerate(sections):
            start = int(section.get("start_bar", section.get("start", 0)))
            if section.get("end_bar", section.get("end")) is not None:
                end = int(section.get("end_bar", section.get("end")))
            else:
                end = start + int(section.get("length", 0))
            if start <= int(bar.bar_index) < end:
                bar.section_label = str(section.get("name", f"section_{index}"))
                bar.section_index = int(index)
                return
