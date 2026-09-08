#!/usr/bin/env python3
"""Music file parsing into clean song/bar/track records."""

from __future__ import annotations

import json
import hashlib
import unicodedata
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from common.config_loader import ConfigView
from data.core import BarRecord, MeasureSpan, NoteEvent, SongRecord, TrackRecord
from data.measure_map import extract_measure_spans, split_tunes
from data.performance_controls import collect_controls
from data.canonical_timeline import (
    CanonicalBarSpan,
    RawSourceNote,
    build_canonical_spans,
    collect_raw_smf_facts,
    fragment_notes,
    resolve_time_signature_chain,
)


MUSIC_SUFFIXES = {".mid", ".midi"}
MIDI_PITCH_CARDINALITY = 128  # zero-based MIDI pitch indices are [0, 127].


@dataclass(frozen=True)
class MusicParserConfig:
    """Configuration for symbolic music parsing."""

    quantize_input: bool = True
    quantize_divisors: tuple[int, ...] = (4,)
    quantize_offsets: bool = True
    quantize_durations: bool = True
    hard_safety_limit: int = 48
    track_retention_policy: str = "error"
    default_velocity: int = 64

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MusicParserConfig":
        """Build parser configuration from the style configuration."""
        section = ConfigView(config).section("music_parser")
        return cls(
            quantize_input=bool(section.get("quantize_input", True)),
            quantize_divisors=tuple(int(x) for x in section.get("quantize_divisors", [4])),
            quantize_offsets=bool(section.get("quantize_offsets", True)),
            quantize_durations=bool(section.get("quantize_durations", True)),
            hard_safety_limit=int(section.get("hard_safety_limit", 48)),
            track_retention_policy=str(section.get("track_retention_policy", "error")),
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
        path = Path(file_path)
        if path.suffix.lower() in {".mid", ".midi"}:
            return self._parse_canonical_midi(path, metadata, transpose_semitones, dataset_root)
        raise ValueError("canonical_smf_required")

    def _parse_canonical_midi(
        self,
        path: Path,
        metadata: Dict[str, Any],
        transpose_semitones: int,
        dataset_root: str | Path | None,
    ) -> List[SongRecord]:
        """Parse one PPQN SMF using the raw-event canonical timeline authority."""
        import mido

        if not self.config.quantize_input or self.config.quantize_divisors != (4,):
            raise ValueError("canonical_quantization_policy_invalid")
        midi = mido.MidiFile(str(path))
        signatures, source_notes = collect_raw_smf_facts(midi)
        if not source_notes:
            raise ValueError("no_note_events")
        chain = resolve_time_signature_chain(signatures)
        ppqn = int(midi.ticks_per_beat)
        spans = build_canonical_spans(chain, ppqn=ppqn, terminal_end_ql=max(note.end_ql(ppqn) for note in source_notes))
        fragments = fragment_notes(source_notes, spans, ppqn=ppqn)
        retained_tracks, retention = self._canonical_track_retention(source_notes)
        retained = set(retained_tracks)
        fragments = [fragment for fragment in fragments if fragment.source_note.physical_track_index in retained]
        if not fragments:
            raise ValueError("no_note_events")

        source_identity = self._source_file_identity(path, dataset_root)
        suffix = f"_T{int(transpose_semitones):+d}" if int(transpose_semitones) != 0 else ""
        song = SongRecord(
            song_id=f"{path.stem}{suffix}",
            file_path=str(path),
            form=None,
            metadata={
                **dict(metadata),
                "transpose_semitones": int(transpose_semitones),
                "source_file_identity": source_identity,
                "tune_index": 0,
                "opus_tune_count": 1,
                "canonical_parser_version": "raw_smf_v1",
                "canonical_span_count": len(spans),
                "track_retention": retention,
                "form_mapping_unavailable": bool(metadata),
                "performance_controls": {"cc64_available": False, "cc64_unavailable_reason": "canonical_raw_controls_pending"},
                "quantization_audit": self._canonical_quantization_audit(fragments),
            },
        )
        song.runtime_diagnostics["quantization_residual_samples"] = self._canonical_residual_samples(fragments)
        by_span: dict[int, list[Any]] = defaultdict(list)
        for fragment in fragments:
            by_span[fragment.canonical_bar_index].append(fragment)
        for span in spans:
            song.bars.append(self._build_canonical_bar(song, span, by_span.get(span.canonical_bar_index, []), retained_tracks, transpose_semitones, ppqn))
        return [song]

    def _canonical_track_retention(self, notes: Sequence[RawSourceNote]) -> tuple[list[int], Dict[str, Any]]:
        """Retain raw physical note tracks or fail under the frozen 48-track policy."""
        counts = Counter(note.physical_track_index for note in notes)
        ranked = sorted(counts, key=lambda index: (-counts[index], index))
        if len(ranked) <= self.config.hard_safety_limit:
            return ranked, {"physical_part_count": len(ranked), "policy": "retain_all", "retained_physical_track_indexes": ranked, "dropped_part_count": 0, "dropped_note_count": 0, "dropped_note_ratio": 0.0}
        if self.config.track_retention_policy != "truncate":
            raise ValueError("track_limit_exceeded")
        retained, dropped = ranked[:self.config.hard_safety_limit], ranked[self.config.hard_safety_limit:]
        dropped_notes = sum(counts[index] for index in dropped)
        return retained, {"physical_part_count": len(ranked), "policy": "truncate", "retained_physical_track_indexes": retained, "dropped_part_count": len(dropped), "dropped_note_count": dropped_notes, "dropped_note_ratio": dropped_notes / max(1, len(notes))}

    @staticmethod
    def _canonical_quantization_audit(fragments: Sequence[Any]) -> Dict[str, Any]:
        """Summarize fragment-local timing residuals by their canonical meter."""
        by_meter: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"onset": [], "end": []})
        for fragment in fragments:
            meter = fragment.meter
            bucket = by_meter[meter]
            bucket["onset"].append(float(abs(fragment.quantized_local_start_ql - fragment.raw_local_start_ql)))
            bucket["end"].append(float(abs(fragment.quantized_local_end_ql - fragment.raw_local_end_ql)))
        def summary(values: list[float]) -> Dict[str, float]:
            ordered = sorted(values)
            return {"max": max(values, default=0.0), "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else 0.0}
        fragment_count = sum(len(item["onset"]) for item in by_meter.values())
        return {"status": "MONITOR", "quantum_ql": 0.25, "source_boundaries_retained": True, "audit_unit": "source_note_fragment", "fragment_count": fragment_count, "event_count": fragment_count, "nonzero_residual_count": sum(value > 1e-9 for item in by_meter.values() for values in item.values() for value in values), "by_meter": {meter: {"fragment_count": len(values["onset"]), "event_count": len(values["onset"]), "nonzero_residual_count": sum(value > 1e-9 for value in values["onset"] + values["end"]), "onset_residual_ql": summary(values["onset"]), "end_residual_ql": summary(values["end"])} for meter, values in by_meter.items()}}

    @staticmethod
    def _canonical_residual_samples(fragments: Sequence[Any]) -> Dict[str, Dict[str, list[float]]]:
        samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"onset_residual_samples_ql": [], "end_residual_samples_ql": []})
        for fragment in fragments:
            meter = fragment.meter
            samples[meter]["onset_residual_samples_ql"].append(float(abs(fragment.quantized_local_start_ql - fragment.raw_local_start_ql)))
            samples[meter]["end_residual_samples_ql"].append(float(abs(fragment.quantized_local_end_ql - fragment.raw_local_end_ql)))
        return dict(samples)

    def _build_canonical_bar(
        self,
        song: SongRecord,
        span: CanonicalBarSpan,
        fragments: Sequence[Any],
        retained_tracks: Sequence[int],
        transpose_semitones: int,
        ppqn: int,
    ) -> BarRecord:
        """Convert bar-local quantized fragments into the codec-facing record."""
        tracks: list[TrackRecord] = []
        for track_index, physical_track_index in enumerate(retained_tracks):
            notes = []
            for fragment in fragments:
                source = fragment.source_note
                if source.physical_track_index != physical_track_index:
                    continue
                notes.append(NoteEvent(
                    pitch=int(source.pitch) + int(transpose_semitones),
                    onset_ql=float(fragment.quantized_local_start_ql),
                    duration_ql=float(fragment.duration_ql),
                    velocity=int(source.velocity),
                    source_file_identity=str(song.metadata["source_file_identity"]),
                    physical_track_index=int(physical_track_index),
                    source_note_ordinal=int(source.source_note_ordinal),
                    source_note_id=f"{song.metadata['source_file_identity']}:{physical_track_index}:{source.source_note_ordinal}",
                    source_onset_ql=float(source.start_ql(ppqn)),
                    continues_from_previous_bar=fragment.continues_from_previous_bar,
                    continues_into_next_bar=fragment.continues_into_next_bar,
                ))
            tracks.append(TrackRecord(track_index, f"track_{physical_track_index}", notes))
        return BarRecord(
            song_id=song.song_id,
            file_path=song.file_path,
            bar_index=span.canonical_bar_index,
            canonical_bar_index=span.canonical_bar_index,
            source_measure_index=span.canonical_bar_index,
            bar_length_ql=float(span.end_ql - span.start_ql),
            time_signature=span.time_signature,
            meter_numerator=span.numerator,
            meter_denominator=span.denominator,
            source_bar_count=int(song.metadata["canonical_span_count"]),
            tracks=tracks,
        )

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

    def _collect_tracks(self, score: stream.Score) -> tuple[List[tuple[int, List[tuple[float, float, int, int, int]]]], Dict[str, Any]]:
        """Collect retained physical-part events and their explicit retention record."""
        parts = list(score.parts) if getattr(score, "parts", None) else []
        if len(parts) > 1:
            tracks = [(index, self._collect_events(part)) for index, part in enumerate(parts)]
            tracks = [track for track in tracks if track[1]]
            return self._select_tracks(tracks)
        events = self._collect_events(score)
        if not events:
            return [], {"physical_part_count": 1, "policy": "retain_all", "retained_physical_track_indexes": [0], "dropped_part_count": 0, "dropped_note_count": 0, "dropped_note_ratio": 0.0}
        return [(0, events)], {"physical_part_count": 1, "policy": "retain_all", "retained_physical_track_indexes": [0], "dropped_part_count": 0, "dropped_note_count": 0, "dropped_note_ratio": 0.0}

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
                events.append((start, start + duration, int(pitch), velocity, source_ordinal * MIDI_PITCH_CARDINALITY + pitch_index))
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

    @staticmethod
    def _quantization_audit(source: Dict[tuple[str, int], tuple[float, float]], quantized: Dict[tuple[str, int], tuple[float, float]], spans: Sequence[MeasureSpan] = ()) -> Dict[str, Any]:
        if set(source) != set(quantized):
            raise ValueError("quantization_event_identity_mismatch")
        onset = [abs(quantized[key][0] - source[key][0]) for key in sorted(source)]
        end = [abs(quantized[key][1] - source[key][1]) for key in sorted(source)]
        def summary(values: List[float]) -> Dict[str, float]:
            ordered = sorted(values)
            return {"max": max(values, default=0.0), "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else 0.0}
        by_meter: Dict[str, Dict[str, List[float]]] = {}
        for key in sorted(source):
            start, source_end = source[key]; meter = next((span.time_signature for span in spans if span.start_ql <= start < span.end_ql), "UNMAPPED")
            bucket = by_meter.setdefault(meter, {"onset": [], "end": []})
            bucket["onset"].append(abs(quantized[key][0] - start)); bucket["end"].append(abs(quantized[key][1] - source_end))
        return {"status": "MONITOR", "quantum_ql": 0.25, "source_boundaries_retained": True, "event_count": len(onset), "nonzero_residual_count": sum(value > 1e-9 for value in onset + end), "onset_residual_ql": summary(onset), "end_residual_ql": summary(end), "by_meter": {meter: {"event_count": len(values["onset"]), "nonzero_residual_count": sum(value > 1e-9 for value in values["onset"] + values["end"]), "onset_residual_ql": summary(values["onset"]), "end_residual_ql": summary(values["end"]), "onset_residual_samples_ql": values["onset"], "end_residual_samples_ql": values["end"]} for meter, values in by_meter.items()}}

    def _element_pitches(self, element: Any) -> List[int]:
        """Extract MIDI pitches from a note or chord element."""
        from music21 import chord, note

        if isinstance(element, note.Note):
            return [int(element.pitch.midi)]
        if isinstance(element, chord.Chord):
            return [int(pitch.midi) for pitch in element.pitches]
        return []

    def _select_tracks(self, tracks: Sequence[tuple[int, List[tuple[float, float, int, int, int]]]]) -> tuple[List[tuple[int, List[tuple[float, float, int, int, int]]]], Dict[str, Any]]:
        """Retain every source part or fail unless explicit truncation is selected."""
        ranked = sorted(tracks, key=lambda item: (-len(item[1]), item[0]))
        if len(ranked) <= self.config.hard_safety_limit:
            record={"physical_part_count": len(ranked), "policy": "retain_all", "retained_physical_track_indexes": [index for index, _ in ranked], "dropped_part_count": 0, "dropped_note_count": 0, "dropped_note_ratio": 0.0}
            return [(physical_index, list(track)) for physical_index, track in ranked], record
        if self.config.track_retention_policy != "truncate":
            raise ValueError("track_limit_exceeded")
        retained, dropped = ranked[: self.config.hard_safety_limit], ranked[self.config.hard_safety_limit:]
        dropped_notes = sum(len(events) for _, events in dropped)
        total_notes = sum(len(events) for _, events in ranked)
        record={"physical_part_count": len(ranked), "policy": "truncate", "retained_physical_track_indexes": [index for index, _ in retained], "dropped_part_count": len(dropped), "dropped_note_count": dropped_notes, "dropped_note_ratio": dropped_notes / max(1, total_notes)}
        return [(physical_index, list(track)) for physical_index, track in retained], record

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
                    source_note_id=(
                        f"{song.metadata['source_file_identity']}:"
                        f"{int(physical_track_index)}:{int(source_note_ordinal)}"
                    ),
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
