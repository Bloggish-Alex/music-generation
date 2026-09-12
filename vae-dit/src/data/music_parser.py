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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from common.config_loader import ConfigView
from data.core import BarRecord, NoteEvent, SongRecord, TrackRecord
from data.performance_controls import collect_raw_smf_controls
from data.canonical_timeline import (
    CanonicalBarSpan,
    RawSourceNote,
    build_canonical_spans,
    collect_raw_smf_facts,
    fragment_notes,
    resolve_time_signature_chain,
)


MUSIC_SUFFIXES = {".mid", ".midi"}
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
    initial_time_signature_policy: str = "error"

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
            initial_time_signature_policy=str(section.get("initial_time_signature_policy", "error")),
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
        performance_controls = collect_raw_smf_controls(midi)
        source_identity = self._source_file_identity(path, dataset_root)
        repairs = []
        signatures, source_notes = collect_raw_smf_facts(midi, repairs=repairs)
        if not source_notes:
            raise ValueError("no_note_events")
        chain = resolve_time_signature_chain(signatures, initial_time_signature_policy=self.config.initial_time_signature_policy)
        ppqn = int(midi.ticks_per_beat)
        spans = build_canonical_spans(chain, ppqn=ppqn, terminal_end_ql=max(note.end_ql(ppqn) for note in source_notes))
        timeline_hash = self._canonical_timeline_hash(spans, ppqn)
        fragments = fragment_notes(source_notes, spans, ppqn=ppqn)
        retained_tracks, retention = self._canonical_track_retention(source_notes)
        retained = set(retained_tracks)
        fragments = [fragment for fragment in fragments if fragment.source_note.physical_track_index in retained]
        if not fragments:
            raise ValueError("no_note_events")

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
                "ppqn": ppqn,
                "canonical_span_count": len(spans),
                "initial_time_signature": {"policy": self.config.initial_time_signature_policy, "origin": chain[0].origin, "injected_at_tick": 0 if chain[0].origin == "smf_default" else None, "meter": f"{chain[0].numerator}/{chain[0].denominator}", "first_real_ts_tick": min((fact.absolute_tick for fact in signatures), default=None)},
                "track_retention": retention,
                "form_mapping_status": self._form_mapping_status(metadata, source_identity, timeline_hash),
                "canonical_timeline_sha256": timeline_hash,
                "performance_controls": performance_controls,
                "paired_source_note_count": len(source_notes),
                "quantization_audit": self._canonical_quantization_audit(fragments, repairs, len(source_notes)),
                "raw_pairing_repairs": [self._pairing_repair_fact(source_identity, path, dataset_root, midi, repair) for repair in repairs],
            },
        )
        fragment_samples = self._canonical_fragment_samples(source_identity, fragments)
        song.runtime_diagnostics["quantization_fragment_samples"] = fragment_samples
        by_span: dict[int, list[Any]] = defaultdict(list)
        for fragment in fragments:
            by_span[fragment.canonical_bar_index].append(fragment)
        for span in spans:
            song.bars.append(self._build_canonical_bar(song, span, by_span.get(span.canonical_bar_index, []), retained_tracks, transpose_semitones, ppqn))
        self._apply_canonical_form_metadata(song, metadata)
        return [song]

    @staticmethod
    def _canonical_timeline_hash(spans: Sequence[CanonicalBarSpan], ppqn: int) -> str:
        """Hash canonical span authority using exact ticks, never display floats."""
        payload = [{"canonical_bar_index": span.canonical_bar_index, "start_tick": span.canonical_start_tick, "end_tick": span.canonical_end_tick, "meter": span.time_signature, "is_partial": span.is_partial} for span in spans]
        encoded = json.dumps({"canonical_parser_version": "raw_smf_v1", "ppqn": ppqn, "spans": payload}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _form_mapping_status(metadata: Mapping[str, Any], source_identity: str, timeline_hash: str) -> str:
        if not metadata:
            return "absent"
        if metadata.get("coordinate_system") != "canonical_bar_index.v1":
            return "unavailable_legacy_measure_index"
        if metadata.get("source_file_identity") != source_identity or metadata.get("canonical_parser_version") != "raw_smf_v1" or metadata.get("canonical_timeline_sha256") != timeline_hash:
            return "unavailable_timeline_mismatch"
        return "mapped"

    def _apply_canonical_form_metadata(self, song: SongRecord, metadata: Mapping[str, Any]) -> None:
        """Attach only bound canonical-bar sections; fail closed on malformed ranges."""
        if song.metadata["form_mapping_status"] != "mapped":
            return
        def reject(status: str) -> None:
            song.metadata["form_mapping_status"] = status
            song.form = None
            for bar in song.bars:
                bar.form = bar.section_label = bar.section_index = None

        sections = metadata.get("sections")
        if not isinstance(sections, list):
            reject("unavailable_timeline_mismatch")
            return
        if not sections:
            reject("unavailable_empty_canonical_sections")
            return
        assignments: dict[int, tuple[str, int]] = {}
        for section_index, section in enumerate(sections):
            if not isinstance(section, Mapping):
                reject("unavailable_timeline_mismatch"); return
            start, end = section.get("canonical_start_bar_index"), section.get("canonical_end_bar_index")
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
                reject("unavailable_timeline_mismatch"); return
            matches = [bar for bar in song.bars if bar.canonical_bar_index is not None and start <= int(bar.canonical_bar_index) < end]
            if len(matches) != end - start or any(int(bar.canonical_bar_index) in assignments for bar in matches):
                reject("unavailable_timeline_mismatch"); return
            for bar in matches:
                assignments[int(bar.canonical_bar_index)] = (str(section.get("name", f"section_{section_index}")), section_index)
        for bar in song.bars:
            assignment = assignments.get(int(bar.canonical_bar_index)) if bar.canonical_bar_index is not None else None
            if assignment is not None:
                bar.form, bar.section_index = assignment
                bar.section_label = bar.form
        song.form = str(metadata.get("form")) if metadata.get("form") is not None else None

    @staticmethod
    def _pairing_repair_fact(source_file_identity: str, path: Path, dataset_root: str | Path | None, midi: Any, repair: Any) -> Dict[str, Any]:
        """Attach file provenance to one canonical raw-pairing repair."""
        return {
            "repair_kind": repair.repair_kind,
            "source_file_identity": source_file_identity,
            "dataset_relative_posix_path": unicodedata.normalize("NFC", path.resolve().relative_to(Path(dataset_root or path.parent).resolve()).as_posix()),
            "tune_index": 0,
            "physical_track_index": repair.physical_track_index,
            "channel": repair.channel,
            "pitch": repair.pitch,
            "on_tick": repair.on_tick,
            "on_event_ordinal": repair.on_event_ordinal,
            "off_tick": repair.off_tick,
            "off_event_ordinal": repair.off_event_ordinal,
            "on_velocity": repair.on_velocity,
            "queue_depth_before": repair.queue_depth_before,
            "same_tick_events": [
                {"event_kind": kind, "event_ordinal": ordinal, "velocity": velocity}
                for kind, ordinal, velocity in repair.same_tick_events
            ],
            "smf_format": int(midi.type),
            "ppqn": int(midi.ticks_per_beat),
        }

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
    def _canonical_quantization_audit(fragments: Sequence[Any], repairs: Sequence[Any], paired_source_note_count: int) -> Dict[str, Any]:
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
        projected = sum(fragment.quantization_repair_kind is not None for fragment in fragments)
        repair_counts = Counter(repair.repair_kind for repair in repairs)
        dropped = repair_counts["same_tick_zero_duration_pair"]
        denominator = paired_source_note_count + dropped
        return {"status": "MONITOR", "quantum_ql": 0.25, "source_boundaries_retained": True, "audit_unit": "source_note_fragment", "fragment_count": fragment_count, "projected_fragment_count": projected, "projected_fragment_rate": projected / fragment_count if fragment_count else 0.0, "raw_pairing_repair_counts": {"same_tick_zero_duration_pair": repair_counts["same_tick_zero_duration_pair"], "redundant_orphan_note_off": repair_counts["redundant_orphan_note_off"]}, "paired_source_note_count": paired_source_note_count, "dropped_source_note_count": dropped, "dropped_source_note_ratio": dropped / denominator if denominator else 0.0, "event_count": fragment_count, "nonzero_residual_count": sum(value > 1e-9 for item in by_meter.values() for values in item.values() for value in values), "by_meter": {meter: {"fragment_count": len(values["onset"]), "projected_fragment_count": sum(fragment.meter == meter and fragment.quantization_repair_kind is not None for fragment in fragments), "event_count": len(values["onset"]), "nonzero_residual_count": sum(value > 1e-9 for value in values["onset"] + values["end"]), "onset_residual_ql": summary(values["onset"]), "end_residual_ql": summary(values["end"])} for meter, values in by_meter.items()}}

    @staticmethod
    def _canonical_fragment_samples(
        source_file_identity: str,
        fragments: Sequence[Any],
    ) -> list[Dict[str, Any]]:
        """Return one complete audit fact for each source-note/bar fragment."""
        samples = []
        for fragment in fragments:
            note = fragment.source_note
            raw_start = float(fragment.raw_local_start_ql)
            raw_end = float(fragment.raw_local_end_ql)
            quantized_start = float(fragment.quantized_local_start_ql)
            quantized_end = float(fragment.quantized_local_end_ql)
            samples.append({
                "source_note_id": f"{source_file_identity}:{note.physical_track_index}:{note.source_note_ordinal}",
                "canonical_bar_index": int(fragment.canonical_bar_index),
                "meter": str(fragment.meter),
                "raw_local_start_tick": int(fragment.raw_local_start_tick),
                "raw_local_end_tick": int(fragment.raw_local_end_tick),
                "ppqn": int(fragment.ppqn),
                "raw_local_start_ql": raw_start,
                "raw_local_end_ql": raw_end,
                "ordinary_quantized_local_start_ql": float(fragment.ordinary_quantized_local_start_ql),
                "ordinary_quantized_local_end_ql": float(fragment.ordinary_quantized_local_end_ql),
                "final_quantized_local_start_ql": quantized_start,
                "final_quantized_local_end_ql": quantized_end,
                "quantization_repair_kind": fragment.quantization_repair_kind,
                "repair_slot_index": fragment.repair_slot_index,
                "projection_overlap_ql": float(fragment.projection_overlap_ql) if fragment.projection_overlap_ql is not None else None,
                "projection_endpoint_error_ql": float(fragment.projection_endpoint_error_ql) if fragment.projection_endpoint_error_ql is not None else None,
                "onset_residual_ql": abs(quantized_start - raw_start),
                "end_residual_ql": abs(quantized_end - raw_end),
            })
        return samples

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
            is_partial=span.is_partial,
            partial_reason=span.partial_reason,
            nominal_meter=span.nominal_meter,
            triggering_ts_tick=span.triggering_ts_tick,
            triggering_ts_provenance=({"physical_track_index": span.triggering_ts_provenance[0], "event_ordinal": span.triggering_ts_provenance[1], "numerator": span.triggering_ts_provenance[2], "denominator": span.triggering_ts_provenance[3]} if span.triggering_ts_provenance is not None else None),
            canonical_start_tick=span.canonical_start_tick,
            canonical_end_tick=span.canonical_end_tick,
            ppqn=span.ppqn,
            source_bar_count=int(song.metadata["canonical_span_count"]),
            tracks=tracks,
        )

    @staticmethod
    def _source_file_identity(path: Path, dataset_root: str | Path | None) -> str:
        """Return a stable dataset-path and raw-byte identity for one source file."""
        root = Path(dataset_root) if dataset_root is not None else path.parent
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        normalized = unicodedata.normalize("NFC", relative)
        inner = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashlib.sha256(f"{normalized}\0{inner}".encode("utf-8")).hexdigest()
