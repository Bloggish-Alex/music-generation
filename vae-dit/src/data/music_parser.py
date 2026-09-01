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
from data.core import BarRecord, NoteEvent, SongRecord, TrackRecord


MUSIC_SUFFIXES = {".mid", ".midi", ".abc", ".krn"}


@dataclass(frozen=True)
class MusicParserConfig:
    """Configuration for symbolic music parsing."""

    bar_length_ql: float = 4.0
    quantize_input: bool = True
    quantize_divisors: tuple[int, ...] = (4, 3)
    quantize_offsets: bool = True
    quantize_durations: bool = True
    max_tracks: int = 3
    register_split_low_max: int = 55
    register_split_mid_max: int = 72
    default_velocity: int = 64

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MusicParserConfig":
        """Build parser configuration from the style configuration."""
        section = ConfigView(config).section("music_parser")
        return cls(
            bar_length_ql=float(section.get("bar_length_ql", 4.0)),
            quantize_input=bool(section.get("quantize_input", True)),
            quantize_divisors=tuple(int(x) for x in section.get("quantize_divisors", [4, 3])),
            quantize_offsets=bool(section.get("quantize_offsets", True)),
            quantize_durations=bool(section.get("quantize_durations", True)),
            max_tracks=int(section.get("max_tracks", 3)),
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
                songs.append(self.parse_file(
                    file_path,
                    form_map.get(file_path.name, {}),
                    transpose_semitones=transpose_semitones,
                    dataset_root=root,
                ))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"Skipping {file_path}: {message}")
                self.failed_files.append({"file_path": str(file_path), "error": message})
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
    ) -> SongRecord:
        """Parse one file into a song record with bar-local tracks."""
        from music21 import converter

        path = Path(file_path)
        score = converter.parse(str(path))
        if int(transpose_semitones) != 0:
            score = score.transpose(int(transpose_semitones), inPlace=False)
        score = self._quantize_score(score)
        raw_tracks = self._collect_tracks(score)
        if not raw_tracks:
            raise ValueError("No note events found.")
        bar_count = self._bar_count(raw_tracks)
        suffix = f"_T{int(transpose_semitones):+d}" if int(transpose_semitones) != 0 else ""
        song = SongRecord(
            song_id=f"{path.stem}{suffix}",
            file_path=str(path),
            form=metadata.get("form"),
            metadata={**dict(metadata), "transpose_semitones": int(transpose_semitones), "source_file_identity": self._source_file_identity(path, dataset_root)},
        )
        for bar_index in range(bar_count):
            bar = self._build_bar(song, raw_tracks, bar_index, bar_count)
            self._assign_form_section(bar, metadata)
            song.bars.append(bar)
        return song

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

    def _collect_tracks(self, score: stream.Score) -> List[List[tuple[float, float, int, int]]]:
        """Collect note events per music21 part, with register splitting fallback."""
        parts = list(score.parts) if getattr(score, "parts", None) else []
        if len(parts) > 1:
            tracks = [self._collect_events(part) for part in parts]
            tracks = [track for track in tracks if track]
            return self._select_tracks(tracks)
        events = self._collect_events(score)
        if not events:
            return []
        return self._split_by_register(events)

    def _collect_events(self, container: Any) -> List[tuple[float, float, int, int]]:
        """Collect flat note events from one part or score."""
        events: List[tuple[float, float, int, int]] = []
        for element in container.flatten().notes:
            pitches = self._element_pitches(element)
            if not pitches:
                continue
            start = float(element.offset)
            duration = float(element.quarterLength)
            if duration <= 0:
                continue
            velocity = int(getattr(getattr(element, "volume", None), "velocity", None) or self.config.default_velocity)
            for pitch in pitches:
                events.append((start, start + duration, int(pitch), velocity))
        return sorted(events, key=lambda item: (item[0], item[2]))

    def _element_pitches(self, element: Any) -> List[int]:
        """Extract MIDI pitches from a note or chord element."""
        from music21 import chord, note

        if isinstance(element, note.Note):
            return [int(element.pitch.midi)]
        if isinstance(element, chord.Chord):
            return [int(pitch.midi) for pitch in element.pitches]
        return []

    def _select_tracks(self, tracks: Sequence[List[tuple[float, float, int, int]]]) -> List[List[tuple[float, float, int, int]]]:
        """Keep the most active tracks when the source has more than max_tracks."""
        ranked = sorted(tracks, key=lambda track: len(track), reverse=True)
        return [list(track) for track in ranked[: max(1, self.config.max_tracks)]]

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

    def _bar_count(self, tracks: Sequence[Sequence[tuple[float, float, int, int]]]) -> int:
        """Compute the number of bars needed to cover all track events."""
        max_end = max(float(event[1]) for track in tracks for event in track)
        return max(1, int(math.ceil(max_end / self.config.bar_length_ql)))

    def _build_bar(
        self,
        song: SongRecord,
        tracks: Sequence[Sequence[tuple[float, float, int, int]]],
        bar_index: int,
        bar_count: int,
    ) -> BarRecord:
        """Build one bar record from global note events."""
        bar_start = float(bar_index) * self.config.bar_length_ql
        bar_end = bar_start + self.config.bar_length_ql
        bar_tracks: List[TrackRecord] = []
        for track_index, track_events in enumerate(tracks[: self.config.max_tracks]):
            notes = []
            for source_note_ordinal, (start, end, pitch, velocity) in enumerate(track_events):
                if end <= bar_start or start >= bar_end:
                    continue
                local_start = max(0.0, float(start) - bar_start)
                local_end = min(self.config.bar_length_ql, float(end) - bar_start)
                notes.append(NoteEvent(
                    pitch=int(pitch),
                    onset_ql=local_start,
                    duration_ql=max(0.0, local_end - local_start),
                    velocity=int(velocity),
                    source_file_identity=str(song.metadata["source_file_identity"]),
                    physical_track_index=int(track_index),
                    source_note_ordinal=int(source_note_ordinal),
                    source_onset_ql=float(start),
                ))
            bar_tracks.append(TrackRecord(track_index=track_index, name=f"track_{track_index}", notes=notes))
        return BarRecord(
            song_id=song.song_id,
            file_path=song.file_path,
            bar_index=int(bar_index),
            bar_length_ql=self.config.bar_length_ql,
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
