#!/usr/bin/env python3
"""Input parsing and bar preprocessing for the symbolic music engine."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from music21 import chord, converter, note, stream

from data.bar_features import BarFeatureExtractor
from common.config_loader import ConfigLoader, ConfigView
from data.core_data import BarRecord, NoteRecord, SongRecord
from diagnostics.diagnostics import TrainingDiagnostics


log = logging.getLogger("music_input")
MUSIC_PATTERNS = ("*.mid", "*.midi", "*.krn", "*.abc", "*.musicxml", "*.xml")


@dataclass(frozen=True)
class InputParserConfig:
    steps_per_bar: int = 16
    bar_length_ql: float = 4.0
    rest_token: int = -1
    sustain_token: int = -2
    polyphonic_strategy: str = "melody_top"
    quantize_input: bool = True
    quantize_divisors: tuple[int, ...] = (4, 3)
    quantize_offsets: bool = True
    quantize_durations: bool = True
    quantize_policy: str = "nearest"
    min_overlap_fraction: float = 0.05


class InputParser:
    """Read music files into SongRecord/BarRecord objects."""

    def __init__(self, config: InputParserConfig) -> None:
        self.config = config
        self.failed_files: List[Dict[str, Any]] = []
        self.feature_extractor = BarFeatureExtractor(
            rest_token=config.rest_token,
            sustain_token=config.sustain_token,
        )

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "InputParser":
        section = ConfigView(config).section("grid_tokenizer")
        return cls(InputParserConfig(
            steps_per_bar=int(section.get("steps_per_bar", 16)),
            bar_length_ql=float(section.get("bar_length_ql", 4.0)),
            rest_token=int(section.get("rest_token", -1)),
            sustain_token=int(section.get("sustain_token", -2)),
            polyphonic_strategy=str(section.get("polyphonic_strategy", "melody_top")),
            quantize_input=bool(section.get("quantize_input", True)),
            quantize_divisors=tuple(int(x) for x in section.get("quantize_divisors", [4, 3])),
            quantize_offsets=bool(section.get("quantize_offsets", True)),
            quantize_durations=bool(section.get("quantize_durations", True)),
            quantize_policy=str(section.get("quantize_policy", "nearest")),
            min_overlap_fraction=float(section.get("min_overlap_fraction", 0.05)),
        ))

    def parse_directory(self, music_dir: str | Path) -> List[SongRecord]:
        music_dir = Path(music_dir)
        form_map = self._load_form_map(music_dir)
        songs: List[SongRecord] = []
        for file_path in self._discover_files(music_dir):
            try:
                songs.append(self.parse_file(file_path, form_map.get(file_path.name, {})))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                log.warning("Skipping %s: %s", file_path, message)
                self.failed_files.append({"file_path": str(file_path), "error": message})
        return songs

    def parse_file(self, file_path: str | Path, metadata: Dict[str, Any]) -> SongRecord:
        file_path = Path(file_path)
        score = converter.parse(str(file_path))
        score = self._quantize_score(score)
        events = self._collect_note_events(score)
        if not events:
            raise ValueError("No note events found.")
        bar_count = int(math.ceil(max(end for _, end, _, _ in events) / self.config.bar_length_ql))
        song = SongRecord(
            song_id=file_path.stem,
            file_path=str(file_path),
            genre=metadata.get("genre"),
            form=metadata.get("form"),
            metadata=metadata,
        )
        for bar_index in range(bar_count):
            bar = self._build_bar(song, events, bar_index, bar_count)
            self._assign_form_section(bar, metadata)
            self._preprocess_bar(bar)
            song.bars.append(bar)
        return song

    def _discover_files(self, music_dir: Path) -> List[Path]:
        files: List[Path] = []
        for pattern in MUSIC_PATTERNS:
            files.extend(music_dir.rglob(pattern))
        return sorted(set(files))

    def _load_form_map(self, music_dir: Path) -> Dict[str, Dict[str, Any]]:
        path = music_dir / "form.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _quantize_score(self, score: stream.Score) -> stream.Score:
        if not self.config.quantize_input:
            return score
        return score.quantize(
            quarterLengthDivisors=self.config.quantize_divisors,
            processOffsets=self.config.quantize_offsets,
            processDurations=self.config.quantize_durations,
            inPlace=False,
        )

    def _collect_note_events(self, score: stream.Score) -> List[tuple[float, float, int, int]]:
        events: List[tuple[float, float, int, int]] = []
        for element in score.flatten().notes:
            pitch = self._event_pitch(element)
            if pitch is None:
                continue
            start = float(element.offset)
            duration = float(element.quarterLength)
            if duration <= 0:
                continue
            velocity = int(getattr(getattr(element, "volume", None), "velocity", None) or 64)
            events.append((start, start + duration, pitch, velocity))
        return sorted(events, key=lambda item: (item[0], item[2]))

    def _event_pitch(self, element: Any) -> Optional[int]:
        if isinstance(element, note.Note):
            return int(element.pitch.midi)
        if isinstance(element, chord.Chord):
            pitches = [int(p.midi) for p in element.pitches]
            if not pitches:
                return None
            if self.config.polyphonic_strategy == "melody_low":
                return min(pitches)
            if self.config.polyphonic_strategy == "chord_root":
                root = element.root()
                return int(root.midi) if root is not None else max(pitches)
            return max(pitches)
        return None

    def _build_bar(
        self,
        song: SongRecord,
        events: Sequence[tuple[float, float, int, int]],
        bar_index: int,
        bar_count: int,
    ) -> BarRecord:
        bar_offset = bar_index * self.config.bar_length_ql
        notes = []
        for start, end, pitch, velocity in events:
            if end <= bar_offset or start >= bar_offset + self.config.bar_length_ql:
                continue
            local_start = max(0.0, start - bar_offset)
            local_end = min(self.config.bar_length_ql, end - bar_offset)
            notes.append(NoteRecord(
                pitch=pitch,
                onset_ql=local_start,
                duration_ql=max(0.0, local_end - local_start),
                velocity=velocity,
            ))
        return BarRecord(
            song_id=song.song_id,
            file_path=song.file_path,
            bar_index=bar_index,
            time_signature="4/4",
            bar_length_ql=self.config.bar_length_ql,
            source_bar_count=bar_count,
            notes=notes,
            genre=song.genre,
            form=song.form,
        )

    def _assign_form_section(self, bar: BarRecord, metadata: Dict[str, Any]) -> None:
        sections = metadata.get("sections") or []
        for idx, section in enumerate(sections):
            start = int(section.get("start_bar", section.get("start", 0)))
            if section.get("end_bar", section.get("end")) is not None:
                end = int(section.get("end_bar", section.get("end")))
            else:
                end = start + int(section.get("length", 0))
            if start <= bar.bar_index < end:
                bar.section_label = str(section.get("name"))
                bar.section_index = idx
                return

    def _preprocess_bar(self, bar: BarRecord) -> None:
        bar.absolute_tokens = self._tokenize_bar(bar)
        bar.relative_tokens = self._relative_tokens(bar.absolute_tokens)
        self.feature_extractor.apply(bar)

    def _tokenize_bar(self, bar: BarRecord) -> List[int]:
        slot_len = self.config.bar_length_ql / self.config.steps_per_bar
        tokens = [self.config.rest_token for _ in range(self.config.steps_per_bar)]
        onset_candidates: List[List[int]] = [[] for _ in range(self.config.steps_per_bar)]
        sustain_present = [False] * self.config.steps_per_bar
        for note_record in bar.notes:
            onset_slot = self._quantize_slot(note_record.onset_ql, slot_len)
            if 0 <= onset_slot < self.config.steps_per_bar:
                onset_candidates[onset_slot].append(note_record.pitch)
            local_end = note_record.onset_ql + note_record.duration_ql
            first_slot = max(0, int(math.floor(note_record.onset_ql / slot_len)))
            last_slot = min(self.config.steps_per_bar - 1, int(math.floor((local_end - 1e-9) / slot_len)))
            for slot in range(first_slot, last_slot + 1):
                slot_start = slot * slot_len
                overlap = max(0.0, min(local_end, slot_start + slot_len) - max(note_record.onset_ql, slot_start))
                if overlap / slot_len >= self.config.min_overlap_fraction:
                    sustain_present[slot] = True
        for slot, pitches in enumerate(onset_candidates):
            if pitches:
                tokens[slot] = min(pitches) if self.config.polyphonic_strategy == "melody_low" else max(pitches)
            elif sustain_present[slot]:
                tokens[slot] = self.config.sustain_token
        return tokens

    def _relative_tokens(self, tokens: Sequence[int]) -> List[int]:
        pitches = [token for token in tokens if token >= 0]
        if not pitches:
            return list(tokens)
        root = min(pitches)
        return [token - root if token >= 0 else token for token in tokens]

    def _quantize_slot(self, onset_ql: float, slot_len: float) -> int:
        if self.config.quantize_policy == "floor":
            return int(math.floor(onset_ql / slot_len))
        return int(round(onset_ql / slot_len))


class InputParserCLI:
    """Standalone CLI for parsing and preprocessing music directories."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Parse music files into SongRecord JSON.")
        parser.add_argument("--music-dir", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--diagnostics-output", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        diagnostics = TrainingDiagnostics()
        parser = InputParser.from_style_config(config)
        songs = parser.parse_directory(args.music_dir)
        bars = [bar for song in songs for bar in song.bars]
        payload = {"songs": [song.to_dict() for song in songs]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        diagnostics.record_input_summary(len(songs), parser.failed_files, len(bars))
        if args.diagnostics_output:
            diagnostics.write(args.diagnostics_output)
        print(f"Wrote {len(songs)} songs / {len(bars)} bars -> {args.output}")


def main() -> None:
    InputParserCLI().run()


if __name__ == "__main__":
    main()
