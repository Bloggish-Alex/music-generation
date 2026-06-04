#!/usr/bin/env python3
"""Convert MIDI-like scores into fixed time-step bar token grids."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from music21 import converter, chord, note, stream

from config_loader import ConfigLoader, ConfigView


log = logging.getLogger("grid_tokenizer")

MUSIC_PATTERNS = ("*.mid", "*.midi", "*.musicxml", "*.xml", "*.krn", "*.abc")


@dataclass(frozen=True)
class GridTokenizerConfig:
    steps_per_bar: int = 16
    bar_length_ql: float = 4.0
    rest_token: int = -1
    sustain_token: int = 0
    polyphonic_strategy: str = "melody_top"
    quantize_input: bool = True
    quantize_divisors: tuple[int, ...] = (4, 3)
    quantize_offsets: bool = True
    quantize_durations: bool = True
    quantize_policy: str = "nearest"
    min_overlap_fraction: float = 0.05


@dataclass
class BarGrid:
    tokens: List[int]
    file_path: str
    bar_index: int
    bar_offset_ql: float
    bar_length_ql: float
    source_index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GridTokenizer:
    """Extract 16-step per-bar token grids from symbolic music files."""

    def __init__(self, config: GridTokenizerConfig) -> None:
        self.config = config

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "GridTokenizer":
        section = ConfigView(config).section("grid_tokenizer")
        return cls(GridTokenizerConfig(
            steps_per_bar=int(section.get("steps_per_bar", 16)),
            bar_length_ql=float(section.get("bar_length_ql", 4.0)),
            rest_token=int(section.get("rest_token", -1)),
            sustain_token=int(section.get("sustain_token", 0)),
            polyphonic_strategy=str(section.get("polyphonic_strategy", "melody_top")),
            quantize_input=bool(section.get("quantize_input", True)),
            quantize_divisors=tuple(int(x) for x in section.get("quantize_divisors", [4, 3])),
            quantize_offsets=bool(section.get("quantize_offsets", True)),
            quantize_durations=bool(section.get("quantize_durations", True)),
            quantize_policy=str(section.get("quantize_policy", "nearest")),
            min_overlap_fraction=float(section.get("min_overlap_fraction", 0.05)),
        ))

    def tokenize_path(self, path: str | Path) -> List[BarGrid]:
        path = Path(path)
        score = converter.parse(str(path))
        score = self._quantize_score(score)
        return self.tokenize_score(score, file_path=str(path))

    def tokenize_directory(
        self,
        music_dir: str | Path,
        patterns: Sequence[str] = MUSIC_PATTERNS,
        limit_files: Optional[int] = None,
    ) -> List[BarGrid]:
        files = self._discover_files(Path(music_dir), patterns)
        if limit_files is not None:
            files = files[:limit_files]
        bars: List[BarGrid] = []
        for file_path in files:
            try:
                file_bars = self.tokenize_path(file_path)
            except Exception as exc:
                log.warning("Skipping %s: %s", file_path, exc)
                continue
            bars.extend(file_bars)
        for idx, bar in enumerate(bars):
            bar.source_index = idx
        return bars

    def tokenize_score(self, score: stream.Score, file_path: str = "") -> List[BarGrid]:
        events = self._collect_note_events(score)
        if not events:
            return []
        max_end = max(end for _, end, _ in events)
        bar_count = int(math.ceil(max_end / self.config.bar_length_ql))
        bars: List[BarGrid] = []
        for bar_index in range(bar_count):
            bar_offset = bar_index * self.config.bar_length_ql
            tokens = self._tokenize_bar(events, bar_offset)
            bars.append(BarGrid(
                tokens=tokens,
                file_path=file_path,
                bar_index=bar_index,
                bar_offset_ql=bar_offset,
                bar_length_ql=self.config.bar_length_ql,
            ))
        return bars

    def save_bars(self, bars: Sequence[BarGrid], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "bars": [bar.to_dict() for bar in bars],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_bars(self, input_path: str | Path) -> List[BarGrid]:
        return self.load_bars_file(input_path)

    @staticmethod
    def load_bars_file(input_path: str | Path) -> List[BarGrid]:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        bars = []
        for idx, item in enumerate(payload.get("bars", [])):
            bar = BarGrid(**item)
            if bar.source_index < 0:
                bar.source_index = idx
            bars.append(bar)
        return bars

    def _discover_files(self, music_dir: Path, patterns: Sequence[str]) -> List[Path]:
        files: List[Path] = []
        for pattern in patterns:
            files.extend(music_dir.rglob(pattern))
        return sorted(set(files))

    def _collect_note_events(self, score: stream.Score) -> List[tuple[float, float, int]]:
        flat = score.flatten()
        events: List[tuple[float, float, int]] = []
        for element in flat.notes:
            pitch = self._event_pitch(element)
            if pitch is None:
                continue
            start = float(element.offset)
            duration = float(element.quarterLength)
            if duration <= 0:
                continue
            events.append((start, start + duration, pitch))
        return sorted(events, key=lambda item: (item[0], item[2]))

    def _quantize_score(self, score: stream.Score) -> stream.Score:
        if not self.config.quantize_input:
            return score
        return score.quantize(
            quarterLengthDivisors=self.config.quantize_divisors,
            processOffsets=self.config.quantize_offsets,
            processDurations=self.config.quantize_durations,
            inPlace=False,
        )

    def _event_pitch(self, element: Any) -> Optional[int]:
        strategy = self.config.polyphonic_strategy
        if isinstance(element, note.Note):
            return int(element.pitch.midi)
        if isinstance(element, chord.Chord):
            pitches = [int(p.midi) for p in element.pitches]
            if not pitches:
                return None
            if strategy == "melody_low":
                return min(pitches)
            if strategy == "chord_root":
                root = element.root()
                return int(root.midi) if root is not None else max(pitches)
            return max(pitches)
        return None

    def _tokenize_bar(
        self,
        events: Sequence[tuple[float, float, int]],
        bar_offset: float,
    ) -> List[int]:
        cfg = self.config
        slot_len = cfg.bar_length_ql / cfg.steps_per_bar
        tokens = [cfg.rest_token for _ in range(cfg.steps_per_bar)]
        onset_candidates: List[List[int]] = [[] for _ in range(cfg.steps_per_bar)]
        sustain_present = [False] * cfg.steps_per_bar

        for start, end, pitch in events:
            if end <= bar_offset or start >= bar_offset + cfg.bar_length_ql:
                continue
            local_start = start - bar_offset
            local_end = end - bar_offset
            onset_slot = self._quantize_slot(local_start, cfg.steps_per_bar, slot_len)
            if 0 <= onset_slot < cfg.steps_per_bar:
                onset_candidates[onset_slot].append(pitch)

            first_slot = max(0, int(math.floor(local_start / slot_len)))
            last_slot = min(cfg.steps_per_bar - 1, int(math.floor((local_end - 1e-9) / slot_len)))
            for slot in range(first_slot, last_slot + 1):
                slot_start = slot * slot_len
                slot_end = slot_start + slot_len
                overlap = max(0.0, min(local_end, slot_end) - max(local_start, slot_start))
                if overlap / slot_len >= cfg.min_overlap_fraction:
                    sustain_present[slot] = True

        for slot, pitches in enumerate(onset_candidates):
            if pitches:
                tokens[slot] = max(pitches) if cfg.polyphonic_strategy != "melody_low" else min(pitches)
            elif sustain_present[slot]:
                tokens[slot] = cfg.sustain_token
        return tokens

    def _quantize_slot(self, local_start: float, steps: int, slot_len: float) -> int:
        if self.config.quantize_policy == "floor":
            return int(math.floor(local_start / slot_len))
        return int(round(local_start / slot_len))


class GridTokenizerCLI:
    """CLI for extracting bar token grids."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Tokenize music files into bar grids.")
        parser.add_argument("--music-dir", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--steps-per-bar", type=int, default=None)
        parser.add_argument("--bar-length-ql", type=float, default=None)
        parser.add_argument("--polyphonic-strategy", default=None)
        parser.add_argument("--no-input-quantize", action="store_true")
        parser.add_argument(
            "--quantize-divisors",
            default=None,
            help="Comma-separated music21 quarterLengthDivisors, e.g. 4,3 or 8,4,3.",
        )
        parser.add_argument("--limit-files", type=int, default=None)
        parser.add_argument("--verbose", action="store_true")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
        config = ConfigLoader().load(args.config)
        tokenizer_config = GridTokenizer.from_style_config(config).config
        updates = {
            "steps_per_bar": args.steps_per_bar,
            "bar_length_ql": args.bar_length_ql,
            "polyphonic_strategy": args.polyphonic_strategy,
            "quantize_input": False if args.no_input_quantize else None,
            "quantize_divisors": (
                tuple(int(x.strip()) for x in args.quantize_divisors.split(",") if x.strip())
                if args.quantize_divisors else None
            ),
        }
        tokenizer_config = GridTokenizerConfig(**{
            **asdict(tokenizer_config),
            **{k: v for k, v in updates.items() if v is not None},
        })
        tokenizer = GridTokenizer(tokenizer_config)
        bars = tokenizer.tokenize_directory(args.music_dir, limit_files=args.limit_files)
        tokenizer.save_bars(bars, args.output)
        print(f"Wrote {len(bars)} bars -> {args.output}")


def main() -> None:
    GridTokenizerCLI().run()


if __name__ == "__main__":
    main()
