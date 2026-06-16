#!/usr/bin/env python3
"""Compact generation-quality report from generation diagnostics.

The generation diagnostics JSON is intentionally rich, but most listening
iterations only need a small set of continuity and sparsity indicators. This
tool extracts those indicators into a Markdown report plus optional CSV/JSON
artifacts so runs can be compared without manually reading the full trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class ReportPaths:
    diagnostics: Path
    markdown: Path
    csv: Path
    summary_json: Path


class GenerationDiagnosticsReader:
    """Read generation diagnostics while tolerating older schema variants."""

    def __init__(self, diagnostics_path: Path) -> None:
        self.diagnostics_path = diagnostics_path
        self.payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))

    def bars(self) -> List[Dict[str, Any]]:
        stages = self.payload.get("stages", {})
        harmonic = stages.get("harmonic_engine", {})
        bars = harmonic.get("bars", [])
        if not isinstance(bars, list):
            raise ValueError("generation diagnostics does not contain stages.harmonic_engine.bars")
        return bars

    def harmonic_engine(self) -> Dict[str, Any]:
        return dict(self.payload.get("stages", {}).get("harmonic_engine", {}))

    def section_plan(self) -> List[Dict[str, Any]]:
        section_stage = self.payload.get("stages", {}).get("section_plan", {})
        sections = section_stage.get("sections", [])
        return sections if isinstance(sections, list) else []


class BoundaryFeatureExtractor:
    """Build per-boundary metrics from realized bars."""

    def rows(self, bars: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for index in range(1, len(bars)):
            previous = bars[index - 1]
            current = bars[index]
            previous_pitches = self._pitches(previous)
            current_pitches = self._pitches(current)
            previous_last = previous_pitches[-1] if previous_pitches else None
            current_first = current_pitches[0] if current_pitches else None
            signed_jump = (
                current_first - previous_last
                if previous_last is not None and current_first is not None
                else None
            )
            rows.append({
                "boundary": f"{index - 1}->{index}",
                "from_bar": index - 1,
                "to_bar": index,
                "section_from": self._section(previous),
                "section_to": self._section(current),
                "same_section": self._section(previous) == self._section(current),
                "section_local_to": self._sampled(current).get("section_local_index"),
                "prev_last_pitch": previous_last,
                "current_first_pitch": current_first,
                "signed_jump": signed_jump,
                "abs_jump": abs(signed_jump) if signed_jump is not None else None,
                "prev_observation_id": self._sampled(previous).get("observation_id"),
                "current_observation_id": self._sampled(current).get("observation_id"),
                "observation_switch": (
                    self._sampled(previous).get("observation_id")
                    != self._sampled(current).get("observation_id")
                ),
                "prev_codebook_id": self._codebook_id(previous),
                "current_codebook_id": self._codebook_id(current),
                "codebook_switch": (
                    self._codebook_id(previous)
                    != self._codebook_id(current)
                ),
                "prev_source_song": self._source(previous),
                "current_source_song": self._source(current),
                "source_switch": self._source(previous) != self._source(current),
                "prev_note_on_ratio": self._density(previous, "note_on_ratio"),
                "current_note_on_ratio": self._density(current, "note_on_ratio"),
                "note_on_ratio_delta": abs(
                    self._density(current, "note_on_ratio")
                    - self._density(previous, "note_on_ratio")
                ),
                "prev_rest_ratio": self._density(previous, "rest_ratio"),
                "current_rest_ratio": self._density(current, "rest_ratio"),
                "rest_ratio_delta": abs(
                    self._density(current, "rest_ratio")
                    - self._density(previous, "rest_ratio")
                ),
                "prev_sustain_ratio": self._density(previous, "sustain_ratio"),
                "current_sustain_ratio": self._density(current, "sustain_ratio"),
                "sustain_ratio_delta": abs(
                    self._density(current, "sustain_ratio")
                    - self._density(previous, "sustain_ratio")
                ),
            })
        return rows

    def _pitches(self, bar: Dict[str, Any]) -> List[int]:
        return [
            int(note["pitch"])
            for note in bar.get("notes", [])
            if isinstance(note, dict) and note.get("pitch") is not None
        ]

    def _sampled(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        sampled = bar.get("sampled", {})
        return sampled if isinstance(sampled, dict) else {}

    def _codebook_id(self, bar: Dict[str, Any]) -> Any:
        return self._sampled(bar).get("codebook_id")

    def _section(self, bar: Dict[str, Any]) -> Optional[str]:
        return self._sampled(bar).get("section") or bar.get("section")

    def _source(self, bar: Dict[str, Any]) -> Optional[str]:
        source = bar.get("codebook_source", {})
        return source.get("source_song") if isinstance(source, dict) else None

    def _density(self, bar: Dict[str, Any], key: str) -> float:
        density = bar.get("codebook_density", {})
        if not isinstance(density, dict):
            return 0.0
        value = density.get(key, 0.0)
        return float(value or 0.0)


class GenerationQualityAnalyzer:
    """Summarize continuity, sparsity, and selector behavior."""

    def __init__(self, jump_warning: int = 7, jump_failure: int = 12, jump_severe: int = 19) -> None:
        self.jump_warning = int(jump_warning)
        self.jump_failure = int(jump_failure)
        self.jump_severe = int(jump_severe)

    def analyze(
        self,
        bars: Sequence[Dict[str, Any]],
        boundary_rows: Sequence[Dict[str, Any]],
        harmonic_engine: Dict[str, Any],
    ) -> Dict[str, Any]:
        valid_boundaries = [row for row in boundary_rows if row.get("abs_jump") is not None]
        abs_jumps = [int(row["abs_jump"]) for row in valid_boundaries]
        selector = harmonic_engine.get("candidate_selector", {})
        selector_events = selector.get("events", []) if isinstance(selector, dict) else []
        selector_probs = [
            float(event.get("selected_probability") or 0.0)
            for event in selector_events
            if isinstance(event, dict)
        ]
        selector_counts = [
            int(event.get("candidate_count") or 0)
            for event in selector_events
            if isinstance(event, dict)
        ]
        return {
            "summary": self._summary(bars, boundary_rows, abs_jumps, selector, selector_probs, selector_counts),
            "jump_buckets": self._jump_buckets(abs_jumps),
            "section_metrics": self._section_metrics(boundary_rows),
            "pause_metrics": self._pause_metrics(bars),
            "observation_profiles": self._observation_profiles(bars, boundary_rows),
            "largest_discontinuities": self._largest_discontinuities(valid_boundaries),
            "diagnosis": self._diagnosis(boundary_rows, abs_jumps, selector_probs),
        }

    def _summary(
        self,
        bars: Sequence[Dict[str, Any]],
        boundary_rows: Sequence[Dict[str, Any]],
        abs_jumps: Sequence[int],
        selector: Any,
        selector_probs: Sequence[float],
        selector_counts: Sequence[int],
    ) -> Dict[str, Any]:
        return {
            "bar_count": len(bars),
            "boundary_count": len(boundary_rows),
            "valid_pitch_boundary_count": len(abs_jumps),
            "mean_abs_jump": self._mean(abs_jumps),
            "median_abs_jump": self._median(abs_jumps),
            "p75_abs_jump": self._percentile(abs_jumps, 0.75),
            "p90_abs_jump": self._percentile(abs_jumps, 0.90),
            "p95_abs_jump": self._percentile(abs_jumps, 0.95),
            "max_abs_jump": max(abs_jumps) if abs_jumps else 0,
            "jump_ge_warning_count": sum(value >= self.jump_warning for value in abs_jumps),
            "jump_ge_warning_ratio": self._ratio(sum(value >= self.jump_warning for value in abs_jumps), len(abs_jumps)),
            "jump_ge_failure_count": sum(value >= self.jump_failure for value in abs_jumps),
            "jump_ge_failure_ratio": self._ratio(sum(value >= self.jump_failure for value in abs_jumps), len(abs_jumps)),
            "jump_ge_severe_count": sum(value >= self.jump_severe for value in abs_jumps),
            "jump_ge_severe_ratio": self._ratio(sum(value >= self.jump_severe for value in abs_jumps), len(abs_jumps)),
            "observation_switch_ratio": self._ratio(sum(row["observation_switch"] for row in boundary_rows), len(boundary_rows)),
            "codebook_switch_ratio": self._ratio(sum(row["codebook_switch"] for row in boundary_rows), len(boundary_rows)),
            "source_song_switch_ratio": self._ratio(sum(row["source_switch"] for row in boundary_rows), len(boundary_rows)),
            "mean_note_on_ratio_delta": self._mean([row["note_on_ratio_delta"] for row in boundary_rows]),
            "p90_note_on_ratio_delta": self._percentile([row["note_on_ratio_delta"] for row in boundary_rows], 0.90),
            "mean_rest_ratio_delta": self._mean([row["rest_ratio_delta"] for row in boundary_rows]),
            "p90_rest_ratio_delta": self._percentile([row["rest_ratio_delta"] for row in boundary_rows], 0.90),
            "mean_sustain_ratio_delta": self._mean([row["sustain_ratio_delta"] for row in boundary_rows]),
            "p90_sustain_ratio_delta": self._percentile([row["sustain_ratio_delta"] for row in boundary_rows], 0.90),
            "candidate_selector_backend": selector.get("backend", "none") if isinstance(selector, dict) else "none",
            "candidate_selector_used_count": selector.get("used_count", 0) if isinstance(selector, dict) else 0,
            "candidate_selected_probability_mean": self._mean(selector_probs),
            "candidate_selected_probability_lt_0_05_ratio": self._ratio(
                sum(value < 0.05 for value in selector_probs),
                len(selector_probs),
            ),
            "candidate_count_mean": self._mean(selector_counts),
        }

    def _jump_buckets(self, abs_jumps: Sequence[int]) -> List[Dict[str, Any]]:
        buckets = [
            ("0-2", lambda value: 0 <= value <= 2),
            ("3-5", lambda value: 3 <= value <= 5),
            ("6-11", lambda value: 6 <= value <= 11),
            ("12-18", lambda value: 12 <= value <= 18),
            ("19+", lambda value: value >= 19),
        ]
        return [
            {
                "bucket": label,
                "count": sum(predicate(value) for value in abs_jumps),
                "ratio": self._ratio(sum(predicate(value) for value in abs_jumps), len(abs_jumps)),
            }
            for label, predicate in buckets
        ]

    def _section_metrics(self, boundary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in boundary_rows:
            label = (
                row["section_to"]
                if row.get("same_section")
                else f"{row['section_from']} -> {row['section_to']}"
            )
            grouped[str(label)].append(row)
        metrics = []
        for label, rows in grouped.items():
            jumps = [row["abs_jump"] for row in rows if row.get("abs_jump") is not None]
            metrics.append({
                "section_or_boundary": label,
                "boundary_count": len(rows),
                "mean_abs_jump": self._mean(jumps),
                "p90_abs_jump": self._percentile(jumps, 0.90),
                "jump_ge_failure_ratio": self._ratio(sum(value >= self.jump_failure for value in jumps), len(jumps)),
                "source_song_switch_ratio": self._ratio(sum(row["source_switch"] for row in rows), len(rows)),
                "mean_note_on_ratio_delta": self._mean([row["note_on_ratio_delta"] for row in rows]),
            })
        return sorted(metrics, key=lambda item: item["section_or_boundary"])

    def _pause_metrics(self, bars: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        rows = []
        for index, bar in enumerate(bars):
            tokens = [int(token) for token in bar.get("relative_tokens", [])]
            sampled = bar.get("sampled", {}) if isinstance(bar.get("sampled", {}), dict) else {}
            density = self._density_from_bar(bar, tokens)
            notes = bar.get("notes", []) if isinstance(bar.get("notes", []), list) else []
            rows.append({
                "bar_index": int(index),
                "section": sampled.get("section") or bar.get("section"),
                "section_local_index": sampled.get("section_local_index"),
                "observation_id": sampled.get("observation_id"),
                "note_on_ratio": density["note_on_ratio"],
                "rest_ratio": density["rest_ratio"],
                "sustain_ratio": density["sustain_ratio"],
                "note_count": len(notes),
                "max_rest_run": self._max_run(tokens, -1),
                "max_sustain_run": self._max_run(tokens, -2),
                "leading_rest_slots": self._leading_run(tokens, -1),
                "trailing_rest_slots": self._trailing_run(tokens, -1),
            })
        return {
            "mean_note_on_ratio": self._mean([row["note_on_ratio"] for row in rows]),
            "mean_rest_ratio": self._mean([row["rest_ratio"] for row in rows]),
            "mean_sustain_ratio": self._mean([row["sustain_ratio"] for row in rows]),
            "empty_decoded_bar_ratio": self._ratio(sum(row["note_on_ratio"] == 0.0 for row in rows), len(rows)),
            "sparse_decoded_bar_ratio_le_2_note_on": self._ratio(
                sum(row["note_on_ratio"] <= 2.0 / 16.0 for row in rows),
                len(rows),
            ),
            "high_rest_bar_ratio_ge_0_5": self._ratio(sum(row["rest_ratio"] >= 0.5 for row in rows), len(rows)),
            "high_sustain_bar_ratio_ge_0_75": self._ratio(sum(row["sustain_ratio"] >= 0.75 for row in rows), len(rows)),
            "max_rest_run": max((row["max_rest_run"] for row in rows), default=0),
            "max_sustain_run": max((row["max_sustain_run"] for row in rows), default=0),
            "top_sparse_bars": sorted(
                rows,
                key=lambda row: (row["note_on_ratio"], -row["rest_ratio"], row["bar_index"]),
            )[:15],
            "top_rest_bars": sorted(
                rows,
                key=lambda row: (row["rest_ratio"], row["max_rest_run"]),
                reverse=True,
            )[:15],
            "top_sustain_bars": sorted(
                rows,
                key=lambda row: (row["sustain_ratio"], row["max_sustain_run"]),
                reverse=True,
            )[:15],
        }

    def _observation_profiles(
        self,
        bars: Sequence[Dict[str, Any]],
        boundary_rows: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for index, bar in enumerate(bars):
            sampled = bar.get("sampled", {}) if isinstance(bar.get("sampled", {}), dict) else {}
            observation_id = sampled.get("observation_id")
            if observation_id is None:
                continue
            tokens = [int(token) for token in bar.get("relative_tokens", [])]
            density = self._density_from_bar(bar, tokens)
            pitches = [
                int(note["pitch"])
                for note in bar.get("notes", [])
                if isinstance(note, dict) and note.get("pitch") is not None
            ]
            grouped[str(observation_id)].append({
                "bar_index": int(index),
                "section": sampled.get("section") or bar.get("section"),
                "section_local_index": sampled.get("section_local_index"),
                "codebook_id": sampled.get("codebook_id"),
                "first_pitch": pitches[0] if pitches else None,
                "last_pitch": pitches[-1] if pitches else None,
                "note_on_ratio": density["note_on_ratio"],
                "rest_ratio": density["rest_ratio"],
                "sustain_ratio": density["sustain_ratio"],
                "note_count": len(pitches),
                "max_rest_run": self._max_run(tokens, -1),
                "max_sustain_run": self._max_run(tokens, -2),
            })
        incoming: Dict[str, List[float]] = defaultdict(list)
        outgoing: Dict[str, List[float]] = defaultdict(list)
        for row in boundary_rows:
            if row.get("abs_jump") is None:
                continue
            current = row.get("current_observation_id")
            previous = row.get("prev_observation_id")
            if current is not None:
                incoming[str(current)].append(float(row["abs_jump"]))
            if previous is not None:
                outgoing[str(previous)].append(float(row["abs_jump"]))
        profiles = []
        for observation_id, rows in grouped.items():
            profiles.append({
                "observation_id": int(observation_id),
                "bar_count": len(rows),
                "sections": sorted({str(row["section"]) for row in rows if row.get("section") is not None}),
                "mean_first_pitch": self._mean([row["first_pitch"] for row in rows if row["first_pitch"] is not None]),
                "mean_last_pitch": self._mean([row["last_pitch"] for row in rows if row["last_pitch"] is not None]),
                "mean_note_on_ratio": self._mean([row["note_on_ratio"] for row in rows]),
                "mean_rest_ratio": self._mean([row["rest_ratio"] for row in rows]),
                "mean_sustain_ratio": self._mean([row["sustain_ratio"] for row in rows]),
                "mean_note_count": self._mean([row["note_count"] for row in rows]),
                "max_rest_run": max((row["max_rest_run"] for row in rows), default=0),
                "max_sustain_run": max((row["max_sustain_run"] for row in rows), default=0),
                "incoming_boundary_count": len(incoming.get(observation_id, [])),
                "incoming_mean_abs_jump": self._mean(incoming.get(observation_id, [])),
                "incoming_p90_abs_jump": self._percentile(incoming.get(observation_id, []), 0.90),
                "incoming_jump_ge_12_ratio": self._ratio(
                    sum(value >= self.jump_failure for value in incoming.get(observation_id, [])),
                    len(incoming.get(observation_id, [])),
                ),
                "outgoing_boundary_count": len(outgoing.get(observation_id, [])),
                "outgoing_mean_abs_jump": self._mean(outgoing.get(observation_id, [])),
                "outgoing_p90_abs_jump": self._percentile(outgoing.get(observation_id, []), 0.90),
                "outgoing_jump_ge_12_ratio": self._ratio(
                    sum(value >= self.jump_failure for value in outgoing.get(observation_id, [])),
                    len(outgoing.get(observation_id, [])),
                ),
                "example_bars": rows[:8],
            })
        return sorted(
            profiles,
            key=lambda item: (
                item["incoming_jump_ge_12_ratio"],
                item["outgoing_jump_ge_12_ratio"],
                item["bar_count"],
            ),
            reverse=True,
        )

    def _density_from_bar(self, bar: Dict[str, Any], tokens: Sequence[int]) -> Dict[str, float]:
        density = bar.get("codebook_density", {})
        if isinstance(density, dict) and density:
            return {
                "note_on_ratio": float(density.get("note_on_ratio", 0.0) or 0.0),
                "rest_ratio": float(density.get("rest_ratio", 0.0) or 0.0),
                "sustain_ratio": float(density.get("sustain_ratio", 0.0) or 0.0),
            }
        total = len(tokens)
        if total == 0:
            return {"note_on_ratio": 0.0, "rest_ratio": 0.0, "sustain_ratio": 0.0}
        return {
            "note_on_ratio": sum(1 for token in tokens if int(token) >= 0) / total,
            "rest_ratio": sum(1 for token in tokens if int(token) == -1) / total,
            "sustain_ratio": sum(1 for token in tokens if int(token) == -2) / total,
        }

    def _leading_run(self, values: Sequence[int], target: int) -> int:
        count = 0
        for value in values:
            if int(value) != int(target):
                break
            count += 1
        return count

    def _trailing_run(self, values: Sequence[int], target: int) -> int:
        count = 0
        for value in reversed(values):
            if int(value) != int(target):
                break
            count += 1
        return count

    def _max_run(self, values: Sequence[int], target: int) -> int:
        best = 0
        current = 0
        for value in values:
            if int(value) == int(target):
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    def _largest_discontinuities(self, valid_boundaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(valid_boundaries, key=lambda row: int(row["abs_jump"]), reverse=True)[:15]

    def _diagnosis(self, boundary_rows: Sequence[Dict[str, Any]], abs_jumps: Sequence[int], selector_probs: Sequence[float]) -> List[str]:
        notes = []
        failure_ratio = self._ratio(sum(value >= self.jump_failure for value in abs_jumps), len(abs_jumps))
        severe_ratio = self._ratio(sum(value >= self.jump_severe for value in abs_jumps), len(abs_jumps))
        source_ratio = self._ratio(sum(row["source_switch"] for row in boundary_rows), len(boundary_rows))
        if failure_ratio >= 0.25:
            notes.append("High octave-scale boundary discontinuity: many boundaries jump by 12+ semitones.")
        if severe_ratio >= 0.15:
            notes.append("Severe register jumps are frequent enough to be audible as hard stitching.")
        if source_ratio >= 0.75:
            notes.append("Source-song switching is high, suggesting bars are stitched from unrelated source contexts.")
        if selector_probs and self._ratio(sum(value < 0.05 for value in selector_probs), len(selector_probs)) >= 0.4:
            notes.append("Candidate selector probabilities are diffuse; selected bars are often not strongly preferred.")
        if not notes:
            notes.append("No single metric crosses the default warning threshold; inspect top discontinuities for local issues.")
        return notes

    def _mean(self, values: Sequence[float]) -> float:
        return float(mean(values)) if values else 0.0

    def _median(self, values: Sequence[float]) -> float:
        return float(median(values)) if values else 0.0

    def _percentile(self, values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
        return float(ordered[index])

    def _ratio(self, count: int, total: int) -> float:
        return float(count) / float(total) if total else 0.0


class MidiSymbolicAnalyzer:
    """Analyze the rendered MIDI that listeners actually hear."""

    def __init__(self, bar_length_ql: float = 4.0) -> None:
        self.bar_length_ql = float(bar_length_ql)

    def analyze_file(self, midi_path: Path) -> Dict[str, Any]:
        notes = self._read_notes(midi_path)
        return self.analyze_notes(notes, source=str(midi_path))

    def analyze_notes(self, notes: Sequence[Dict[str, float]], source: str = "") -> Dict[str, Any]:
        if not notes:
            return {
                "source": source,
                "note_count": 0,
                "diagnosis": ["No MIDI notes were found."],
            }
        sorted_notes = sorted(notes, key=lambda item: (item["offset"], item["pitch"]))
        pitches = [note["pitch"] for note in sorted_notes]
        durations = [note["duration"] for note in sorted_notes]
        offsets = [note["offset"] for note in sorted_notes]
        total_duration = max(note["offset"] + note["duration"] for note in sorted_notes)
        onset_counts = Counter(round(note["offset"], 6) for note in sorted_notes)
        intervals = [
            abs(int(sorted_notes[index]["pitch"]) - int(sorted_notes[index - 1]["pitch"]))
            for index in range(1, len(sorted_notes))
        ]
        bar_metrics = self._bar_metrics(sorted_notes, total_duration)
        return {
            "source": source,
            "note_count": len(sorted_notes),
            "total_duration_ql": total_duration,
            "pitch": {
                "min": min(pitches),
                "max": max(pitches),
                "range": max(pitches) - min(pitches),
                "mean": self._mean(pitches),
                "std": self._std(pitches),
                "unique_count": len(set(pitches)),
                "pitch_class_entropy": self._entropy([pitch % 12 for pitch in pitches]),
            },
            "duration": {
                "mean": self._mean(durations),
                "median": self._median(durations),
                "std": self._std(durations),
                "entropy": self._entropy([round(duration, 3) for duration in durations]),
                "short_note_ratio_le_0_25ql": self._ratio(sum(duration <= 0.25 for duration in durations), len(durations)),
                "long_note_ratio_ge_4ql": self._ratio(sum(duration >= 4.0 for duration in durations), len(durations)),
            },
            "rhythm_density": {
                "onset_count": len(onset_counts),
                "mean_notes_per_onset": self._mean(list(onset_counts.values())),
                "note_density_per_ql": len(sorted_notes) / total_duration if total_duration else 0.0,
                "onset_density_per_ql": len(onset_counts) / total_duration if total_duration else 0.0,
                "silence_ratio": self._silence_ratio(sorted_notes, total_duration),
                "silence_gaps": self._silence_gaps(sorted_notes, total_duration),
            },
            "polyphony": self._polyphony(sorted_notes, total_duration),
            "melodic_intervals": {
                "mean_abs_interval": self._mean(intervals),
                "median_abs_interval": self._median(intervals),
                "p90_abs_interval": self._percentile(intervals, 0.90),
                "max_abs_interval": max(intervals) if intervals else 0,
                "interval_ge_7_ratio": self._ratio(sum(value >= 7 for value in intervals), len(intervals)),
                "interval_ge_12_ratio": self._ratio(sum(value >= 12 for value in intervals), len(intervals)),
            },
            "bar_level": bar_metrics,
            "histograms": {
                "pitch_class": self._normalized_counter([str(pitch % 12) for pitch in pitches]),
                "duration_bucket": self._duration_histogram(durations),
                "interval_bucket": self._interval_histogram(intervals),
            },
            "diagnosis": self._diagnosis(sorted_notes, total_duration, bar_metrics, intervals),
        }

    def _read_notes(self, midi_path: Path) -> List[Dict[str, float]]:
        import mido

        midi = mido.MidiFile(str(midi_path))
        ticks_per_beat = float(midi.ticks_per_beat or 480)
        notes: List[Dict[str, float]] = []
        for track_index, track in enumerate(midi.tracks):
            absolute_tick = 0
            active: Dict[tuple[int, int], List[int]] = defaultdict(list)
            for message in track:
                absolute_tick += int(message.time)
                if message.type == "note_on" and message.velocity > 0:
                    channel = int(getattr(message, "channel", 0))
                    active[(channel, int(message.note))].append(absolute_tick)
                elif message.type in {"note_off", "note_on"}:
                    channel = int(getattr(message, "channel", 0))
                    key = (channel, int(message.note))
                    if active.get(key):
                        start_tick = active[key].pop(0)
                        duration_tick = max(0, absolute_tick - start_tick)
                        if duration_tick == 0:
                            continue
                        notes.append({
                            "pitch": float(message.note),
                            "offset": float(start_tick) / ticks_per_beat,
                            "duration": float(duration_tick) / ticks_per_beat,
                            "track": float(track_index),
                        })
            for (_channel, pitch), starts in active.items():
                for start_tick in starts:
                    duration_tick = max(0, absolute_tick - start_tick)
                    if duration_tick == 0:
                        continue
                    notes.append({
                        "pitch": float(pitch),
                        "offset": float(start_tick) / ticks_per_beat,
                        "duration": float(duration_tick) / ticks_per_beat,
                        "track": float(track_index),
                    })
        return notes

    def _bar_metrics(self, notes: Sequence[Dict[str, float]], total_duration: float) -> Dict[str, Any]:
        bar_count = max(1, int(math.ceil(total_duration / self.bar_length_ql)))
        per_bar_counts = []
        boundary_jumps = []
        for bar_index in range(bar_count):
            start = bar_index * self.bar_length_ql
            end = start + self.bar_length_ql
            bar_notes = [note for note in notes if start <= note["offset"] < end]
            per_bar_counts.append(len(bar_notes))
            if bar_index > 0:
                prev_start = (bar_index - 1) * self.bar_length_ql
                prev_notes = [
                    note for note in notes
                    if prev_start <= note["offset"] < start
                ]
                if prev_notes and bar_notes:
                    prev_last = max(prev_notes, key=lambda item: (item["offset"], item["pitch"]))
                    current_first = min(bar_notes, key=lambda item: (item["offset"], item["pitch"]))
                    boundary_jumps.append(abs(int(current_first["pitch"]) - int(prev_last["pitch"])))
        return {
            "bar_length_ql": self.bar_length_ql,
            "estimated_bar_count": bar_count,
            "mean_notes_per_bar": self._mean(per_bar_counts),
            "p90_notes_per_bar": self._percentile(per_bar_counts, 0.90),
            "empty_bar_ratio": self._ratio(sum(count == 0 for count in per_bar_counts), len(per_bar_counts)),
            "sparse_bar_ratio_le_2_notes": self._ratio(sum(count <= 2 for count in per_bar_counts), len(per_bar_counts)),
            "boundary_mean_abs_jump": self._mean(boundary_jumps),
            "boundary_p90_abs_jump": self._percentile(boundary_jumps, 0.90),
            "boundary_jump_ge_12_ratio": self._ratio(sum(value >= 12 for value in boundary_jumps), len(boundary_jumps)),
        }

    def _polyphony(self, notes: Sequence[Dict[str, float]], total_duration: float) -> Dict[str, Any]:
        points = sorted({0.0, total_duration} | {
            point
            for note in notes
            for point in (note["offset"], note["offset"] + note["duration"])
        })
        weighted_active = 0.0
        active_spans = 0
        mono_spans = 0
        max_active = 0
        for start, end in zip(points, points[1:]):
            if end <= start:
                continue
            active = sum(note["offset"] <= start < note["offset"] + note["duration"] for note in notes)
            duration = end - start
            weighted_active += active * duration
            active_spans += duration if active > 0 else 0
            mono_spans += duration if active == 1 else 0
            max_active = max(max_active, active)
        return {
            "mean_active_notes": weighted_active / total_duration if total_duration else 0.0,
            "max_active_notes": max_active,
            "monophonic_active_ratio": mono_spans / active_spans if active_spans else 0.0,
        }

    def _silence_ratio(self, notes: Sequence[Dict[str, float]], total_duration: float) -> float:
        if total_duration <= 0:
            return 0.0
        intervals = sorted((note["offset"], note["offset"] + note["duration"]) for note in notes)
        covered = 0.0
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                covered += max(0.0, current_end - current_start)
                current_start, current_end = start, end
        covered += max(0.0, current_end - current_start)
        return max(0.0, 1.0 - covered / total_duration)

    def _silence_gaps(self, notes: Sequence[Dict[str, float]], total_duration: float) -> Dict[str, Any]:
        if total_duration <= 0:
            return {
                "gap_count": 0,
                "mean_gap_ql": 0.0,
                "p90_gap_ql": 0.0,
                "max_gap_ql": 0.0,
                "gap_ge_0_5ql_count": 0,
                "gap_ge_1ql_count": 0,
                "top_gaps": [],
            }
        intervals = sorted((float(note["offset"]), float(note["offset"]) + float(note["duration"])) for note in notes)
        merged = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        gaps = []
        cursor = 0.0
        for start, end in merged:
            if start > cursor:
                gaps.append({"start_ql": cursor, "end_ql": start, "duration_ql": start - cursor})
            cursor = max(cursor, end)
        if cursor < total_duration:
            gaps.append({"start_ql": cursor, "end_ql": total_duration, "duration_ql": total_duration - cursor})
        durations = [float(gap["duration_ql"]) for gap in gaps]
        return {
            "gap_count": len(gaps),
            "mean_gap_ql": self._mean(durations),
            "p90_gap_ql": self._percentile(durations, 0.90),
            "max_gap_ql": max(durations) if durations else 0.0,
            "gap_ge_0_5ql_count": sum(value >= 0.5 for value in durations),
            "gap_ge_1ql_count": sum(value >= 1.0 for value in durations),
            "top_gaps": sorted(gaps, key=lambda item: float(item["duration_ql"]), reverse=True)[:10],
        }

    def _diagnosis(
        self,
        notes: Sequence[Dict[str, float]],
        total_duration: float,
        bar_metrics: Dict[str, Any],
        intervals: Sequence[int],
    ) -> List[str]:
        notes_list = list(notes)
        notes_per_ql = len(notes_list) / total_duration if total_duration else 0.0
        interval_ge_12 = self._ratio(sum(value >= 12 for value in intervals), len(intervals))
        notes_text = []
        if bar_metrics.get("empty_bar_ratio", 0.0) > 0.05:
            notes_text.append("Rendered MIDI contains empty bars.")
        if bar_metrics.get("sparse_bar_ratio_le_2_notes", 0.0) > 0.25:
            notes_text.append("Rendered MIDI has many sparse bars.")
        if interval_ge_12 > 0.20:
            notes_text.append("Rendered melodic event order contains frequent octave-scale jumps.")
        if notes_per_ql < 0.5:
            notes_text.append("Overall note density is low.")
        gaps = self._silence_gaps(notes_list, total_duration)
        if float(gaps.get("max_gap_ql", 0.0)) >= 1.0:
            notes_text.append("Rendered MIDI contains a silence gap of at least one quarter note.")
        if not notes_text:
            notes_text.append("No rendered-MIDI metric crosses the initial warning thresholds.")
        return notes_text

    def _duration_histogram(self, durations: Sequence[float]) -> Dict[str, float]:
        def bucket(value: float) -> str:
            if value <= 0.25:
                return "<=0.25"
            if value <= 0.5:
                return "<=0.5"
            if value <= 1.0:
                return "<=1"
            if value <= 2.0:
                return "<=2"
            if value <= 4.0:
                return "<=4"
            return ">4"
        return self._normalized_counter(bucket(value) for value in durations)

    def _interval_histogram(self, intervals: Sequence[int]) -> Dict[str, float]:
        def bucket(value: int) -> str:
            if value <= 2:
                return "0-2"
            if value <= 5:
                return "3-5"
            if value <= 11:
                return "6-11"
            if value <= 18:
                return "12-18"
            return "19+"
        return self._normalized_counter(bucket(value) for value in intervals)

    def _normalized_counter(self, values: Iterable[Any]) -> Dict[str, float]:
        counts = Counter(str(value) for value in values)
        total = sum(counts.values())
        return {key: count / total for key, count in sorted(counts.items())} if total else {}

    def _entropy(self, values: Iterable[Any]) -> float:
        counts = Counter(values)
        total = sum(counts.values())
        if not total:
            return 0.0
        return -sum((count / total) * math.log(count / total) for count in counts.values())

    def _mean(self, values: Sequence[float]) -> float:
        return float(mean(values)) if values else 0.0

    def _median(self, values: Sequence[float]) -> float:
        return float(median(values)) if values else 0.0

    def _std(self, values: Sequence[float]) -> float:
        return float(pstdev(values)) if len(values) > 1 else 0.0

    def _percentile(self, values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
        return float(ordered[index])

    def _ratio(self, count: int, total: int) -> float:
        return float(count) / float(total) if total else 0.0


class ReferenceCorpusAnalyzer:
    """Compare generated MIDI statistics with a reference MIDI corpus."""

    def __init__(self, bar_length_ql: float = 4.0, max_files: int = 30) -> None:
        self.midi_analyzer = MidiSymbolicAnalyzer(bar_length_ql=bar_length_ql)
        self.max_files = int(max_files)

    def analyze(self, reference_dir: Path, generated_midi_metrics: Dict[str, Any]) -> Dict[str, Any]:
        midi_files = self._midi_files(reference_dir)
        file_metrics = []
        for midi_file in midi_files[: self.max_files]:
            try:
                file_metrics.append(self.midi_analyzer.analyze_file(midi_file))
            except Exception as exc:  # pragma: no cover - diagnostics should not fail the whole report.
                file_metrics.append({"source": str(midi_file), "error": str(exc)})
        valid = [metrics for metrics in file_metrics if "error" not in metrics and metrics.get("note_count", 0) > 0]
        aggregate = self._aggregate(valid)
        return {
            "reference_dir": str(reference_dir),
            "requested_file_count": len(midi_files),
            "analyzed_file_count": len(valid),
            "failed_file_count": len(file_metrics) - len(valid),
            "failed_files": [metrics for metrics in file_metrics if "error" in metrics],
            "aggregate": aggregate,
            "generated_vs_reference": self._compare(generated_midi_metrics, aggregate),
        }

    def _midi_files(self, reference_dir: Path) -> List[Path]:
        patterns = ["*.mid", "*.midi", "*.MID", "*.MIDI"]
        files: List[Path] = []
        for pattern in patterns:
            files.extend(reference_dir.rglob(pattern))
        return sorted(set(files))

    def _aggregate(self, metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        scalar_paths = {
            "pitch.mean": ("pitch", "mean"),
            "pitch.range": ("pitch", "range"),
            "duration.mean": ("duration", "mean"),
            "rhythm_density.note_density_per_ql": ("rhythm_density", "note_density_per_ql"),
            "rhythm_density.silence_ratio": ("rhythm_density", "silence_ratio"),
            "polyphony.mean_active_notes": ("polyphony", "mean_active_notes"),
            "polyphony.max_active_notes": ("polyphony", "max_active_notes"),
            "melodic_intervals.interval_ge_12_ratio": ("melodic_intervals", "interval_ge_12_ratio"),
            "bar_level.sparse_bar_ratio_le_2_notes": ("bar_level", "sparse_bar_ratio_le_2_notes"),
            "bar_level.boundary_jump_ge_12_ratio": ("bar_level", "boundary_jump_ge_12_ratio"),
        }
        aggregate: Dict[str, Any] = {}
        for name, path in scalar_paths.items():
            values = [self._nested(metrics_item, path) for metrics_item in metrics]
            numeric = [float(value) for value in values if value is not None]
            aggregate[name] = {
                "mean": float(mean(numeric)) if numeric else 0.0,
                "std": float(pstdev(numeric)) if len(numeric) > 1 else 0.0,
            }
        aggregate["pitch_class_histogram"] = self._mean_histogram(metrics, ("histograms", "pitch_class"))
        aggregate["duration_bucket_histogram"] = self._mean_histogram(metrics, ("histograms", "duration_bucket"))
        aggregate["interval_bucket_histogram"] = self._mean_histogram(metrics, ("histograms", "interval_bucket"))
        return aggregate

    def _compare(self, generated: Dict[str, Any], aggregate: Dict[str, Any]) -> Dict[str, Any]:
        comparisons = {}
        scalar_paths = {
            "pitch.mean": ("pitch", "mean"),
            "pitch.range": ("pitch", "range"),
            "duration.mean": ("duration", "mean"),
            "rhythm_density.note_density_per_ql": ("rhythm_density", "note_density_per_ql"),
            "rhythm_density.silence_ratio": ("rhythm_density", "silence_ratio"),
            "polyphony.mean_active_notes": ("polyphony", "mean_active_notes"),
            "melodic_intervals.interval_ge_12_ratio": ("melodic_intervals", "interval_ge_12_ratio"),
            "bar_level.sparse_bar_ratio_le_2_notes": ("bar_level", "sparse_bar_ratio_le_2_notes"),
            "bar_level.boundary_jump_ge_12_ratio": ("bar_level", "boundary_jump_ge_12_ratio"),
        }
        for name, path in scalar_paths.items():
            generated_value = self._nested(generated, path)
            reference = aggregate.get(name, {})
            reference_mean = reference.get("mean", 0.0)
            reference_std = reference.get("std", 0.0)
            delta = float(generated_value or 0.0) - float(reference_mean or 0.0)
            comparisons[name] = {
                "generated": float(generated_value or 0.0),
                "reference_mean": float(reference_mean or 0.0),
                "reference_std": float(reference_std or 0.0),
                "delta": delta,
                "z_like_delta": delta / reference_std if reference_std else 0.0,
            }
        comparisons["pitch_class_kl_to_reference"] = self._kl_divergence(
            generated.get("histograms", {}).get("pitch_class", {}),
            aggregate.get("pitch_class_histogram", {}),
        )
        comparisons["duration_bucket_kl_to_reference"] = self._kl_divergence(
            generated.get("histograms", {}).get("duration_bucket", {}),
            aggregate.get("duration_bucket_histogram", {}),
        )
        comparisons["interval_bucket_kl_to_reference"] = self._kl_divergence(
            generated.get("histograms", {}).get("interval_bucket", {}),
            aggregate.get("interval_bucket_histogram", {}),
        )
        return comparisons

    def _mean_histogram(self, metrics: Sequence[Dict[str, Any]], path: Sequence[str]) -> Dict[str, float]:
        keys = sorted({
            key
            for metrics_item in metrics
            for key in (self._nested(metrics_item, path) or {}).keys()
        })
        if not keys:
            return {}
        result = {}
        for key in keys:
            result[key] = float(mean([
                float((self._nested(metrics_item, path) or {}).get(key, 0.0))
                for metrics_item in metrics
            ]))
        total = sum(result.values())
        return {key: value / total for key, value in result.items()} if total else result

    def _nested(self, payload: Dict[str, Any], path: Sequence[str]) -> Any:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _kl_divergence(self, left: Dict[str, float], right: Dict[str, float]) -> float:
        keys = sorted(set(left) | set(right))
        if not keys:
            return 0.0
        epsilon = 1e-9
        total_left = sum(float(left.get(key, 0.0)) for key in keys) + epsilon * len(keys)
        total_right = sum(float(right.get(key, 0.0)) for key in keys) + epsilon * len(keys)
        value = 0.0
        for key in keys:
            p = (float(left.get(key, 0.0)) + epsilon) / total_left
            q = (float(right.get(key, 0.0)) + epsilon) / total_right
            value += p * math.log(p / q)
        return value


class GenerationQualityReport:
    """Render Markdown and structured files."""

    def write(
        self,
        paths: ReportPaths,
        summary: Dict[str, Any],
        boundary_rows: Sequence[Dict[str, Any]],
    ) -> None:
        paths.markdown.parent.mkdir(parents=True, exist_ok=True)
        paths.csv.parent.mkdir(parents=True, exist_ok=True)
        paths.summary_json.parent.mkdir(parents=True, exist_ok=True)
        self._write_csv(paths.csv, boundary_rows)
        paths.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        paths.markdown.write_text(self._markdown(paths, summary), encoding="utf-8")

    def _write_csv(self, output_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        if not rows:
            output_path.write_text("", encoding="utf-8")
            return
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _markdown(self, paths: ReportPaths, payload: Dict[str, Any]) -> str:
        lines = [
            "# Generation Quality Report",
            "",
            f"- diagnostics: `{paths.diagnostics}`",
            f"- boundary_csv: `{paths.csv}`",
            f"- summary_json: `{paths.summary_json}`",
            "",
            "## Summary",
            "",
        ]
        lines.extend(self._table(["Metric", "Value"], self._summary_rows(payload["summary"])))
        if payload.get("midi"):
            lines.extend(["", "## Rendered MIDI Quality", ""])
            lines.extend(self._table(["Metric", "Value"], self._midi_rows(payload["midi"])))
        lines.extend(["", "## Boundary Jump Buckets", ""])
        lines.extend(self._table(
            ["Abs jump semitones", "Count", "Ratio"],
            [
                [item["bucket"], item["count"], self._fmt(item["ratio"])]
                for item in payload["jump_buckets"]
            ],
        ))
        if payload.get("pause_metrics"):
            lines.extend(["", "## Pause / Sparse-Bar Diagnostics", ""])
            lines.extend(self._table(["Metric", "Value"], self._pause_rows(payload["pause_metrics"])))
            lines.extend(["", "### Top Sparse Bars", ""])
            lines.extend(self._table(
                [
                    "Bar",
                    "Section",
                    "Local",
                    "Obs",
                    "Note-on ratio",
                    "Rest ratio",
                    "Sustain ratio",
                    "Notes",
                    "Max rest run",
                    "Max sustain run",
                ],
                [
                    [
                        row["bar_index"],
                        row["section"],
                        row["section_local_index"],
                        row["observation_id"],
                        self._fmt(row["note_on_ratio"]),
                        self._fmt(row["rest_ratio"]),
                        self._fmt(row["sustain_ratio"]),
                        row["note_count"],
                        row["max_rest_run"],
                        row["max_sustain_run"],
                    ]
                    for row in payload["pause_metrics"].get("top_sparse_bars", [])[:10]
                ],
            ))
        if payload.get("observation_profiles"):
            lines.extend(["", "## Observation Diagnostics", ""])
            lines.extend(self._table(
                [
                    "Obs",
                    "Bars",
                    "Mean first",
                    "Mean last",
                    "Note-on",
                    "Rest",
                    "Sustain",
                    "In jump mean",
                    "In >=12",
                    "Out jump mean",
                    "Out >=12",
                ],
                [
                    [
                        item["observation_id"],
                        item["bar_count"],
                        self._fmt(item["mean_first_pitch"]),
                        self._fmt(item["mean_last_pitch"]),
                        self._fmt(item["mean_note_on_ratio"]),
                        self._fmt(item["mean_rest_ratio"]),
                        self._fmt(item["mean_sustain_ratio"]),
                        self._fmt(item["incoming_mean_abs_jump"]),
                        self._fmt(item["incoming_jump_ge_12_ratio"]),
                        self._fmt(item["outgoing_mean_abs_jump"]),
                        self._fmt(item["outgoing_jump_ge_12_ratio"]),
                    ]
                    for item in payload["observation_profiles"][:20]
                ],
            ))
        lines.extend(["", "## Section-Level Continuity", ""])
        lines.extend(self._table(
            [
                "Section / Boundary",
                "Boundaries",
                "Mean abs jump",
                "P90 abs jump",
                "Jump >=12 ratio",
                "Source switch ratio",
                "Mean note-on delta",
            ],
            [
                [
                    item["section_or_boundary"],
                    item["boundary_count"],
                    self._fmt(item["mean_abs_jump"]),
                    self._fmt(item["p90_abs_jump"]),
                    self._fmt(item["jump_ge_failure_ratio"]),
                    self._fmt(item["source_song_switch_ratio"]),
                    self._fmt(item["mean_note_on_ratio_delta"]),
                ]
                for item in payload["section_metrics"]
            ],
        ))
        lines.extend(["", "## Largest Discontinuities", ""])
        lines.extend(self._table(
            [
                "Boundary",
                "Section",
                "Abs jump",
                "Signed",
                "Prev last",
                "Current first",
                "Prev obs",
                "Current obs",
                "Source switch",
                "Note-on delta",
            ],
            [
                [
                    row["boundary"],
                    row["section_to"] if row.get("same_section") else f"{row['section_from']} -> {row['section_to']}",
                    row["abs_jump"],
                    row["signed_jump"],
                    row["prev_last_pitch"],
                    row["current_first_pitch"],
                    row["prev_observation_id"],
                    row["current_observation_id"],
                    row["source_switch"],
                    self._fmt(row["note_on_ratio_delta"]),
                ]
                for row in payload["largest_discontinuities"]
            ],
        ))
        if payload.get("reference"):
            lines.extend(["", "## Reference Corpus Comparison", ""])
            lines.extend(self._reference_section(payload["reference"]))
        lines.extend(["", "## Diagnosis", ""])
        lines.extend([f"- {note}" for note in payload["diagnosis"]])
        if payload.get("midi", {}).get("diagnosis"):
            lines.extend([f"- {note}" for note in payload["midi"]["diagnosis"]])
        if payload.get("reference", {}).get("generated_vs_reference"):
            lines.extend(self._reference_diagnosis(payload["reference"]["generated_vs_reference"]))
        lines.append("")
        return "\n".join(lines)

    def _summary_rows(self, summary: Dict[str, Any]) -> List[List[Any]]:
        keys = [
            "bar_count",
            "boundary_count",
            "valid_pitch_boundary_count",
            "mean_abs_jump",
            "median_abs_jump",
            "p90_abs_jump",
            "max_abs_jump",
            "jump_ge_warning_ratio",
            "jump_ge_failure_ratio",
            "jump_ge_severe_ratio",
            "observation_switch_ratio",
            "codebook_switch_ratio",
            "source_song_switch_ratio",
            "mean_note_on_ratio_delta",
            "p90_note_on_ratio_delta",
            "mean_rest_ratio_delta",
            "p90_rest_ratio_delta",
            "mean_sustain_ratio_delta",
            "p90_sustain_ratio_delta",
            "candidate_selector_backend",
            "candidate_selector_used_count",
            "candidate_selected_probability_mean",
            "candidate_selected_probability_lt_0_05_ratio",
            "candidate_count_mean",
        ]
        return [[key, self._fmt(summary.get(key))] for key in keys]

    def _midi_rows(self, midi: Dict[str, Any]) -> List[List[Any]]:
        if not midi or midi.get("note_count", 0) == 0:
            return [["note_count", 0]]
        pairs = [
            ("note_count", midi.get("note_count")),
            ("total_duration_ql", midi.get("total_duration_ql")),
            ("pitch.min", midi.get("pitch", {}).get("min")),
            ("pitch.max", midi.get("pitch", {}).get("max")),
            ("pitch.range", midi.get("pitch", {}).get("range")),
            ("pitch.mean", midi.get("pitch", {}).get("mean")),
            ("pitch.std", midi.get("pitch", {}).get("std")),
            ("duration.mean", midi.get("duration", {}).get("mean")),
            ("duration.entropy", midi.get("duration", {}).get("entropy")),
            ("duration.short_note_ratio_le_0_25ql", midi.get("duration", {}).get("short_note_ratio_le_0_25ql")),
            ("rhythm_density.note_density_per_ql", midi.get("rhythm_density", {}).get("note_density_per_ql")),
            ("rhythm_density.onset_density_per_ql", midi.get("rhythm_density", {}).get("onset_density_per_ql")),
            ("rhythm_density.silence_ratio", midi.get("rhythm_density", {}).get("silence_ratio")),
            ("rhythm_density.silence_gaps.gap_count", midi.get("rhythm_density", {}).get("silence_gaps", {}).get("gap_count")),
            ("rhythm_density.silence_gaps.max_gap_ql", midi.get("rhythm_density", {}).get("silence_gaps", {}).get("max_gap_ql")),
            ("rhythm_density.silence_gaps.gap_ge_0_5ql_count", midi.get("rhythm_density", {}).get("silence_gaps", {}).get("gap_ge_0_5ql_count")),
            ("rhythm_density.silence_gaps.gap_ge_1ql_count", midi.get("rhythm_density", {}).get("silence_gaps", {}).get("gap_ge_1ql_count")),
            ("polyphony.mean_active_notes", midi.get("polyphony", {}).get("mean_active_notes")),
            ("polyphony.max_active_notes", midi.get("polyphony", {}).get("max_active_notes")),
            ("polyphony.monophonic_active_ratio", midi.get("polyphony", {}).get("monophonic_active_ratio")),
            ("melodic_intervals.mean_abs_interval", midi.get("melodic_intervals", {}).get("mean_abs_interval")),
            ("melodic_intervals.p90_abs_interval", midi.get("melodic_intervals", {}).get("p90_abs_interval")),
            ("melodic_intervals.interval_ge_12_ratio", midi.get("melodic_intervals", {}).get("interval_ge_12_ratio")),
            ("bar_level.estimated_bar_count", midi.get("bar_level", {}).get("estimated_bar_count")),
            ("bar_level.mean_notes_per_bar", midi.get("bar_level", {}).get("mean_notes_per_bar")),
            ("bar_level.empty_bar_ratio", midi.get("bar_level", {}).get("empty_bar_ratio")),
            ("bar_level.sparse_bar_ratio_le_2_notes", midi.get("bar_level", {}).get("sparse_bar_ratio_le_2_notes")),
            ("bar_level.boundary_mean_abs_jump", midi.get("bar_level", {}).get("boundary_mean_abs_jump")),
            ("bar_level.boundary_jump_ge_12_ratio", midi.get("bar_level", {}).get("boundary_jump_ge_12_ratio")),
        ]
        return [[key, self._fmt(value)] for key, value in pairs]

    def _pause_rows(self, pause: Dict[str, Any]) -> List[List[Any]]:
        keys = [
            "mean_note_on_ratio",
            "mean_rest_ratio",
            "mean_sustain_ratio",
            "empty_decoded_bar_ratio",
            "sparse_decoded_bar_ratio_le_2_note_on",
            "high_rest_bar_ratio_ge_0_5",
            "high_sustain_bar_ratio_ge_0_75",
            "max_rest_run",
            "max_sustain_run",
        ]
        return [[key, self._fmt(pause.get(key))] for key in keys]

    def _reference_section(self, reference: Dict[str, Any]) -> List[str]:
        lines = [
            f"- reference_dir: `{reference.get('reference_dir')}`",
            f"- analyzed_file_count: `{reference.get('analyzed_file_count')}`",
            f"- failed_file_count: `{reference.get('failed_file_count')}`",
            "",
        ]
        comparisons = reference.get("generated_vs_reference", {})
        rows = []
        for key, value in comparisons.items():
            if not isinstance(value, dict):
                continue
            rows.append([
                key,
                self._fmt(value.get("generated")),
                self._fmt(value.get("reference_mean")),
                self._fmt(value.get("reference_std")),
                self._fmt(value.get("delta")),
                self._fmt(value.get("z_like_delta")),
            ])
        lines.extend(self._table(
            ["Metric", "Generated", "Reference mean", "Reference std", "Delta", "Z-like delta"],
            rows,
        ))
        kl_rows = [
            ["pitch_class_kl_to_reference", self._fmt(comparisons.get("pitch_class_kl_to_reference"))],
            ["duration_bucket_kl_to_reference", self._fmt(comparisons.get("duration_bucket_kl_to_reference"))],
            ["interval_bucket_kl_to_reference", self._fmt(comparisons.get("interval_bucket_kl_to_reference"))],
        ]
        lines.extend(["", "### Distribution Divergence", ""])
        lines.extend(self._table(["Metric", "KL divergence"], kl_rows))
        return lines

    def _reference_diagnosis(self, comparisons: Dict[str, Any]) -> List[str]:
        notes = []
        for key, value in comparisons.items():
            if not isinstance(value, dict):
                continue
            z_like = abs(float(value.get("z_like_delta") or 0.0))
            if z_like >= 2.0:
                notes.append(f"Reference deviation: `{key}` differs from corpus mean by about {z_like:.2f} standard deviations.")
        return notes

    def _table(self, headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return lines

    def _fmt(self, value: Any) -> Any:
        if isinstance(value, float):
            return f"{value:.6f}"
        return value


class GenerationQualityCLI:
    """CLI entrypoint."""

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self._parser().parse_args(argv)
        diagnostics_path = self._resolve_diagnostics(args.input)
        output_dir = args.output_dir or diagnostics_path.parent
        prefix = args.prefix or diagnostics_path.stem.replace(".generation_diagnostics", "")
        paths = ReportPaths(
            diagnostics=diagnostics_path,
            markdown=output_dir / f"{prefix}.quality_report.md",
            csv=output_dir / f"{prefix}.boundary_details.csv",
            summary_json=output_dir / f"{prefix}.quality_summary.json",
        )
        reader = GenerationDiagnosticsReader(diagnostics_path)
        bars = reader.bars()
        boundary_rows = BoundaryFeatureExtractor().rows(bars)
        analyzer = GenerationQualityAnalyzer(
            jump_warning=args.jump_warning,
            jump_failure=args.jump_failure,
            jump_severe=args.jump_severe,
        )
        summary = analyzer.analyze(bars, boundary_rows, reader.harmonic_engine())
        midi_path = args.midi or self._resolve_midi(args.input, diagnostics_path)
        if midi_path is not None:
            midi_metrics = MidiSymbolicAnalyzer(bar_length_ql=args.bar_length_ql).analyze_file(midi_path)
            summary["midi"] = midi_metrics
            if args.reference_dir:
                summary["reference"] = ReferenceCorpusAnalyzer(
                    bar_length_ql=args.bar_length_ql,
                    max_files=args.reference_max_files,
                ).analyze(args.reference_dir, midi_metrics)
        GenerationQualityReport().write(paths, summary, boundary_rows)
        print(f"Report -> {paths.markdown}")
        print(f"Boundary CSV -> {paths.csv}")
        print(f"Summary JSON -> {paths.summary_json}")
        if midi_path is not None:
            print(f"MIDI analyzed -> {midi_path}")

    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "input",
            type=Path,
            help="Generation diagnostics JSON or a directory containing one *.generation_diagnostics.json file.",
        )
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--prefix", type=str, default=None)
        parser.add_argument("--jump-warning", type=int, default=7)
        parser.add_argument("--jump-failure", type=int, default=12)
        parser.add_argument("--jump-severe", type=int, default=19)
        parser.add_argument("--midi", type=Path, default=None)
        parser.add_argument("--reference-dir", type=Path, default=None)
        parser.add_argument("--reference-max-files", type=int, default=30)
        parser.add_argument("--bar-length-ql", type=float, default=4.0)
        return parser

    def _resolve_diagnostics(self, path: Path) -> Path:
        if path.is_file():
            return path
        candidates = sorted(path.glob("*.generation_diagnostics.json"))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one *.generation_diagnostics.json in {path}, found {len(candidates)}."
            )
        return candidates[0]

    def _resolve_midi(self, input_path: Path, diagnostics_path: Path) -> Optional[Path]:
        search_dir = input_path if input_path.is_dir() else diagnostics_path.parent
        stem = diagnostics_path.name.replace(".generation_diagnostics.json", "")
        preferred = search_dir / f"{stem}.mid"
        if preferred.exists():
            return preferred
        midi_files = sorted(list(search_dir.glob("*.mid")) + list(search_dir.glob("*.midi")))
        if len(midi_files) == 1:
            return midi_files[0]
        return None


def main() -> None:
    GenerationQualityCLI().run()


if __name__ == "__main__":
    main()
