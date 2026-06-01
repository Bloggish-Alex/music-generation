#!/usr/bin/env python3
"""Evaluate section, matrix, and hybrid timeline generation traces."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EPS = 1.0e-9


def build_runs(labels: Sequence[int]) -> List[Tuple[int, int]]:
    if not labels:
        return []
    runs: List[Tuple[int, int]] = []
    current = int(labels[0])
    count = 1
    for label in labels[1:]:
        label = int(label)
        if label == current:
            count += 1
        else:
            runs.append((current, count))
            current = label
            count = 1
    runs.append((current, count))
    return runs


def _entropy(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def _duration_probability(model: Any, cluster: int, duration: int) -> float:
    values = list(model.persistence_duration.run_lengths.get(int(cluster), []))
    if not values:
        return EPS
    counts = Counter(int(v) for v in values)
    return float((counts.get(int(duration), 0) + EPS) / (len(values) + EPS * (len(counts) + 1)))


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    return float(mean(values)) if values else None


def evaluate_labels(
    labels: Sequence[int],
    event_log: Sequence[Mapping[str, Any]],
    model: Any,
) -> Dict[str, Any]:
    labels = [int(x) for x in labels]
    runs = build_runs(labels)

    start_prob = None
    if labels:
        probs = getattr(model.start_distribution, "start_probs", [])
        if 0 <= labels[0] < len(probs):
            start_prob = float(max(float(probs[labels[0]]), EPS))

    transition_probs: List[float] = []
    zero_transition_count = 0
    for (src, _src_len), (dst, _dst_len) in zip(runs, runs[1:]):
        matrix = model.transition_matrix.prob_matrix
        prob = float(matrix[int(src), int(dst)]) if src < matrix.shape[0] and dst < matrix.shape[1] else 0.0
        if prob <= 0.0:
            zero_transition_count += 1
        transition_probs.append(max(prob, EPS))

    duration_probs = [
        _duration_probability(model, cluster, duration)
        for cluster, duration in runs
    ]

    run_lengths = [duration for _cluster, duration in runs]
    clusters = [cluster for cluster, _duration in runs]
    training_run_means = [
        float(model.persistence_duration.stats[cluster][0])
        for cluster in clusters
        if 0 <= cluster < len(model.persistence_duration.stats)
    ]
    generated_run_means = [float(duration) for _cluster, duration in runs]

    return {
        "length": len(labels),
        "unique_clusters": len(set(labels)),
        "label_entropy": _entropy(labels),
        "run_count": len(runs),
        "run_mean": _safe_mean([float(x) for x in run_lengths]),
        "run_std": float(np.std(run_lengths)) if run_lengths else None,
        "run_max": max(run_lengths) if run_lengths else None,
        "start_probability": start_prob,
        "start_nll": -math.log(start_prob) if start_prob else None,
        "transition_count": len(transition_probs),
        "transition_zero_rate": (
            float(zero_transition_count / len(transition_probs))
            if transition_probs else None
        ),
        "transition_mean_probability": _safe_mean(transition_probs),
        "transition_nll": (
            float(-sum(math.log(p) for p in transition_probs) / len(transition_probs))
            if transition_probs else None
        ),
        "duration_nll": (
            float(-sum(math.log(p) for p in duration_probs) / len(duration_probs))
            if duration_probs else None
        ),
        "run_mean_abs_delta_vs_training": (
            float(np.mean([
                abs(gen - train)
                for gen, train in zip(generated_run_means, training_run_means)
            ]))
            if generated_run_means and training_run_means else None
        ),
        "events": evaluate_events(event_log, total_length=len(labels)),
    }


def evaluate_events(
    event_log: Sequence[Mapping[str, Any]],
    total_length: int,
) -> Dict[str, Any]:
    kind_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    content_source_counts: Counter[str] = Counter()
    kind_lengths: Counter[str] = Counter()
    section_lengths: List[int] = []
    by_label: Dict[str, List[List[int]]] = defaultdict(list)

    for event in event_log:
        kind = str(event.get("kind", "UNKNOWN"))
        role = str(event.get("role", "n/a"))
        length = int(event.get("length", 0) or 0)
        kind_counts[kind] += 1
        role_counts[role] += 1
        kind_lengths[kind] += length
        if event.get("content_source") is not None:
            content_source_counts[str(event.get("content_source"))] += 1
        if kind == "SECTION":
            section_lengths.append(length)
            label = str(event.get("label", ""))
            labels = [int(x) for x in event.get("labels", [])]
            by_label[label].append(labels)

    recurrence_scores: List[float] = []
    for occurrences in by_label.values():
        if len(occurrences) < 2 or not occurrences[0]:
            continue
        source = occurrences[0]
        for candidate in occurrences[1:]:
            n = min(len(source), len(candidate))
            if n <= 0:
                continue
            recurrence_scores.append(
                sum(1 for a, b in zip(source[:n], candidate[:n]) if a == b) / float(n)
            )

    return {
        "kind_counts": dict(kind_counts),
        "role_counts": dict(role_counts),
        "content_source_counts": dict(content_source_counts),
        "kind_length_ratios": {
            kind: float(length / max(1, total_length))
            for kind, length in kind_lengths.items()
        },
        "section_count": kind_counts.get("SECTION", 0),
        "timeline_run_count": kind_counts.get("TIMELINE_RUN", 0),
        "section_mean_length": _safe_mean([float(x) for x in section_lengths]),
        "section_recurrence_exact_ratio": _safe_mean(recurrence_scores),
    }


def evaluate_trace(trace: Mapping[str, Any], model: Any) -> Dict[str, Any]:
    timeline = trace.get("timeline") if isinstance(trace.get("timeline"), Mapping) else {}
    labels = timeline.get("labels") if isinstance(timeline.get("labels"), Sequence) else []
    events = timeline.get("events") if isinstance(timeline.get("events"), Sequence) else []
    requested = trace.get("requested") if isinstance(trace.get("requested"), Mapping) else {}
    metrics = evaluate_labels(labels, events, model)
    metrics["timeline_mode"] = requested.get("timeline_mode", "section")
    metrics["seed"] = requested.get("seed")
    metrics["target_measures"] = requested.get("target_measures")
    metrics["note_metrics"] = evaluate_notes(trace)
    return metrics


def evaluate_notes(trace: Mapping[str, Any]) -> Dict[str, Any]:
    notes_root = trace.get("notes") if isinstance(trace.get("notes"), Mapping) else {}
    final = notes_root.get("final") if isinstance(notes_root.get("final"), Sequence) else []
    counts: List[int] = []
    melody_counts: List[int] = []
    intervals: List[int] = []
    pitch_values: List[int] = []
    empty_bars = 0
    for bar in final:
        if not isinstance(bar, Mapping):
            continue
        count = int(bar.get("count", 0) or 0)
        melody_count = int(bar.get("melody_count", 0) or 0)
        counts.append(count)
        melody_counts.append(melody_count)
        if count <= 0:
            empty_bars += 1
        pitches = [int(x) for x in bar.get("pitches", [])]
        pitch_values.extend(pitches)
        intervals.extend(abs(b - a) for a, b in zip(pitches, pitches[1:]))

    return {
        "bars": len(final),
        "empty_bar_ratio": float(empty_bars / len(final)) if final else None,
        "note_count_mean": _safe_mean([float(x) for x in counts]),
        "melody_count_mean": _safe_mean([float(x) for x in melody_counts]),
        "pitch_min": min(pitch_values) if pitch_values else None,
        "pitch_max": max(pitch_values) if pitch_values else None,
        "mean_abs_melodic_interval": _safe_mean([float(x) for x in intervals]),
    }


def aggregate_metrics(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("timeline_mode", "unknown"))].append(item)

    result: Dict[str, Any] = {}
    scalar_keys = [
        "transition_nll",
        "duration_nll",
        "transition_zero_rate",
        "run_mean",
        "run_mean_abs_delta_vs_training",
        "label_entropy",
    ]
    note_keys = ["empty_bar_ratio", "note_count_mean", "mean_abs_melodic_interval"]
    for mode, rows in grouped.items():
        summary: Dict[str, Any] = {"runs": len(rows)}
        for key in scalar_keys:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            summary[f"{key}_mean"] = _safe_mean(values)
        for key in note_keys:
            values = [
                float(row.get("note_metrics", {}).get(key))
                for row in rows
                if isinstance(row.get("note_metrics"), Mapping)
                and row.get("note_metrics", {}).get(key) is not None
            ]
            summary[f"{key}_mean"] = _safe_mean(values)
        event_values = [
            row.get("events", {}).get("section_recurrence_exact_ratio")
            for row in rows
            if isinstance(row.get("events"), Mapping)
            and row.get("events", {}).get("section_recurrence_exact_ratio") is not None
        ]
        summary["section_recurrence_exact_ratio_mean"] = _safe_mean([float(x) for x in event_values])
        result[mode] = summary
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    aggregate = report.get("aggregate", {})
    runs = report.get("runs", [])
    lines: List[str] = ["# Timeline Evaluation Report", ""]
    lines.append("## Aggregate")
    lines.append("")
    lines.append(
        "| Mode | Runs | Transition NLL | Duration NLL | Zero Transition | Run Mean | Run Delta | Section Recurrence | Empty Bars | Note Count |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    if isinstance(aggregate, Mapping):
        for mode, summary in aggregate.items():
            if not isinstance(summary, Mapping):
                continue
            lines.append(
                "| {mode} | {runs} | {tnll} | {dnll} | {zero} | {run_mean} | {delta} | {rec} | {empty} | {notes} |".format(
                    mode=mode,
                    runs=_fmt(summary.get("runs")),
                    tnll=_fmt(summary.get("transition_nll_mean")),
                    dnll=_fmt(summary.get("duration_nll_mean")),
                    zero=_fmt(summary.get("transition_zero_rate_mean")),
                    run_mean=_fmt(summary.get("run_mean_mean")),
                    delta=_fmt(summary.get("run_mean_abs_delta_vs_training_mean")),
                    rec=_fmt(summary.get("section_recurrence_exact_ratio_mean")),
                    empty=_fmt(summary.get("empty_bar_ratio_mean")),
                    notes=_fmt(summary.get("note_count_mean_mean")),
                )
            )
    lines.append("")
    lines.append("## Per Run")
    lines.append("")
    lines.append(
        "| Mode | Seed | Length | Runs | Transition NLL | Duration NLL | Event Kinds | Roles | Note Bars | Empty Bars |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |")
    if isinstance(runs, Sequence):
        for row in runs:
            if not isinstance(row, Mapping):
                continue
            events = row.get("events", {}) if isinstance(row.get("events"), Mapping) else {}
            notes = row.get("note_metrics", {}) if isinstance(row.get("note_metrics"), Mapping) else {}
            lines.append(
                "| {mode} | {seed} | {length} | {runs} | {tnll} | {dnll} | {kinds} | {roles} | {note_bars} | {empty} |".format(
                    mode=row.get("timeline_mode", "unknown"),
                    seed=_fmt(row.get("seed")),
                    length=_fmt(row.get("length")),
                    runs=_fmt(row.get("run_count")),
                    tnll=_fmt(row.get("transition_nll")),
                    dnll=_fmt(row.get("duration_nll")),
                    kinds=_dict_text(events.get("kind_counts", {})),
                    roles=_dict_text(events.get("role_counts", {})),
                    note_bars=_fmt(notes.get("bars")),
                    empty=_fmt(notes.get("empty_bar_ratio")),
                )
            )
    lines.append("")
    lines.append("## Reading The Metrics")
    lines.append("")
    lines.append("- Lower `Transition NLL` means the run-to-run cluster transitions are closer to the learned transition matrix.")
    lines.append("- Lower `Duration NLL` means generated run lengths are closer to the learned persistence distributions.")
    lines.append("- `Section Recurrence` is exact cluster-label similarity between later section occurrences and the first occurrence of the same label; matrix-only has no section recurrence by design.")
    lines.append("- `Empty Bars` and `Note Count` are note-generation sanity checks, not musical quality scores.")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _dict_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "n/a"
    return ", ".join(f"{k}:{v}" for k, v in value.items()) or "n/a"


def load_trace(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

