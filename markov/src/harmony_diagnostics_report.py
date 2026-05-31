"""Diagnostic reporting for generated harmony JSON files.

The generator writes ``<output>.harmony.json`` as an observation surface for
the learned harmony, theme, and candidate-scoring layers.  This module turns
that raw JSON into a compact report that answers three practical questions:

1. Is the whole generation healthy?
2. Which bars are likely responsible for audible problems?
3. Which parameter families are worth inspecting next?

The thresholds below are not generation rules.  They are diagnostic guardrails
used to rank symptoms and explain likely causes.  They are CLI-tunable so the
report can be made stricter or looser for different corpora and profiles.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Thresholds used only to classify diagnostics in the report."""

    low_chord_tone_ratio: float = 0.45
    low_strong_beat_ratio: float = 0.85
    high_resolution_cost: float = 0.55
    high_unresolved_cost: float = 0.45
    bad_harmony_score: float = -0.75
    very_bad_harmony_score: float = -1.20
    low_candidate_count: int = 8
    low_proposal_count: int = 1
    strong_dual_damping: float = 0.50
    high_exact_copy_cost: float = 0.12
    high_partner_error: float = 0.13


@dataclass(frozen=True)
class BarIssue:
    bar: int
    severity: float
    code: str
    message: str
    suggestion: str


def _as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return statistics.fmean(clean) if clean else math.nan


def _minimum(values: Iterable[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return min(clean) if clean else math.nan


def _maximum(values: Iterable[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return max(clean) if clean else math.nan


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _bar_number(bar: Mapping[str, Any], fallback: int) -> int:
    return _as_int(bar.get("bar"), fallback)


def _diagnostics(bar: Mapping[str, Any]) -> Mapping[str, Any]:
    value = bar.get("diagnostics")
    return value if isinstance(value, Mapping) else {}


def _components(bar: Mapping[str, Any]) -> Mapping[str, Any]:
    value = bar.get("candidate_score_components")
    if isinstance(value, Mapping):
        return value

    # Older diagnostics duplicated score components under candidate payloads.
    for key in ("dual_theme_candidate", "repeat_harmony_candidate"):
        candidate = bar.get(key)
        if isinstance(candidate, Mapping):
            nested = candidate.get("score_components")
            if isinstance(nested, Mapping):
                return nested
    return {}


def _candidate_payload(bar: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = bar.get(key)
    return value if isinstance(value, Mapping) else {}


def _selected_components(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = payload.get("selected_components")
    return value if isinstance(value, Mapping) else {}


def _collect_metric(
    bars: Sequence[Mapping[str, Any]],
    getter,
) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for index, bar in enumerate(bars, start=1):
        value = _as_float(getter(bar))
        if not math.isnan(value):
            values.append((_bar_number(bar, index), value))
    return values


def _metric_stats(values: Sequence[tuple[int, float]]) -> JsonDict:
    raw = [v for _, v in values]
    return {
        "mean": _mean(raw),
        "min": _minimum(raw),
        "max": _maximum(raw),
    }


def _worst_low(values: Sequence[tuple[int, float]], limit: int) -> list[tuple[int, float]]:
    return sorted(values, key=lambda item: item[1])[:limit]


def _worst_high(values: Sequence[tuple[int, float]], limit: int) -> list[tuple[int, float]]:
    return sorted(values, key=lambda item: item[1], reverse=True)[:limit]


def _issue(
    issues: list[BarIssue],
    bar: int,
    severity: float,
    code: str,
    message: str,
    suggestion: str,
) -> None:
    issues.append(
        BarIssue(
            bar=bar,
            severity=severity,
            code=code,
            message=message,
            suggestion=suggestion,
        )
    )


def _analyze_bar_issues(
    bars: Sequence[Mapping[str, Any]],
    thresholds: DiagnosticThresholds,
) -> list[BarIssue]:
    issues: list[BarIssue] = []

    for index, bar in enumerate(bars, start=1):
        bar_no = _bar_number(bar, index)
        diag = _diagnostics(bar)
        comps = _components(bar)
        dual = _candidate_payload(bar, "dual_theme_candidate")
        repeat = _candidate_payload(bar, "repeat_harmony_candidate")

        chord_ratio = _as_float(diag.get("chord_tone_ratio"))
        strong_ratio = _as_float(diag.get("strong_beat_chord_tone_ratio"))
        unresolved = _as_float(diag.get("unresolved_dissonance_cost"))
        resolution = _as_float(diag.get("non_chord_resolution_cost"))
        harmony_score = _as_float(diag.get("score"))

        if not math.isnan(harmony_score) and harmony_score <= thresholds.very_bad_harmony_score:
            _issue(
                issues,
                bar_no,
                abs(harmony_score - thresholds.very_bad_harmony_score) + 2.0,
                "very_bad_harmony",
                f"harmony score is very low ({_fmt(harmony_score)}).",
                "Inspect candidate pool first; if candidate_count is already high, this is a scorer/prior issue rather than a simple weight issue.",
            )
        elif not math.isnan(harmony_score) and harmony_score <= thresholds.bad_harmony_score:
            _issue(
                issues,
                bar_no,
                abs(harmony_score - thresholds.bad_harmony_score) + 1.0,
                "bad_harmony",
                f"harmony score is low ({_fmt(harmony_score)}).",
                "Check chord-tone ratio, resolution cost, and whether harmony_score is being outweighed by development or conditional score.",
            )

        if not math.isnan(chord_ratio) and chord_ratio < thresholds.low_chord_tone_ratio:
            _issue(
                issues,
                bar_no,
                thresholds.low_chord_tone_ratio - chord_ratio,
                "low_chord_tone_ratio",
                f"chord-tone ratio is low ({_fmt(chord_ratio)}).",
                "Prefer improving harmony-aware proposal/candidate generation before raising harmony weight.",
            )

        if not math.isnan(strong_ratio) and strong_ratio < thresholds.low_strong_beat_ratio:
            _issue(
                issues,
                bar_no,
                thresholds.low_strong_beat_ratio - strong_ratio,
                "weak_strong_beat",
                f"strong-beat chord-tone ratio is weak ({_fmt(strong_ratio)}).",
                "This usually points to note-level conditional sampling or candidate scoring, not section structure.",
            )

        if not math.isnan(unresolved) and unresolved > thresholds.high_unresolved_cost:
            _issue(
                issues,
                bar_no,
                unresolved - thresholds.high_unresolved_cost,
                "unresolved_dissonance",
                f"unresolved dissonance cost is high ({_fmt(unresolved)}).",
                "Look for learned resolution priors or candidate scoring features; avoid fixing individual pitches by rule.",
            )

        if not math.isnan(resolution) and resolution > thresholds.high_resolution_cost:
            _issue(
                issues,
                bar_no,
                resolution - thresholds.high_resolution_cost,
                "non_chord_resolution",
                f"non-chord resolution cost is high ({_fmt(resolution)}).",
                "If repeated across many bars, train/score transition-to-resolution behavior rather than adding local repairs.",
            )

        candidate_count = max(
            _as_int(dual.get("candidate_count")),
            _as_int(repeat.get("candidate_count")),
        )
        proposal_count = max(
            _as_int(dual.get("proposal_count")),
            _as_int(repeat.get("proposal_count")),
        )
        if candidate_count and candidate_count <= thresholds.low_candidate_count:
            _issue(
                issues,
                bar_no,
                (thresholds.low_candidate_count - candidate_count) / max(thresholds.low_candidate_count, 1),
                "small_candidate_pool",
                f"candidate pool is small ({candidate_count}).",
                "Increase candidate budget only for weak contexts; global budget increases usually add noise and runtime.",
            )
        if candidate_count and proposal_count <= thresholds.low_proposal_count and bar.get("section_role") in {"REPEAT", "RETURN"}:
            _issue(
                issues,
                bar_no,
                0.4,
                "few_structural_proposals",
                f"repeat/return bar has few structural proposals ({proposal_count}).",
                "Inspect early_repeat_adaptation or dual_theme_proposal coverage for this context.",
            )

        damping = _as_float(comps.get("dual_theme_harmony_damping"))
        raw_dual = _as_float(comps.get("raw_dual_theme_score"))
        dual_score = _as_float(comps.get("dual_theme_score"))
        if (
            not math.isnan(damping)
            and damping < thresholds.strong_dual_damping
            and not math.isnan(raw_dual)
            and raw_dual > 0
        ):
            _issue(
                issues,
                bar_no,
                thresholds.strong_dual_damping - damping,
                "dual_theme_damped",
                f"dual-theme score is strongly damped ({_fmt(raw_dual)} -> {_fmt(dual_score)}).",
                "Good signal: identity wanted this candidate, harmony rejected it. Improve harmony-compatible theme proposals before changing weights.",
            )

        for payload_key, label in (
            ("dual_theme_candidate", "dual theme"),
            ("repeat_harmony_candidate", "repeat harmony"),
        ):
            payload = _candidate_payload(bar, payload_key)
            selected = _selected_components(payload)
            exact_copy = _as_float(selected.get("exact_copy_cost"))
            partner_error = _as_float(selected.get("partner_abs_error"))
            if not math.isnan(exact_copy) and exact_copy > thresholds.high_exact_copy_cost:
                _issue(
                    issues,
                    bar_no,
                    exact_copy - thresholds.high_exact_copy_cost,
                    "excessive_exact_copy",
                    f"{label} exact-copy cost is high ({_fmt(exact_copy)}).",
                    "The repetition may sound pasted; prefer learned variation candidates over stronger copy penalties.",
                )
            if not math.isnan(partner_error) and partner_error > thresholds.high_partner_error:
                _issue(
                    issues,
                    bar_no,
                    partner_error - thresholds.high_partner_error,
                    "weak_partner_match",
                    f"{label} partner error is high ({_fmt(partner_error)}).",
                    "Theme relation exists, but the chosen realization is drifting from the partner contour/anchor.",
                )

    return sorted(issues, key=lambda item: item.severity, reverse=True)


def analyze_harmony_json(
    path: Path,
    thresholds: DiagnosticThresholds | None = None,
    top_n: int = 12,
) -> JsonDict:
    thresholds = thresholds or DiagnosticThresholds()
    data = json.loads(path.read_text(encoding="utf-8"))
    bars_raw = data.get("bars", [])
    if not isinstance(bars_raw, list):
        raise ValueError(f"{path} does not contain a list field named 'bars'.")
    bars: list[Mapping[str, Any]] = [bar for bar in bars_raw if isinstance(bar, Mapping)]

    score_values = _collect_metric(bars, lambda b: _diagnostics(b).get("score"))
    chord_values = _collect_metric(bars, lambda b: _diagnostics(b).get("chord_tone_ratio"))
    strong_values = _collect_metric(bars, lambda b: _diagnostics(b).get("strong_beat_chord_tone_ratio"))
    unresolved_values = _collect_metric(bars, lambda b: _diagnostics(b).get("unresolved_dissonance_cost"))
    resolution_values = _collect_metric(bars, lambda b: _diagnostics(b).get("non_chord_resolution_cost"))
    total_score_values = _collect_metric(bars, lambda b: _components(b).get("total_score"))
    development_values = _collect_metric(bars, lambda b: _components(b).get("development_score"))
    conditional_values = _collect_metric(bars, lambda b: _components(b).get("conditional_score"))
    weighted_harmony_values = _collect_metric(bars, lambda b: _components(b).get("weighted_harmony_score"))
    raw_dual_values = _collect_metric(bars, lambda b: _components(b).get("raw_dual_theme_score"))
    dual_values = _collect_metric(bars, lambda b: _components(b).get("dual_theme_score"))
    reranker_probability_values = _collect_metric(
        bars,
        lambda b: (_components(b).get("reranker_probability")
                   if _components(b).get("reranker_probability") is not None
                   else (b.get("candidate_reranker") or {}).get("probability")
                   if isinstance(b.get("candidate_reranker"), Mapping) else None),
    )
    reranker_weighted_values = _collect_metric(
        bars,
        lambda b: (_components(b).get("reranker_weighted_score")
                   if _components(b).get("reranker_weighted_score") is not None
                   else (b.get("candidate_reranker") or {}).get("weighted_score")
                   if isinstance(b.get("candidate_reranker"), Mapping) else None),
    )
    raw_reranker_probability_values = _collect_metric(
        bars,
        lambda b: (_components(b).get("raw_reranker_probability")
                   if _components(b).get("raw_reranker_probability") is not None
                   else (b.get("candidate_reranker") or {}).get("raw_probability")
                   if isinstance(b.get("candidate_reranker"), Mapping) else None),
    )
    reranker_calibration_adjustment_values = _collect_metric(
        bars,
        lambda b: (_components(b).get("reranker_calibration_adjustment")
                   if _components(b).get("reranker_calibration_adjustment") is not None
                   else (b.get("candidate_reranker") or {}).get("calibration_adjustment")
                   if isinstance(b.get("candidate_reranker"), Mapping) else None),
    )
    rhythm_score_values = _collect_metric(
        bars,
        lambda b: (_components(b).get("rhythm_score")
                   if _components(b).get("rhythm_score") is not None
                   else (b.get("rhythm_candidate") or {}).get("score")
                   if isinstance(b.get("rhythm_candidate"), Mapping) else None),
    )
    rhythm_source_values = _collect_metric(
        bars,
        lambda b: ((b.get("rhythm_candidate") or {}).get("source_similarity")
                   if isinstance(b.get("rhythm_candidate"), Mapping) else None),
    )
    rhythm_copy_values = _collect_metric(
        bars,
        lambda b: ((b.get("rhythm_candidate") or {}).get("exact_repeat_cost")
                   if isinstance(b.get("rhythm_candidate"), Mapping) else None),
    )
    rhythm_contour_values = _collect_metric(
        bars,
        lambda b: ((b.get("rhythm_candidate") or {}).get("phrase_contour_score")
                   if isinstance(b.get("rhythm_candidate"), Mapping) else None),
    )
    rhythm_note_count_values = _collect_metric(
        bars,
        lambda b: ((b.get("rhythm_candidate") or {}).get("note_count")
                   if isinstance(b.get("rhythm_candidate"), Mapping) else None),
    )
    rhythm_prior_probability_values = _collect_metric(
        bars,
        lambda b: ((b.get("rhythm_candidate") or {}).get("learned_prior_probability")
                   if isinstance(b.get("rhythm_candidate"), Mapping) else None),
    )
    rhythm_prior_weighted_values = _collect_metric(
        bars,
        lambda b: ((b.get("rhythm_candidate") or {}).get("learned_prior_weighted")
                   if isinstance(b.get("rhythm_candidate"), Mapping) else None),
    )

    candidate_counts: list[tuple[int, int]] = []
    proposal_counts: list[tuple[int, int]] = []
    selected_from_proposal = 0
    proposal_decisions = 0
    proposal_kind_counts: dict[str, int] = {}
    for index, bar in enumerate(bars, start=1):
        bar_no = _bar_number(bar, index)
        dual = _candidate_payload(bar, "dual_theme_candidate")
        repeat = _candidate_payload(bar, "repeat_harmony_candidate")
        candidate_count = max(_as_int(dual.get("candidate_count")), _as_int(repeat.get("candidate_count")))
        proposal_count = max(_as_int(dual.get("proposal_count")), _as_int(repeat.get("proposal_count")))
        if candidate_count:
            candidate_counts.append((bar_no, candidate_count))
        if proposal_count:
            proposal_counts.append((bar_no, proposal_count))
        for payload in (dual, repeat):
            if not payload:
                continue
            if "selected_from_proposal" in payload:
                proposal_decisions += 1
                if bool(payload.get("selected_from_proposal")):
                    selected_from_proposal += 1
            kind = payload.get("selected_proposal_kind")
            if kind:
                proposal_kind_counts[str(kind)] = proposal_kind_counts.get(str(kind), 0) + 1

    section_role_counts: dict[str, int] = {}
    narrative_role_counts: dict[str, int] = {}
    roman_counts: dict[str, int] = {}
    for bar in bars:
        for source, target in (
            (bar.get("section_role"), section_role_counts),
            (bar.get("narrative_role"), narrative_role_counts),
        ):
            if source:
                target[str(source)] = target.get(str(source), 0) + 1
        roman = _diagnostics(bar).get("roman") or (bar.get("harmony") or {}).get("roman")
        if roman:
            roman_counts[str(roman)] = roman_counts.get(str(roman), 0) + 1

    issues = _analyze_bar_issues(bars, thresholds)
    issue_counts: dict[str, int] = {}
    for item in issues:
        issue_counts[item.code] = issue_counts.get(item.code, 0) + 1

    return {
        "path": str(path),
        "summary": data.get("summary", {}),
        "bar_count": len(bars),
        "metrics": {
            "harmony_score": _metric_stats(score_values),
            "chord_tone_ratio": _metric_stats(chord_values),
            "strong_beat_chord_tone_ratio": _metric_stats(strong_values),
            "unresolved_dissonance_cost": _metric_stats(unresolved_values),
            "non_chord_resolution_cost": _metric_stats(resolution_values),
            "total_candidate_score": _metric_stats(total_score_values),
            "development_score": _metric_stats(development_values),
            "conditional_score": _metric_stats(conditional_values),
            "weighted_harmony_score": _metric_stats(weighted_harmony_values),
            "raw_dual_theme_score": _metric_stats(raw_dual_values),
            "dual_theme_score": _metric_stats(dual_values),
            "reranker_probability": _metric_stats(reranker_probability_values),
            "raw_reranker_probability": _metric_stats(raw_reranker_probability_values),
            "reranker_calibration_adjustment": _metric_stats(reranker_calibration_adjustment_values),
            "reranker_weighted_score": _metric_stats(reranker_weighted_values),
            "rhythm_score": _metric_stats(rhythm_score_values),
            "rhythm_source_similarity": _metric_stats(rhythm_source_values),
            "rhythm_exact_repeat_cost": _metric_stats(rhythm_copy_values),
            "rhythm_phrase_contour_score": _metric_stats(rhythm_contour_values),
            "rhythm_note_count": _metric_stats(rhythm_note_count_values),
            "rhythm_prior_probability": _metric_stats(rhythm_prior_probability_values),
            "rhythm_prior_weighted_score": _metric_stats(rhythm_prior_weighted_values),
        },
        "distributions": {
            "section_role_counts": section_role_counts,
            "narrative_role_counts": narrative_role_counts,
            "roman_counts": roman_counts,
            "proposal_kind_counts": proposal_kind_counts,
        },
        "candidate_pool": {
            "candidate_count_mean": _mean([float(v) for _, v in candidate_counts]),
            "candidate_count_min": min((v for _, v in candidate_counts), default=0),
            "candidate_count_max": max((v for _, v in candidate_counts), default=0),
            "proposal_count_mean": _mean([float(v) for _, v in proposal_counts]),
            "proposal_count_min": min((v for _, v in proposal_counts), default=0),
            "proposal_count_max": max((v for _, v in proposal_counts), default=0),
            "selected_from_proposal_ratio": (
                selected_from_proposal / proposal_decisions if proposal_decisions else math.nan
            ),
        },
        "worst": {
            "lowest_harmony_score": _worst_low(score_values, top_n),
            "lowest_chord_tone_ratio": _worst_low(chord_values, top_n),
            "highest_unresolved_cost": _worst_high(unresolved_values, top_n),
            "highest_resolution_cost": _worst_high(resolution_values, top_n),
            "lowest_total_candidate_score": _worst_low(total_score_values, top_n),
            "lowest_development_score": _worst_low(development_values, top_n),
            "lowest_conditional_score": _worst_low(conditional_values, top_n),
            "lowest_reranker_probability": _worst_low(reranker_probability_values, top_n),
            "lowest_rhythm_score": _worst_low(rhythm_score_values, top_n),
            "highest_rhythm_exact_repeat_cost": _worst_high(rhythm_copy_values, top_n),
            "lowest_rhythm_note_count": _worst_low(rhythm_note_count_values, top_n),
            "lowest_rhythm_prior_probability": _worst_low(rhythm_prior_probability_values, top_n),
        },
        "issues": [
            {
                "bar": item.bar,
                "severity": item.severity,
                "code": item.code,
                "message": item.message,
                "suggestion": item.suggestion,
            }
            for item in issues[:top_n]
        ],
        "issue_counts": issue_counts,
    }


def _health_label(report: Mapping[str, Any], thresholds: DiagnosticThresholds) -> str:
    metrics = report.get("metrics", {})
    harmony = metrics.get("harmony_score", {}) if isinstance(metrics, Mapping) else {}
    chord = metrics.get("chord_tone_ratio", {}) if isinstance(metrics, Mapping) else {}
    strong = metrics.get("strong_beat_chord_tone_ratio", {}) if isinstance(metrics, Mapping) else {}
    score_min = _as_float(harmony.get("min") if isinstance(harmony, Mapping) else None)
    chord_mean = _as_float(chord.get("mean") if isinstance(chord, Mapping) else None)
    strong_mean = _as_float(strong.get("mean") if isinstance(strong, Mapping) else None)
    issue_counts = report.get("issue_counts", {})
    issue_total = sum(issue_counts.values()) if isinstance(issue_counts, Mapping) else 0

    if not math.isnan(score_min) and score_min <= thresholds.very_bad_harmony_score:
        return "risky"
    if not math.isnan(chord_mean) and chord_mean < 0.55:
        return "needs attention"
    if not math.isnan(strong_mean) and strong_mean < 0.95:
        return "needs attention"
    if issue_total >= 12:
        return "needs attention"
    return "healthy enough for listening test"


def _render_dict_counts(counts: Mapping[str, Any], limit: int = 12) -> str:
    if not counts:
        return "n/a"
    items = sorted(counts.items(), key=lambda item: (-_as_int(item[1]), str(item[0])))[:limit]
    return ", ".join(f"{key}:{value}" for key, value in items)


def _render_metric_table(metrics: Mapping[str, Any]) -> list[str]:
    lines = ["| metric | mean | min | max |", "|---|---:|---:|---:|"]
    for key, value in metrics.items():
        if not isinstance(value, Mapping):
            continue
        lines.append(
            f"| {key} | {_fmt(_as_float(value.get('mean')))} | "
            f"{_fmt(_as_float(value.get('min')))} | {_fmt(_as_float(value.get('max')))} |"
        )
    return lines


def _render_worst_table(title: str, values: Sequence[Sequence[Any]]) -> list[str]:
    lines = [f"### {title}", "| bar | value |", "|---:|---:|"]
    if not values:
        lines.append("| n/a | n/a |")
        return lines
    for bar, value in values:
        lines.append(f"| {bar} | {_fmt(_as_float(value))} |")
    return lines


def render_markdown_report(report: Mapping[str, Any], thresholds: DiagnosticThresholds) -> str:
    lines: list[str] = []
    path = report.get("path", "unknown")
    summary = report.get("summary", {})
    metrics = report.get("metrics", {})
    distributions = report.get("distributions", {})
    pool = report.get("candidate_pool", {})
    worst = report.get("worst", {})
    issues = report.get("issues", [])

    lines.append(f"# Harmony Diagnostics Report")
    lines.append("")
    lines.append(f"Source: `{path}`")
    lines.append(f"Health: **{_health_label(report, thresholds)}**")
    lines.append("")

    if isinstance(summary, Mapping):
        lines.append("## Summary")
        for key in (
            "actual_mode",
            "conditional_note_model_active",
            "conditional_note_model_version",
            "candidate_reranker_active",
            "rhythm_development_active",
            "rhythm_candidate_prior_active",
            "dual_theme_candidate_scoring_active",
            "bar_count",
            "dual_theme_active_bars",
            "mean_rhythm_score",
            "mean_rhythm_source_similarity",
            "mean_rhythm_note_count",
            "mean_chord_tone_ratio",
            "mean_strong_beat_chord_tone_ratio",
            "bass_root_or_fifth_ratio",
        ):
            if key in summary:
                lines.append(f"- `{key}`: `{_fmt(summary[key])}`")
        lines.append("")

    if isinstance(metrics, Mapping):
        lines.append("## Score Overview")
        lines.extend(_render_metric_table(metrics))
        lines.append("")

    if isinstance(pool, Mapping):
        lines.append("## Candidate Pool")
        lines.append(f"- candidate count: mean `{_fmt(_as_float(pool.get('candidate_count_mean')))}`, min `{pool.get('candidate_count_min')}`, max `{pool.get('candidate_count_max')}`")
        lines.append(f"- proposal count: mean `{_fmt(_as_float(pool.get('proposal_count_mean')))}`, min `{pool.get('proposal_count_min')}`, max `{pool.get('proposal_count_max')}`")
        lines.append(f"- selected from proposal ratio: `{_fmt(_as_float(pool.get('selected_from_proposal_ratio')))}`")
        lines.append("")

    if isinstance(distributions, Mapping):
        lines.append("## Distributions")
        lines.append(f"- section roles: {_render_dict_counts(distributions.get('section_role_counts', {}))}")
        lines.append(f"- narrative roles: {_render_dict_counts(distributions.get('narrative_role_counts', {}))}")
        lines.append(f"- roman chords: {_render_dict_counts(distributions.get('roman_counts', {}))}")
        lines.append(f"- selected proposal kinds: {_render_dict_counts(distributions.get('proposal_kind_counts', {}))}")
        lines.append("")

    if isinstance(worst, Mapping):
        lines.append("## Worst Bars")
        for key, title in (
            ("lowest_harmony_score", "Lowest Harmony Score"),
            ("lowest_chord_tone_ratio", "Lowest Chord Tone Ratio"),
            ("highest_unresolved_cost", "Highest Unresolved Dissonance Cost"),
            ("highest_resolution_cost", "Highest Non-Chord Resolution Cost"),
            ("lowest_total_candidate_score", "Lowest Total Candidate Score"),
            ("lowest_development_score", "Lowest Development Score"),
            ("lowest_conditional_score", "Lowest Conditional Score"),
            ("lowest_reranker_probability", "Lowest Candidate Reranker Probability"),
            ("lowest_rhythm_score", "Lowest Rhythm Score"),
            ("highest_rhythm_exact_repeat_cost", "Highest Rhythm Exact Repeat Cost"),
            ("lowest_rhythm_note_count", "Lowest Rhythm Note Count"),
            ("lowest_rhythm_prior_probability", "Lowest Rhythm Prior Probability"),
        ):
            lines.extend(_render_worst_table(title, worst.get(key, [])))
            lines.append("")

    lines.append("## Likely Problems")
    if not issues:
        lines.append("No major threshold-based issues were detected. Use listening tests and compare multiple seeds before changing parameters.")
    else:
        lines.append("| bar | issue | severity | observation | next check |")
        lines.append("|---:|---|---:|---|---|")
        for item in issues:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                f"| {item.get('bar')} | `{item.get('code')}` | "
                f"{_fmt(_as_float(item.get('severity')))} | "
                f"{item.get('message')} | {item.get('suggestion')} |"
            )
    lines.append("")

    lines.append("## Adjustment Priority")
    lines.append("1. If candidate counts are low in bad repeat/return bars, tune candidate budget or proposal coverage first.")
    lines.append("2. If candidate counts are high but harmony remains bad, tune candidate scoring or add learned reranker features.")
    lines.append("3. If raw dual-theme score is often damped by harmony, improve harmony-compatible theme variation proposals.")
    lines.append("4. If conditional/development scores dominate bad bars, inspect learned priors instead of increasing harmony weight.")
    lines.append("5. If rhythm_score is low in repeat/return bars, inspect rhythm identity and density before changing narrative structure.")
    lines.append("6. Avoid one-bar special rules; repeated symptoms should become features or training targets.")
    lines.append("")
    return "\n".join(lines)


def _write_or_print(text: str, output: Path | None) -> None:
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically so an interrupted report never leaves a half-written
    # file.  This also avoids a few Windows file-locking edge cases when
    # overwriting an existing report that an editor has just released.
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=str(output.parent),
        prefix=f".{output.name}.",
        suffix=".tmp",
    ) as handle:
        temp_name = handle.name
        handle.write(text)
    os.replace(temp_name, output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a generated <output>.harmony.json file and write a tuning report.",
    )
    parser.add_argument("json_file", type=Path, help="Path to a generated .harmony.json file.")
    parser.add_argument("-o", "--output", type=Path, help="Optional report output path.")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--top", type=int, default=12, help="Number of worst bars/issues to show.")
    parser.add_argument("--low-chord-tone-ratio", type=float, default=DiagnosticThresholds.low_chord_tone_ratio)
    parser.add_argument("--low-strong-beat-ratio", type=float, default=DiagnosticThresholds.low_strong_beat_ratio)
    parser.add_argument("--high-resolution-cost", type=float, default=DiagnosticThresholds.high_resolution_cost)
    parser.add_argument("--high-unresolved-cost", type=float, default=DiagnosticThresholds.high_unresolved_cost)
    parser.add_argument("--bad-harmony-score", type=float, default=DiagnosticThresholds.bad_harmony_score)
    parser.add_argument("--very-bad-harmony-score", type=float, default=DiagnosticThresholds.very_bad_harmony_score)
    parser.add_argument("--low-candidate-count", type=int, default=DiagnosticThresholds.low_candidate_count)
    parser.add_argument("--low-proposal-count", type=int, default=DiagnosticThresholds.low_proposal_count)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    thresholds = DiagnosticThresholds(
        low_chord_tone_ratio=args.low_chord_tone_ratio,
        low_strong_beat_ratio=args.low_strong_beat_ratio,
        high_resolution_cost=args.high_resolution_cost,
        high_unresolved_cost=args.high_unresolved_cost,
        bad_harmony_score=args.bad_harmony_score,
        very_bad_harmony_score=args.very_bad_harmony_score,
        low_candidate_count=args.low_candidate_count,
        low_proposal_count=args.low_proposal_count,
    )
    report = analyze_harmony_json(args.json_file, thresholds=thresholds, top_n=max(args.top, 1))

    if args.format == "json":
        text = json.dumps(report, ensure_ascii=False, indent=2)
    else:
        text = render_markdown_report(report, thresholds)
    _write_or_print(text, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
