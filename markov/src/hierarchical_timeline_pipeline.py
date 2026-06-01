#!/usr/bin/env python3
"""Timeline generation strategies for HierarchicalGenerator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from narrative_planner import NarrativePlanner


@dataclass
class TimelineGenerationContext:
    """Mutable state passed through one timeline generation strategy."""

    generator: Any
    target_measures: int
    start_states: Optional[List[int]]
    template_file: Optional[Union[int, str]]
    variation_strength: float
    seed: Optional[int]
    rng: np.random.RandomState
    labels: List[int] = field(default_factory=list)
    event_log: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.target_measures - len(self.labels)


class TimelineGenerationStrategy:
    """Base class for timeline generation modes."""

    mode = ""
    requires_grammar = False

    def generate(self, ctx: TimelineGenerationContext) -> Tuple[List[int], List[Dict[str, Any]]]:
        raise NotImplementedError


class MatrixTimelineStrategy(TimelineGenerationStrategy):
    """Generate a complete timeline from PhraseGenerator only."""

    mode = "matrix"

    def generate(self, ctx: TimelineGenerationContext) -> Tuple[List[int], List[Dict[str, Any]]]:
        labels, events = self._generate_runs(
            ctx.generator,
            ctx.remaining,
            ctx.rng,
            start_offset=len(ctx.labels),
        )
        ctx.labels.extend(labels)
        ctx.event_log.extend(events)
        return finalize_timeline(ctx.labels, ctx.event_log, ctx.target_measures)

    @staticmethod
    def _generate_runs(
        generator: Any,
        target_measures: int,
        rng: np.random.RandomState,
        start_offset: int = 0,
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        labels = generator.phrase_gen.generate(
            target_measures,
            seed=int(rng.randint(0, 2 ** 31 - 1)),
        )
        return MatrixTimelineStrategy._generate_runs_from_labels(labels, start_offset=start_offset)

    @staticmethod
    def _generate_runs_from_labels(
        labels: List[int],
        start_offset: int = 0,
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        return labels, MatrixTimelineStrategy._events_from_runs(
            labels,
            label_prefix="T",
            start_offset=start_offset,
        )

    @staticmethod
    def _events_from_runs(
        labels: List[int],
        *,
        label_prefix: str,
        start_offset: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        event_log: List[Dict[str, Any]] = []
        if not labels:
            return event_log

        run_index = 0
        current = labels[0]
        run_start = 0
        for idx, label in enumerate(labels[1:], start=1):
            if label == current:
                continue
            run_labels = labels[run_start:idx]
            event = {
                "kind": "TIMELINE_RUN",
                "label": f"{label_prefix}{start_offset + run_index:03d}_C{current}",
                "cluster": int(current),
                "role": "NEW",
                "length": len(run_labels),
                "labels": run_labels,
            }
            if extra:
                event.update(extra)
            event_log.append(event)
            run_index += 1
            current = label
            run_start = idx
        run_labels = labels[run_start:]
        event = {
            "kind": "TIMELINE_RUN",
            "label": f"{label_prefix}{start_offset + run_index:03d}_C{current}",
            "cluster": int(current),
            "role": "NEW",
            "length": len(run_labels),
            "labels": run_labels,
        }
        if extra:
            event.update(extra)
        event_log.append(event)
        return event_log


class MatrixMinedTimelineStrategy(TimelineGenerationStrategy):
    """Generate matrix labels, then mine repeated segments as pseudo sections."""

    mode = "matrix_mined"

    def generate(self, ctx: TimelineGenerationContext) -> Tuple[List[int], List[Dict[str, Any]]]:
        labels = ctx.generator.phrase_gen.generate(
            ctx.remaining,
            seed=int(ctx.rng.randint(0, 2 ** 31 - 1)),
        )
        mined_events = self._mine_events(labels, ctx)
        ctx.labels.extend(labels)
        ctx.event_log.extend(mined_events)
        return finalize_timeline(ctx.labels, ctx.event_log, ctx.target_measures)

    def _mine_events(
        self,
        labels: List[int],
        ctx: TimelineGenerationContext,
    ) -> List[Dict[str, Any]]:
        if not labels:
            return []
        cfg = ctx.generator.config.get("matrix_mined_timeline", {})
        if not isinstance(cfg, dict):
            cfg = {}
        min_len = int(cfg.get("min_len", 4))
        max_len = int(cfg.get("max_len", 12))
        min_occurrences = int(cfg.get("min_occurrences", 2))
        max_families = int(cfg.get("max_families", 8))
        min_unique_clusters = int(cfg.get("min_unique_clusters", 2))
        min_len = max(2, min_len)
        max_len = max(min_len, min(max_len, len(labels) // 2 if len(labels) >= 4 else len(labels)))

        intervals = self._select_repeated_intervals(
            labels,
            min_len=min_len,
            max_len=max_len,
            min_occurrences=max(2, min_occurrences),
            max_families=max(1, max_families),
            min_unique_clusters=max(1, min_unique_clusters),
        )
        if not intervals:
            return MatrixTimelineStrategy._generate_runs_from_labels(labels)[1]

        by_start = {start: (family, length, occurrence) for start, length, family, occurrence in intervals}
        events: List[Dict[str, Any]] = []
        pos = 0
        while pos < len(labels):
            item = by_start.get(pos)
            if item is None:
                next_section_start = min(
                    [start for start in by_start if start > pos] or [len(labels)]
                )
                free_labels = labels[pos:next_section_start]
                events.extend(MatrixTimelineStrategy._events_from_runs(
                    free_labels,
                    label_prefix=f"M{pos:03d}R",
                    start_offset=0,
                    extra={"content_source": "matrix", "mined": False},
                ))
                pos = next_section_start
                continue

            family, length, occurrence = item
            section_labels = labels[pos:pos + length]
            events.append({
                "kind": "SECTION",
                "label": family,
                "role": "NEW" if occurrence == 0 else "RETURN",
                "cycle": 0,
                "length": length,
                "labels": section_labels,
                "content_source": "matrix",
                "mined": True,
                "mined_occurrence": occurrence,
            })
            pos += length
        return events

    @staticmethod
    def _select_repeated_intervals(
        labels: List[int],
        *,
        min_len: int,
        max_len: int,
        min_occurrences: int,
        max_families: int,
        min_unique_clusters: int,
    ) -> List[Tuple[int, int, str, int]]:
        candidates: List[Tuple[int, int, Tuple[int, ...], List[int]]] = []
        for length in range(max_len, min_len - 1, -1):
            starts_by_pattern: Dict[Tuple[int, ...], List[int]] = {}
            for start in range(0, len(labels) - length + 1):
                pattern = tuple(labels[start:start + length])
                if len(set(pattern)) < min_unique_clusters:
                    continue
                starts_by_pattern.setdefault(pattern, []).append(start)
            for pattern, starts in starts_by_pattern.items():
                non_overlapping = MatrixMinedTimelineStrategy._non_overlapping_starts(starts, length)
                if len(non_overlapping) >= min_occurrences:
                    score = length * (len(non_overlapping) - 1)
                    candidates.append((score, length, pattern, non_overlapping))

        occupied = [False] * len(labels)
        selected: List[Tuple[int, int, str, int]] = []
        family_index = 0
        for _score, length, _pattern, starts in sorted(candidates, reverse=True):
            usable = [
                start for start in starts
                if not any(occupied[start:start + length])
            ]
            if len(usable) < min_occurrences:
                continue
            family = _family_label(family_index)
            family_index += 1
            for occurrence, start in enumerate(usable):
                for idx in range(start, start + length):
                    occupied[idx] = True
                selected.append((start, length, family, occurrence))
            if family_index >= max_families:
                break
        return sorted(selected)

    @staticmethod
    def _non_overlapping_starts(starts: List[int], length: int) -> List[int]:
        result: List[int] = []
        last_end = -1
        for start in sorted(starts):
            if start >= last_end:
                result.append(start)
                last_end = start + length
        return result


class FlatTimelineStrategy(TimelineGenerationStrategy):
    """Fallback when no SectionGrammar is available."""

    mode = "flat"

    def generate(self, ctx: TimelineGenerationContext) -> Tuple[List[int], List[Dict[str, Any]]]:
        extra = ctx.generator.phrase_gen.generate(
            ctx.remaining,
            seed=int(ctx.rng.randint(0, 2 ** 31 - 1)),
        )
        ctx.labels.extend(extra)
        ctx.event_log.append({
            "kind": "FLAT",
            "length": len(extra),
            "labels": extra,
        })
        return finalize_timeline(ctx.labels, ctx.event_log, ctx.target_measures)


class SectionTimelineStrategy(TimelineGenerationStrategy):
    """Existing SectionGrammar timeline path."""

    mode = "section"
    requires_grammar = True

    def generate(self, ctx: TimelineGenerationContext) -> Tuple[List[int], List[Dict[str, Any]]]:
        generator = ctx.generator
        grammar = generator.grammar
        fs = select_template(grammar, ctx.template_file, ctx.rng)

        label_seen: set[str] = set()
        cycle = 0
        section_labels = fs.label_sequence
        n_gaps = len(section_labels) - 1

        while len(ctx.labels) < ctx.target_measures:
            free_lengths = grammar._sample_free_lengths(n_gaps, ctx.rng)

            for i, sec_label in enumerate(section_labels):
                if len(ctx.labels) >= ctx.target_measures:
                    break

                role, vary = assign_role(
                    sec_label, i, cycle, section_labels,
                    label_seen, ctx.variation_strength,
                )
                content = grammar.generate_section_content(
                    sec_label,
                    fs,
                    vary=vary,
                    variation_strength=ctx.variation_strength,
                    seed=int(ctx.rng.randint(0, 2 ** 31 - 1)),
                )
                ctx.labels.extend(content)
                ctx.event_log.append({
                    "kind": "SECTION",
                    "label": sec_label,
                    "role": role,
                    "cycle": cycle,
                    "length": len(content),
                    "labels": content,
                })

                if i < len(section_labels) - 1 and len(ctx.labels) < ctx.target_measures:
                    self._append_free_block(ctx, free_lengths[i] if i < len(free_lengths) else 4)

                label_seen.add(sec_label)

            cycle += 1

        self._end_with_return(ctx, fs)
        self._rebalance_narrative_timeline(ctx, fs)
        return finalize_timeline(ctx.labels, ctx.event_log, ctx.target_measures)

    def _append_free_block(self, ctx: TimelineGenerationContext, free_len: int) -> None:
        free_labels = ctx.generator.phrase_gen.generate(
            free_len,
            seed=int(ctx.rng.randint(0, 2 ** 31 - 1)),
        )
        ctx.labels.extend(free_labels)
        ctx.event_log.append({
            "kind": "FREE",
            "length": free_len,
            "labels": free_labels,
        })

        remainder = len(ctx.labels) % 4
        if 0 < remainder <= 3:
            pad = 4 - remainder
            pad_labels = ctx.generator.phrase_gen.generate(
                pad,
                seed=int(ctx.rng.randint(0, 2 ** 31 - 1)),
            )
            ctx.labels.extend(pad_labels)
            ctx.event_log.append({
                "kind": "FREE",
                "length": pad,
                "labels": pad_labels,
                "grid_pad": True,
            })

    def _end_with_return(self, ctx: TimelineGenerationContext, fs: Any) -> None:
        if not ctx.event_log or ctx.event_log[-1]["kind"] != "FREE":
            return

        grammar = ctx.generator.grammar
        primary_label = fs.label_sequence[0]

        free_len = 0
        while ctx.event_log and ctx.event_log[-1]["kind"] == "FREE":
            free_len += int(ctx.event_log[-1]["length"])
            ctx.event_log.pop()
        del ctx.labels[-free_len:]

        content: List[int] = []
        while len(content) < free_len:
            content.extend(grammar.generate_section_content(
                primary_label,
                fs,
                vary=True,
                variation_strength=ctx.variation_strength,
                seed=int(ctx.rng.randint(0, 2 ** 31 - 1)),
            ))
        fitted = content[:free_len]
        ctx.labels.extend(fitted)
        ctx.event_log.append({
            "kind": "SECTION",
            "label": primary_label,
            "role": "RETURN",
            "length": free_len,
            "labels": fitted,
        })

    def _rebalance_narrative_timeline(self, ctx: TimelineGenerationContext, fs: Any) -> None:
        config = ctx.generator.config
        narrative_cfg = config.get("narrative", {})
        if not isinstance(narrative_cfg, dict) or not narrative_cfg.get("enabled", True):
            return
        rebalance_cfg = narrative_cfg.get("timeline_rebalance", {})
        if not isinstance(rebalance_cfg, dict) or not rebalance_cfg.get("enabled", True):
            return

        primary = fs.label_sequence[0] if fs.label_sequence else "A"
        secondary_labels = [
            label for label in fs.label_sequence
            if label != primary and label in fs.prototypes
        ]
        if not secondary_labels:
            return
        secondary = secondary_labels[0]

        total_len = sum(int(ev.get("length", 0)) for ev in ctx.event_log)
        if total_len <= 0:
            return
        current_secondary = sum(
            int(ev.get("length", 0))
            for ev in ctx.event_log
            if ev.get("kind") == "SECTION" and ev.get("label") == secondary
        )
        min_ratio = float(rebalance_cfg.get("min_secondary_ratio", 0.16))
        target_secondary = int(round(total_len * min_ratio))
        needed = max(0, target_secondary - current_secondary)
        if needed <= 0:
            ctx.labels = ctx.labels[:ctx.target_measures]
            return

        allowed_regions = set(rebalance_cfg.get("secondary_regions", ["CONTRAST", "DEVELOPMENT", "CLIMAX"]))
        max_replaced = int(round(total_len * float(rebalance_cfg.get("max_replaced_ratio", 0.18))))
        replaced = 0
        starts = event_starts(ctx.event_log)
        candidates: List[Tuple[float, int]] = []
        for idx, ev in enumerate(ctx.event_log):
            if ev.get("kind") != "FREE" or ev.get("grid_pad"):
                continue
            start = starts[idx]
            length = int(ev.get("length", 0))
            if length <= 0:
                continue
            pos = (start + 0.5 * length) / max(1, total_len - 1)
            macro = NarrativePlanner._macro_role(
                pos,
                float(narrative_cfg.get("contrast_position", 0.24)),
                float(narrative_cfg.get("development_position", 0.42)),
                float(narrative_cfg.get("climax_position", 0.72)),
                float(narrative_cfg.get("recap_position", 0.84)),
                float(narrative_cfg.get("coda_position", 0.94)),
            )
            if macro in allowed_regions:
                priority = abs(pos - 0.50) - 0.01 * length + ctx.rng.random() * 0.001
                candidates.append((priority, idx))

        for _, idx in sorted(candidates):
            if needed <= 0 or replaced >= max_replaced:
                break
            ev = ctx.event_log[idx]
            length = int(ev.get("length", 0))
            secondary_seen_before = any(
                prior.get("kind") == "SECTION" and prior.get("label") == secondary
                for prior in ctx.event_log[:idx]
            )
            content = ctx.generator.grammar.generate_section_content(
                secondary,
                fs,
                vary=secondary_seen_before,
                variation_strength=ctx.variation_strength,
                seed=int(ctx.rng.randint(0, 2 ** 31 - 1)),
            )
            if not content:
                continue
            fitted: List[int] = []
            while len(fitted) < length:
                fitted.extend(content)
            fitted = fitted[:length]
            ctx.event_log[idx] = {
                "kind": "SECTION",
                "label": secondary,
                "role": "RETURN" if secondary_seen_before else "NEW",
                "cycle": ev.get("cycle", 0),
                "length": length,
                "labels": fitted,
                "narrative_rebalanced": True,
                "replaced_kind": "FREE",
            }
            needed -= length
            replaced += length

        ctx.event_log = trim_events_to_length(ctx.event_log, ctx.target_measures)
        ctx.labels = labels_from_events(ctx.event_log)[:ctx.target_measures]


class HybridTimelineStrategy(TimelineGenerationStrategy):
    """Use SectionGrammar boundaries with matrix-generated section content."""

    mode = "hybrid"
    requires_grammar = True

    def generate(self, ctx: TimelineGenerationContext) -> Tuple[List[int], List[Dict[str, Any]]]:
        grammar = ctx.generator.grammar
        fs = select_template(grammar, ctx.template_file, ctx.rng)
        label_seen: set[str] = set()
        cycle = 0
        section_labels = fs.label_sequence
        n_gaps = len(section_labels) - 1

        while len(ctx.labels) < ctx.target_measures:
            free_lengths = grammar._sample_free_lengths(n_gaps, ctx.rng)

            for i, sec_label in enumerate(section_labels):
                if len(ctx.labels) >= ctx.target_measures:
                    break

                role, _vary = assign_role(
                    sec_label, i, cycle, section_labels,
                    label_seen, ctx.variation_strength,
                )
                prototype = fs.prototypes.get(sec_label, [])
                section_len = max(1, len(prototype))
                content = ctx.generator.phrase_gen.generate(
                    section_len,
                    seed=int(ctx.rng.randint(0, 2 ** 31 - 1)),
                )
                ctx.labels.extend(content)
                ctx.event_log.append({
                    "kind": "SECTION",
                    "label": sec_label,
                    "role": role,
                    "cycle": cycle,
                    "length": len(content),
                    "labels": content,
                    "content_source": "matrix",
                    "prototype_length": len(prototype),
                })

                if i < len(section_labels) - 1 and len(ctx.labels) < ctx.target_measures:
                    free_len = free_lengths[i] if i < len(free_lengths) else 4
                    free_labels = ctx.generator.phrase_gen.generate(
                        free_len,
                        seed=int(ctx.rng.randint(0, 2 ** 31 - 1)),
                    )
                    ctx.labels.extend(free_labels)
                    ctx.event_log.append({
                        "kind": "FREE",
                        "length": free_len,
                        "labels": free_labels,
                        "content_source": "matrix",
                    })

                label_seen.add(sec_label)
            cycle += 1

        return finalize_timeline(ctx.labels, ctx.event_log, ctx.target_measures)


class TimelineGenerationPipeline:
    """Dispatch timeline generation to a named strategy."""

    def __init__(self, generator: Any) -> None:
        self.generator = generator
        self.strategies: Dict[str, TimelineGenerationStrategy] = {
            SectionTimelineStrategy.mode: SectionTimelineStrategy(),
            MatrixTimelineStrategy.mode: MatrixTimelineStrategy(),
            MatrixMinedTimelineStrategy.mode: MatrixMinedTimelineStrategy(),
            HybridTimelineStrategy.mode: HybridTimelineStrategy(),
            FlatTimelineStrategy.mode: FlatTimelineStrategy(),
        }

    def generate(
        self,
        target_measures: int,
        start_states: Optional[List[int]] = None,
        template_file: Optional[Union[int, str]] = None,
        variation_strength: float = 0.3,
        seed: Optional[int] = None,
        timeline_mode: str = "section",
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        rng = np.random.RandomState(seed)
        ctx = TimelineGenerationContext(
            generator=self.generator,
            target_measures=target_measures,
            start_states=start_states,
            template_file=template_file,
            variation_strength=variation_strength,
            seed=seed,
            rng=rng,
        )
        if start_states:
            ctx.labels.extend(start_states)
            ctx.event_log.append({
                "kind": "USER_START",
                "length": len(start_states),
                "labels": list(start_states),
            })
        if ctx.remaining <= 0:
            return finalize_timeline(ctx.labels, ctx.event_log, target_measures)

        mode = str(timeline_mode or "section").lower()
        public_modes = {
            name for name, strategy in self.strategies.items()
            if strategy.mode != "flat"
        }
        if mode not in public_modes:
            raise ValueError(f"timeline_mode must be one of: {', '.join(sorted(public_modes))}")
        strategy = self.strategies[mode]
        if strategy.requires_grammar:
            grammar = self.generator.grammar
            if grammar is None or not grammar.files:
                strategy = self.strategies["flat"]
        return strategy.generate(ctx)


def finalize_timeline(
    labels: List[int],
    event_log: List[Dict[str, Any]],
    target_measures: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    if len(labels) > target_measures:
        labels = labels[:target_measures]
        event_log = trim_events_to_length(event_log, target_measures)
    return labels, event_log


def trim_events_to_length(
    event_log: List[Dict[str, Any]],
    target_measures: int,
) -> List[Dict[str, Any]]:
    trimmed: List[Dict[str, Any]] = []
    used = 0
    for ev in event_log:
        if used >= target_measures:
            break
        length = int(ev.get("length", 0))
        keep = min(length, target_measures - used)
        if keep <= 0:
            break
        new_ev = dict(ev)
        if keep < length:
            new_ev["length"] = keep
            new_ev["labels"] = list(ev.get("labels", []))[:keep]
            new_ev["truncated"] = True
        trimmed.append(new_ev)
        used += keep
    return trimmed


def event_starts(event_log: List[Dict[str, Any]]) -> List[int]:
    starts: List[int] = []
    pos = 0
    for ev in event_log:
        starts.append(pos)
        pos += int(ev.get("length", 0))
    return starts


def labels_from_events(event_log: List[Dict[str, Any]]) -> List[int]:
    labels: List[int] = []
    for ev in event_log:
        labels.extend(int(x) for x in ev.get("labels", []))
    return labels


def _family_label(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return alphabet[index]
    return f"S{index + 1}"


def select_template(
    grammar: Any,
    template_file: Optional[Union[int, str]],
    rng: np.random.RandomState,
) -> Any:
    """Pick a template file, preferring multi-family and grid-aligned."""
    if template_file is not None:
        if isinstance(template_file, int):
            return grammar.files[template_file % len(grammar.files)]
        match = next(
            (f for f in grammar.files
             if f.filename == template_file
             or f.filename.endswith(template_file)
             or Path(f.filename).stem == template_file),
            None,
        )
        if match is None:
            raise KeyError(f"No file matching '{template_file}'")
        return match

    multi = [f for f in grammar.files if f.n_families >= 2]
    by_grid = {0: [], 1: [], 2: []}
    candidates = multi if multi and rng.random() < 0.7 else grammar.files
    for f in candidates:
        lengths = [len(seq) for seq in f.prototypes.values()]
        aligned = sum(1 for length in lengths if length % 4 == 0)
        if aligned == len(lengths) and lengths:
            by_grid[0].append(f)
        elif aligned >= len(lengths) // 2:
            by_grid[1].append(f)
        else:
            by_grid[2].append(f)
    pool = (by_grid[0] * 7 + by_grid[1] * 2 + by_grid[2]) or grammar.files
    return pool[rng.randint(0, len(pool))]


def assign_role(
    sec_label: str,
    i: int,
    cycle: int,
    section_labels: List[str],
    label_seen: set[str],
    variation_strength: float,
) -> Tuple[str, bool]:
    """Determine the structural role of a section occurrence."""
    if cycle == 0 and i == 0:
        return "NEW", False
    if (i > 0 and sec_label == section_labels[i - 1]) \
            or (i == 0 and sec_label == section_labels[-1]):
        return "REPEAT", False
    if sec_label in label_seen:
        return "RETURN", variation_strength > 0
    return "NEW", False
