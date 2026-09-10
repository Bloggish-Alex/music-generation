"""Exact raw-SMF facts and canonical bar spans for Codec V2.

This module deliberately has no music21 dependency.  Raw MIDI ticks are the
authority for time-signature resolution and note pairing; float QL values are
only created by the parser boundary that consumes these records.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Sequence


QUANTUM_QL = Fraction(1, 4)


class CanonicalTimelineError(ValueError):
    """A frozen canonical-parser failure identified by its stable error code."""


@dataclass(frozen=True)
class RawTimeSignatureFact:
    """One raw SMF time-signature meta event, before conflict resolution."""

    physical_track_index: int
    event_ordinal: int
    absolute_tick: int
    numerator: int
    denominator_exponent: int

    @property
    def denominator(self) -> int:
        return 2 ** self.denominator_exponent


@dataclass(frozen=True)
class RawSourceNote:
    """One successfully paired source note with an immutable raw-file ordinal."""

    physical_track_index: int
    channel: int
    source_note_ordinal: int
    start_tick: int
    end_tick: int
    pitch: int
    velocity: int
    tune_index: int = 0

    def start_ql(self, ppqn: int) -> Fraction:
        return Fraction(self.start_tick, ppqn)

    def end_ql(self, ppqn: int) -> Fraction:
        return Fraction(self.end_tick, ppqn)


@dataclass(frozen=True)
class RawPairingRepair:
    """One narrowly authorized raw-pairing normalization fact."""

    repair_kind: str
    physical_track_index: int
    channel: int
    pitch: int
    on_tick: int | None
    on_event_ordinal: int | None
    off_tick: int
    off_event_ordinal: int
    on_velocity: int | None
    queue_depth_before: int
    same_tick_events: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class ResolvedTimeSignature:
    """One merged meter value and all raw facts that declared it at one tick."""

    absolute_tick: int
    numerator: int
    denominator_exponent: int
    contributors: tuple[RawTimeSignatureFact, ...]
    origin: str = "smf"

    @property
    def denominator(self) -> int:
        return 2 ** self.denominator_exponent


@dataclass(frozen=True)
class CanonicalBarSpan:
    """A gap-free canonical bar in exact quarter-length coordinates."""

    canonical_bar_index: int
    start_ql: Fraction
    end_ql: Fraction
    numerator: int
    denominator: int
    time_signature_tick: int
    is_partial: bool = False
    partial_reason: str | None = None
    nominal_meter: str | None = None
    triggering_ts_tick: int | None = None
    triggering_ts_provenance: tuple[int, int, int, int] | None = None
    canonical_start_tick: int = 0
    canonical_end_tick: int = 0
    ppqn: int = 0

    @property
    def time_signature(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass(frozen=True)
class BarLocalQuantizedNote:
    """One source-note fragment quantized against a single canonical bar."""

    source_note: RawSourceNote
    canonical_bar_index: int
    meter: str
    raw_local_start_tick: int
    raw_local_end_tick: int
    ppqn: int
    raw_local_start_ql: Fraction
    raw_local_end_ql: Fraction
    ordinary_quantized_local_start_ql: Fraction
    ordinary_quantized_local_end_ql: Fraction
    quantized_local_start_ql: Fraction
    quantized_local_end_ql: Fraction
    quantization_repair_kind: str | None
    repair_slot_index: int | None
    projection_overlap_ql: Fraction | None
    projection_endpoint_error_ql: Fraction | None
    continues_from_previous_bar: bool
    continues_into_next_bar: bool

    @property
    def duration_ql(self) -> Fraction:
        return self.quantized_local_end_ql - self.quantized_local_start_ql


def _failure(code: str) -> CanonicalTimelineError:
    return CanonicalTimelineError(code)


def _denominator_exponent(denominator: int) -> int:
    if denominator < 1 or denominator & (denominator - 1):
        raise _failure("time_signature_invalid")
    return denominator.bit_length() - 1


def collect_raw_smf_facts(midi: Any, *, tune_index: int = 0, repairs: list[RawPairingRepair] | None = None) -> tuple[list[RawTimeSignatureFact], list[RawSourceNote]]:
    """Collect exact TS facts and FIFO-paired notes from a PPQN SMF 0/1 file."""
    if int(getattr(midi, "type", -1)) == 2:
        raise _failure("smf_format_2_unsupported")
    if int(getattr(midi, "type", -1)) not in {0, 1}:
        raise _failure("smf_format_unsupported")
    if int(getattr(midi, "ticks_per_beat", 0)) <= 0:
        raise _failure("smpte_division_unsupported")

    signatures: list[RawTimeSignatureFact] = []
    paired: list[tuple[int, int, int, int, int, int, int]] = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        pending: dict[tuple[int, int], deque[tuple[int, int, int]]] = defaultdict(deque)
        note_events: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
        track_events: list[tuple[int, int, Any]] = []
        for event_ordinal, event in enumerate(track):
            absolute_tick += int(event.time)
            track_events.append((absolute_tick, event_ordinal, event))
            if event.type in {"note_on", "note_off"}:
                channel, pitch = int(event.channel), int(event.note)
                is_on = event.type == "note_on" and int(event.velocity) > 0
                note_events[absolute_tick].append(("note_on" if is_on else "note_off", event_ordinal, int(event.velocity)))
        for absolute_tick, event_ordinal, event in track_events:
            if event.type == "time_signature":
                denominator = int(event.denominator)
                exponent = _denominator_exponent(denominator)
                if not 1 <= int(event.numerator) <= 255 or not 0 <= exponent <= 7:
                    raise _failure("time_signature_invalid")
                signatures.append(RawTimeSignatureFact(track_index, event_ordinal, absolute_tick, int(event.numerator), exponent))
                continue
            if event.type not in {"note_on", "note_off"}:
                continue
            channel, pitch = int(event.channel), int(event.note)
            key = (channel, pitch)
            is_on = event.type == "note_on" and int(event.velocity) > 0
            if is_on:
                pending[key].append((absolute_tick, event_ordinal, int(event.velocity)))
                continue
            if not pending[key]:
                if repairs is not None:
                    repairs.append(RawPairingRepair("redundant_orphan_note_off", track_index, channel, pitch, None, None, absolute_tick, event_ordinal, None, 0, tuple(note_events[absolute_tick])))
                    continue
                raise _failure("orphan_note_off")
            start_tick, onset_ordinal, velocity = pending[key].popleft()
            if absolute_tick == start_tick and len(pending[key]) == 0:
                if repairs is not None:
                    repairs.append(RawPairingRepair("same_tick_zero_duration_pair", track_index, channel, pitch, start_tick, onset_ordinal, absolute_tick, event_ordinal, velocity, 1, tuple(note_events[absolute_tick])))
                    continue
                raise _failure("note_nonpositive_duration")
            if absolute_tick < start_tick or absolute_tick == start_tick:
                raise _failure("note_nonpositive_duration")
            paired.append((track_index, channel, start_tick, absolute_tick, pitch, velocity, onset_ordinal))
        if any(pending.values()):
            raise _failure("unterminated_note_on")

    ordered = sorted(paired, key=lambda item: (tune_index, item[2], item[0], item[6]))
    notes = [
        RawSourceNote(track, channel, ordinal, start, end, pitch, velocity, tune_index)
        for ordinal, (track, channel, start, end, pitch, velocity, _) in enumerate(ordered)
    ]
    return signatures, notes


def resolve_time_signature_chain(facts: Iterable[RawTimeSignatureFact], *, initial_time_signature_policy: str = "error") -> tuple[ResolvedTimeSignature, ...]:
    """Merge duplicate raw declarations and reject same-tick meter conflicts."""
    if initial_time_signature_policy not in {"error", "smf_default_4_4"}:
        raise _failure("time_signature_initial_policy_invalid")
    by_tick: dict[int, list[RawTimeSignatureFact]] = defaultdict(list)
    for fact in facts:
        by_tick[int(fact.absolute_tick)].append(fact)
    injected = False
    if 0 not in by_tick:
        if initial_time_signature_policy == "error":
            raise _failure("time_signature_initial_missing")
        by_tick[0].append(RawTimeSignatureFact(-1, -1, 0, 4, 2))
        injected = True
    resolved: list[ResolvedTimeSignature] = []
    for tick in sorted(by_tick):
        contributors = tuple(sorted(by_tick[tick], key=lambda item: (item.physical_track_index, item.event_ordinal)))
        meters = {(item.numerator, item.denominator_exponent) for item in contributors}
        if len(meters) != 1:
            raise _failure("time_signature_conflict")
        numerator, exponent = next(iter(meters))
        resolved.append(ResolvedTimeSignature(tick, numerator, exponent, contributors, "smf_default" if injected and tick == 0 else "smf"))
    return tuple(resolved)


def build_canonical_spans(
    chain: Sequence[ResolvedTimeSignature],
    *, ppqn: int,
    terminal_end_ql: Fraction,
) -> tuple[CanonicalBarSpan, ...]:
    """Emit canonical bars through the bar containing the last raw note end."""
    if ppqn <= 0:
        raise _failure("smpte_division_unsupported")
    if terminal_end_ql <= 0:
        raise _failure("no_note_events")
    if not chain or chain[0].absolute_tick != 0:
        raise _failure("time_signature_initial_missing")

    spans: list[CanonicalBarSpan] = []
    current_start = Fraction(0)
    active = chain[0]
    chain_index = 1
    while current_start < terminal_end_ql or not spans:
        next_change = Fraction(chain[chain_index].absolute_tick, ppqn) if chain_index < len(chain) else None
        length = Fraction(active.numerator * 4, active.denominator)
        end = current_start + length
        if next_change is not None and next_change <= terminal_end_ql and current_start < next_change < end:
            change = chain[chain_index]
            start_tick = current_start * ppqn
            if start_tick.denominator != 1:
                raise _failure("canonical_span_tick_invariant")
            spans.append(CanonicalBarSpan(len(spans), current_start, next_change, active.numerator, active.denominator, active.absolute_tick, True, "time_signature_change", f"{active.numerator}/{active.denominator}", change.absolute_tick, (change.contributors[0].physical_track_index, change.contributors[0].event_ordinal, change.numerator, change.denominator), int(start_tick), change.absolute_tick, ppqn))
            current_start = next_change
            active = chain[chain_index]; chain_index += 1
            continue
        start_tick, end_tick = current_start * ppqn, end * ppqn
        if start_tick.denominator != 1 or end_tick.denominator != 1:
            raise _failure("canonical_span_tick_invariant")
        spans.append(CanonicalBarSpan(len(spans), current_start, end, active.numerator, active.denominator, active.absolute_tick, False, None, f"{active.numerator}/{active.denominator}", None, None, int(start_tick), int(end_tick), ppqn))
        current_start = end
        if next_change is not None and next_change == current_start:
            active = chain[chain_index]
            chain_index += 1
        if current_start >= terminal_end_ql:
            break
    return tuple(spans)


def _nearest_boundary(value: Fraction, boundaries: Sequence[Fraction]) -> Fraction:
    """Choose nearest boundary; exact half ties move away from local zero."""
    distance = min(abs(value - candidate) for candidate in boundaries)
    return max(candidate for candidate in boundaries if abs(value - candidate) == distance)


def _slot_boundaries(length: Fraction) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    starts = tuple(
        Fraction(index) * QUANTUM_QL
        for index in range((length.numerator * 4 + length.denominator - 1) // length.denominator)
        if Fraction(index) * QUANTUM_QL < length
    )
    return starts, (*starts, length)


def _minimum_representable_slot_projection(
    raw_start: Fraction, raw_end: Fraction, starts: Sequence[Fraction], ends: Sequence[Fraction]
) -> tuple[int, Fraction, Fraction, Fraction, Fraction]:
    """Select the frozen best-overlap bar-local slot for a collapsed fragment."""
    candidates = []
    for index, start in enumerate(starts):
        end = ends[index + 1]
        overlap = max(Fraction(0), min(raw_end, end) - max(raw_start, start))
        error = abs(start - raw_start) + abs(end - raw_end)
        candidates.append((overlap, error, index, start, end))
    overlap, error, index, start, end = max(candidates, key=lambda item: (item[0], -item[1], item[2]))
    if overlap <= 0:
        raise _failure("quantization_projection_without_raw_overlap")
    return index, start, end, overlap, error


def fragment_note(note: RawSourceNote, span: CanonicalBarSpan, *, ppqn: int) -> BarLocalQuantizedNote | None:
    """Clip one raw note to one span and quantize it on that span's local grid."""
    source_start, source_end = note.start_ql(ppqn), note.end_ql(ppqn)
    start, end = max(source_start, span.start_ql), min(source_end, span.end_ql)
    if end <= start:
        return None
    length = span.end_ql - span.start_ql
    raw_local_start, raw_local_end = start - span.start_ql, end - span.start_ql
    starts, ends = _slot_boundaries(length)
    ordinary_start = _nearest_boundary(raw_local_start, starts)
    ordinary_end = _nearest_boundary(raw_local_end, ends)
    quantized_start, quantized_end = ordinary_start, ordinary_end
    repair_kind = None; repair_slot_index = None; overlap = None; endpoint_error = None
    if quantized_end <= quantized_start:
        repair_slot_index, quantized_start, quantized_end, overlap, endpoint_error = _minimum_representable_slot_projection(raw_local_start, raw_local_end, starts, ends)
        repair_kind = "minimum_representable_slot_projection"
    return BarLocalQuantizedNote(
        source_note=note, canonical_bar_index=span.canonical_bar_index, meter=span.time_signature,
        raw_local_start_tick=int(raw_local_start * ppqn), raw_local_end_tick=int(raw_local_end * ppqn), ppqn=ppqn,
        raw_local_start_ql=raw_local_start, raw_local_end_ql=raw_local_end,
        ordinary_quantized_local_start_ql=ordinary_start, ordinary_quantized_local_end_ql=ordinary_end,
        quantized_local_start_ql=quantized_start, quantized_local_end_ql=quantized_end,
        quantization_repair_kind=repair_kind, repair_slot_index=repair_slot_index,
        projection_overlap_ql=overlap, projection_endpoint_error_ql=endpoint_error,
        continues_from_previous_bar=source_start < span.start_ql, continues_into_next_bar=source_end > span.end_ql,
    )


def fragment_notes(notes: Iterable[RawSourceNote], spans: Iterable[CanonicalBarSpan], *, ppqn: int) -> tuple[BarLocalQuantizedNote, ...]:
    """Return all independently quantized note/span intersections."""
    return tuple(fragment for note in notes for span in spans if (fragment := fragment_note(note, span, ppqn=ppqn)) is not None)
