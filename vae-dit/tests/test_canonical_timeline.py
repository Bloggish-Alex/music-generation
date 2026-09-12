from fractions import Fraction

import pytest

from data.canonical_timeline import (
    CanonicalTimelineError,
    build_canonical_spans,
    collect_raw_smf_facts,
    fragment_notes,
    resolve_time_signature_chain,
)


def _midi(*tracks):
    import mido
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    for events in tracks:
        track = mido.MidiTrack(); midi.tracks.append(track)
        track.extend(events)
    return midi


def _ts(numerator, denominator, time=0):
    import mido
    return mido.MetaMessage("time_signature", numerator=numerator, denominator=denominator, time=time)


def _note_on(note, velocity, time=0):
    import mido
    return mido.Message("note_on", note=note, velocity=velocity, time=time)


def _note_off(note, time=0):
    import mido
    return mido.Message("note_off", note=note, velocity=0, time=time)


def test_duplicate_ts_facts_merge_and_canonical_spans_are_exact() -> None:
    midi = _midi([_ts(5, 32)], [_ts(5, 32), _note_on(60, 90, 360), _note_off(60, 120)])
    facts, notes = collect_raw_smf_facts(midi)
    chain = resolve_time_signature_chain(facts)
    spans = build_canonical_spans(chain, ppqn=480, terminal_end_ql=notes[0].end_ql(480))
    assert len(chain) == 1 and len(chain[0].contributors) == 2
    assert [(span.start_ql, span.end_ql) for span in spans] == [(Fraction(0), Fraction(5, 8)), (Fraction(5, 8), Fraction(5, 4))]


def test_fragment_quantization_uses_the_local_bar_phase() -> None:
    midi = _midi([_ts(5, 32)], [_note_on(60, 90, 360), _note_off(60, 120)])
    facts, notes = collect_raw_smf_facts(midi)
    spans = build_canonical_spans(resolve_time_signature_chain(facts), ppqn=480, terminal_end_ql=notes[0].end_ql(480))
    fragments = fragment_notes(notes, spans, ppqn=480)
    assert len(fragments) == 1
    fragment = fragments[0]
    assert fragment.canonical_bar_index == 1
    assert (fragment.raw_local_start_ql, fragment.raw_local_end_ql) == (Fraction(1, 8), Fraction(3, 8))
    assert (fragment.quantized_local_start_ql, fragment.quantized_local_end_ql) == (Fraction(1, 4), Fraction(1, 2))


def test_cross_bar_note_has_one_fragment_per_span_and_physical_continuation() -> None:
    midi = _midi([_ts(3, 4)], [_note_on(60, 90), _note_off(60, 1920)])
    facts, notes = collect_raw_smf_facts(midi)
    spans = build_canonical_spans(resolve_time_signature_chain(facts), ppqn=480, terminal_end_ql=notes[0].end_ql(480))
    fragments = fragment_notes(notes, spans, ppqn=480)
    assert len(fragments) == 2
    assert fragments[0].continues_into_next_bar is True
    assert fragments[1].continues_from_previous_bar is True
    assert fragments[1].quantized_local_start_ql == 0


def test_fifo_pairing_and_file_scoped_ordinals_are_deterministic() -> None:
    midi = _midi([_ts(4, 4)], [_note_on(60, 90), _note_on(60, 80, 10), _note_off(60, 10), _note_off(60, 10)], [_note_on(64, 70), _note_off(64, 30)])
    _, notes = collect_raw_smf_facts(midi)
    same_pitch = [note for note in notes if note.pitch == 60]
    assert [(note.start_tick, note.end_tick) for note in same_pitch] == [(0, 20), (10, 30)]
    assert [note.source_note_ordinal for note in notes] == [0, 1, 2]


def test_frozen_pairing_normalization_discards_only_authorized_redundancies() -> None:
    midi = _midi([_ts(4, 4)], [_note_on(60, 90), _note_off(60), _note_on(69, 70), _note_off(69, 10), _note_on(60, 80), _note_off(60, 10), _note_off(62)])
    repairs = []
    _, notes = collect_raw_smf_facts(midi, repairs=repairs)
    assert [(note.pitch, note.start_tick, note.end_tick, note.source_note_ordinal) for note in notes] == [(69, 0, 10, 0), (60, 10, 20, 1)]
    assert [repair.repair_kind for repair in repairs] == ["same_tick_zero_duration_pair", "redundant_orphan_note_off"]
    assert [event[0] for event in repairs[0].same_tick_events] == ["note_on", "note_off", "note_on"]


def test_same_tick_pair_with_fifo_depth_above_one_remains_strict_failure() -> None:
    midi = _midi([_ts(4, 4)], [_note_on(60, 90), _note_on(60, 80), _note_off(60)])
    with pytest.raises(CanonicalTimelineError, match="note_nonpositive_duration"):
        collect_raw_smf_facts(midi, repairs=[])


@pytest.mark.parametrize("events, error", [
    ([_ts(4, 4, 1)], "time_signature_initial_missing"),
    ([_ts(4, 4), _note_off(60)], "orphan_note_off"),
])
def test_invalid_raw_facts_fail_without_fallback(events, error) -> None:
    midi = _midi(events)
    with pytest.raises(CanonicalTimelineError, match=error):
        facts, _ = collect_raw_smf_facts(midi)
        resolve_time_signature_chain(facts)


def test_resolve_time_signature_chain_can_explicitly_inject_smf_default_4_4() -> None:
    facts, _ = collect_raw_smf_facts(_midi([_ts(3, 4, 480)]))
    chain = resolve_time_signature_chain(facts, initial_time_signature_policy="smf_default_4_4")
    assert [(item.absolute_tick, item.numerator, item.denominator, item.origin) for item in chain] == [(0, 4, 4, "smf_default"), (480, 3, 4, "smf")]


def test_resolve_time_signature_chain_can_inject_when_smf_has_no_ts_events() -> None:
    chain = resolve_time_signature_chain([], initial_time_signature_policy="smf_default_4_4")
    assert [(item.absolute_tick, item.numerator, item.denominator, item.origin) for item in chain] == [(0, 4, 4, "smf_default")]
    with pytest.raises(CanonicalTimelineError, match="time_signature_initial_missing"):
        resolve_time_signature_chain([])


def test_mid_bar_time_signature_change_materializes_exact_partial_span() -> None:
    midi = _midi([_ts(4, 4), _ts(3, 4, 480)], [_note_on(60, 90), _note_off(60, 2400)])
    facts, notes = collect_raw_smf_facts(midi)
    spans = build_canonical_spans(resolve_time_signature_chain(facts), ppqn=480, terminal_end_ql=notes[0].end_ql(480))
    assert [(span.start_ql, span.end_ql, span.time_signature, span.is_partial) for span in spans] == [(Fraction(0), Fraction(1), "4/4", True), (Fraction(1), Fraction(4), "3/4", False), (Fraction(4), Fraction(7), "3/4", False)]
    assert spans[0].partial_reason == "time_signature_change"


def test_bar_boundary_time_signature_change_keeps_preceding_span_full() -> None:
    """A TS at an exact old-meter boundary must not manufacture a partial span."""
    midi = _midi([_ts(4, 4), _ts(3, 4, 1920)], [_note_on(60, 90), _note_off(60, 3360)])
    facts, notes = collect_raw_smf_facts(midi)
    spans = build_canonical_spans(resolve_time_signature_chain(facts), ppqn=480, terminal_end_ql=notes[0].end_ql(480))
    assert [(span.start_ql, span.end_ql, span.time_signature, span.is_partial) for span in spans] == [
        (Fraction(0), Fraction(4), "4/4", False),
        (Fraction(4), Fraction(7), "3/4", False),
    ]


def test_terminal_time_signature_change_materializes_the_preceding_partial_span() -> None:
    """A TS exactly at terminal end closes the preceding partial span."""
    midi = _midi([_ts(4, 4), _ts(3, 4, 480)], [_note_on(60, 90), _note_off(60, 480)])
    facts, notes = collect_raw_smf_facts(midi)
    spans = build_canonical_spans(resolve_time_signature_chain(facts), ppqn=480, terminal_end_ql=notes[0].end_ql(480))
    assert len(spans) == 1 and spans[0].is_partial and spans[0].end_ql == Fraction(1)


def test_post_terminal_time_signature_does_not_remove_terminal_rest_tail() -> None:
    midi = _midi([_ts(4, 4), _ts(3, 4, 960)], [_note_on(60, 90), _note_off(60, 480)])
    facts, notes = collect_raw_smf_facts(midi)
    spans = build_canonical_spans(resolve_time_signature_chain(facts), ppqn=480, terminal_end_ql=notes[0].end_ql(480))
    assert [(span.start_ql, span.end_ql, span.time_signature, span.is_partial) for span in spans] == [(Fraction(0), Fraction(4), "4/4", False)]


@pytest.mark.parametrize("raw_start,raw_end,expected_slot,expected_end", [
    (53, 58, 0, Fraction(1, 4)),
    (96, 144, 1, Fraction(1, 2)),
    (115, 125, 1, Fraction(1, 2)),
])
def test_collapsed_fragment_projects_to_frozen_best_local_slot(raw_start, raw_end, expected_slot, expected_end) -> None:
    midi = _midi([_ts(4, 4)], [_note_on(60, 90, raw_start), _note_off(60, raw_end - raw_start)])
    facts, notes = collect_raw_smf_facts(midi)
    span = build_canonical_spans(resolve_time_signature_chain(facts), ppqn=480, terminal_end_ql=notes[0].end_ql(480))[0]
    fragment = fragment_notes(notes, [span], ppqn=480)[0]
    assert fragment.quantization_repair_kind == "minimum_representable_slot_projection"
    assert fragment.repair_slot_index == expected_slot
    assert fragment.quantized_local_end_ql == expected_end
