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


@pytest.mark.parametrize("events, error", [
    ([_ts(4, 4, 1)], "time_signature_initial_missing"),
    ([_ts(4, 4), _note_off(60)], "orphan_note_off"),
])
def test_invalid_raw_facts_fail_without_fallback(events, error) -> None:
    midi = _midi(events)
    with pytest.raises(CanonicalTimelineError, match=error):
        facts, _ = collect_raw_smf_facts(midi)
        resolve_time_signature_chain(facts)


def test_mid_bar_time_signature_change_fails_exactly() -> None:
    midi = _midi([_ts(4, 4), _ts(3, 4, 480)], [_note_on(60, 90), _note_off(60, 2400)])
    facts, notes = collect_raw_smf_facts(midi)
    with pytest.raises(CanonicalTimelineError, match="time_signature_mid_bar_change"):
        build_canonical_spans(resolve_time_signature_chain(facts), ppqn=480, terminal_end_ql=notes[0].end_ql(480))


def test_terminal_does_not_hide_a_later_mid_bar_time_signature_change() -> None:
    """A post-terminal fact is validated even though it emits no trailing bar."""
    midi = _midi([_ts(4, 4), _ts(3, 4, 480)], [_note_on(60, 90), _note_off(60, 480)])
    facts, notes = collect_raw_smf_facts(midi)
    with pytest.raises(CanonicalTimelineError, match="time_signature_mid_bar_change"):
        build_canonical_spans(resolve_time_signature_chain(facts), ppqn=480, terminal_end_ql=notes[0].end_ql(480))


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
