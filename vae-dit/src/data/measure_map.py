"""music21-specific real-measure extraction for final Codec V2 parsing."""
from __future__ import annotations

from typing import Any, Sequence

from data.core import MeasureSpan

EPSILON_QL = 1.0e-6


def split_tunes(parsed: Any) -> list[Any]:
    """Return independent scores for a score or music21 Opus."""
    from music21 import stream
    return list(parsed.scores) if isinstance(parsed, stream.Opus) else [parsed]


def extract_measure_spans(score: Any) -> list[MeasureSpan]:
    """Extract a validated, contiguous measure map from the first physical part."""
    from music21 import stream
    parts = list(score.parts) if getattr(score, "parts", None) else [score]
    measures = list(parts[0].recurse().getElementsByClass(stream.Measure))
    if not measures:
        raise ValueError("measure_map_unavailable")
    result: list[MeasureSpan] = []
    previous_end: float | None = None
    for position, measure in enumerate(measures):
        signature = measure.timeSignature or measure.getContextByClass("TimeSignature")
        if signature is None:
            raise ValueError("measure_map_unavailable")
        start = float(measure.offset)
        duration = float(measure.duration.quarterLength)
        if duration <= 0:
            raise ValueError("measure_map_invalid_duration")
        if previous_end is not None and abs(start - previous_end) > EPSILON_QL:
            raise ValueError("measure_map_noncontiguous")
        numerator, denominator = int(signature.numerator), int(signature.denominator)
        nominal = float(signature.barDuration.quarterLength)
        result.append(MeasureSpan(position, start, start + duration, f"{numerator}/{denominator}", numerator, denominator, position == 0 and duration < nominal - EPSILON_QL))
        previous_end = start + duration
    reference = [(span.start_ql, span.end_ql, span.time_signature) for span in result]
    for part in parts[1:]:
        part_measures = list(part.recurse().getElementsByClass(stream.Measure))
        if len(part_measures) != len(reference):
            raise ValueError("measure_map_part_mismatch")
        for expected, measure in zip(reference, part_measures):
            signature = measure.timeSignature or measure.getContextByClass("TimeSignature")
            observed = (float(measure.offset), float(measure.offset + measure.duration.quarterLength), f"{signature.numerator}/{signature.denominator}" if signature is not None else None)
            if observed[2] != expected[2] or abs(observed[0] - expected[0]) > EPSILON_QL or abs(observed[1] - expected[1]) > EPSILON_QL:
                raise ValueError("measure_map_part_mismatch")
    return result
