"""Shared deterministic V2 source-note role assignment."""

from __future__ import annotations

from typing import Any, Iterable


def identity(note: Any) -> str:
    """Return the source identity used as the final deterministic tie-break."""
    if isinstance(note, dict):
        return str(note.get("source_note_id") or (
            f"{note.get('physical_track_index', note.get('track_index', 0))}:"
            f"{note.get('source_note_ordinal', 0)}"
        ))
    return (
        f"{note.source_file_identity or ''}:{note.physical_track_index or 0}:"
        f"{note.source_note_ordinal or 0}"
    )


def value(note: Any, name: str, default: Any = 0) -> Any:
    """Read a source field from either a raw mapping or a parsed note."""
    return note.get(name, default) if isinstance(note, dict) else getattr(note, name, default)


def assign(
    active: Iterable[Any],
    previous: Any | None,
    tolerance: int = 7,
) -> tuple[Any | None, Any | None, list[Any]]:
    """Assign melody, bass, and sorted harmony using the V2 contract rules."""
    notes = list(active)
    if not notes:
        return None, None, []
    highest = min(
        notes,
        key=lambda note: (-int(value(note, "pitch")), -int(value(note, "velocity")), identity(note)),
    )
    melody = (
        previous
        if previous in notes and int(value(previous, "pitch")) >= int(value(highest, "pitch")) - tolerance
        else highest
    )
    remaining = [note for note in notes if note is not melody]
    bass = (
        min(
            remaining,
            key=lambda note: (int(value(note, "pitch")), -int(value(note, "velocity")), identity(note)),
        )
        if remaining
        else None
    )
    harmony = [note for note in remaining if note is not bass]
    harmony.sort(
        key=lambda note: (
            int(value(note, "pitch")),
            int(value(note, "physical_track_index", value(note, "track_index", 0))),
            float(value(note, "duration_ql")),
            float(value(note, "source_onset_ql", value(note, "onset_ql"))),
            int(value(note, "velocity")),
            identity(note),
        )
    )
    return melody, bass, harmony
