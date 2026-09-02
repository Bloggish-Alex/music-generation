#!/usr/bin/env python3
"""Core data records for parsed symbolic music and encoded bars."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MeasureSpan:
    """One validated source-measure interval in quarter lengths."""

    source_measure_index: int
    start_ql: float
    end_ql: float
    time_signature: str
    numerator: int
    denominator: int
    is_pickup: bool = False


@dataclass
class NoteEvent:
    """One note event in bar-local quarter-length time."""

    pitch: int
    onset_ql: float
    duration_ql: float
    velocity: int = 64
    source_file_identity: str | None = None
    physical_track_index: int | None = None
    source_note_ordinal: int | None = None
    # Persisted, stable identity of the un-clipped source note.  A continued
    # note deliberately keeps this value in every bar it intersects.
    source_note_id: str | None = None
    source_onset_ql: float | None = None
    continues_from_previous_bar: bool = False
    continues_into_next_bar: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert the note event to a JSON-safe dictionary."""
        return asdict(self)


@dataclass
class TrackRecord:
    """One voice/track inside a bar."""

    track_index: int
    name: str
    notes: List[NoteEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the track record to a JSON-safe dictionary."""
        return {
            "track_index": int(self.track_index),
            "name": self.name,
            "notes": [note.to_dict() for note in self.notes],
        }


@dataclass
class BarRecord:
    """A parsed bar with up to several raw tracks before tensor encoding."""

    song_id: str
    file_path: str
    bar_index: int
    bar_length_ql: float
    time_signature: str = "4/4"
    source_measure_index: int | None = None
    meter_numerator: int | None = None
    meter_denominator: int | None = None
    is_pickup: bool = False
    source_bar_count: Optional[int] = None
    form: Optional[str] = None
    section_label: Optional[str] = None
    section_index: Optional[int] = None
    tracks: List[TrackRecord] = field(default_factory=list)
    action: Optional[str] = None
    action_reason: Optional[str] = None

    def all_notes(self) -> List[NoteEvent]:
        """Return all notes from all tracks sorted by onset and pitch."""
        notes = [note for track in self.tracks for note in track.notes]
        return sorted(notes, key=lambda item: (float(item.onset_ql), int(item.pitch)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert the bar record to a JSON-safe dictionary."""
        return {
            "song_id": self.song_id,
            "file_path": self.file_path,
            "bar_index": int(self.bar_index),
            "bar_length_ql": float(self.bar_length_ql),
            "time_signature": self.time_signature,
            "source_measure_index": self.source_measure_index,
            "meter_numerator": self.meter_numerator,
            "meter_denominator": self.meter_denominator,
            "is_pickup": self.is_pickup,
            "source_bar_count": self.source_bar_count,
            "form": self.form,
            "section_label": self.section_label,
            "section_index": self.section_index,
            "tracks": [track.to_dict() for track in self.tracks],
            "action": self.action,
            "action_reason": self.action_reason,
        }


@dataclass
class SongRecord:
    """One parsed input file and its bar sequence."""

    song_id: str
    file_path: str
    form: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    runtime_diagnostics: Dict[str, Any] = field(default_factory=dict, repr=False)
    bars: List[BarRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the song record to a JSON-safe dictionary."""
        return {
            "song_id": self.song_id,
            "file_path": self.file_path,
            "form": self.form,
            "metadata": self.metadata,
            "bars": [bar.to_dict() for bar in self.bars],
        }


@dataclass
class BarTensorRecord:
    """Encoded tensor and diagnostics for one bar."""

    song_id: str
    bar_index: int
    tensor_shape: List[int]
    tensor: Any
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the tensor record to a JSON-safe dictionary."""
        tensor_value = self.tensor.tolist() if hasattr(self.tensor, "tolist") else self.tensor
        return {
            "song_id": self.song_id,
            "bar_index": int(self.bar_index),
            "tensor_shape": [int(value) for value in self.tensor_shape],
            "tensor": tensor_value,
            "diagnostics": self.diagnostics,
        }


@dataclass
class LatentBarRecord:
    """Metadata for one bar encoded by a trained latent model."""

    row_index: int
    tensor_key: str
    song_id: str
    bar_index: int
    action: Optional[str] = None
    form: Optional[str] = None
    section_label: Optional[str] = None
    section_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the latent row metadata to a JSON-safe dictionary."""
        return {
            "row_index": int(self.row_index),
            "tensor_key": self.tensor_key,
            "song_id": self.song_id,
            "bar_index": int(self.bar_index),
            "action": self.action,
            "form": self.form,
            "section_label": self.section_label,
            "section_index": self.section_index,
        }
