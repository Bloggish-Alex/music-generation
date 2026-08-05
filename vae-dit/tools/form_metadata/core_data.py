#!/usr/bin/env python3
"""Shared data structures for the DFA/HMM music engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NoteRecord:
    """One parsed symbolic note event stored in bar-local quarter lengths."""

    pitch: int
    onset_ql: float
    duration_ql: float
    velocity: int = 64

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this note without changing quarter-length units."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NoteRecord":
        """Reconstruct a note from its JSON-safe field mapping."""
        return cls(**payload)


@dataclass
class BarRecord:
    """Core bar object enriched stage by stage during training.

    Raw input fields are populated by parsing. Grid fields are populated by
    preprocessing. Cluster and observation fields are populated by clustering.
    """

    song_id: str
    file_path: str
    bar_index: int
    time_signature: str
    bar_length_ql: float
    source_bar_count: Optional[int] = None
    notes: List[NoteRecord] = field(default_factory=list)
    genre: Optional[str] = None
    form: Optional[str] = None
    section_label: Optional[str] = None
    section_index: Optional[int] = None
    absolute_tokens: List[int] = field(default_factory=list)
    relative_tokens: List[int] = field(default_factory=list)
    token_variance: float = 0.0
    sharing_score: float = 1.0
    feature_vector: List[float] = field(default_factory=list)
    pitch_intervals: List[int] = field(default_factory=list)
    codebook_id: Optional[int] = None
    kmeans_id: Optional[int] = None
    composite_key: Optional[str] = None
    observation_id: Optional[int] = None

    def tokens_for_edit_distance(self, strategy: str = "relative") -> List[int]:
        """Return the token sequence used by edit-distance strategies."""
        if strategy == "absolute":
            return list(self.absolute_tokens)
        return list(self.relative_tokens)

    def composite_parts(self) -> Dict[str, Optional[int]]:
        """Return optional historical cluster identifiers for compatibility."""
        return {
            "codebook_id": self.codebook_id,
            "kmeans_id": self.kmeans_id,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the bar and its nested note records."""
        payload = asdict(self)
        payload["notes"] = [note.to_dict() for note in self.notes]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BarRecord":
        """Reconstruct a bar and nested notes from a JSON-safe mapping."""
        data = dict(payload)
        data["notes"] = [NoteRecord.from_dict(item) for item in data.get("notes", [])]
        return cls(**data)


@dataclass
class SongRecord:
    """One parsed training file plus optional form metadata."""

    song_id: str
    file_path: str
    genre: Optional[str] = None
    form: Optional[str] = None
    bars: List[BarRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the song, metadata, and ordered bar list."""
        return {
            "song_id": self.song_id,
            "file_path": self.file_path,
            "genre": self.genre,
            "form": self.form,
            "bars": [bar.to_dict() for bar in self.bars],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SongRecord":
        """Reconstruct a song and its ordered nested bars."""
        return cls(
            song_id=payload["song_id"],
            file_path=payload["file_path"],
            genre=payload.get("genre"),
            form=payload.get("form"),
            bars=[BarRecord.from_dict(item) for item in payload.get("bars", [])],
            metadata=payload.get("metadata", {}),
        )
