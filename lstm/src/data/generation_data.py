#!/usr/bin/env python3
"""Typed data objects for generation and harmonic realization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from data.bar_density import TokenDensityAnalyzer, TokenDensityMetrics
from data.core_data import NoteRecord


@dataclass(frozen=True)
class CodebookCandidate:
    """One physical training bar assigned to a global codebook label."""

    source_song: Optional[str]
    source_file: Optional[str]
    source_bar_index: Optional[int]
    relative_tokens: List[int]
    absolute_tokens: List[int]
    density: Optional[TokenDensityMetrics] = None
    token_variance: float = 0.0
    sharing_score: float = 1.0
    kmeans_id: Optional[int] = None
    observation_id: Optional[int] = None
    position_ratio: float = 0.0
    latent_vector: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.density is not None:
            payload["density"] = self.density.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CodebookCandidate":
        density_payload = payload.get("density")
        source_bar_index = payload.get("source_bar_index")
        kmeans_id = payload.get("kmeans_id")
        observation_id = payload.get("observation_id")
        relative_tokens = [int(x) for x in payload.get("relative_tokens", [])]
        token_variance = float(payload.get("token_variance", CodebookEntry._token_variance(relative_tokens)))
        return cls(
            source_song=payload.get("source_song"),
            source_file=payload.get("source_file"),
            source_bar_index=int(source_bar_index) if source_bar_index is not None else None,
            relative_tokens=relative_tokens,
            absolute_tokens=[int(x) for x in payload.get("absolute_tokens", [])],
            density=(
                TokenDensityMetrics.from_dict(density_payload)
                if density_payload is not None
                else TokenDensityAnalyzer().analyze(relative_tokens)
            ),
            token_variance=token_variance,
            sharing_score=float(payload.get("sharing_score", CodebookEntry._sharing_score(token_variance))),
            kmeans_id=int(kmeans_id) if kmeans_id is not None else None,
            observation_id=int(observation_id) if observation_id is not None else None,
            position_ratio=float(payload.get("position_ratio", 0.0)),
            latent_vector=(
                [float(x) for x in payload["latent_vector"]]
                if payload.get("latent_vector") is not None
                else None
            ),
        )

    def source(self) -> "CodebookSource":
        return CodebookSource(
            source_song=self.source_song,
            source_file=self.source_file,
            source_bar_index=self.source_bar_index,
        )


@dataclass(frozen=True)
class CodebookEntry:
    """One global codebook entry used for harmonic realization."""

    codebook_id: int
    source_song: Optional[str]
    source_file: Optional[str]
    source_bar_index: Optional[int]
    relative_tokens: List[int]
    absolute_tokens: List[int]
    density: Optional[TokenDensityMetrics] = None
    token_variance: float = 0.0
    sharing_score: float = 1.0
    candidates: List[CodebookCandidate] = field(default_factory=list)
    latent_vector: Optional[List[float]] = None
    position_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.density is not None:
            payload["density"] = self.density.to_dict()
        payload["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CodebookEntry":
        source_bar_index = payload.get("source_bar_index")
        relative_tokens = [int(x) for x in payload.get("relative_tokens", [])]
        token_variance = float(payload.get("token_variance", cls._token_variance(relative_tokens)))
        density_payload = payload.get("density")
        return cls(
            codebook_id=int(payload.get("codebook_id", 0)),
            source_song=payload.get("source_song"),
            source_file=payload.get("source_file"),
            source_bar_index=int(source_bar_index) if source_bar_index is not None else None,
            relative_tokens=relative_tokens,
            absolute_tokens=[int(x) for x in payload.get("absolute_tokens", [])],
            density=(
                TokenDensityMetrics.from_dict(density_payload)
                if density_payload is not None
                else TokenDensityAnalyzer().analyze(relative_tokens)
            ),
            token_variance=token_variance,
            sharing_score=float(payload.get("sharing_score", cls._sharing_score(token_variance))),
            candidates=[
                CodebookCandidate.from_dict(item)
                for item in payload.get("candidates", [])
            ],
            latent_vector=(
                [float(x) for x in payload["latent_vector"]]
                if payload.get("latent_vector") is not None
                else None
            ),
            position_ratio=float(payload.get("position_ratio", 0.0)),
        )

    @staticmethod
    def _token_variance(tokens: List[int]) -> float:
        values = [float(token) for token in tokens if int(token) >= 0]
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return float(sum((value - mean) ** 2 for value in values) / len(values))

    @staticmethod
    def _sharing_score(variance: float) -> float:
        return float(1.0 / (1.0 + max(0.0, variance)))

    def source(self) -> "CodebookSource":
        return CodebookSource(
            source_song=self.source_song,
            source_file=self.source_file,
            source_bar_index=self.source_bar_index,
        )


@dataclass(frozen=True)
class SectionPlanItem:
    """One DFA/form section to be generated."""

    state_id: int
    name: str
    bars: int
    source: Optional[str] = None
    pitch_offset: int = 0
    cadence: str = "none"
    start_degree: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SectionPlanItem":
        return cls(
            state_id=int(payload["state_id"]),
            name=str(payload["name"]),
            bars=int(payload["bars"]),
            source=payload.get("source"),
            pitch_offset=int(payload.get("pitch_offset", 0) or 0),
            cadence=str(payload.get("cadence", "none")),
            start_degree=payload.get("start_degree"),
        )


@dataclass(frozen=True)
class SampledBar:
    """One sampled observation before harmonic realization."""

    output_bar_index: int
    section: str
    section_local_index: int
    hidden_state: int
    observation_id: int
    composite_key: str
    emission_probability: float
    source_file: str
    source_bar_index: int
    codebook_id: int
    kmeans_id: Optional[int]
    absolute_tokens: List[int]
    relative_tokens: List[int]
    selection_mode: str = "sampled_bar"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SampledBar":
        kmeans_id = payload.get("kmeans_id")
        return cls(
            output_bar_index=int(payload["output_bar_index"]),
            section=str(payload["section"]),
            section_local_index=int(payload["section_local_index"]),
            hidden_state=int(payload["hidden_state"]),
            observation_id=int(payload["observation_id"]),
            composite_key=str(payload["composite_key"]),
            emission_probability=float(payload["emission_probability"]),
            source_file=str(payload["source_file"]),
            source_bar_index=int(payload["source_bar_index"]),
            codebook_id=int(payload["codebook_id"]),
            kmeans_id=int(kmeans_id) if kmeans_id is not None else None,
            absolute_tokens=[int(x) for x in payload.get("absolute_tokens", [])],
            relative_tokens=[int(x) for x in payload.get("relative_tokens", [])],
            selection_mode=str(payload.get("selection_mode", "sampled_bar")),
        )


@dataclass(frozen=True)
class HarmonyBarPlan:
    """Per-bar harmonic plan produced by the Markov bridge."""

    section: str
    section_local_index: int
    degree: str
    degree_index: int
    section_base_pitch: int
    bar_base_pitch: int
    cadence: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HarmonyBarPlan":
        return cls(
            section=str(payload["section"]),
            section_local_index=int(payload["section_local_index"]),
            degree=str(payload["degree"]),
            degree_index=int(payload["degree_index"]),
            section_base_pitch=int(payload["section_base_pitch"]),
            bar_base_pitch=int(payload["bar_base_pitch"]),
            cadence=str(payload.get("cadence", "none")),
        )


@dataclass(frozen=True)
class CodebookSource:
    """Source metadata for the codebook medoid used to realize a bar."""

    source_song: Optional[str]
    source_file: Optional[str]
    source_bar_index: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CodebookSource":
        source_bar_index = payload.get("source_bar_index")
        return cls(
            source_song=payload.get("source_song"),
            source_file=payload.get("source_file"),
            source_bar_index=int(source_bar_index) if source_bar_index is not None else None,
        )


@dataclass(frozen=True)
class RealizedBar:
    """One generated bar after harmonic realization."""

    sampled: SampledBar
    harmony: HarmonyBarPlan
    codebook_source: CodebookSource
    codebook_density: Optional[TokenDensityMetrics]
    relative_tokens: List[int]
    token_variance: float
    sharing_score: float
    voice_sharing_applied: bool
    notes: List[NoteRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sampled": self.sampled.to_dict(),
            "harmony": self.harmony.to_dict(),
            "codebook_source": self.codebook_source.to_dict(),
            "codebook_density": self.codebook_density.to_dict() if self.codebook_density is not None else None,
            "relative_tokens": list(self.relative_tokens),
            "token_variance": float(self.token_variance),
            "sharing_score": float(self.sharing_score),
            "voice_sharing_applied": bool(self.voice_sharing_applied),
            "notes": [note.to_dict() for note in self.notes],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RealizedBar":
        sampled_payload = payload.get("sampled", payload)
        harmony_payload = payload.get("harmony", payload)
        return cls(
            sampled=SampledBar.from_dict(sampled_payload),
            harmony=HarmonyBarPlan.from_dict(harmony_payload),
            codebook_source=CodebookSource.from_dict(payload.get("codebook_source", {})),
            codebook_density=(
                TokenDensityMetrics.from_dict(payload["codebook_density"])
                if payload.get("codebook_density") is not None
                else None
            ),
            relative_tokens=[int(x) for x in payload.get("relative_tokens", [])],
            token_variance=float(payload.get("token_variance", 0.0)),
            sharing_score=float(payload.get("sharing_score", 1.0)),
            voice_sharing_applied=bool(payload.get("voice_sharing_applied", False)),
            notes=[NoteRecord.from_dict(item) for item in payload.get("notes", [])],
        )


@dataclass
class GenerationResult:
    """Complete generation payload, with harmonic bars optional until realized."""

    form: str
    seed: Optional[int]
    section_plan: List[SectionPlanItem]
    sampled_bars: List[SampledBar]
    harmonic_bars: List[RealizedBar] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "form": self.form,
            "seed": self.seed,
            "section_plan": [section.to_dict() for section in self.section_plan],
            "sampled_bars": [bar.to_dict() for bar in self.sampled_bars],
        }
        if self.harmonic_bars:
            payload["harmonic_bars"] = [bar.to_dict() for bar in self.harmonic_bars]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GenerationResult":
        return cls(
            form=str(payload["form"]),
            seed=payload.get("seed"),
            section_plan=[
                SectionPlanItem.from_dict(item) for item in payload.get("section_plan", [])
            ],
            sampled_bars=[
                SampledBar.from_dict(item) for item in payload.get("sampled_bars", [])
            ],
            harmonic_bars=[
                RealizedBar.from_dict(item) for item in payload.get("harmonic_bars", [])
            ],
        )
