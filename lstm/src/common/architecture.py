#!/usr/bin/env python3
"""Stable architecture interfaces for encoder, decoder, and renderer layers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from data.core_data import ObservationVocab, SongRecord
from data.generation_data import CodebookEntry, GenerationResult, RealizedBar, SectionPlanItem


SymbolID = int
CodebookID = int


@dataclass(frozen=True)
class DecodeContext:
    """Context supplied by the decoder when it asks for valid symbols."""

    section: Optional[str] = None
    section_local_index: Optional[int] = None
    phrase_position: Optional[int] = None
    position_context: Optional[str] = None
    hidden_state: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SymbolDescriptor:
    """Encoder-owned description of a symbol.

    Decoder modules should treat symbol_id as opaque. This descriptor is for
    Encoder/Renderer/Diagnostics code that needs to resolve the symbol back to
    codebook and conditioning metadata.
    """

    symbol_id: SymbolID
    codebook_id: CodebookID
    descriptor_key: str
    feature_cluster_id: Optional[int] = None
    phrase_position: Optional[int] = None
    period_role: Optional[str] = None
    position_context: Optional[str] = None
    position_strategy: Optional[str] = None
    position_modulo: Optional[int] = None
    raw_parts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SymbolDescriptor":
        return cls(
            symbol_id=int(payload["symbol_id"]),
            codebook_id=int(payload["codebook_id"]),
            descriptor_key=str(payload.get("descriptor_key", payload["symbol_id"])),
            feature_cluster_id=_optional_int(payload.get("feature_cluster_id")),
            phrase_position=_optional_int(payload.get("phrase_position")),
            period_role=payload.get("period_role"),
            position_context=payload.get("position_context"),
            position_strategy=payload.get("position_strategy"),
            position_modulo=_optional_int(payload.get("position_modulo")),
            raw_parts=dict(payload.get("raw_parts", {})),
        )


@dataclass
class GlobalCodebook:
    """The central finite vocabulary of reusable bar material."""

    entries: Dict[CodebookID, CodebookEntry] = field(default_factory=dict)

    def get(self, codebook_id: CodebookID) -> CodebookEntry:
        return self.entries[int(codebook_id)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            str(codebook_id): entry.to_dict()
            for codebook_id, entry in sorted(self.entries.items())
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GlobalCodebook":
        return cls({
            int(codebook_id): CodebookEntry.from_dict({**entry, "codebook_id": int(codebook_id)})
            for codebook_id, entry in payload.items()
        })


@dataclass
class SymbolVocabulary:
    """Opaque SymbolID vocabulary owned by the encoder layer."""

    symbol_to_descriptor: Dict[SymbolID, SymbolDescriptor] = field(default_factory=dict)
    descriptor_to_symbol: Dict[str, SymbolID] = field(default_factory=dict)

    @classmethod
    def from_observation_vocab(cls, vocab: ObservationVocab) -> "SymbolVocabulary":
        descriptors: Dict[SymbolID, SymbolDescriptor] = {}
        descriptor_to_symbol: Dict[str, SymbolID] = {}
        for symbol_id, descriptor_key in vocab.observation_to_composite.items():
            symbol = int(symbol_id)
            key = str(descriptor_key)
            parts = dict(vocab.composite_parts.get(key, {}))
            descriptor = SymbolDescriptor(
                symbol_id=symbol,
                codebook_id=int(parts.get("codebook_id", -1)),
                descriptor_key=key,
                feature_cluster_id=_optional_int(parts.get("kmeans_id")),
                phrase_position=_optional_int(parts.get("phrase_position")),
                period_role=parts.get("period_role"),
                position_context=(
                    str(parts["position_context"])
                    if parts.get("position_context") is not None
                    else None
                ),
                position_strategy=parts.get("position_strategy"),
                position_modulo=_optional_int(parts.get("position_modulo")),
                raw_parts=parts,
            )
            descriptors[symbol] = descriptor
            descriptor_to_symbol[key] = symbol
        return cls(descriptors, descriptor_to_symbol)

    def descriptor_for(self, symbol_id: SymbolID) -> SymbolDescriptor:
        return self.symbol_to_descriptor[int(symbol_id)]

    def codebook_id_for(self, symbol_id: SymbolID) -> CodebookID:
        return self.descriptor_for(symbol_id).codebook_id

    def descriptor_key_for(self, symbol_id: SymbolID) -> str:
        return self.descriptor_for(symbol_id).descriptor_key

    def symbol_for_descriptor_key(self, descriptor_key: str) -> SymbolID:
        return int(self.descriptor_to_symbol[str(descriptor_key)])

    def symbols_for_codebook_id(self, codebook_id: CodebookID) -> List[SymbolID]:
        return [
            int(symbol_id)
            for symbol_id, descriptor in self.symbol_to_descriptor.items()
            if int(descriptor.codebook_id) == int(codebook_id)
        ]

    def symbols_for_context(self, context: DecodeContext, position_strategy: Optional[str] = None) -> List[SymbolID]:
        result: List[SymbolID] = []
        for symbol_id, descriptor in self.symbol_to_descriptor.items():
            if position_strategy is not None and descriptor.position_strategy != position_strategy:
                continue
            if context.position_context is not None and descriptor.position_context != str(context.position_context):
                continue
            if context.phrase_position is not None and descriptor.phrase_position != int(context.phrase_position):
                continue
            result.append(int(symbol_id))
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol_to_descriptor": {
                str(symbol_id): descriptor.to_dict()
                for symbol_id, descriptor in sorted(self.symbol_to_descriptor.items())
            },
            "descriptor_to_symbol": {
                key: int(value)
                for key, value in sorted(self.descriptor_to_symbol.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SymbolVocabulary":
        descriptors = {
            int(symbol_id): SymbolDescriptor.from_dict(descriptor)
            for symbol_id, descriptor in payload.get("symbol_to_descriptor", {}).items()
        }
        descriptor_to_symbol = {
            str(key): int(value)
            for key, value in payload.get("descriptor_to_symbol", {}).items()
        }
        return cls(descriptors, descriptor_to_symbol)


@dataclass
class EncoderModel:
    """Stable encoder output consumed by decoder and renderer layers."""

    codebook: GlobalCodebook
    vocabulary: SymbolVocabulary
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(
        cls,
        codebook: Dict[int, CodebookEntry],
        observation_vocab: ObservationVocab,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EncoderModel":
        return cls(
            codebook=GlobalCodebook(dict(codebook)),
            vocabulary=SymbolVocabulary.from_observation_vocab(observation_vocab),
            metadata=metadata or {},
        )

    def descriptor_for_symbol(self, symbol_id: SymbolID) -> SymbolDescriptor:
        return self.vocabulary.descriptor_for(symbol_id)

    def codebook_entry_for_symbol(self, symbol_id: SymbolID) -> CodebookEntry:
        return self.codebook.get(self.vocabulary.codebook_id_for(symbol_id))

    def symbols_for_context(self, context: DecodeContext, position_strategy: Optional[str] = None) -> List[SymbolID]:
        return self.vocabulary.symbols_for_context(context, position_strategy=position_strategy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "codebook": self.codebook.to_dict(),
            "vocabulary": self.vocabulary.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EncoderModel":
        return cls(
            codebook=GlobalCodebook.from_dict(payload.get("codebook", {})),
            vocabulary=SymbolVocabulary.from_dict(payload.get("vocabulary", {})),
            metadata=dict(payload.get("metadata", {})),
        )


class Encoder(Protocol):
    """Training-side layer that builds the global codebook and symbol vocabulary."""

    def fit(self, songs: Sequence[SongRecord]) -> EncoderModel:
        ...


class Decoder(Protocol):
    """Symbol sequence layer. It should output opaque SymbolID values."""

    def generate(self, form_name: str, seed: Optional[int] = None) -> GenerationResult:
        ...


class Renderer(Protocol):
    """Physical realization layer that turns a symbolic plan into notes/MIDI."""

    def realize(self, generation: GenerationResult) -> GenerationResult:
        ...


def descriptor_key(parts: Dict[str, Any]) -> str:
    """Canonical internal key for a symbol descriptor strategy.

    This helper is for encoder strategies only. Decoder and renderer modules
    should never parse or construct descriptor keys directly.
    """
    return json.dumps(parts, sort_keys=True, separators=(",", ":"))


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)
