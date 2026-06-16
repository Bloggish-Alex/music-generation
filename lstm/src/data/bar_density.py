#!/usr/bin/env python3
"""Token density diagnostics for bar codebook entries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

from common.config_loader import ConfigView


@dataclass(frozen=True)
class TokenDensityMetrics:
    token_count: int
    rest_count: int
    sustain_count: int
    note_on_count: int
    rest_ratio: float
    sustain_ratio: float
    note_on_ratio: float
    realized_note_count: int
    active_duration_ql: float
    silent_duration_ql: float
    is_sparse: bool
    is_semi_sparse: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TokenDensityMetrics":
        return cls(
            token_count=int(payload.get("token_count", 0)),
            rest_count=int(payload.get("rest_count", 0)),
            sustain_count=int(payload.get("sustain_count", 0)),
            note_on_count=int(payload.get("note_on_count", 0)),
            rest_ratio=float(payload.get("rest_ratio", 0.0)),
            sustain_ratio=float(payload.get("sustain_ratio", 0.0)),
            note_on_ratio=float(payload.get("note_on_ratio", 0.0)),
            realized_note_count=int(payload.get("realized_note_count", 0)),
            active_duration_ql=float(payload.get("active_duration_ql", 0.0)),
            silent_duration_ql=float(payload.get("silent_duration_ql", 0.0)),
            is_sparse=bool(payload.get("is_sparse", False)),
            is_semi_sparse=bool(payload.get("is_semi_sparse", False)),
        )


@dataclass(frozen=True)
class TokenDensityConfig:
    rest_token: int = -1
    sustain_token: int = -2
    bar_length_ql: float = 4.0
    sparse_active_duration_ql: float = 2.0
    sparse_note_on_count: int = 2
    sparse_rest_ratio: float = 0.5
    semi_sparse_active_duration_ql: float = 3.5
    semi_sparse_note_on_count: int = 6
    semi_sparse_rest_ratio: float = 0.25


class TokenDensityAnalyzer:
    """Compute audibility-oriented density metrics from grid tokens."""

    def __init__(self, config: TokenDensityConfig | None = None) -> None:
        self.config = config or TokenDensityConfig()

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "TokenDensityAnalyzer":
        grid = ConfigView(config).section("grid_tokenizer")
        density = ConfigView(config).section("bar_density")
        return cls(TokenDensityConfig(
            rest_token=int(grid.get("rest_token", -1)),
            sustain_token=int(grid.get("sustain_token", -2)),
            bar_length_ql=float(grid.get("bar_length_ql", 4.0)),
            sparse_active_duration_ql=float(density.get("sparse_active_duration_ql", 2.0)),
            sparse_note_on_count=int(density.get("sparse_note_on_count", 2)),
            sparse_rest_ratio=float(density.get("sparse_rest_ratio", 0.5)),
            semi_sparse_active_duration_ql=float(density.get("semi_sparse_active_duration_ql", 3.5)),
            semi_sparse_note_on_count=int(density.get("semi_sparse_note_on_count", 6)),
            semi_sparse_rest_ratio=float(density.get("semi_sparse_rest_ratio", 0.25)),
        ))

    def analyze(self, tokens: Sequence[int]) -> TokenDensityMetrics:
        token_values = [int(token) for token in tokens]
        token_count = len(token_values)
        rest_count = sum(1 for token in token_values if token == self.config.rest_token)
        sustain_count = sum(1 for token in token_values if token == self.config.sustain_token)
        note_on_count = sum(1 for token in token_values if token >= 0)
        step_ql = self.config.bar_length_ql / max(1, token_count)
        realized_note_count = 0
        active_slots = 0
        current_length = 0
        has_current_note = False
        for token in token_values:
            if token >= 0:
                if has_current_note:
                    realized_note_count += 1
                    active_slots += max(1, current_length)
                has_current_note = True
                current_length = 1
            elif token == self.config.sustain_token and has_current_note:
                current_length += 1
            else:
                if has_current_note:
                    realized_note_count += 1
                    active_slots += max(1, current_length)
                has_current_note = False
                current_length = 0
        if has_current_note:
            realized_note_count += 1
            active_slots += max(1, current_length)
        active_duration = round(float(active_slots * step_ql), 6)
        silent_duration = round(float(max(0.0, self.config.bar_length_ql - active_duration)), 6)
        rest_ratio = self._ratio(rest_count, token_count)
        sustain_ratio = self._ratio(sustain_count, token_count)
        note_on_ratio = self._ratio(note_on_count, token_count)
        is_sparse = (
            active_duration < self.config.sparse_active_duration_ql
            or realized_note_count <= self.config.sparse_note_on_count
            or rest_ratio >= self.config.sparse_rest_ratio
        )
        is_semi_sparse = (
            is_sparse
            or active_duration < self.config.semi_sparse_active_duration_ql
            or realized_note_count <= self.config.semi_sparse_note_on_count
            or rest_ratio >= self.config.semi_sparse_rest_ratio
        )
        return TokenDensityMetrics(
            token_count=token_count,
            rest_count=rest_count,
            sustain_count=sustain_count,
            note_on_count=note_on_count,
            rest_ratio=rest_ratio,
            sustain_ratio=sustain_ratio,
            note_on_ratio=note_on_ratio,
            realized_note_count=realized_note_count,
            active_duration_ql=active_duration,
            silent_duration_ql=silent_duration,
            is_sparse=is_sparse,
            is_semi_sparse=is_semi_sparse,
        )

    def _ratio(self, count: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return round(float(count) / float(total), 6)
