#!/usr/bin/env python3
"""Latent-token sequence model contracts.

The first learned decoder should output logits over the whole global codebook
vocabulary. The temporal graph planner then applies a dynamic mask before
sampling or beam search.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from common.architecture import SymbolID
from decoder.temporal_graph import CandidateTokenMask, TemporalActionState


@dataclass(frozen=True)
class LatentTokenContext:
    """Recent token sequence represented by codebook prototype latents."""

    token_ids: List[SymbolID]
    latent_sequence: List[List[float]]
    action_state: Optional[TemporalActionState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.action_state is not None:
            payload["action_state"] = self.action_state.to_dict()
        return payload


@dataclass(frozen=True)
class TokenLogitResult:
    """Raw and masked logits for next-token selection."""

    vocab_size: int
    logits: List[float]
    masked_logits: List[float]
    allowed_token_ids: List[SymbolID]
    top_tokens: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def probability_for(self, token_id: SymbolID) -> float:
        probs = self.masked_probabilities()
        token = int(token_id)
        return float(probs[token]) if 0 <= token < len(probs) else 0.0

    def masked_probabilities(self) -> List[float]:
        values = np.asarray(self.masked_logits, dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            return [0.0 for _ in self.masked_logits]
        max_value = float(np.max(values[finite]))
        exp_values = np.zeros_like(values)
        exp_values[finite] = np.exp(values[finite] - max_value)
        total = float(np.sum(exp_values))
        if total <= 0:
            return [0.0 for _ in self.masked_logits]
        return [float(item / total) for item in exp_values]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LatentTokenModel(Protocol):
    """Sequence model that scores the next global-codebook token."""

    def score_next(
        self,
        context: LatentTokenContext,
        candidate_mask: CandidateTokenMask,
    ) -> TokenLogitResult:
        ...


class TokenLogitMasker:
    """Apply graph-planner masks to model logits."""

    def apply(
        self,
        logits: Sequence[float],
        candidate_mask: CandidateTokenMask,
        top_k: int = 10,
    ) -> TokenLogitResult:
        raw = [float(value) for value in logits]
        allowed = sorted({int(token_id) for token_id in candidate_mask.allowed_token_ids})
        allowed_set = set(allowed)
        masked = [
            value if index in allowed_set else -math.inf
            for index, value in enumerate(raw)
        ]
        top_tokens = self._top_tokens(masked, top_k=top_k)
        return TokenLogitResult(
            vocab_size=len(raw),
            logits=raw,
            masked_logits=masked,
            allowed_token_ids=allowed,
            top_tokens=top_tokens,
            diagnostics={
                "allowed_count": len(allowed),
                "mask_reason": candidate_mask.reason,
            },
        )

    def _top_tokens(self, masked_logits: Sequence[float], top_k: int) -> List[Dict[str, Any]]:
        probs = TokenLogitResult(
            vocab_size=len(masked_logits),
            logits=list(masked_logits),
            masked_logits=list(masked_logits),
            allowed_token_ids=[],
        ).masked_probabilities()
        ordered = sorted(
            [
                (index, float(logit), float(probs[index]))
                for index, logit in enumerate(masked_logits)
                if math.isfinite(float(logit))
            ],
            key=lambda item: item[2],
            reverse=True,
        )
        return [
            {"token_id": int(index), "logit": logit, "probability": probability}
            for index, logit, probability in ordered[: max(0, int(top_k))]
        ]


class NullLatentTokenModel:
    """Uniform baseline over the graph-planner mask.

    This is useful for integration before the actual LSTM is trained.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)
        self.masker = TokenLogitMasker()

    def score_next(
        self,
        context: LatentTokenContext,
        candidate_mask: CandidateTokenMask,
    ) -> TokenLogitResult:
        result = self.masker.apply(
            logits=[0.0 for _ in range(self.vocab_size)],
            candidate_mask=candidate_mask,
        )
        diagnostics = dict(result.diagnostics)
        diagnostics.update({
            "model": self.__class__.__name__,
            "context_token_count": len(context.token_ids),
            "latent_dim": len(context.latent_sequence[-1]) if context.latent_sequence else 0,
        })
        return TokenLogitResult(
            vocab_size=result.vocab_size,
            logits=result.logits,
            masked_logits=result.masked_logits,
            allowed_token_ids=result.allowed_token_ids,
            top_tokens=result.top_tokens,
            diagnostics=diagnostics,
        )
