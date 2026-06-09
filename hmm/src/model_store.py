#!/usr/bin/env python3
"""Model bundle persistence for the DFA/HMM engine."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from architecture import EncoderModel, SymbolVocabulary
from core_data import BarRecord, ObservationVocab, SongRecord
from form_hmm import LeftToRightFormHMM
from generation_data import CodebookEntry


class ObservationBarPoolBuilder:
    """Build concrete bar pools indexed by contiguous HMM observation ID."""

    def build(self, songs: Sequence[SongRecord]) -> Dict[int, List[BarRecord]]:
        pools: Dict[int, List[BarRecord]] = defaultdict(list)
        for song in songs:
            for bar in song.bars:
                if bar.observation_id is not None:
                    pools[int(bar.observation_id)].append(bar)
        return dict(pools)

    def diagnostics(
        self,
        pools: Dict[int, List[BarRecord]],
        vocab: ObservationVocab,
    ) -> Dict[str, Any]:
        expected = set(vocab.observation_to_composite)
        actual = set(pools)
        missing = sorted(expected - actual)
        counts = {str(obs): len(bars) for obs, bars in sorted(pools.items())}
        return {
            "expected_observation_count": len(expected),
            "pooled_observation_count": len(actual),
            "missing_observation_ids": missing,
            "pool_size_by_observation": counts,
            "min_pool_size": min(counts.values()) if counts else 0,
            "max_pool_size": max(counts.values()) if counts else 0,
        }


@dataclass
class ModelBundle:
    """Saved model data needed for form-driven generation."""

    config: Dict[str, Any]
    observation_vocab: ObservationVocab
    form_models: Dict[str, LeftToRightFormHMM]
    form_templates: Dict[str, Any]
    global_codebook: Dict[int, CodebookEntry]
    observation_to_bars: Dict[int, List[BarRecord]]
    training_summary: Dict[str, Any]
    encoder_model: EncoderModel | None = None

    @classmethod
    def from_training(
        cls,
        config: Dict[str, Any],
        songs: Sequence[SongRecord],
        vocab: ObservationVocab,
        form_models: Dict[str, LeftToRightFormHMM],
        training_summary: Dict[str, Any],
        form_templates: Dict[str, Any] | None = None,
        global_codebook: Dict[int, CodebookEntry] | None = None,
        encoder_model: EncoderModel | None = None,
    ) -> "ModelBundle":
        pools = ObservationBarPoolBuilder().build(songs)
        templates = form_templates or {
            name: {
                "sections": [
                    {
                        "name": model.state_role_map.get(index, f"State_{index}"),
                        "length": model.section_lengths[index] if index < len(model.section_lengths) else 1,
                    }
                    for index in range(model.n_states)
                ]
            }
            for name, model in form_models.items()
        }
        codebook = global_codebook or {}
        encoder_model = encoder_model or EncoderModel.from_legacy(
            codebook,
            vocab,
            metadata={
                "source": "legacy_bar_clustering",
                "symbol_id_field": "observation_id",
                "codebook_id_field": "codebook_id",
            },
        )
        return cls(
            config,
            vocab,
            form_models,
            templates,
            codebook,
            dict(pools),
            training_summary,
            encoder_model,
        )

    @property
    def symbol_vocabulary(self) -> SymbolVocabulary:
        if self.encoder_model is not None:
            return self.encoder_model.vocabulary
        return SymbolVocabulary.from_observation_vocab(self.observation_vocab)

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "observation_vocab": self.observation_vocab.to_dict(),
            "form_models": {name: model.to_dict() for name, model in self.form_models.items()},
            "form_templates": self.form_templates,
            "global_codebook": {
                str(key): value.to_dict() for key, value in self.global_codebook.items()
            },
            "encoder_model": (
                self.encoder_model.to_dict()
                if self.encoder_model is not None
                else EncoderModel.from_legacy(
                    self.global_codebook,
                    self.observation_vocab,
                    metadata={"source": "legacy_save_adapter"},
                ).to_dict()
            ),
            "observation_to_bars": {
                str(obs): [bar.to_dict() for bar in bars]
                for obs, bars in self.observation_to_bars.items()
            },
            "training_summary": self.training_summary,
        }
        (model_dir / "model_bundle.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: str | Path) -> "ModelBundle":
        payload = json.loads((Path(model_dir) / "model_bundle.json").read_text(encoding="utf-8"))
        observation_vocab = ObservationVocab.from_dict(payload["observation_vocab"])
        global_codebook = {
            int(key): CodebookEntry.from_dict({**value, "codebook_id": int(key)})
            for key, value in payload.get("global_codebook", {}).items()
        }
        encoder_payload = payload.get("encoder_model")
        encoder_model = (
            EncoderModel.from_dict(encoder_payload)
            if encoder_payload is not None
            else EncoderModel.from_legacy(
                global_codebook,
                observation_vocab,
                metadata={"source": "legacy_load_adapter"},
            )
        )
        return cls(
            config=payload["config"],
            observation_vocab=observation_vocab,
            form_models={
                name: LeftToRightFormHMM.from_dict(model_payload)
                for name, model_payload in payload.get("form_models", {}).items()
            },
            form_templates=payload.get("form_templates", {}),
            global_codebook=global_codebook,
            observation_to_bars={
                int(obs): [BarRecord.from_dict(item) for item in bars]
                for obs, bars in payload.get("observation_to_bars", {}).items()
            },
            training_summary=payload.get("training_summary", {}),
            encoder_model=encoder_model,
        )
