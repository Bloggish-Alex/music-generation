#!/usr/bin/env python3
"""Renderer layer facade for physical realization and MIDI output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from decoder.candidate_selector import CandidateSelectorModel
from data.generation_data import CodebookEntry, GenerationResult
from renderer.harmonic_engine import HarmonicEngine, HarmonicMidiRenderer


@dataclass
class RenderResult:
    """Renderer output plus diagnostics."""

    generation: GenerationResult
    diagnostics: Dict[str, Any]


class HarmonicPhysicalRenderer:
    """Adapter around the current harmonic engine and MIDI renderer."""

    def __init__(
        self,
        config: Dict[str, Any],
        codebook: Dict[int, CodebookEntry],
        candidate_selector_model: Optional[CandidateSelectorModel] = None,
    ) -> None:
        self.config = config
        self.codebook = codebook
        self.candidate_selector_model = candidate_selector_model

    def realize(self, generation: GenerationResult) -> RenderResult:
        engine = HarmonicEngine(self.config, self.codebook, self.candidate_selector_model)
        realized = engine.realize(generation)
        return RenderResult(realized, dict(engine.diagnostics))

    def write_midi(self, generation: GenerationResult, midi_path: Path) -> Dict[str, Any]:
        return HarmonicMidiRenderer.from_style_config(self.config).write(
            generation.harmonic_bars,
            midi_path,
        )
