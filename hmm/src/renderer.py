#!/usr/bin/env python3
"""Renderer layer facade for physical realization and MIDI output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from generation_data import CodebookEntry, GenerationResult
from harmonic_engine import HarmonicEngine, HarmonicMidiRenderer


@dataclass
class RenderResult:
    """Renderer output plus diagnostics."""

    generation: GenerationResult
    diagnostics: Dict[str, Any]


class HarmonicPhysicalRenderer:
    """Adapter around the current harmonic engine and MIDI renderer."""

    def __init__(self, config: Dict[str, Any], codebook: Dict[int, CodebookEntry]) -> None:
        self.config = config
        self.codebook = codebook

    def realize(self, generation: GenerationResult) -> RenderResult:
        engine = HarmonicEngine(self.config, self.codebook)
        realized = engine.realize(generation)
        return RenderResult(realized, dict(engine.diagnostics))

    def write_midi(self, generation: GenerationResult, midi_path: Path) -> Dict[str, Any]:
        return HarmonicMidiRenderer.from_style_config(self.config).write(
            generation.harmonic_bars,
            midi_path,
        )
