"""Output writer for generated symbolic bars and rendered MIDI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from common.model_store import ModelBundle
from data.generation_data import GenerationResult
from diagnostics.diagnostics import GenerationDiagnostics
from renderer.renderer import HarmonicPhysicalRenderer


class GenerationOutputWriter:
    """Persist generation artifacts after decoder sampling is complete."""

    def __init__(
        self,
        bundle: ModelBundle,
        config: Dict[str, Any],
        diagnostics: Optional[GenerationDiagnostics] = None,
    ) -> None:
        self.bundle = bundle
        self.config = config
        self.diagnostics = diagnostics

    def write(self, generation: GenerationResult, json_path: Path, midi_path: Path) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        codebook = (
            self.bundle.encoder_model.codebook.entries
            if self.bundle.encoder_model is not None
            else self.bundle.global_codebook
        )
        renderer = HarmonicPhysicalRenderer(
            self.config,
            codebook,
            candidate_selector_model=self.bundle.candidate_selector_model,
        )
        render_result = renderer.realize(generation)
        realized = render_result.generation
        render_diag = renderer.write_midi(realized, midi_path)
        render_result.diagnostics["render"] = render_diag
        json_path.write_text(json.dumps(realized.to_dict(), indent=2), encoding="utf-8")
        if self.diagnostics is None:
            return
        self.diagnostics.record_stage("harmonic_engine", render_result.diagnostics)
        for event in render_result.diagnostics.get("rare_bar_selection", {}).get("events", []):
            self.diagnostics.record_rare_bar_selection(event)
