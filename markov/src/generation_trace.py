#!/usr/bin/env python3
"""Generation trace recorder for hierarchical generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from hierarchical_types import NoteEvent

log = logging.getLogger("hierarchical_generator")


class GenerationTraceRecorder:
    """Collect and write an end-to-end generation data-flow trace."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.modules: List[Dict[str, Any]] = []
        self.note_bars: List[Dict[str, Any]] = []

    def record_module(self, run: Any, module_name: str) -> None:
        if not self.enabled:
            return
        self.modules.append({
            "module": module_name,
            "labels": len(run.labels),
            "events": len(run.event_log),
            "measure_context": len(run.measure_context),
            "notes_bars": len(run.all_notes),
            "notes_total": sum(len(bar_notes) for bar_notes in run.all_notes),
        })

    def before_note_stage(self, bar: Any) -> Dict[str, Any]:
        return self.note_snapshot(bar)

    def after_note_stage(self, bar: Any) -> Dict[str, Any]:
        return self.note_snapshot(bar)

    def record_note_stage(
        self,
        bar: Any,
        stage_name: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        while len(self.note_bars) <= bar.i:
            self.note_bars.append({
                "bar": len(self.note_bars) + 1,
                "cluster": None,
                "section_label": None,
                "role": None,
                "local_bar": None,
                "stages": [],
            })
        entry = self.note_bars[bar.i]
        entry.update({
            "cluster": int(bar.cluster_id),
            "section_label": bar.sl or None,
            "role": bar.role or None,
            "local_bar": int(bar.bi) if bar.sl else None,
        })
        entry["stages"].append({
            "stage": stage_name,
            "before": before,
            "after": after,
            "changed": before != after,
        })

    def write(self, run: Any, generator: Any) -> None:
        if not self.enabled:
            return
        output_path = Path(run.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path = output_path.with_suffix(".generation.json")
        report = {
            "version": 1,
            "output_path": str(output_path),
            "requested": {
                "target_measures": run.target_measures,
                "template_file": run.template_file,
                "variation_strength": run.variation_strength,
                "time_signature": list(run.time_signature),
                "tempo": run.tempo,
                "seed": run.seed,
                "enable_variation": run.enable_variation,
            },
            "executed_modules": getattr(generator, "_last_midi_generation_modules", []),
            "executed_note_modules": getattr(generator, "_last_note_generation_modules", []),
            "module_trace": self.modules,
            "timeline": {
                "labels": [int(x) for x in run.labels],
                "events": self.json_safe(run.event_log),
            },
            "measure_context": self.measure_context(run),
            "composition": self.composition(run),
            "harmony": {
                "requested_mode": run.harmony_mode,
                "actual_mode": run.actual_harmony_mode,
                "has_learned_model": run.learned_harmony is not None,
                "bars": self.harmony(run),
            },
            "notes": {
                "bars": self.json_safe(self.note_bars),
                "final": [self.notes(bar_notes) for bar_notes in run.all_notes],
            },
        }
        with open(trace_path, "w") as f:
            json.dump(self.json_safe(report), f, indent=2)
        log.info("Wrote generation trace to %s", trace_path)

    def measure_context(self, run: Any) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for i, item in enumerate(run.measure_context):
            label, local_bar, role, occurrence_id, section_len = item
            rows.append({
                "bar": i + 1,
                "cluster": int(run.labels[i]) if i < len(run.labels) else None,
                "section_label": label,
                "local_bar": int(local_bar),
                "role": role,
                "occurrence_id": int(occurrence_id),
                "section_len": int(section_len),
            })
        return rows

    def composition(self, run: Any) -> Dict[str, Any]:
        plan = run.composition_plan
        if plan is None:
            return {}
        return {
            "global_tonic_pc": int(plan.global_tonic_pc),
            "measure_targets": {
                str(k): int(v) for k, v in plan.measure_targets.items()
            },
            "measure_affects": {
                str(k): self.json_safe(v) for k, v in plan.measure_affects.items()
            },
            "development_roles": {
                str(k): v for k, v in plan.development_roles.items()
            },
        }

    def harmony(self, run: Any) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for i in range(len(run.labels)):
            harmony = run.harmonic_plan.get(i)
            rows.append({
                "bar": i + 1,
                "harmony": harmony.to_dict() if harmony is not None else None,
            })
        return rows

    def note_snapshot(self, bar: Any) -> Dict[str, Any]:
        return {
            "context_ready": bool(bar.sl),
            "bar_target_ready": bar.bar_target is not None,
            "rhythm_target_ready": bar.rhythm_target is not None,
            "notes": self.note_summary(bar.notes),
            "reused_motif": bool(bar.reused_motif),
            "generated_new_theme": bool(bar.generated_new_theme),
            "target_pitch": int(bar.target_pitch) if bar.target_pitch is not None else None,
            "affect_keys": sorted(str(k) for k in bar.affect.keys()),
            "dual_theme_active": isinstance(bar.affect.get("dual_theme"), dict),
        }

    def note_summary(self, notes: List[NoteEvent]) -> Dict[str, Any]:
        melody = [n for n in notes if n.voice == "melody" and n.pitch >= 0]
        bass = [n for n in notes if n.voice == "bass" and n.pitch >= 0]
        return {
            "count": len(notes),
            "melody_count": len(melody),
            "bass_count": len(bass),
            "pitches": [int(n.pitch) for n in melody[:16]],
            "rhythm": [
                [float(n.beat_offset), float(n.duration_ql)]
                for n in sorted(melody, key=lambda x: (x.beat_offset, x.pitch))[:16]
            ],
        }

    def notes(self, notes: List[NoteEvent]) -> Dict[str, Any]:
        melody = [n for n in notes if n.voice == "melody" and n.pitch >= 0]
        bass = [n for n in notes if n.voice == "bass" and n.pitch >= 0]
        return {
            "count": len(notes),
            "melody_count": len(melody),
            "bass_count": len(bass),
            "pitches": [int(n.pitch) for n in melody[:16]],
            "events": [
                {
                    "pitch": int(n.pitch),
                    "duration": float(n.duration_ql),
                    "offset": float(n.beat_offset),
                    "velocity": int(n.velocity),
                    "voice": n.voice,
                }
                for n in sorted(notes, key=lambda x: (x.beat_offset, x.voice, x.pitch))
            ],
        }

    @classmethod
    def json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): cls.json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.json_safe(v) for v in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return cls.json_safe(value.tolist())
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
