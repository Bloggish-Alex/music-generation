#!/usr/bin/env python3
"""Decoder layer helpers for form-conditioned symbol generation."""

from __future__ import annotations

from typing import List

from generation_data import SectionPlanItem
from model_store import ModelBundle


class SectionPlanBuilder:
    """Build a deterministic section plan from a trained form template."""

    def build(self, bundle: ModelBundle, form_name: str) -> List[SectionPlanItem]:
        template = bundle.form_templates.get(form_name, {})
        sections = template.get("sections", [])
        if not sections:
            model = bundle.form_models[form_name]
            sections = [
                {
                    "name": model.state_role_map.get(index, f"State_{index}"),
                    "length": model.section_lengths[index] if index < len(model.section_lengths) else 1,
                    "source": None,
                }
                for index in range(model.n_states)
            ]
        plan = []
        for state_id, section in enumerate(sections):
            plan.append(SectionPlanItem(
                state_id=int(state_id),
                name=str(section.get("name", f"State_{state_id}")),
                bars=int(section.get("length", 1)),
                source=section.get("source"),
                pitch_offset=int(section.get("pitch_offset", 0) or 0),
                cadence=str(section.get("cadence", "none")),
                start_degree=section.get("start_degree"),
            ))
        return plan
