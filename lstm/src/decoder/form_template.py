#!/usr/bin/env python3
"""Form templates and lightweight form metadata models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from common.config_loader import ConfigView


@dataclass(frozen=True)
class FormSection:
    name: str
    length: int
    source: Optional[str] = None
    pitch_offset: int = 0
    cadence: str = "none"
    start_degree: Optional[str] = None


@dataclass(frozen=True)
class FormTemplate:
    name: str
    sections: List[FormSection]


class FormTemplateLibrary:
    """Load form section templates from style config."""

    def __init__(self, templates: Dict[str, FormTemplate]) -> None:
        self.templates = templates

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "FormTemplateLibrary":
        forms = ConfigView(config).section("forms")
        templates = {}
        for form_name, payload in forms.items():
            sections = [
                FormSection(
                    name=str(item["name"]),
                    length=int(item["length"]),
                    source=item.get("source"),
                    pitch_offset=int(item.get("pitch_offset", 0) or 0),
                    cadence=str(item.get("cadence", "none")),
                    start_degree=item.get("start_degree"),
                )
                for item in payload.get("sections", [])
            ]
            templates[str(form_name)] = FormTemplate(str(form_name), sections)
        return cls(templates)

    def require(self, form_name: str) -> FormTemplate:
        if form_name not in self.templates:
            raise ValueError(f"Unknown form '{form_name}'. Available: {sorted(self.templates)}")
        return self.templates[form_name]


class TemplateFormModel:
    """Non-probabilistic form metadata model used by template/graph decoders."""

    def __init__(self, name: str, sections: List[FormSection], n_observations: int) -> None:
        self.name = str(name)
        self.n_states = len(sections)
        self.n_observations = int(n_observations)
        self.section_lengths = [int(section.length) for section in sections]
        self.state_role_map = {index: section.name for index, section in enumerate(sections)}
        self.emissionprob = np.full(
            (max(1, self.n_states), max(1, self.n_observations)),
            1.0 / max(1, self.n_observations),
            dtype=np.float64,
        )
        self.diagnostics = {
            "model_type": "template_form_model",
            "learned_sequence_model": False,
        }

    @classmethod
    def from_template(cls, template: FormTemplate, n_observations: int) -> "TemplateFormModel":
        return cls(template.name, list(template.sections), int(n_observations))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TemplateFormModel":
        sections = [
            FormSection(
                name=str(item.get("name", f"State_{index}")),
                length=int(item.get("length", 1)),
                source=item.get("source"),
                pitch_offset=int(item.get("pitch_offset", 0) or 0),
                cadence=str(item.get("cadence", "none")),
                start_degree=item.get("start_degree"),
            )
            for index, item in enumerate(payload.get("sections", []))
        ]
        if not sections:
            sections = [
                FormSection(
                    name=str(payload.get("state_role_map", {}).get(str(index), f"State_{index}")),
                    length=int(length),
                )
                for index, length in enumerate(payload.get("section_lengths", []))
            ]
        return cls(
            str(payload.get("name", "")),
            sections,
            int(payload.get("n_observations", 0)),
        )

    def sample_from_state(self, state: int, rng: np.random.Generator) -> tuple[int, float]:
        probs = self.emissionprob[int(state)]
        observation = int(rng.choice(len(probs), p=probs))
        return observation, float(probs[observation])

    def to_dict(self) -> Dict[str, Any]:
        sections = []
        for index, length in enumerate(self.section_lengths):
            sections.append({
                "name": self.state_role_map.get(index, f"State_{index}"),
                "length": int(length),
            })
        return {
            "model_type": "template_form_model",
            "name": self.name,
            "n_states": int(self.n_states),
            "n_observations": int(self.n_observations),
            "section_lengths": [int(value) for value in self.section_lengths],
            "state_role_map": {str(key): value for key, value in self.state_role_map.items()},
            "sections": sections,
            "diagnostics": dict(self.diagnostics),
        }
