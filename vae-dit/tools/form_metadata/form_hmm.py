#!/usr/bin/env python3
"""Form-template data structures required by metadata classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config_loader import ConfigView


@dataclass(frozen=True)
class FormSection:
    """Describe one named section in a form template.

    Length is measured in template bars. The metadata generator scales these
    relative lengths to the observed number of bars in each source piece.
    """

    name: str
    length: int
    source: Optional[str] = None
    pitch_offset: int = 0
    cadence: str = "none"
    start_degree: Optional[str] = None


@dataclass(frozen=True)
class FormTemplate:
    """Group an ordered list of sections under a form name."""

    name: str
    sections: List[FormSection]


class FormTemplateLibrary:
    """Load and validate the form templates used by the classifier."""

    def __init__(self, templates: Dict[str, FormTemplate]) -> None:
        """Store templates keyed by the exact configured form name."""
        self.templates = templates

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "FormTemplateLibrary":
        """Build templates from the top-level ``forms`` configuration mapping."""
        forms = ConfigView(config).section("forms")
        templates: Dict[str, FormTemplate] = {}
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
        """Return a named template or raise with the available alternatives."""
        if form_name not in self.templates:
            raise ValueError(f"Unknown form '{form_name}'. Available: {sorted(self.templates)}")
        return self.templates[form_name]
