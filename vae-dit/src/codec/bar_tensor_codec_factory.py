#!/usr/bin/env python3
"""Factory for selectable bar tensor codec backends."""

from __future__ import annotations

from typing import Any, Dict

from common.config_loader import ConfigView
from codec.bar_tensor_codec import BarTensorCodec
from codec.semantic_bar_tensor_codec import SemanticBarTensorCodec
from codec.semantic_harmony_set_codec import SemanticHarmonySetCodec


class BarTensorCodecFactory:
    """Create the configured bar tensor codec."""

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """Return a codec with an encode(bar) method."""
        section = ConfigView(config).section("bar_tensor")
        backend = str(section.get("backend", "legacy_physical")).strip().lower()
        if backend in {"legacy", "legacy_physical", "physical"}:
            return BarTensorCodec.from_config(config)
        if backend in {"semantic_3voice", "semantic", "melody_harmony_bass"}:
            return SemanticBarTensorCodec.from_config(config)
        if backend == "semantic_harmony_set_v2":
            return SemanticHarmonySetCodec.from_config(config)
        raise ValueError(f"Unsupported bar_tensor.backend: {backend}")
