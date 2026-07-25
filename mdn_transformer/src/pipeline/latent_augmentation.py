#!/usr/bin/env python3
"""Training-only latent-space augmentation for sequence samples."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Tuple

import numpy as np

from common.config_loader import ConfigView


@dataclass(frozen=True)
class LatentAugmentationConfig:
    """Configuration for latent training augmentation."""

    enabled: bool = True
    mixup_ratio: float = 0.25
    primary_weight: float = 0.7
    same_action: bool = True
    same_position: bool = True
    random_seed_offset: int = 1009

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LatentAugmentationConfig":
        """Build augmentation config from style config."""
        section = ConfigView(config).section("latent_augmentation")
        return cls(
            enabled=bool(section.get("enabled", True)),
            mixup_ratio=float(section.get("mixup_ratio", 0.25)),
            primary_weight=float(section.get("primary_weight", 0.7)),
            same_action=bool(section.get("same_action", True)),
            same_position=bool(section.get("same_position", True)),
            random_seed_offset=int(section.get("random_seed_offset", 1009)),
        )


@dataclass
class LatentAugmentationResult:
    """Augmented dataset and diagnostics."""

    dataset: Any
    diagnostics: Dict[str, Any]


class LatentTrainingAugmenter:
    """Apply latent mixup to training samples without touching validation data."""

    def __init__(self, config: LatentAugmentationConfig, random_seed: int) -> None:
        self.config = config
        self.random_seed = int(random_seed)

    @classmethod
    def from_config(cls, config: Dict[str, Any], random_seed: int) -> "LatentTrainingAugmenter":
        """Build augmenter from global style config."""
        return cls(LatentAugmentationConfig.from_config(config), random_seed=random_seed)

    def augment_dataset(self, dataset: Any) -> LatentAugmentationResult:
        """Return a dataset of the same class with augmented training samples appended."""
        samples = list(getattr(dataset, "samples", []))
        if not self.config.enabled:
            return LatentAugmentationResult(dataset=dataset, diagnostics=self._diagnostics(samples, [], {}))
        synthetic = self._mixup(samples)
        augmented_dataset = dataset.__class__([*samples, *synthetic])
        return LatentAugmentationResult(
            dataset=augmented_dataset,
            diagnostics=self._diagnostics(samples, synthetic, self._group_counts(samples)),
        )

    def _mixup(self, samples: List[Any]) -> List[Any]:
        """Create same-condition latent Lerp samples."""
        groups = self._groups(samples)
        usable_keys = [key for key, members in groups.items() if len(members) >= 2]
        if not usable_keys:
            return []
        rng = np.random.default_rng(self.random_seed + int(self.config.random_seed_offset))
        target_count = max(0, int(round(len(samples) * max(0.0, float(self.config.mixup_ratio)))))
        weights = np.asarray([len(groups[key]) for key in usable_keys], dtype=np.float64)
        weights = weights / weights.sum()
        alpha = float(np.clip(self.config.primary_weight, 0.0, 1.0))
        beta = 1.0 - alpha
        synthetic: List[Any] = []
        for index in range(target_count):
            key = usable_keys[int(rng.choice(len(usable_keys), p=weights))]
            members = groups[key]
            pair_indices = rng.choice(len(members), size=2, replace=False)
            primary = members[int(pair_indices[0])]
            secondary = members[int(pair_indices[1])]
            synthetic.append(self._lerp_sample(primary, secondary, alpha, beta, index))
        return synthetic

    def _lerp_sample(self, primary: Any, secondary: Any, alpha: float, beta: float, index: int) -> Any:
        """Blend latent context and target while keeping primary discrete conditions."""
        context_mu = (
            alpha * np.asarray(primary.context_mu, dtype=np.float32)
            + beta * np.asarray(secondary.context_mu, dtype=np.float32)
        ).astype(np.float32)
        target_mu = (
            alpha * np.asarray(primary.target_mu, dtype=np.float32)
            + beta * np.asarray(secondary.target_mu, dtype=np.float32)
        ).astype(np.float32)
        updates = {
            "context_mu": context_mu,
            "target_mu": target_mu,
            "song_id": f"{primary.song_id}__latent_mixup_{index}",
            "base_song_id": f"{primary.base_song_id}__latent_mixup",
            "target_bar_index": -1,
        }
        if getattr(primary, "theme_embedding", None) is not None and getattr(secondary, "theme_embedding", None) is not None:
            updates["theme_embedding"] = (
                alpha * np.asarray(primary.theme_embedding, dtype=np.float32)
                + beta * np.asarray(secondary.theme_embedding, dtype=np.float32)
            ).astype(np.float32)
        if getattr(primary, "theme_tokens", None) is not None and getattr(secondary, "theme_tokens", None) is not None:
            updates["theme_tokens"] = (
                alpha * np.asarray(primary.theme_tokens, dtype=np.float32)
                + beta * np.asarray(secondary.theme_tokens, dtype=np.float32)
            ).astype(np.float32)
        return replace(
            primary,
            **updates,
        )

    def _groups(self, samples: List[Any]) -> Dict[Tuple[int, int], List[Any]]:
        """Group samples by the configured discrete condition key."""
        groups: Dict[Tuple[int, int], List[Any]] = {}
        for sample in samples:
            action = int(sample.target_action_id) if self.config.same_action else -1
            position = int(sample.target_position_id) if self.config.same_position else -1
            groups.setdefault((action, position), []).append(sample)
        return groups

    def _group_counts(self, samples: List[Any]) -> Dict[str, int]:
        """Return JSON-safe group-size diagnostics."""
        return {
            f"action_{key[0]}__position_{key[1]}": int(len(value))
            for key, value in sorted(self._groups(samples).items())
        }

    def _diagnostics(self, original: List[Any], synthetic: List[Any], group_counts: Dict[str, int]) -> Dict[str, Any]:
        """Return augmentation diagnostics."""
        usable_group_count = int(sum(1 for value in group_counts.values() if value >= 2))
        return {
            "enabled": bool(self.config.enabled),
            "strategy": "same_condition_lerp_mixup",
            "original_count": int(len(original)),
            "synthetic_count": int(len(synthetic)),
            "output_count": int(len(original) + len(synthetic)),
            "mixup_ratio": float(self.config.mixup_ratio),
            "primary_weight": float(self.config.primary_weight),
            "secondary_weight": float(1.0 - self.config.primary_weight),
            "same_action": bool(self.config.same_action),
            "same_position": bool(self.config.same_position),
            "group_count": int(len(group_counts)),
            "usable_group_count": usable_group_count,
            "group_counts": group_counts,
        }
