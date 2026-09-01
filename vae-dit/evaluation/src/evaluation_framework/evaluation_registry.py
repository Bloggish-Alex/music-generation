"""Flat registry for evaluation modules."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

from .evaluation_api import EvaluationModule


class EvaluationModuleRegistry:
    """Registry changes only when a new module is explicitly registered."""

    def __init__(self) -> None:
        self._modules: Dict[str, EvaluationModule] = {}

    def register(self, module: EvaluationModule) -> None:
        if module.test_point in self._modules:
            raise ValueError(f"Evaluation module already registered: {module.test_point}")
        self._modules[module.test_point] = module

    def get(self, test_point: str) -> EvaluationModule:
        try:
            return self._modules[test_point]
        except KeyError as error:
            raise KeyError(f"Unknown evaluation module: {test_point}") from error

    def select(self, requested: Sequence[str] | str) -> list[EvaluationModule]:
        if requested == "all":
            return [self._modules[name] for name in sorted(self._modules)]
        names = [name for name in requested if name]
        return [self.get(name) for name in names]

    def names(self) -> list[str]:
        return sorted(self._modules)


DEFAULT_MODULE_REGISTRY = EvaluationModuleRegistry()


def _register_builtin_modules() -> None:
    # Imports stay at the registry boundary so individual evaluators do not
    # mutate global state as a side effect of being imported by tests.
    from .evaluation_trajectory_teacher_forced import TRAJECTORY_TEACHER_FORCED_MODULE
    from .evaluation_trajectory_reference_frame import TRAJECTORY_REFERENCE_FRAME_MODULE
    from .evaluation_trajectory_history_swap import TRAJECTORY_HISTORY_SWAP_MODULE
    from .evaluation_trajectory_one_step_oracle import TRAJECTORY_ONE_STEP_ORACLE_MODULE
    from .evaluation_trajectory_rollout import TRAJECTORY_ROLLOUT_MODULE
    from .evaluation_dataset_tonality import DATASET_TONALITY_MODULE
    from .evaluation_anchor_transport import ANCHOR_TRANSPORT_MODULE
    from .evaluation_codec_fidelity import CODEC_FIDELITY_MODULE
    from .evaluation_dvae_fidelity import DVAE_FIDELITY_MODULE
    from .evaluation_dvae_pitch_diagnostics import (
        DVAE_PITCH_GRADIENT_PROBE_MODULE,
        DVAE_PITCH_SUPERVISION_AUDIT_MODULE,
    )
    from .evaluation_latent_probe import LATENT_PROBE_MODULE
    from .evaluation_trajectory_anchor_context import TRAJECTORY_ANCHOR_CONTEXT_MODULE
    from .evaluation_physical_trajectory_objective import PHYSICAL_TRAJECTORY_OBJECTIVE_MODULE
    from .evaluation_renderer_consistency import RENDERER_CONSISTENCY_MODULE
    from .evaluation_attribution import ATTRIBUTION_MODULE
    from .evaluation_oracle_ladder import ORACLE_LADDER_MODULE

    DEFAULT_MODULE_REGISTRY.register(DATASET_TONALITY_MODULE)
    DEFAULT_MODULE_REGISTRY.register(ANCHOR_TRANSPORT_MODULE)
    DEFAULT_MODULE_REGISTRY.register(CODEC_FIDELITY_MODULE)
    DEFAULT_MODULE_REGISTRY.register(DVAE_FIDELITY_MODULE)
    DEFAULT_MODULE_REGISTRY.register(DVAE_PITCH_SUPERVISION_AUDIT_MODULE)
    DEFAULT_MODULE_REGISTRY.register(DVAE_PITCH_GRADIENT_PROBE_MODULE)
    DEFAULT_MODULE_REGISTRY.register(LATENT_PROBE_MODULE)
    DEFAULT_MODULE_REGISTRY.register(TRAJECTORY_ANCHOR_CONTEXT_MODULE)
    DEFAULT_MODULE_REGISTRY.register(PHYSICAL_TRAJECTORY_OBJECTIVE_MODULE)
    DEFAULT_MODULE_REGISTRY.register(TRAJECTORY_HISTORY_SWAP_MODULE)
    DEFAULT_MODULE_REGISTRY.register(TRAJECTORY_ONE_STEP_ORACLE_MODULE)
    DEFAULT_MODULE_REGISTRY.register(TRAJECTORY_REFERENCE_FRAME_MODULE)
    DEFAULT_MODULE_REGISTRY.register(TRAJECTORY_TEACHER_FORCED_MODULE)
    DEFAULT_MODULE_REGISTRY.register(TRAJECTORY_ROLLOUT_MODULE)
    DEFAULT_MODULE_REGISTRY.register(RENDERER_CONSISTENCY_MODULE)
    DEFAULT_MODULE_REGISTRY.register(ATTRIBUTION_MODULE)
    DEFAULT_MODULE_REGISTRY.register(ORACLE_LADDER_MODULE)


_register_builtin_modules()
