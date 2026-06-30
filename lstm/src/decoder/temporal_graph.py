#!/usr/bin/env python3
"""Interpretable temporal theme graph planning contracts.

This module intentionally contains data structures and minimal planning
interfaces only. It does not implement graph mining or generation logic yet.
The goal is to keep the next decoder stage centered on explicit temporal
actions and opaque codebook token IDs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from common.architecture import SymbolID


class TemporalAction(str, Enum):
    """Human-readable musical development actions."""

    INTRODUCE = "INTRODUCE"
    REPEAT = "REPEAT"
    VARY = "VARY"
    VARY_LIGHTLY = "VARY_LIGHTLY"
    SEQUENCE = "SEQUENCE"
    CONTRAST = "CONTRAST"
    DEVELOP = "DEVELOP"
    RETURN = "RETURN"
    CADENCE = "CADENCE"
    CADENCE_STRONG = "CADENCE_STRONG"


class MaterialRole(str, Enum):
    """Role of a material node in the theme graph."""

    THEME_SEED = "THEME_SEED"
    REPEAT = "REPEAT"
    VARIATION = "VARIATION"
    CONTRAST = "CONTRAST"
    DEVELOPMENT = "DEVELOPMENT"
    SEQUENCE = "SEQUENCE"
    TRANSITION = "TRANSITION"
    CADENCE = "CADENCE"
    RETURN = "RETURN"


@dataclass(frozen=True)
class MaterialNode:
    """Interpretable music material group used by the graph planner.

    A material node is a semantic group, not necessarily one concrete bar.
    Concrete generation still happens through `candidate_token_ids`.
    """

    material_id: str
    role: MaterialRole
    owner_theme: Optional[str] = None
    candidate_token_ids: List[SymbolID] = field(default_factory=list)
    prototype_token_ids: List[SymbolID] = field(default_factory=list)
    tags: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["role"] = self.role.value
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MaterialNode":
        return cls(
            material_id=str(payload["material_id"]),
            role=MaterialRole(str(payload["role"])),
            owner_theme=payload.get("owner_theme"),
            candidate_token_ids=[int(item) for item in payload.get("candidate_token_ids", [])],
            prototype_token_ids=[int(item) for item in payload.get("prototype_token_ids", [])],
            tags=dict(payload.get("tags", {})),
            diagnostics=dict(payload.get("diagnostics", {})),
        )


@dataclass(frozen=True)
class ThemeAnchor:
    """Remembered anchor for a theme introduced during generation."""

    theme_id: str
    seed_material_id: str
    seed_token_ids: List[SymbolID] = field(default_factory=list)
    cadence_material_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ThemeAnchor":
        return cls(
            theme_id=str(payload["theme_id"]),
            seed_material_id=str(payload["seed_material_id"]),
            seed_token_ids=[int(item) for item in payload.get("seed_token_ids", [])],
            cadence_material_id=payload.get("cadence_material_id"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class ThemeMemory:
    """Runtime memory for one generated piece.

    The graph planner uses this to know what has been introduced, varied, and
    should later return.
    """

    theme_anchors: Dict[str, ThemeAnchor] = field(default_factory=dict)
    used_material_ids: List[str] = field(default_factory=list)
    recent_token_ids: List[SymbolID] = field(default_factory=list)
    recent_material_ids: List[str] = field(default_factory=list)
    return_targets: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def remember_theme(self, anchor: ThemeAnchor) -> None:
        self.theme_anchors[str(anchor.theme_id)] = anchor

    def remember_selection(
        self,
        material_id: str,
        token_id: SymbolID,
        max_recent: int = 16,
    ) -> None:
        self.used_material_ids.append(str(material_id))
        self.recent_material_ids.append(str(material_id))
        self.recent_token_ids.append(int(token_id))
        if max_recent > 0:
            self.recent_material_ids = self.recent_material_ids[-max_recent:]
            self.recent_token_ids = self.recent_token_ids[-max_recent:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theme_anchors": {
                key: anchor.to_dict()
                for key, anchor in sorted(self.theme_anchors.items())
            },
            "used_material_ids": list(self.used_material_ids),
            "recent_token_ids": [int(item) for item in self.recent_token_ids],
            "recent_material_ids": list(self.recent_material_ids),
            "return_targets": dict(self.return_targets),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TemporalActionState:
    """Current temporal planning state for one output bar."""

    action: TemporalAction
    section: str
    section_local_index: int
    section_length: int
    phrase_position: Optional[int] = None
    target_theme_id: Optional[str] = None
    target_cadence: Optional[str] = None
    harmony_degree: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        return payload


@dataclass(frozen=True)
class TemporalActionSpan:
    """Configured action span inside a form section."""

    section: str
    action: TemporalAction
    bars: int
    theme: Optional[str] = None
    target_cadence: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        return payload


@dataclass(frozen=True)
class TemporalGraphTemplate:
    """Form-specific temporal graph prior loaded from style config."""

    form_name: str
    material_policy: Dict[TemporalAction, List[MaterialRole]]
    action_script: Dict[str, List[TemporalActionSpan]]
    theme_memory_config: Dict[str, Any] = field(default_factory=dict)

    def action_state_for(
        self,
        section: str,
        section_local_index: int,
        section_length: int,
        harmony_degree: Optional[str] = None,
    ) -> TemporalActionState:
        spans = self.action_script.get(section, [])
        cursor = 0
        for span in spans:
            next_cursor = cursor + max(0, int(span.bars))
            if int(section_local_index) < next_cursor:
                return TemporalActionState(
                    action=span.action,
                    section=section,
                    section_local_index=int(section_local_index),
                    section_length=int(section_length),
                    phrase_position=int(section_local_index),
                    target_theme_id=span.theme,
                    target_cadence=span.target_cadence,
                    harmony_degree=harmony_degree,
                    metadata={
                        **span.metadata,
                        "form": self.form_name,
                        "span_start": cursor,
                        "span_end": next_cursor,
                    },
                )
            cursor = next_cursor
        fallback_action = TemporalAction.CADENCE if section_local_index >= section_length - 2 else TemporalAction.DEVELOP
        return TemporalActionState(
            action=fallback_action,
            section=section,
            section_local_index=int(section_local_index),
            section_length=int(section_length),
            phrase_position=int(section_local_index),
            harmony_degree=harmony_degree,
            metadata={"form": self.form_name, "source": "temporal_graph_fallback"},
        )

    def roles_for_action(self, action: TemporalAction) -> List[MaterialRole]:
        return list(self.material_policy.get(action, []))

    @classmethod
    def from_config(cls, form_name: str, config: Dict[str, Any]) -> "TemporalGraphTemplate":
        templates = config.get("temporal_graph_templates", {})
        templates = templates if isinstance(templates, dict) else {}
        global_policy = _parse_material_policy(templates.get("material_policy", {}))
        form_payload = templates.get(form_name, {})
        form_payload = form_payload if isinstance(form_payload, dict) else {}
        form_policy = _parse_material_policy(form_payload.get("material_policy", {}))
        material_policy = {**global_policy, **form_policy}
        return cls(
            form_name=str(form_name),
            material_policy=material_policy,
            action_script=_parse_action_script(form_payload.get("action_script", {})),
            theme_memory_config=dict(form_payload.get("theme_memory", {})),
        )


@dataclass(frozen=True)
class CandidateTokenMask:
    """Allowed token IDs proposed by the graph planner."""

    allowed_token_ids: List[SymbolID]
    material_ids: List[str] = field(default_factory=list)
    reason: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def contains(self, token_id: SymbolID) -> bool:
        return int(token_id) in set(int(item) for item in self.allowed_token_ids)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalPlanningStep:
    """One graph-planning decision before LSTM scoring."""

    state: TemporalActionState
    mask: CandidateTokenMask
    material_nodes: List[MaterialNode] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "mask": self.mask.to_dict(),
            "material_nodes": [node.to_dict() for node in self.material_nodes],
        }


class TemporalGraphPlanner(Protocol):
    """Planner that turns action state and memory into a token mask."""

    def plan_next(
        self,
        state: TemporalActionState,
        memory: ThemeMemory,
    ) -> TemporalPlanningStep:
        ...


class StaticTemporalGraphPlanner:
    """Small deterministic planner useful for tests and first integration.

    It maps actions to material roles and expands matching material nodes into
    allowed token IDs. More advanced planners can replace this without changing
    the LSTM interface.
    """

    DEFAULT_ACTION_ROLE_MAP: Dict[TemporalAction, List[MaterialRole]] = {
        TemporalAction.INTRODUCE: [MaterialRole.THEME_SEED],
        TemporalAction.REPEAT: [MaterialRole.REPEAT, MaterialRole.THEME_SEED],
        TemporalAction.VARY: [MaterialRole.VARIATION],
        TemporalAction.VARY_LIGHTLY: [MaterialRole.VARIATION, MaterialRole.REPEAT],
        TemporalAction.SEQUENCE: [MaterialRole.SEQUENCE, MaterialRole.DEVELOPMENT],
        TemporalAction.CONTRAST: [MaterialRole.CONTRAST],
        TemporalAction.DEVELOP: [MaterialRole.DEVELOPMENT, MaterialRole.SEQUENCE],
        TemporalAction.RETURN: [MaterialRole.RETURN, MaterialRole.THEME_SEED],
        TemporalAction.CADENCE: [MaterialRole.CADENCE],
        TemporalAction.CADENCE_STRONG: [MaterialRole.CADENCE],
    }

    def __init__(
        self,
        material_nodes: Sequence[MaterialNode],
        template: Optional[TemporalGraphTemplate] = None,
    ) -> None:
        self.material_nodes = list(material_nodes)
        self.template = template

    def plan_next(
        self,
        state: TemporalActionState,
        memory: ThemeMemory,
    ) -> TemporalPlanningStep:
        roles = set(
            self.template.roles_for_action(state.action)
            if self.template is not None
            else self.DEFAULT_ACTION_ROLE_MAP.get(state.action, [])
        )
        nodes = [
            node for node in self.material_nodes
            if node.role in roles and self._matches_theme(node, state)
        ]
        allowed = sorted({
            int(token_id)
            for node in nodes
            for token_id in node.candidate_token_ids
        })
        return TemporalPlanningStep(
            state=state,
            mask=CandidateTokenMask(
                allowed_token_ids=allowed,
                material_ids=[node.material_id for node in nodes],
                reason=f"action={state.action.value}",
                diagnostics={
                    "node_count": len(nodes),
                    "memory_recent_token_count": len(memory.recent_token_ids),
                },
            ),
            material_nodes=nodes,
        )

    def _matches_theme(self, node: MaterialNode, state: TemporalActionState) -> bool:
        if state.target_theme_id is None:
            return True
        return node.owner_theme in {None, state.target_theme_id}


@dataclass(frozen=True)
class RuntimeThemeGraphConfig:
    """Runtime knobs for the first Theme Graph decoder integration."""

    enabled: bool = False
    candidate_top_k: int = 32
    min_candidates: int = 8
    memory_window: int = 16
    memory_bias: float = 2.0
    anchor_span_bars: int = 4
    aligned_anchor_bias: float = 4.0
    same_theme_memory_bias: float = 1.5
    memory_density_filter_enabled: bool = True
    memory_density_note_on_tolerance: int = 4
    latent_trajectory_enabled: bool = True
    prototype_blend_weight: Dict[str, float] = field(default_factory=dict)
    movement_blend_weight: Dict[str, float] = field(default_factory=dict)
    decode_with_planned_latent: bool = True
    continuation_rerank_enabled: bool = True
    continuation_weight: float = 0.20
    continuation_sigma: float = 3.0
    prototype_rerank_weight: float = 0.35
    prototype_rerank_sigma: float = 2.5
    density_rerank_weight: float = 0.08
    density_rerank_sigma: float = 4.0
    planning_enabled: bool = True
    planning_window_bars: int = 4
    planning_beam_width: int = 4
    planning_branch_top_k: int = 8
    planning_path_temperature: float = 0.75
    action_temperature: Dict[str, float] = field(default_factory=dict)
    diagnostics_top_k: int = 8

    @classmethod
    def from_decoder_config(cls, decoder_config: Dict[str, Any]) -> "RuntimeThemeGraphConfig":
        section = decoder_config.get("temporal_graph", {})
        section = section if isinstance(section, dict) else {}
        action_temperature = section.get("action_temperature", {})
        action_temperature = action_temperature if isinstance(action_temperature, Mapping) else {}
        latent_section = section.get("latent_trajectory", {})
        latent_section = latent_section if isinstance(latent_section, Mapping) else {}
        prototype_blend = latent_section.get("prototype_blend_weight", {})
        prototype_blend = prototype_blend if isinstance(prototype_blend, Mapping) else {}
        movement_blend = latent_section.get("movement_blend_weight", {})
        movement_blend = movement_blend if isinstance(movement_blend, Mapping) else {}
        return cls(
            enabled=bool(section.get("enabled", False)),
            candidate_top_k=int(section.get("candidate_top_k", 32)),
            min_candidates=int(section.get("min_candidates", 8)),
            memory_window=int(section.get("memory_window", 16)),
            memory_bias=float(section.get("memory_bias", 2.0)),
            anchor_span_bars=int(section.get("anchor_span_bars", 4)),
            aligned_anchor_bias=float(section.get("aligned_anchor_bias", section.get("memory_bias", 2.0))),
            same_theme_memory_bias=float(section.get("same_theme_memory_bias", 1.5)),
            memory_density_filter_enabled=bool(section.get("memory_density_filter_enabled", True)),
            memory_density_note_on_tolerance=int(section.get("memory_density_note_on_tolerance", 4)),
            latent_trajectory_enabled=bool(latent_section.get("enabled", True)),
            prototype_blend_weight={str(key): float(value) for key, value in prototype_blend.items()},
            movement_blend_weight={str(key): float(value) for key, value in movement_blend.items()},
            decode_with_planned_latent=bool(latent_section.get("decode_with_planned_latent", True)),
            continuation_rerank_enabled=bool(latent_section.get("continuation_rerank_enabled", True)),
            continuation_weight=float(latent_section.get("continuation_weight", 0.20)),
            continuation_sigma=float(latent_section.get("continuation_sigma", 3.0)),
            prototype_rerank_weight=float(latent_section.get("prototype_rerank_weight", 0.35)),
            prototype_rerank_sigma=float(latent_section.get("prototype_rerank_sigma", 2.5)),
            density_rerank_weight=float(latent_section.get("density_rerank_weight", 0.08)),
            density_rerank_sigma=float(latent_section.get("density_rerank_sigma", 4.0)),
            planning_enabled=bool(section.get("planning_enabled", True)),
            planning_window_bars=int(section.get("planning_window_bars", 4)),
            planning_beam_width=int(section.get("planning_beam_width", 4)),
            planning_branch_top_k=int(section.get("planning_branch_top_k", 8)),
            planning_path_temperature=float(section.get("planning_path_temperature", action_temperature.get("default", 0.75))),
            action_temperature={
                str(key): float(value)
                for key, value in action_temperature.items()
            },
            diagnostics_top_k=int(section.get("diagnostics_top_k", 8)),
        )

    def temperature_for_action(self, action: TemporalAction | str, fallback: float = 0.75) -> float:
        action_name = action.value if isinstance(action, TemporalAction) else str(action)
        value = self.action_temperature.get(action_name, self.action_temperature.get("default", fallback))
        return max(1.0e-6, float(value))

    def prototype_weight_for_action(self, action: TemporalAction | str, fallback: float = 0.25) -> float:
        action_name = action.value if isinstance(action, TemporalAction) else str(action)
        value = self.prototype_blend_weight.get(action_name, self.prototype_blend_weight.get("default", fallback))
        return min(1.0, max(0.0, float(value)))

    def movement_weight_for_action(self, action: TemporalAction | str, fallback: float = 0.10) -> float:
        action_name = action.value if isinstance(action, TemporalAction) else str(action)
        value = self.movement_blend_weight.get(action_name, self.movement_blend_weight.get("default", fallback))
        return min(1.0, max(0.0, float(value)))


class RuntimeThemeGraphController:
    """Stateful graph planner used during one generation run.

    This first runtime does not mine material nodes. Instead, it uses explicit
    temporal actions plus remembered theme token sequences to produce a dynamic
    mask over the global codebook vocabulary. The LSTM still ranks the masked
    candidates.
    """

    MEMORY_ACTIONS = {
        TemporalAction.REPEAT,
        TemporalAction.VARY,
        TemporalAction.VARY_LIGHTLY,
        TemporalAction.RETURN,
    }
    INTRODUCTION_ACTIONS = {
        TemporalAction.INTRODUCE,
        TemporalAction.CONTRAST,
    }

    def __init__(
        self,
        template: TemporalGraphTemplate,
        vocab_size: int,
        config: RuntimeThemeGraphConfig,
        token_density: Optional[Dict[SymbolID, Dict[str, Any]]] = None,
    ) -> None:
        self.template = template
        self.vocab_size = int(vocab_size)
        self.config = config
        self.token_density = {int(key): dict(value) for key, value in (token_density or {}).items()}
        self.memory = ThemeMemory(metadata={"theme_sequences": {}})

    def action_state(
        self,
        section: str,
        section_local_index: int,
        section_length: int,
        harmony_degree: Optional[str] = None,
    ) -> TemporalActionState:
        return self.template.action_state_for(
            section=section,
            section_local_index=int(section_local_index),
            section_length=int(section_length),
            harmony_degree=harmony_degree,
        )

    def plan_from_logits(
        self,
        state: TemporalActionState,
        logits: Sequence[float],
    ) -> TemporalPlanningStep:
        top_ids = self._top_ids(logits, max(1, self.config.candidate_top_k))
        memory_ids = self._memory_candidate_ids(state)
        aligned_anchor = self.aligned_anchor_token_id(state)
        bias_map = self.memory_biases_for_state(state)
        allowed = self._allowed_ids(state, top_ids, memory_ids)
        density_diag = self._density_filter_diagnostics(state)
        return TemporalPlanningStep(
            state=state,
            mask=CandidateTokenMask(
                allowed_token_ids=allowed,
                material_ids=[],
                reason=f"temporal_graph:{state.action.value}",
                diagnostics={
                    "candidate_source": "theme_memory_plus_lstm_top_k",
                    "top_k_count": len(top_ids),
                    "memory_candidate_count": len(memory_ids),
                    "allowed_count": len(allowed),
                    **density_diag,
                    "memory_bias": float(self.config.memory_bias),
                    "aligned_anchor_bias": float(self.config.aligned_anchor_bias),
                    "same_theme_memory_bias": float(self.config.same_theme_memory_bias),
                    "anchor_span_bars": int(self.config.anchor_span_bars),
                    "aligned_anchor_token_id": int(aligned_anchor) if aligned_anchor is not None else None,
                    "biased_token_count": len(bias_map),
                    "theme_memory": self.memory.to_dict(),
                    "top_lstm_token_ids": top_ids[: max(0, self.config.diagnostics_top_k)],
                    "memory_token_ids": memory_ids[: max(0, self.config.diagnostics_top_k)],
                    "biased_token_ids": [
                        {"token_id": int(token_id), "bias": float(bias)}
                        for token_id, bias in sorted(
                            bias_map.items(),
                            key=lambda item: float(item[1]),
                            reverse=True,
                        )[: max(0, self.config.diagnostics_top_k)]
                    ],
                },
            ),
            material_nodes=[],
        )

    def memory_token_ids_for_state(self, state: TemporalActionState) -> List[SymbolID]:
        return self._memory_candidate_ids(state)

    def aligned_anchor_token_id(self, state: TemporalActionState) -> Optional[SymbolID]:
        values = self._density_filtered_theme_sequence(state)
        if not values:
            return None
        index = self._anchor_index(state, len(values))
        return int(values[index])

    def memory_biases_for_state(self, state: TemporalActionState) -> Dict[SymbolID, float]:
        if state.action not in self.MEMORY_ACTIONS:
            return {}
        values = self._theme_sequence(state)
        if not values:
            return {}
        result = {
            int(token_id): float(self.config.same_theme_memory_bias)
            for token_id in values
        }
        aligned = self.aligned_anchor_token_id(state)
        if aligned is not None:
            result[int(aligned)] = max(
                float(result.get(int(aligned), 0.0)),
                float(self.config.aligned_anchor_bias),
            )
        return result

    def remember_selection(
        self,
        state: TemporalActionState,
        token_id: SymbolID,
        latent_vector: Optional[Sequence[float]] = None,
        token_density: Optional[Dict[str, Any]] = None,
    ) -> None:
        theme_id = state.target_theme_id
        if not theme_id:
            return
        sequences = self.memory.metadata.setdefault("theme_sequences", {})
        density_profiles = self.memory.metadata.setdefault("theme_density_profiles", {})
        latent_profiles = self.memory.metadata.setdefault("theme_latent_profiles", {})
        sequence = sequences.setdefault(str(theme_id), [])
        if (
            state.action in self.INTRODUCTION_ACTIONS
            and int(state.section_local_index) < max(1, int(self.config.anchor_span_bars))
        ) or not sequence:
            sequence.append(int(token_id))
            self._remember_theme_density(density_profiles, str(theme_id), int(token_id))
            self._remember_theme_latent(latent_profiles, str(theme_id), int(token_id), latent_vector, token_density)
            if len(sequence) > max(1, self.config.memory_window):
                del sequence[:-max(1, self.config.memory_window)]
            if str(theme_id) not in self.memory.theme_anchors:
                self.memory.remember_theme(ThemeAnchor(
                    theme_id=str(theme_id),
                    seed_material_id=f"{theme_id}:runtime_anchor",
                    seed_token_ids=[int(token_id)],
                    metadata={
                        "introduced_in": state.section,
                        "section_local_index": int(state.section_local_index),
                        "density_profile": self._theme_density_profile(str(theme_id)),
                        "latent_profile": self._theme_latent_profile(str(theme_id)),
                    },
                ))
        self.memory.remember_selection(
            material_id=f"{theme_id}:{state.action.value}",
            token_id=int(token_id),
            max_recent=max(1, self.config.memory_window),
        )

    def planned_latent_for_selection(
        self,
        state: TemporalActionState,
        token_id: SymbolID,
        selected_latent: Optional[Sequence[float]],
        previous_latent: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        if not self.config.latent_trajectory_enabled or selected_latent is None:
            return {
                "enabled": bool(self.config.latent_trajectory_enabled),
                "used": False,
                "reason": "disabled_or_missing_selected_latent",
            }
        selected = [float(value) for value in selected_latent]
        prototype = self.prototype_latent_for_state(state)
        proto_weight = self.config.prototype_weight_for_action(state.action)
        movement_weight = self.config.movement_weight_for_action(state.action)
        planned = list(selected)
        used_sources = ["selected_token_latent"]
        if state.action in self.MEMORY_ACTIONS and prototype is not None and proto_weight > 0.0:
            planned = self._blend_vectors(planned, prototype, proto_weight)
            used_sources.append("theme_prototype")
        movement = self.prototype_movement_for_state(state)
        if state.action in self.MEMORY_ACTIONS and previous_latent is not None and movement is not None and movement_weight > 0.0:
            moved = [
                float(previous_latent[index]) + float(movement[index])
                for index in range(min(len(previous_latent), len(movement)))
            ]
            planned = self._blend_vectors(planned, moved, movement_weight)
            used_sources.append("theme_prototype_movement")
        return {
            "enabled": True,
            "used": bool(len(used_sources) > 1 and self.config.decode_with_planned_latent),
            "decode_with_planned_latent": bool(self.config.decode_with_planned_latent),
            "token_id": int(token_id),
            "action": state.action.value,
            "target_theme_id": state.target_theme_id,
            "prototype_weight": float(proto_weight),
            "movement_weight": float(movement_weight),
            "has_prototype": prototype is not None,
            "has_movement": movement is not None,
            "source": "+".join(used_sources),
            "latent_vector": [float(value) for value in planned],
            "selected_latent_norm": self._vector_norm(selected),
            "planned_latent_norm": self._vector_norm(planned),
            "prototype_distance": self._vector_distance(selected, prototype) if prototype is not None else None,
        }

    def prototype_latent_for_state(self, state: TemporalActionState) -> Optional[List[float]]:
        if state.target_theme_id is None:
            return None
        profile = self._theme_latent_profile(str(state.target_theme_id))
        vectors = profile.get("latent_vectors", [])
        if not isinstance(vectors, list) or not vectors:
            return None
        index = self._anchor_index(state, len(vectors))
        vector = vectors[index]
        return [float(value) for value in vector] if isinstance(vector, list) else None

    def prototype_movement_for_state(self, state: TemporalActionState) -> Optional[List[float]]:
        if state.target_theme_id is None:
            return None
        profile = self._theme_latent_profile(str(state.target_theme_id))
        vectors = profile.get("latent_vectors", [])
        if not isinstance(vectors, list) or len(vectors) < 2:
            return None
        index = self._anchor_index(state, len(vectors))
        previous_index = max(0, index - 1)
        current = vectors[index]
        previous = vectors[previous_index]
        if not isinstance(current, list) or not isinstance(previous, list):
            return None
        length = min(len(current), len(previous))
        return [float(current[i]) - float(previous[i]) for i in range(length)]

    def density_target_for_state(self, state: TemporalActionState) -> Optional[float]:
        return self._theme_anchor_note_on_count(state)

    def _allowed_ids(
        self,
        state: TemporalActionState,
        top_ids: Sequence[SymbolID],
        memory_ids: Sequence[SymbolID],
    ) -> List[SymbolID]:
        allowed = []
        if state.action in self.MEMORY_ACTIONS and memory_ids:
            allowed.extend(int(token_id) for token_id in memory_ids)
        allowed.extend(int(token_id) for token_id in top_ids)
        deduped = []
        seen = set()
        for token_id in allowed:
            if 0 <= int(token_id) < self.vocab_size and int(token_id) not in seen:
                deduped.append(int(token_id))
                seen.add(int(token_id))
        if len(deduped) < max(1, self.config.min_candidates):
            for token_id in top_ids:
                if int(token_id) not in seen:
                    deduped.append(int(token_id))
                    seen.add(int(token_id))
                if len(deduped) >= max(1, self.config.min_candidates):
                    break
        return deduped

    def _memory_candidate_ids(self, state: TemporalActionState) -> List[SymbolID]:
        theme_id = state.target_theme_id
        if not theme_id:
            return []
        values = self._density_filtered_theme_sequence(state)
        if not values:
            return []
        if state.action in self.MEMORY_ACTIONS:
            index = self._anchor_index(state, len(values))
            return [values[index], *values]
        return []

    def _theme_sequence(self, state: TemporalActionState) -> List[SymbolID]:
        theme_id = state.target_theme_id
        if not theme_id:
            return []
        sequences = self.memory.metadata.get("theme_sequences", {})
        sequence = sequences.get(str(theme_id), []) if isinstance(sequences, dict) else []
        if not sequence:
            anchor = self.memory.theme_anchors.get(str(theme_id))
            sequence = anchor.seed_token_ids if anchor is not None else []
        return [
            int(token_id)
            for token_id in sequence
            if 0 <= int(token_id) < self.vocab_size
        ]

    def _density_filtered_theme_sequence(self, state: TemporalActionState) -> List[SymbolID]:
        values = self._theme_sequence(state)
        if not values or not self._should_filter_density(state):
            return values
        anchor_count = self._theme_anchor_note_on_count(state)
        if anchor_count is None:
            return values
        tolerance = max(0, int(self.config.memory_density_note_on_tolerance))
        filtered = [
            int(token_id)
            for token_id in values
            if self._token_note_on_count(token_id) is not None
            and abs(float(self._token_note_on_count(token_id)) - float(anchor_count)) <= tolerance
        ]
        return filtered or values

    def _should_filter_density(self, state: TemporalActionState) -> bool:
        return (
            bool(self.config.memory_density_filter_enabled)
            and state.action in {TemporalAction.RETURN, TemporalAction.VARY_LIGHTLY}
            and state.target_theme_id is not None
        )

    def _remember_theme_density(self, density_profiles: Dict[str, Any], theme_id: str, token_id: SymbolID) -> None:
        note_on_count = self._token_note_on_count(token_id)
        if note_on_count is None:
            return
        profile = density_profiles.setdefault(str(theme_id), {"note_on_counts": []})
        counts = profile.setdefault("note_on_counts", [])
        counts.append(int(note_on_count))
        if len(counts) > max(1, int(self.config.anchor_span_bars)):
            del counts[:-max(1, int(self.config.anchor_span_bars))]
        profile["mean_note_on_count"] = float(sum(counts) / max(1, len(counts)))

    def _remember_theme_latent(
        self,
        latent_profiles: Dict[str, Any],
        theme_id: str,
        token_id: SymbolID,
        latent_vector: Optional[Sequence[float]],
        token_density: Optional[Dict[str, Any]],
    ) -> None:
        if latent_vector is None:
            return
        profile = latent_profiles.setdefault(str(theme_id), {"token_ids": [], "latent_vectors": [], "density": []})
        token_ids = profile.setdefault("token_ids", [])
        vectors = profile.setdefault("latent_vectors", [])
        densities = profile.setdefault("density", [])
        token_ids.append(int(token_id))
        vectors.append([float(value) for value in latent_vector])
        densities.append(dict(token_density or {}))
        limit = max(1, int(self.config.anchor_span_bars))
        if len(token_ids) > limit:
            del token_ids[:-limit]
            del vectors[:-limit]
            del densities[:-limit]
        profile["latent_dim"] = int(len(vectors[-1])) if vectors else 0

    def _theme_latent_profile(self, theme_id: str) -> Dict[str, Any]:
        profiles = self.memory.metadata.get("theme_latent_profiles", {})
        profile = profiles.get(str(theme_id), {}) if isinstance(profiles, dict) else {}
        return dict(profile) if isinstance(profile, dict) else {}

    def _theme_density_profile(self, theme_id: str) -> Dict[str, Any]:
        profiles = self.memory.metadata.get("theme_density_profiles", {})
        profile = profiles.get(str(theme_id), {}) if isinstance(profiles, dict) else {}
        return dict(profile) if isinstance(profile, dict) else {}

    def _theme_anchor_note_on_count(self, state: TemporalActionState) -> Optional[float]:
        if state.target_theme_id is None:
            return None
        value = self._theme_density_profile(str(state.target_theme_id)).get("mean_note_on_count")
        return float(value) if value is not None else None

    def _token_note_on_count(self, token_id: SymbolID) -> Optional[int]:
        density = self.token_density.get(int(token_id), {})
        value = density.get("note_on_count")
        return int(value) if value is not None else None

    def _blend_vectors(self, base: Sequence[float], target: Sequence[float], weight: float) -> List[float]:
        length = min(len(base), len(target))
        if length <= 0:
            return [float(value) for value in base]
        blended = [
            (1.0 - float(weight)) * float(base[index]) + float(weight) * float(target[index])
            for index in range(length)
        ]
        if len(base) > length:
            blended.extend(float(value) for value in base[length:])
        return blended

    def _vector_distance(
        self,
        left: Optional[Sequence[float]],
        right: Optional[Sequence[float]],
    ) -> Optional[float]:
        if left is None or right is None:
            return None
        length = min(len(left), len(right))
        if length <= 0:
            return None
        return sum((float(left[index]) - float(right[index])) ** 2 for index in range(length)) ** 0.5

    def _vector_norm(self, vector: Sequence[float]) -> float:
        return sum(float(value) ** 2 for value in vector) ** 0.5

    def _density_filter_diagnostics(self, state: TemporalActionState) -> Dict[str, Any]:
        raw = self._theme_sequence(state)
        filtered = self._density_filtered_theme_sequence(state)
        anchor_count = self._theme_anchor_note_on_count(state)
        filtered_set = set(int(token_id) for token_id in filtered)
        filtered_out = [int(token_id) for token_id in raw if int(token_id) not in filtered_set]
        return {
            "memory_density_filter_enabled": bool(self.config.memory_density_filter_enabled),
            "memory_density_filter_applied": bool(self._should_filter_density(state) and anchor_count is not None),
            "theme_anchor_note_on_count": anchor_count,
            "memory_density_note_on_tolerance": int(self.config.memory_density_note_on_tolerance),
            "memory_density_raw_count": len(raw),
            "memory_density_filtered_count": len(filtered),
            "memory_density_filtered_out_count": len(filtered_out),
            "memory_density_filtered_out_token_ids": filtered_out[: max(0, self.config.diagnostics_top_k)],
        }

    def _anchor_index(self, state: TemporalActionState, length: int) -> int:
        if length <= 0:
            return 0
        span_start = int(state.metadata.get("span_start", 0) or 0)
        local_in_span = max(0, int(state.section_local_index) - span_start)
        return int(local_in_span % length)

    def _top_ids(self, logits: Sequence[float], top_k: int) -> List[SymbolID]:
        pairs = [
            (int(index), float(value))
            for index, value in enumerate(logits)
            if value == value and value not in {float("inf"), -float("inf")}
        ]
        pairs.sort(key=lambda item: item[1], reverse=True)
        return [index for index, _value in pairs[: max(1, int(top_k))]]


def _parse_material_policy(payload: Any) -> Dict[TemporalAction, List[MaterialRole]]:
    if not isinstance(payload, dict):
        return {}
    result: Dict[TemporalAction, List[MaterialRole]] = {}
    for action_name, role_names in payload.items():
        roles = role_names if isinstance(role_names, list) else []
        result[TemporalAction(str(action_name))] = [
            MaterialRole(str(role_name))
            for role_name in roles
        ]
    return result


def _parse_action_script(payload: Any) -> Dict[str, List[TemporalActionSpan]]:
    if not isinstance(payload, dict):
        return {}
    result: Dict[str, List[TemporalActionSpan]] = {}
    for section, items in payload.items():
        spans = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            metadata = {
                key: value
                for key, value in item.items()
                if key not in {"action", "bars", "theme", "target_cadence"}
            }
            spans.append(TemporalActionSpan(
                section=str(section),
                action=TemporalAction(str(item["action"])),
                bars=int(item.get("bars", 1)),
                theme=item.get("theme"),
                target_cadence=item.get("target_cadence"),
                metadata=metadata,
            ))
        result[str(section)] = spans
    return result
