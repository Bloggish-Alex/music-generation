#!/usr/bin/env python3
"""Note-generation pipeline stages for HierarchicalGenerator.

This module keeps the long per-bar note pipeline out of
``hierarchical_generator.py`` while preserving the same method body and
self-based helper calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from harmonic_planner import HarmonicPlanner
from hierarchical_planning import _stable_hash
from hierarchical_types import NoteEvent, ThemeIdentity, ThemeSkeleton
from rhythm_development import RhythmMemory, RhythmMotifModel, RhythmVariation

VARIATION_PROGRESSION_DENOM = 3

@dataclass
class NotePipelineContext:
    run: Any
    labels: List[int]
    measure_context: List[Tuple[str, int, str, int, int]]
    breathing: List[bool]
    structure_graph: List[Any]
    composition_plan: Any
    harmonic_plan: Dict[int, Any]
    base_seed: int
    time_signature: Tuple[int, int]
    enable_variation: bool
    all_notes: List[List[NoteEvent]] = field(default_factory=list)
    occurrence_count: Dict[str, int] = field(default_factory=dict)
    motif_memory: Dict[str, List[List[NoteEvent]]] = field(default_factory=dict)
    rhythm_memory: RhythmMemory = field(default_factory=RhythmMemory)
    theme_identities: Dict[str, ThemeIdentity] = field(default_factory=dict)
    theme_skeletons: Dict[str, ThemeSkeleton] = field(default_factory=dict)
    recent_rhythm_cells: List[Tuple[float, ...]] = field(default_factory=list)
    return_variation_plans: Dict[int, List[str]] = field(default_factory=dict)
    melodic_skeleton: Dict[int, int] = field(default_factory=dict)
    previous_melody_pitch: Optional[int] = None
    select_transform: Any = None
    apply_variation: Any = None


@dataclass
class NoteBarState:
    i: int
    cluster_id: int
    skip_remaining: bool = False
    sl: str = ""
    bi: int = 0
    role: str = ""
    occurrence_id: int = 0
    section_len: int = 0
    is_end: bool = False
    phrase_role: str = ""
    target_pitch: Optional[int] = None
    affect: Dict[str, Any] = field(default_factory=dict)
    bar_target: Any = None
    rhythm_target: Any = None
    measure_seed: int = 0
    notes: List[NoteEvent] = field(default_factory=list)
    reused_motif: bool = False
    generated_new_theme: bool = False


class NoteGenerationStage:
    name = ""

    def run(
        self,
        generator: "HierarchicalNotePipelineMixin",
        ctx: NotePipelineContext,
        bar: NoteBarState,
    ) -> None:
        raise NotImplementedError


class BarContextStage(NoteGenerationStage):
    name = "bar_context"

    def run(self, generator: "HierarchicalNotePipelineMixin", ctx: NotePipelineContext, bar: NoteBarState) -> None:
        bar.sl, bar.bi, bar.role, bar.occurrence_id, bar.section_len = ctx.measure_context[bar.i]
        bar.is_end = ctx.breathing[bar.i]
        bar.phrase_role = generator._phrase_role(bar.bi, bar.section_len, bar.role)
        bar.target_pitch = ctx.melodic_skeleton.get(bar.i)
        bar.affect = dict(ctx.composition_plan.measure_affects.get(bar.i) or {})
        harmony = ctx.harmonic_plan.get(bar.i)
        if harmony is not None:
            bar.affect["harmony"] = harmony.to_dict()
        ctx.composition_plan.measure_affects[bar.i] = bar.affect
        if bar.role in ("FREE", "FLAT"):
            bar.measure_seed = _stable_hash(ctx.base_seed, "FREE", bar.i)
        else:
            bar.measure_seed = _stable_hash(ctx.base_seed, bar.sl, bar.bi)


class DualThemeTargetStage(NoteGenerationStage):
    name = "dual_theme_target"

    def run(self, generator: "HierarchicalNotePipelineMixin", ctx: NotePipelineContext, bar: NoteBarState) -> None:
        dual_theme = generator.dual_theme_development.target_for_bar(
            label=bar.sl,
            local_bar=bar.bi,
            section_len=bar.section_len,
            narrative_tension=float(bar.affect.get("narrative_tension", bar.affect.get("tension", 0.35))),
            narrative_intensity=float(bar.affect.get("narrative_intensity", bar.affect.get("intensity", 0.35))),
            theme_identities=ctx.theme_identities,
            theme_skeletons=ctx.theme_skeletons,
        )
        if dual_theme is not None:
            bar.affect["dual_theme"] = dual_theme
        ctx.composition_plan.measure_affects[bar.i] = bar.affect


class BarTargetStage(NoteGenerationStage):
    name = "bar_target"

    def run(self, generator: "HierarchicalNotePipelineMixin", ctx: NotePipelineContext, bar: NoteBarState) -> None:
        bar.bar_target = generator._build_bar_generation_target(
            bar_index=bar.i,
            label=bar.sl,
            local_bar=bar.bi,
            target_pitch=bar.target_pitch,
            affect=bar.affect,
            structure_graph=ctx.structure_graph,
            theme_skeletons=ctx.theme_skeletons,
            composition_plan=ctx.composition_plan,
        )


class RhythmTargetStage(NoteGenerationStage):
    name = "rhythm_target"

    def run(self, generator: "HierarchicalNotePipelineMixin", ctx: NotePipelineContext, bar: NoteBarState) -> None:
        bar.rhythm_target = generator.rhythm_phrase_planner.plan(
            label=bar.sl,
            local_bar=bar.bi,
            section_role=bar.role,
            phrase_role=bar.phrase_role,
            narrative_role=str(bar.affect.get("narrative_role", "")),
            tension=float(bar.affect.get("narrative_tension", bar.affect.get("tension", 0.35))),
            source_cell=ctx.rhythm_memory.get(bar.sl, bar.bi),
        )


class MotifReuseOrGenerateStage(NoteGenerationStage):
    name = "motif_reuse_or_generate"

    def run(self, generator: "HierarchicalNotePipelineMixin", ctx: NotePipelineContext, bar: NoteBarState) -> None:
        if bar.role not in ("FREE", "FLAT", "NEW") and bar.sl in ctx.motif_memory and bar.bi < len(ctx.motif_memory[bar.sl]):
            bar.notes = generator._clone_notes(ctx.motif_memory[bar.sl][bar.bi])
            bar.reused_motif = True
            return
        bar.notes = generator._generate_scored_measure(
            cluster_label=bar.cluster_id,
            time_signature=ctx.time_signature,
            is_section_end=bar.is_end,
            phrase_role=bar.phrase_role,
            target_pitch=bar.target_pitch,
            previous_pitch=ctx.previous_melody_pitch,
            affect=bar.affect,
            bar_target=bar.bar_target,
            seed=bar.measure_seed,
            bar_index=bar.i,
            rhythm_target=bar.rhythm_target,
            recent_rhythm_cells=ctx.recent_rhythm_cells,
        )
        bar.generated_new_theme = bar.role == "NEW"


class NewThemeMemoryStage(NoteGenerationStage):
    name = "new_theme_memory"

    def run(self, generator: "HierarchicalNotePipelineMixin", ctx: NotePipelineContext, bar: NoteBarState) -> None:
        if not bar.generated_new_theme:
            return
        ctx.motif_memory.setdefault(bar.sl, [])
        while len(ctx.motif_memory[bar.sl]) <= bar.bi:
            ctx.motif_memory[bar.sl].append([])
        ctx.motif_memory[bar.sl][bar.bi] = generator._clone_notes(bar.notes)
        ctx.rhythm_memory.remember(bar.sl, bar.bi, bar.notes)
        generator._update_theme_identity(
            ctx.theme_identities, bar.sl, ctx.motif_memory[bar.sl], bar.section_len,
        )
        generator._update_theme_skeleton(
            ctx.theme_skeletons, bar.sl, ctx.motif_memory[bar.sl],
            bar.section_len, ctx.composition_plan.global_tonic_pc,
        )


class ControlledVariationStage(NoteGenerationStage):
    name = "controlled_variation"

    def run(self, generator: "HierarchicalNotePipelineMixin", ctx: NotePipelineContext, bar: NoteBarState) -> None:
        if (
            ctx.select_transform is None
            or ctx.apply_variation is None
            or bar.role in ("FREE", "FLAT", "NEW", "REPEAT")
        ):
            return
        if bar.bi == 0:
            k = ctx.occurrence_count.get(bar.sl, 0)
            if k > 0:
                sec_labels = ctx.labels[bar.i:bar.i + bar.section_len]
                variation_rng = np.random.RandomState(
                    _stable_hash(ctx.base_seed, bar.sl, bar.occurrence_id, "variation")
                )
                s = (generator._section_entropy(sec_labels) / generator._max_entropy) * (
                    k / (k + VARIATION_PROGRESSION_DENOM)
                )
                generator._current_variation_profile = ctx.select_transform(
                    generator.model.clusterer.centroids,
                    sec_labels,
                    s,
                    rng=variation_rng,
                ) if s > 0.02 else None
            else:
                generator._current_variation_profile = None
        if generator._current_variation_profile:
            variation_rng = np.random.RandomState(
                _stable_hash(ctx.base_seed, bar.sl, bar.occurrence_id, bar.bi, "variation")
            )
            bar.notes = ctx.apply_variation(bar.notes, generator._current_variation_profile, rng=variation_rng)


class ReturnMotifCandidatesStage(NoteGenerationStage):
    name = "return_motif_candidates"

    def run(self, generator: "HierarchicalNotePipelineMixin", ctx: NotePipelineContext, bar: NoteBarState) -> None:
        generator._run_return_motif_candidates(ctx, bar)


class OccurrenceTrackingStage(NoteGenerationStage):
    name = "occurrence_tracking"

    def run(self, generator: "HierarchicalNotePipelineMixin", ctx: NotePipelineContext, bar: NoteBarState) -> None:
        if bar.role not in ("FREE", "FLAT") and bar.bi == 0:
            ctx.occurrence_count[bar.sl] = ctx.occurrence_count.get(bar.sl, 0) + 1


class AppendBarStateStage(NoteGenerationStage):
    name = "append_bar_state"

    def run(self, generator: "HierarchicalNotePipelineMixin", ctx: NotePipelineContext, bar: NoteBarState) -> None:
        ctx.all_notes.append(bar.notes)
        ctx.recent_rhythm_cells.append(RhythmMotifModel.cell(bar.notes))
        ctx.recent_rhythm_cells = ctx.recent_rhythm_cells[-4:]
        ctx.previous_melody_pitch = generator._last_melody_pitch(bar.notes, ctx.previous_melody_pitch)


NOTE_GENERATION_CHAIN: Tuple[NoteGenerationStage, ...] = (
    BarContextStage(),
    DualThemeTargetStage(),
    BarTargetStage(),
    RhythmTargetStage(),
    MotifReuseOrGenerateStage(),
    NewThemeMemoryStage(),
    ControlledVariationStage(),
    ReturnMotifCandidatesStage(),
    OccurrenceTrackingStage(),
    AppendBarStateStage(),
)
NOTE_GENERATION_MODULES: Tuple[str, ...] = tuple(stage.name for stage in NOTE_GENERATION_CHAIN)
DEFAULT_NOTE_GENERATION_MODULES: Tuple[str, ...] = NOTE_GENERATION_MODULES
DEFAULT_NOTE_GENERATION_STOP_AFTER: Optional[str] = None


class HierarchicalNotePipelineMixin:
    def _run_note_generation_module(self, run: "MidiGenerationRun") -> None:
        ctx = self._make_note_pipeline_context(run)
        selected_note_modules = self._resolve_note_generation_modules(
            run.note_modules,
            run.note_stop_after_module
            if run.note_stop_after_module is not None
            else DEFAULT_NOTE_GENERATION_STOP_AFTER,
        )
        self._last_note_generation_modules = selected_note_modules
        chain = self._note_generation_chain(selected_note_modules)

        for i, cluster_id in enumerate(ctx.labels):
            bar = NoteBarState(i=i, cluster_id=cluster_id)
            for stage in chain:
                before = ctx.run.trace.before_note_stage(bar)
                stage.run(self, ctx, bar)
                after = ctx.run.trace.after_note_stage(bar)
                ctx.run.trace.record_note_stage(bar, stage.name, before, after)
                if bar.skip_remaining:
                    break

        run.all_notes = ctx.all_notes

    def _make_note_pipeline_context(self, run: "MidiGenerationRun") -> NotePipelineContext:
        ctx = NotePipelineContext(
            run=run,
            labels=run.labels,
            measure_context=run.measure_context,
            breathing=run.breathing,
            structure_graph=run.structure_graph,
            composition_plan=run.composition_plan,
            harmonic_plan=run.harmonic_plan,
            base_seed=run.base_seed,
            time_signature=run.time_signature,
            enable_variation=run.enable_variation,
        )
        self._last_dual_theme_scores: Dict[int, Dict[str, Any]] = {}
        self._last_repeat_harmony_scores: Dict[int, Dict[str, Any]] = {}
        self._last_candidate_score_components: Dict[int, Dict[str, float]] = {}
        self._last_candidate_reranker_scores: Dict[int, Dict[str, Any]] = {}
        self._last_rhythm_scores: Dict[int, Dict[str, float]] = {}
        ctx.melodic_skeleton = self._build_melodic_skeleton(
            ctx.labels, ctx.measure_context, ctx.base_seed, ctx.composition_plan,
        )
        if ctx.enable_variation:
            from note_transform import select_transforms, apply_variation
            ctx.select_transform = select_transforms
            ctx.apply_variation = apply_variation
        return ctx

    def _note_generation_chain(self, selected_modules: List[str]) -> List[NoteGenerationStage]:
        stages = {stage.name: stage for stage in NOTE_GENERATION_CHAIN}
        return [stages[name] for name in selected_modules]

    def _run_return_motif_candidates(self, ctx: NotePipelineContext, bar: NoteBarState) -> None:
        if not (
            bar.reused_motif
            and ctx.enable_variation
            and bar.role in ("RETURN", "VARIANT", "REPEAT")
        ):
            return

        motif_cfg = self.config.get("motif_return", {})
        base_strength = float(motif_cfg.get("base_strength", 0.10))
        occurrence_growth = float(motif_cfg.get("occurrence_growth", 0.06))
        max_strength = float(motif_cfg.get("max_strength", 0.38))
        scorer_cfg = self.config.get("development_scorer", {})
        candidate_count = int(scorer_cfg.get("candidate_count", 5))
        scorer_weights = scorer_cfg.get("weights", {}) if isinstance(scorer_cfg, dict) else {}
        harmony_return_weight = float(
            scorer_weights.get("harmony", 1.15)
            if isinstance(scorer_weights, dict) else 1.15
        )
        candidate_count = self._return_candidate_count(
            candidate_count,
            bar.bar_target,
            bar.affect,
        )
        if bar.role == "REPEAT":
            base_strength *= float(motif_cfg.get("repeat_strength_scale", 0.55))
            occurrence_growth *= float(motif_cfg.get("repeat_growth_scale", 0.35))

        k = ctx.occurrence_count.get(bar.sl, 1)
        plan = ctx.return_variation_plans.setdefault(
            bar.occurrence_id,
            self._build_return_variation_plan(
                bar.section_len,
                bar.role,
                np.random.RandomState(
                    _stable_hash(ctx.base_seed, bar.sl, bar.occurrence_id, "return-plan")
                ),
            ),
        )
        variation_mode = plan[min(bar.bi, len(plan) - 1)] if plan else "CONTOUR"
        identity = ctx.theme_identities.get(bar.sl)
        source_notes = self._clone_notes(bar.notes)
        partner_notes = self._partner_theme_notes(
            ctx.motif_memory,
            bar.bar_target.dual_theme,
            bar.bi,
        )
        previous_notes = ctx.all_notes[-1] if ctx.all_notes else None
        proposal_count = self.dual_theme_proposal.count(bar.bar_target, partner_notes)
        repeat_harmony_count = self.repeat_harmony_proposal.count(
            bar.role,
            bar.bar_target,
            partner_notes,
        )
        base_candidate_count = max(1, candidate_count)
        total_candidate_count = base_candidate_count + proposal_count + repeat_harmony_count
        strength = min(
            max_strength,
            (base_strength + occurrence_growth * k) * bar.bar_target.development_strength,
        )

        best_notes: Optional[List[NoteEvent]] = None
        best_dual_theme_diag: Optional[Dict[str, float]] = None
        best_early_repeat_diag: Optional[Dict[str, float]] = None
        best_rhythm_diag: Optional[Dict[str, float]] = None
        best_score_components: Optional[Dict[str, float]] = None
        best_candidate_index: Optional[int] = None
        best_proposal_kind: Optional[str] = None
        best_repeat_harmony_kind: Optional[str] = None
        best_score = -1e9

        for candidate_i in range(total_candidate_count):
            variation_rng = np.random.RandomState(
                _stable_hash(
                    ctx.base_seed,
                    bar.sl,
                    bar.occurrence_id,
                    bar.bi,
                    "motif-return",
                    candidate_i,
                )
            )
            candidate_strength = strength * float(variation_rng.uniform(0.85, 1.15))
            candidate_notes = self._vary_return_motif(
                source_notes,
                strength=candidate_strength,
                rng=variation_rng,
                target_pitch=bar.target_pitch,
                variation_mode=variation_mode,
                phrase_role=bar.phrase_role,
                development_role=bar.bar_target.development_role,
                target_attraction=bar.bar_target.target_attraction,
                rhythm_change_scale=bar.bar_target.rhythm_change_scale,
            )
            repeat_harmony_index = None
            if candidate_i >= base_candidate_count and candidate_i < base_candidate_count + proposal_count:
                proposal_index = candidate_i - base_candidate_count
                candidate_notes = self.dual_theme_proposal.propose(
                    candidate_notes,
                    source_notes=source_notes,
                    partner_notes=partner_notes,
                    target=bar.bar_target,
                    rng=variation_rng,
                    proposal_index=proposal_index,
                )
            elif candidate_i >= base_candidate_count + proposal_count:
                proposal_index = None
                repeat_harmony_index = candidate_i - base_candidate_count - proposal_count
                candidate_notes = self.repeat_harmony_proposal.propose(
                    candidate_notes,
                    target=bar.bar_target,
                    rng=variation_rng,
                    proposal_index=repeat_harmony_index,
                )
            else:
                proposal_index = None

            if identity is not None and bar.bar_target.development_role not in ("FRAGMENT", "INTENSIFY"):
                candidate_notes = self._apply_theme_identity(
                    candidate_notes,
                    identity=identity,
                    bar_index=bar.bi,
                    section_len=bar.section_len,
                    target_pitch=bar.target_pitch,
                    rng=variation_rng,
                )
            if bar.rhythm_target is not None:
                candidate_notes = RhythmVariation.apply(
                    candidate_notes,
                    bar.rhythm_target,
                    variation_rng,
                    bar_length=float(ctx.time_signature[0]) * (4.0 / float(ctx.time_signature[1])),
                )
            candidate_notes = self._fit_notes_to_bar_target(
                candidate_notes,
                bar.bar_target,
                variation_rng,
            )
            score, score_components, dual_theme_diag, early_repeat_diag, rhythm_diag = (
                self._score_return_candidate(
                    candidate_notes,
                    bar,
                    source_notes=source_notes,
                    partner_notes=partner_notes,
                    previous_notes=previous_notes,
                    previous_cells=ctx.recent_rhythm_cells,
                    harmony_return_weight=harmony_return_weight,
                    proposal_kind=(
                        self.dual_theme_proposal.kind(proposal_index)
                        if proposal_index is not None else None
                    ) or (
                        self.repeat_harmony_proposal.kind(repeat_harmony_index)
                        if repeat_harmony_index is not None else None
                    ),
                )
            )
            if score > best_score:
                best_score = score
                best_notes = candidate_notes
                best_dual_theme_diag = dual_theme_diag
                best_early_repeat_diag = early_repeat_diag
                best_rhythm_diag = rhythm_diag
                best_score_components = score_components
                best_candidate_index = candidate_i
                best_proposal_kind = (
                    self.dual_theme_proposal.kind(proposal_index)
                    if proposal_index is not None else None
                )
                best_repeat_harmony_kind = (
                    self.repeat_harmony_proposal.kind(repeat_harmony_index)
                    if repeat_harmony_index is not None else None
                )

        if best_notes is not None:
            best_notes, best_score_components, best_dual_theme_diag, best_early_repeat_diag, best_rhythm_diag = (
                self._maybe_accept_harmony_repair(
                    best_notes,
                    best_score,
                    best_score_components,
                    best_dual_theme_diag,
                    best_early_repeat_diag,
                    best_rhythm_diag,
                    bar,
                    source_notes=source_notes,
                    partner_notes=partner_notes,
                    previous_notes=previous_notes,
                    previous_cells=ctx.recent_rhythm_cells,
                    harmony_return_weight=harmony_return_weight,
                )
            )
            self._record_return_candidate_diagnostics(
                bar.i,
                total_candidate_count,
                proposal_count,
                repeat_harmony_count,
                base_candidate_count,
                partner_notes,
                best_candidate_index,
                best_proposal_kind,
                best_repeat_harmony_kind,
                best_dual_theme_diag,
                best_early_repeat_diag,
                best_rhythm_diag,
                best_score_components,
            )
            bar.notes = best_notes

    def _return_candidate_count(
        self,
        candidate_count: int,
        bar_target: Any,
        affect: Dict[str, Any],
    ) -> int:
        harmony_cfg = self.config.get("harmony", {})
        if not (isinstance(harmony_cfg, dict) and harmony_cfg.get("enabled", False) and bar_target.harmony):
            return candidate_count
        multiplier = float(harmony_cfg.get("return_candidate_count_multiplier", 1.6))
        if bar_target.cadence_strength > 0.5 or bar_target.harmony.get("cadence_role") == "CADENCE":
            multiplier = float(harmony_cfg.get("cadence_return_candidate_count_multiplier", 2.2))
        candidate_count = max(candidate_count, int(round(candidate_count * multiplier)))
        boost_cfg = harmony_cfg.get("weak_context_candidate_boost", {})
        if isinstance(boost_cfg, dict) and boost_cfg.get("enabled", True):
            boosted_roles = set(str(x) for x in boost_cfg.get(
                "narrative_roles",
                ["DEVELOPMENT", "CLIMAX", "RECAP"],
            ))
            if str(affect.get("narrative_role", "")) in boosted_roles:
                boost = float(boost_cfg.get("multiplier", 1.6))
                max_candidates = int(boost_cfg.get("max_candidates", 24))
                candidate_count = min(
                    max_candidates,
                    max(candidate_count, int(round(candidate_count * boost))),
                )
        return candidate_count

    def _score_return_candidate(
        self,
        candidate_notes: List[NoteEvent],
        bar: NoteBarState,
        *,
        source_notes: List[NoteEvent],
        partner_notes: Optional[List[NoteEvent]],
        previous_notes: Optional[List[NoteEvent]],
        previous_cells: List[Tuple[float, ...]],
        harmony_return_weight: float,
        proposal_kind: Optional[str],
    ) -> Tuple[float, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        rhythm_diag = self.rhythm_scorer.score(
            candidate_notes,
            bar.rhythm_target,
            previous_cells=previous_cells,
        )
        rhythm_score = float(rhythm_diag.get("score", 0.0))
        development_score = self.development_scorer.score(
            candidate_notes,
            bar.bar_target,
            bar.cluster_id,
            source_notes=source_notes,
            previous_notes=previous_notes,
        ).total
        harmony_score = HarmonicPlanner.score_melody(
            candidate_notes,
            bar.bar_target.harmony,
            self.config,
        )
        conditional_score = self._conditional_return_score(candidate_notes, bar)
        dual_theme_diag = self.dual_theme_scorer.diagnostics(
            candidate_notes,
            source_notes=source_notes,
            partner_notes=partner_notes,
            target=bar.bar_target,
        )
        raw_dual_theme_score = float(dual_theme_diag.get("score", 0.0))
        dual_theme_damping = self._dual_theme_harmony_damping(harmony_score)
        dual_theme_score = raw_dual_theme_score * dual_theme_damping
        early_repeat_diag = self.early_repeat_scorer.diagnostics(
            candidate_notes,
            role=bar.role,
            target=bar.bar_target,
            partner_notes=partner_notes,
            config=self.config,
        )
        early_repeat_score = float(early_repeat_diag.get("score", 0.0))
        base_score = (
            development_score
            + rhythm_score
            + harmony_return_weight * harmony_score
            + conditional_score
            + dual_theme_score
            + early_repeat_score
        )
        score_components = {
            "base_score": float(base_score),
            "development_score": float(development_score),
            "rhythm_score": float(rhythm_score),
            "harmony_score": float(harmony_score),
            "harmony_weight": float(harmony_return_weight),
            "weighted_harmony_score": float(harmony_return_weight * harmony_score),
            "conditional_score": float(conditional_score),
            "dual_theme_score": float(dual_theme_score),
            "raw_dual_theme_score": float(raw_dual_theme_score),
            "dual_theme_harmony_damping": float(dual_theme_damping),
            "early_repeat_score": float(early_repeat_score),
        }
        reranker_score = self._score_with_candidate_reranker(
            candidate_notes,
            bar.bar_target,
            bar.cluster_id,
            source_notes=source_notes,
            partner_notes=partner_notes,
            score_components=score_components,
            proposal_kind=proposal_kind,
        )
        score = base_score + reranker_score.weighted
        score_components.update({
            "total_score": float(score),
            "reranker_probability": float(reranker_score.probability),
            "reranker_logit": float(reranker_score.logit),
            "raw_reranker_probability": float(reranker_score.raw_probability),
            "raw_reranker_logit": float(reranker_score.raw_logit),
            "calibrated_reranker_probability": float(reranker_score.calibrated_probability),
            "calibrated_reranker_logit": float(reranker_score.calibrated_logit),
            "reranker_calibration_adjustment": float(reranker_score.calibration_adjustment),
            "reranker_good_cadence_confidence": float(reranker_score.good_cadence_confidence),
            "reranker_weighted_score": float(reranker_score.weighted),
        })
        return score, score_components, dual_theme_diag, early_repeat_diag, rhythm_diag

    def _conditional_return_score(self, notes: List[NoteEvent], bar: NoteBarState) -> float:
        conditional_model = getattr(self.model, "conditional_note_model", None)
        conditional_cfg = self.config.get("conditional_note_model", {})
        if (
            conditional_model is None
            or not isinstance(conditional_cfg, dict)
            or not conditional_cfg.get("enabled", True)
        ):
            return 0.0
        return float(conditional_cfg.get("return_score_weight", 0.85)) * (
            conditional_model.score_candidate(
                notes,
                bar.bar_target,
                bar.cluster_id,
                self.config,
            )
        )

    def _maybe_accept_harmony_repair(
        self,
        best_notes: List[NoteEvent],
        best_score: float,
        best_score_components: Optional[Dict[str, float]],
        best_dual_theme_diag: Optional[Dict[str, float]],
        best_early_repeat_diag: Optional[Dict[str, float]],
        best_rhythm_diag: Optional[Dict[str, float]],
        bar: NoteBarState,
        *,
        source_notes: List[NoteEvent],
        partner_notes: Optional[List[NoteEvent]],
        previous_notes: Optional[List[NoteEvent]],
        previous_cells: List[Tuple[float, ...]],
        harmony_return_weight: float,
    ) -> Tuple[
        List[NoteEvent],
        Optional[Dict[str, float]],
        Optional[Dict[str, float]],
        Optional[Dict[str, float]],
        Optional[Dict[str, float]],
    ]:
        repaired_notes = self._repair_harmony_mismatch(best_notes, bar.bar_target)
        if repaired_notes is best_notes:
            return best_notes, best_score_components, best_dual_theme_diag, best_early_repeat_diag, best_rhythm_diag
        repaired_score, repaired_components, repaired_dual, repaired_early, repaired_rhythm = (
            self._score_return_candidate(
                repaired_notes,
                bar,
                source_notes=source_notes,
                partner_notes=partner_notes,
                previous_notes=previous_notes,
                previous_cells=previous_cells,
                harmony_return_weight=harmony_return_weight,
                proposal_kind="adaptive_repair",
            )
        )
        if repaired_score <= best_score:
            return best_notes, best_score_components, best_dual_theme_diag, best_early_repeat_diag, best_rhythm_diag
        return repaired_notes, repaired_components, repaired_dual, repaired_early, repaired_rhythm

    def _record_return_candidate_diagnostics(
        self,
        bar_index: int,
        total_candidate_count: int,
        proposal_count: int,
        repeat_harmony_count: int,
        base_candidate_count: int,
        partner_notes: Optional[List[NoteEvent]],
        best_candidate_index: Optional[int],
        best_proposal_kind: Optional[str],
        best_repeat_harmony_kind: Optional[str],
        best_dual_theme_diag: Optional[Dict[str, float]],
        best_early_repeat_diag: Optional[Dict[str, float]],
        best_rhythm_diag: Optional[Dict[str, float]],
        best_score_components: Optional[Dict[str, float]],
    ) -> None:
        self._last_dual_theme_scores[bar_index] = {
            "candidate_count": int(total_candidate_count),
            "proposal_count": int(proposal_count),
            "partner_available": bool(partner_notes),
            "selected_candidate_index": best_candidate_index,
            "selected_from_proposal": (
                best_candidate_index is not None
                and best_candidate_index >= base_candidate_count
                and best_candidate_index < base_candidate_count + proposal_count
            ),
            "selected_proposal_kind": best_proposal_kind,
            "selected_score": float(
                best_dual_theme_diag.get("score", 0.0)
                if best_dual_theme_diag is not None else 0.0
            ),
            "selected_components": best_dual_theme_diag or {"active": 0.0, "score": 0.0},
            "score_components": best_score_components,
        }
        self._last_repeat_harmony_scores[bar_index] = {
            "candidate_count": int(total_candidate_count),
            "proposal_count": int(repeat_harmony_count),
            "selected_candidate_index": best_candidate_index,
            "selected_from_proposal": (
                best_candidate_index is not None
                and best_candidate_index >= base_candidate_count + proposal_count
            ),
            "selected_proposal_kind": best_repeat_harmony_kind,
            "selected_score": float(
                best_early_repeat_diag.get("score", 0.0)
                if best_early_repeat_diag is not None else 0.0
            ),
            "selected_components": best_early_repeat_diag or {"active": 0.0, "score": 0.0},
            "score_components": best_score_components,
        }
        if best_score_components is None:
            return
        self._last_candidate_score_components[bar_index] = best_score_components
        if best_rhythm_diag is not None:
            self._last_rhythm_scores[bar_index] = best_rhythm_diag
        self._last_candidate_reranker_scores[bar_index] = {
            "enabled": bool(self.config.get("candidate_reranker", {}).get("enabled", True))
            if isinstance(self.config.get("candidate_reranker", {}), dict) else True,
            "model_available": bool(
                self.candidate_reranker is not None
                and getattr(self.candidate_reranker, "available", False)
            ),
            "probability": best_score_components.get("reranker_probability"),
            "logit": best_score_components.get("reranker_logit"),
            "raw_probability": best_score_components.get("raw_reranker_probability"),
            "raw_logit": best_score_components.get("raw_reranker_logit"),
            "calibration_adjustment": best_score_components.get("reranker_calibration_adjustment"),
            "good_cadence_confidence": best_score_components.get("reranker_good_cadence_confidence"),
            "weighted_score": best_score_components.get("reranker_weighted_score"),
        }

