#!/usr/bin/env python3
"""
Hierarchical Generator — three-tier music generation from a trained MusicModel.

Tier 1 (Section):  SectionGrammar → section label sequence + FREE placeholders
Tier 2 (State):    Prototypes fill SECTION blocks; PhraseGenerator fills FREE blocks
Tier 3 (Note):     ClusterNoteSampler generates pitches, durations, velocities
                   per measure from cluster centroids + pitch-class histograms

Usage::

    from music_model import MusicModel
    from hierarchical_generator import HierarchicalGenerator

    model = MusicModel.load("./models/corelli")
    gen = HierarchicalGenerator(model)
    gen.generate_midi("output.mid", target_measures=120, seed=42)

    # With user-specified starting bars:
    gen.generate_midi("output.mid", target_measures=120,
                      start_states=[2, 2, 2, 0, 0], seed=42)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from hierarchical_render import (
    clamp_measure_bounds,
    clamp_overlaps,
    ensure_final_bar_end,
    write_midi,
)
from development_scorer import DevelopmentCandidateScorer
from dual_theme_development import DualThemeDevelopment
from dual_theme_proposal import DualThemeProposal
from dual_theme_scorer import DualThemeCandidateScorer
from early_repeat_scorer import EarlyRepeatCandidateScorer
from harmonic_planner import HarmonicPlanner
from candidate_reranker import CandidateReranker
from hierarchical_planning import HierarchicalPlanningMixin
from narrative_planner import NarrativePlanner
from hierarchical_types import (
    BarGenerationTarget,
    BarSkeleton,
    CompositionPlan,
    NoteEvent,
    SectionAffect,
    StructureEdge,
    ThemeIdentity,
    ThemeSkeleton,
)
from hierarchical_sampler import ClusterNoteSampler
from music_model import MusicModel
from phrase_generator import PhraseGenerator
from repeat_harmony_proposal import RepeatHarmonyProposal
from rhythm_development import (
    RhythmCandidateScorer,
    RhythmMemory,
    RhythmMotifModel,
    RhythmPhrasePlanner,
    RhythmVariation,
)


def _stable_hash(*args: object) -> int:
    """Deterministic hash — same result across Python processes."""
    h = hashlib.md5()
    for a in args:
        h.update(str(a).encode())
    return int(h.hexdigest()[:8], 16)

log = logging.getLogger("hierarchical_generator")

# ---------------------------------------------------------------------------
# Duration categories (quarterLength), same as measure_clustering.py
# ---------------------------------------------------------------------------

_DUR_VALUES: List[float] = [
    4.0, 2.0, 1.0, 0.5, 0.25, 0.125,
    3.0, 1.5, 0.75, 2.0 / 3.0, 1.0 / 3.0,
]

_SHORT_DUR_INDICES = {3, 4, 5, 8, 9, 10}  # <= 0.5 ql

# ---------------------------------------------------------------------------
# Musical Style Parameters
# ---------------------------------------------------------------------------
# These are co-adapted through empirical tuning on the Corelli corpus.
# Changing any single value may require re-tuning others.  Each parameter
# is documented with its musical meaning and the expected range.
#
# To adapt to a new corpus: adjust here, regenerate, evaluate with the
# same metrics (16-bar block similarity, cadence gaps, polyphony).

# ---- breathing / cadence ----

CADENCE_GAP = 0.5          # beats of silence at section boundaries (4/4 eighth-note)

# ---- melody / pitch walk ----

# Step-size distribution: now derived per-cluster from training data.
# See clusterer.step_histograms — computed during MusicModel.fit().
STEP_UPWARD_BIAS = 0.55          # probability of ascending vs descending
REGISTER_WINDOW = 10             # semi-octave range per measure (±10 from centre)
REGISTER_LO = 28                 # MIDI E2 — absolute floor
REGISTER_HI = 96                 # MIDI C7 — absolute ceiling
REGISTER_CENTRE_LO = 40          # MIDI E2 — centre pitch minimum
REGISTER_CENTRE_HI = 84          # MIDI C6 — centre pitch maximum
OCTAVE_WEIGHTS = [3, 3, 4, 4, 4, 5]  # weight toward middle register (octave 4)
PC_REJECT_THRESHOLD = 4.0        # if histogram[pc] * this < random, reduce leap
PC_REJECT_SCALE = 0.5            # how much to reduce rejected leap

# ---- rhythm / note generation ----

NOTE_DENSITY_SCALE = 2.0          # density → raw note count multiplier
ENTROPY_JITTER_SCALE = 1.5        # entropy → standard deviation of note count
MIN_NOTES_PER_MEASURE = 2
MAX_NOTES_PER_MEASURE = 24        # ~6 notes/beat in 4/4
MIN_DURATION = 0.25               # sixteenth note — shortest allowed
MIN_REMAINING = 0.03              # stop generating when less than this remains
DURATION_COUNT_WEIGHTS = [2, 2, 3, 3, 4]  # 2–4 preferred durations per measure
REST_PROBABILITY_SCALE = 0.5      # multiplier on silence_ratio for actual rests
MAX_REST_PROB = 0.6               # cap on rest probability per note

# ---- velocity / dynamics ----

VELOCITY_PEAK_MEAN, VELOCITY_PEAK_STD = 100, 8
VELOCITY_TROUGH_MEAN, VELOCITY_TROUGH_STD = 55, 10
VELOCITY_PEAK_FRAC_MIN, VELOCITY_PEAK_FRAC_MAX = 0.25, 0.55
VELOCITY_JITTER = 5               # Gaussian std added to velocity arc

# ---- offbeat / swing ----

OFFBEAT_SCALE = 0.3               # offbeat_ratio → probability of swing placement
SWING_MIN, SWING_MAX = 0.02, 0.12 # beat_offset jitter range (quarterLength)

# ---- variation / transforms ----

MAX_ENTROPY = 3.5                 # normalizer for centroid entropy (theoretical max)
MIN_VARIATION_STRENGTH = 0.02     # strength below this → skip transforms
VARIATION_PROGRESSION_DENOM = 3   # k/(k+N) curve steepness

# ---- section structure / grid ----

GRID_SIZE = 4                     # snap to this many bars
MAX_GRID_PAD = 3                  # max bars padded to reach grid boundary
MULTI_FAMILY_PREFERENCE = 0.7     # probability of using multi-family template
GRID_WEIGHTS = [7, 2, 1]          # grid-aligned, mostly-grid, irregular template ratio

# ---- MIDI output ----

TICKS_PER_BEAT = 480

# ---------------------------------------------------------------------------
# ClusterNoteSampler
# ---------------------------------------------------------------------------


# Duration categories used by return-motif variation.
_USABLE_DUR_VALUES: List[float] = [4.0, 2.0, 1.0, 0.5, 0.25, 3.0, 1.5, 0.75]

def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``base``."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _config_root() -> Path:
    """Default deployment config root.

    In the final layout the CLI runs from ``bin/`` and config lives in the
    sibling ``config/`` directory, so this resolves to ``../config``.
    """
    return Path.cwd().parent / "config"


def _available_profiles() -> List[str]:
    profile_dir = _config_root() / "profiles"
    if not profile_dir.exists():
        return []
    return sorted(p.stem for p in profile_dir.glob("*.yaml"))


def _load_style_config(path: str | Path | None, profile: str | None = None) -> Dict[str, Any]:
    """Load style parameters from a YAML file, falling back to defaults."""
    import yaml
    config_root = _config_root()
    default_path = config_root / "style_defaults.yaml"
    config: Dict[str, Any] = {}
    if default_path.exists():
        with open(default_path) as f:
            config = yaml.safe_load(f) or {}
    if profile:
        profile_path = config_root / "profiles" / f"{profile.lower()}.yaml"
        if not profile_path.exists():
            available = _available_profiles()
            suffix = f" Available: {', '.join(available)}" if available else ""
            raise ValueError(
                f"Unknown composer profile '{profile}'.{suffix}"
            )
        with open(profile_path) as f:
            _deep_update(config, yaml.safe_load(f) or {})
    if path:
        with open(path) as f:
            _deep_update(config, yaml.safe_load(f) or {})
    return config


class HierarchicalGenerator(HierarchicalPlanningMixin):
    """Three-tier music generator.

    Combines SectionGrammar (macro form), PhraseGenerator (FREE block
    states), and ClusterNoteSampler (per-measure notes) into a single
    end-to-end pipeline.

    Args:
        model: Trained MusicModel.
        config_path: Optional path to a YAML style config file.
            See ``config/style_defaults.yaml`` for available parameters.
            Falls back to the defaults if not provided.
    """

    def __init__(
        self,
        model: MusicModel,
        config_path: str | Path | None = None,
        composer_profile: str | None = None,
    ) -> None:
        self.model = model
        self.phrase_gen = PhraseGenerator(model)

        # Load style config first — needed by sampler
        self.config = _load_style_config(config_path, composer_profile)

        centroids = model.clusterer.centroids
        if centroids is None:
            raise ValueError("Model clusterer has no centroids — is it trained?")

        pitch_hists = getattr(model.clusterer, "pitch_histograms", None)
        step_hists = getattr(model.clusterer, "step_histograms", None)
        bass_hists = getattr(model.clusterer, "bass_histograms", None)
        phrase_role_stats = getattr(model.clusterer, "phrase_role_stats", None)
        self.note_sampler = ClusterNoteSampler(
            centroids, pitch_hists, step_hists, bass_hists,
            phrase_role_stats=phrase_role_stats,
            bass_config=self.config.get("bass", {}),
        )
        self.development_scorer = DevelopmentCandidateScorer(
            step_hists,
            pitch_hists,
            self.config,
        )
        self.harmonic_planner = HarmonicPlanner(self.config)
        self.dual_theme_development = DualThemeDevelopment(self.config)
        self.dual_theme_proposal = DualThemeProposal(self.config)
        self.dual_theme_scorer = DualThemeCandidateScorer(self.config)
        self.early_repeat_scorer = EarlyRepeatCandidateScorer(self.config)
        self.repeat_harmony_proposal = RepeatHarmonyProposal(self.config)
        self.candidate_reranker = getattr(model, "candidate_reranker", None)
        self.rhythm_phrase_planner = RhythmPhrasePlanner(self.config)
        self.rhythm_candidate_prior = getattr(model, "rhythm_candidate_prior", None)
        self.rhythm_scorer = RhythmCandidateScorer(self.config, self.rhythm_candidate_prior)
        self._current_variation_profile: Optional[List] = None
        self._max_entropy = float(centroids[:, 7].max()) if centroids is not None else 3.5

    @property
    def grammar(self):
        return self.model.grammar

    # ------------------------------------------------------------------
    # Timeline generation
    # ------------------------------------------------------------------

    def generate_timeline(
        self,
        target_measures: int,
        start_states: Optional[List[int]] = None,
        template_file: Optional[Union[int, str]] = None,
        variation_strength: float = 0.3,
        seed: Optional[int] = None,
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """Generate a full cluster-label timeline.

        Args:
            target_measures: Desired total length in measures.
            start_states: Optional cluster labels to prepend.
            template_file: Index or filename stem for section template.
            variation_strength: RETURN/VARIANT deviation from prototype.
            seed: Random seed.

        Returns:
            labels: Full cluster label sequence.
            event_log: Description of each segment (SECTION, FREE, USER_START).
        """
        rng = np.random.RandomState(seed)
        labels: List[int] = []
        event_log: List[Dict[str, Any]] = []

        # Phase 1: User-specified start bars
        if start_states:
            labels.extend(start_states)
            event_log.append({
                "kind": "USER_START",
                "length": len(start_states),
                "labels": list(start_states),
            })

        remaining = target_measures - len(labels)
        if remaining <= 0:
            return labels[:target_measures], event_log

        # Phase 2: Section grammar + PhraseGenerator
        if self.grammar is not None and self.grammar.files:
            labels, event_log = self._generate_with_grammar(
                target_measures, labels, event_log,
                template_file, variation_strength, rng,
            )
        else:
            log.info("No section grammar — using flat phrase generation.")
            extra = self.phrase_gen.generate(
                remaining, seed=int(rng.randint(0, 2 ** 31 - 1)),
            )
            labels.extend(extra)
            event_log.append({
                "kind": "FLAT",
                "length": len(extra),
                "labels": extra,
            })

        if len(labels) > target_measures:
            labels = labels[:target_measures]
            event_log = self._trim_events_to_length(event_log, target_measures)

        log.info(
            "Timeline: %d measures, %d events.",
            len(labels), len(event_log),
        )
        return labels, event_log

    # ------------------------------------------------------------------
    # MIDI generation
    # ------------------------------------------------------------------

    def generate_midi(
        self,
        output_path: Union[str, Path],
        target_measures: int = 120,
        start_states: Optional[List[int]] = None,
        template_file: Optional[Union[int, str]] = None,
        variation_strength: float = 0.3,
        time_signature: Tuple[int, int] = (4, 4),
        tempo: int = 120,
        seed: Optional[int] = None,
        enable_variation: bool = True,
    ) -> List[int]:
        """Generate a full MIDI file.

        Args:
            enable_variation: If True (default), apply controlled transforms
                to non-NEW section occurrences.  Set to False for
                exact repeats only.

        Returns the cluster label timeline used.
        """
        # 1. Timeline
        labels, event_log = self.generate_timeline(
            target_measures=target_measures,
            start_states=start_states,
            template_file=template_file,
            variation_strength=variation_strength,
            seed=seed,
        )

        # 2. Build measure-level context and breathing points
        measure_context = self._build_measure_context(event_log)
        breathing = self._compute_breathing(measure_context)
        n = len(measure_context)
        base_seed = seed if seed is not None else 0
        structure_graph = self._build_structure_graph(measure_context)
        composition_plan = self._build_composition_plan(
            labels, measure_context, base_seed=base_seed,
        )
        self._last_composition_plan = composition_plan
        harmonic_cfg = self.config.get("harmony", {})
        harmony_mode = str(harmonic_cfg.get("mode", "auto")).lower() if isinstance(harmonic_cfg, dict) else "auto"
        learned_harmony = getattr(self.model, "harmonic_model", None)
        actual_harmony_mode = "disabled"
        if self.harmonic_planner.enabled and learned_harmony is not None and harmony_mode in ("auto", "learned"):
            harmonic_plan = learned_harmony.build_plan(
                measure_context,
                composition_plan.measure_affects,
                composition_plan.global_tonic_pc,
                seed=base_seed,
                config=self.config,
            )
            actual_harmony_mode = "learned"
        elif harmony_mode == "learned" and learned_harmony is None:
            log.warning("harmony.mode=learned but model has no learned harmony; using rule planner fallback.")
            harmonic_plan = self.harmonic_planner.build_plan(
                measure_context,
                composition_plan.measure_affects,
                composition_plan.global_tonic_pc,
            )
            actual_harmony_mode = "rule_fallback" if self.harmonic_planner.enabled else "disabled"
        else:
            harmonic_plan = self.harmonic_planner.build_plan(
                measure_context,
                composition_plan.measure_affects,
                composition_plan.global_tonic_pc,
            )
            actual_harmony_mode = "rule" if self.harmonic_planner.enabled else "disabled"

        # 3. Per-measure note generation
        all_notes: List[List[NoteEvent]] = []
        occurrence_count: Dict[str, int] = {}
        motif_memory: Dict[str, List[List[NoteEvent]]] = {}
        rhythm_memory = RhythmMemory()
        theme_identities: Dict[str, ThemeIdentity] = {}
        theme_skeletons: Dict[str, ThemeSkeleton] = {}
        recent_rhythm_cells: List[Tuple[float, ...]] = []
        return_variation_plans: Dict[int, List[str]] = {}
        self._last_dual_theme_scores: Dict[int, Dict[str, Any]] = {}
        self._last_repeat_harmony_scores: Dict[int, Dict[str, Any]] = {}
        self._last_candidate_score_components: Dict[int, Dict[str, float]] = {}
        self._last_candidate_reranker_scores: Dict[int, Dict[str, Any]] = {}
        self._last_rhythm_scores: Dict[int, Dict[str, float]] = {}
        melodic_skeleton = self._build_melodic_skeleton(
            labels, measure_context, base_seed, composition_plan,
        )
        previous_melody_pitch: Optional[int] = None

        _select = _apply = None
        if enable_variation:
            from note_transform import select_transforms, apply_variation
            _select = select_transforms
            _apply = apply_variation

        for i, cluster_id in enumerate(labels):
            sl, bi, role, occurrence_id, section_len = measure_context[i]
            is_end = breathing[i]

            phrase_role = self._phrase_role(bi, section_len, role)
            target_pitch = melodic_skeleton.get(i)
            affect = dict(composition_plan.measure_affects.get(i) or {})
            harmony = harmonic_plan.get(i)
            if harmony is not None:
                affect["harmony"] = harmony.to_dict()
            dual_theme = self.dual_theme_development.target_for_bar(
                label=sl,
                local_bar=bi,
                section_len=section_len,
                narrative_tension=float(affect.get("narrative_tension", affect.get("tension", 0.35))),
                narrative_intensity=float(affect.get("narrative_intensity", affect.get("intensity", 0.35))),
                theme_identities=theme_identities,
                theme_skeletons=theme_skeletons,
            )
            if dual_theme is not None:
                affect["dual_theme"] = dual_theme
            composition_plan.measure_affects[i] = affect
            bar_target = self._build_bar_generation_target(
                bar_index=i,
                label=sl,
                local_bar=bi,
                target_pitch=target_pitch,
                affect=affect,
                structure_graph=structure_graph,
                theme_skeletons=theme_skeletons,
                composition_plan=composition_plan,
            )
            rhythm_target = self.rhythm_phrase_planner.plan(
                label=sl,
                local_bar=bi,
                section_role=role,
                phrase_role=phrase_role,
                narrative_role=str(affect.get("narrative_role", "")),
                tension=float(affect.get("narrative_tension", affect.get("tension", 0.35))),
                source_cell=rhythm_memory.get(sl, bi),
            )

            # Seed from section identity (FREE/FLAT use per-measure seed)
            if role in ("FREE", "FLAT"):
                measure_seed = _stable_hash(base_seed, "FREE", i)
            else:
                measure_seed = _stable_hash(base_seed, sl, bi)

            reused_motif = False
            if role not in ("FREE", "FLAT", "NEW") and sl in motif_memory and bi < len(motif_memory[sl]):
                notes = self._clone_notes(motif_memory[sl][bi])
                reused_motif = True
            else:
                notes = self._generate_scored_measure(
                    cluster_label=cluster_id,
                    time_signature=time_signature,
                    is_section_end=is_end,
                    phrase_role=phrase_role,
                    target_pitch=target_pitch,
                    previous_pitch=previous_melody_pitch,
                    affect=affect,
                    bar_target=bar_target,
                    seed=measure_seed,
                    bar_index=i,
                    rhythm_target=rhythm_target,
                    recent_rhythm_cells=recent_rhythm_cells,
                )
                if role == "NEW":
                    motif_memory.setdefault(sl, [])
                    while len(motif_memory[sl]) <= bi:
                        motif_memory[sl].append([])
                    motif_memory[sl][bi] = self._clone_notes(notes)
                    rhythm_memory.remember(sl, bi, notes)
                    self._update_theme_identity(
                        theme_identities, sl, motif_memory[sl], section_len,
                    )
                    self._update_theme_skeleton(
                        theme_skeletons, sl, motif_memory[sl],
                        section_len, composition_plan.global_tonic_pc,
                    )

            # Controlled variation: non-NEW sections get transforms.
            # Strength s is data-driven: average cluster entropy scaled
            # by an information-theoretic progression k/(k+1).
            if (_select is not None and _apply is not None
                    and role not in ("FREE", "FLAT", "NEW", "REPEAT")):
                if bi == 0:
                    k = occurrence_count.get(sl, 0)
                    if k > 0:
                        sec_labels = labels[i:i + section_len]
                        variation_rng = np.random.RandomState(
                            _stable_hash(base_seed, sl, occurrence_id, "variation")
                        )
                        s = (self._section_entropy(sec_labels) / self._max_entropy
                             ) * (k / (k + VARIATION_PROGRESSION_DENOM))
                        self._current_variation_profile = _select(
                            self.model.clusterer.centroids,
                            sec_labels, s, rng=variation_rng,
                        ) if s > 0.02 else None
                    else:
                        self._current_variation_profile = None
                if self._current_variation_profile:
                    variation_rng = np.random.RandomState(
                        _stable_hash(base_seed, sl, occurrence_id, bi, "variation")
                    )
                    notes = _apply(notes, self._current_variation_profile, rng=variation_rng)
            if reused_motif and enable_variation and role in ("RETURN", "VARIANT", "REPEAT"):
                k = occurrence_count.get(sl, 1)
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
                harmony_cfg = self.config.get("harmony", {})
                if isinstance(harmony_cfg, dict) and harmony_cfg.get("enabled", False) and bar_target.harmony:
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
                        narrative_role = str(affect.get("narrative_role", ""))
                        if narrative_role in boosted_roles:
                            boost = float(boost_cfg.get("multiplier", 1.6))
                            max_candidates = int(boost_cfg.get("max_candidates", 24))
                            candidate_count = min(
                                max_candidates,
                                max(candidate_count, int(round(candidate_count * boost))),
                            )
                if role == "REPEAT":
                    base_strength *= float(motif_cfg.get("repeat_strength_scale", 0.55))
                    occurrence_growth *= float(motif_cfg.get("repeat_growth_scale", 0.35))
                plan = return_variation_plans.setdefault(
                    occurrence_id,
                    self._build_return_variation_plan(
                        section_len,
                        role,
                        np.random.RandomState(
                            _stable_hash(base_seed, sl, occurrence_id, "return-plan")
                        ),
                    ),
                )
                variation_mode = plan[min(bi, len(plan) - 1)] if plan else "CONTOUR"
                identity = theme_identities.get(sl)
                skeleton = theme_skeletons.get(sl)
                source_notes = self._clone_notes(notes)
                partner_notes = self._partner_theme_notes(
                    motif_memory,
                    bar_target.dual_theme,
                    bi,
                )
                previous_notes = all_notes[-1] if all_notes else None
                best_notes: Optional[List[NoteEvent]] = None
                best_dual_theme_diag: Optional[Dict[str, float]] = None
                best_early_repeat_diag: Optional[Dict[str, float]] = None
                best_rhythm_diag: Optional[Dict[str, float]] = None
                best_score_components: Optional[Dict[str, float]] = None
                best_candidate_index: Optional[int] = None
                best_proposal_kind: Optional[str] = None
                best_repeat_harmony_kind: Optional[str] = None
                best_score = -1e9
                strength = min(
                    max_strength,
                    (base_strength + occurrence_growth * k) * bar_target.development_strength,
                )
                proposal_count = self.dual_theme_proposal.count(bar_target, partner_notes)
                repeat_harmony_count = self.repeat_harmony_proposal.count(
                    role,
                    bar_target,
                    partner_notes,
                )
                base_candidate_count = max(1, candidate_count)
                total_candidate_count = base_candidate_count + proposal_count + repeat_harmony_count
                for candidate_i in range(total_candidate_count):
                    variation_rng = np.random.RandomState(
                        _stable_hash(base_seed, sl, occurrence_id, bi, "motif-return", candidate_i)
                    )
                    candidate_strength = strength * float(
                        variation_rng.uniform(0.85, 1.15)
                    )
                    candidate_notes = self._vary_return_motif(
                        source_notes,
                        strength=candidate_strength,
                        rng=variation_rng,
                        target_pitch=target_pitch,
                        variation_mode=variation_mode,
                        phrase_role=phrase_role,
                        development_role=bar_target.development_role,
                        target_attraction=bar_target.target_attraction,
                        rhythm_change_scale=bar_target.rhythm_change_scale,
                    )
                    repeat_harmony_index = None
                    if candidate_i >= base_candidate_count and candidate_i < base_candidate_count + proposal_count:
                        proposal_index = candidate_i - base_candidate_count
                        candidate_notes = self.dual_theme_proposal.propose(
                            candidate_notes,
                            source_notes=source_notes,
                            partner_notes=partner_notes,
                            target=bar_target,
                            rng=variation_rng,
                            proposal_index=proposal_index,
                        )
                    elif candidate_i >= base_candidate_count + proposal_count:
                        proposal_index = None
                        repeat_harmony_index = candidate_i - base_candidate_count - proposal_count
                        candidate_notes = self.repeat_harmony_proposal.propose(
                            candidate_notes,
                            target=bar_target,
                            rng=variation_rng,
                            proposal_index=repeat_harmony_index,
                        )
                    else:
                        proposal_index = None
                    if identity is not None and bar_target.development_role not in ("FRAGMENT", "INTENSIFY"):
                        candidate_notes = self._apply_theme_identity(
                            candidate_notes,
                            identity=identity,
                            bar_index=bi,
                            section_len=section_len,
                            target_pitch=target_pitch,
                            rng=variation_rng,
                        )
                    candidate_notes = RhythmVariation.apply(
                        candidate_notes,
                        rhythm_target,
                        variation_rng,
                        bar_length=float(time_signature[0]) * (4.0 / float(time_signature[1])),
                    )
                    candidate_notes = self._fit_notes_to_bar_target(
                        candidate_notes, bar_target, variation_rng,
                    )
                    rhythm_diag = self.rhythm_scorer.score(
                        candidate_notes,
                        rhythm_target,
                        previous_cells=recent_rhythm_cells,
                    )
                    rhythm_score = float(rhythm_diag.get("score", 0.0))
                    development_score = self.development_scorer.score(
                        candidate_notes,
                        bar_target,
                        cluster_id,
                        source_notes=source_notes,
                        previous_notes=previous_notes,
                    ).total
                    harmony_score = HarmonicPlanner.score_melody(
                        candidate_notes,
                        bar_target.harmony,
                        self.config,
                    )
                    conditional_score = 0.0
                    conditional_model = getattr(self.model, "conditional_note_model", None)
                    conditional_cfg = self.config.get("conditional_note_model", {})
                    if (
                        conditional_model is not None
                        and isinstance(conditional_cfg, dict)
                        and conditional_cfg.get("enabled", True)
                    ):
                        conditional_score = float(conditional_cfg.get("return_score_weight", 0.85)) * (
                            conditional_model.score_candidate(
                                candidate_notes,
                                bar_target,
                                cluster_id,
                                self.config,
                            )
                        )
                    dual_theme_diag = self.dual_theme_scorer.diagnostics(
                        candidate_notes,
                        source_notes=source_notes,
                        partner_notes=partner_notes,
                        target=bar_target,
                    )
                    raw_dual_theme_score = float(dual_theme_diag.get("score", 0.0))
                    dual_theme_damping = self._dual_theme_harmony_damping(harmony_score)
                    dual_theme_score = raw_dual_theme_score * dual_theme_damping
                    early_repeat_diag = self.early_repeat_scorer.diagnostics(
                        candidate_notes,
                        role=role,
                        target=bar_target,
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
                    proposal_kind = (
                        self.dual_theme_proposal.kind(proposal_index)
                        if proposal_index is not None else None
                    )
                    repeat_harmony_kind = (
                        self.repeat_harmony_proposal.kind(repeat_harmony_index)
                        if repeat_harmony_index is not None else None
                    )
                    reranker_score = self._score_with_candidate_reranker(
                        candidate_notes,
                        bar_target,
                        cluster_id,
                        source_notes=source_notes,
                        partner_notes=partner_notes,
                        score_components=score_components,
                        proposal_kind=proposal_kind or repeat_harmony_kind,
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
                    if score > best_score:
                        best_score = score
                        best_notes = candidate_notes
                        best_dual_theme_diag = dual_theme_diag
                        best_early_repeat_diag = early_repeat_diag
                        best_rhythm_diag = rhythm_diag
                        best_score_components = score_components
                        best_candidate_index = candidate_i
                        best_proposal_kind = proposal_kind
                        best_repeat_harmony_kind = repeat_harmony_kind
                if best_notes is not None:
                    repaired_notes = self._repair_harmony_mismatch(best_notes, bar_target)
                    if repaired_notes is not best_notes:
                        repaired_development = self.development_scorer.score(
                            repaired_notes,
                            bar_target,
                            cluster_id,
                            source_notes=source_notes,
                            previous_notes=previous_notes,
                        ).total
                        repaired_harmony = HarmonicPlanner.score_melody(
                            repaired_notes,
                            bar_target.harmony,
                            self.config,
                        )
                        repaired_conditional = 0.0
                        conditional_model = getattr(self.model, "conditional_note_model", None)
                        conditional_cfg = self.config.get("conditional_note_model", {})
                        if (
                            conditional_model is not None
                            and isinstance(conditional_cfg, dict)
                            and conditional_cfg.get("enabled", True)
                        ):
                            repaired_conditional = float(conditional_cfg.get("return_score_weight", 0.85)) * (
                                conditional_model.score_candidate(
                                    repaired_notes,
                                    bar_target,
                                    cluster_id,
                                    self.config,
                                )
                            )
                        repaired_dual_theme_diag = self.dual_theme_scorer.diagnostics(
                            repaired_notes,
                            source_notes=source_notes,
                            partner_notes=partner_notes,
                            target=bar_target,
                        )
                        repaired_raw_dual = float(repaired_dual_theme_diag.get("score", 0.0))
                        repaired_dual_damping = self._dual_theme_harmony_damping(repaired_harmony)
                        repaired_dual = repaired_raw_dual * repaired_dual_damping
                        repaired_early_repeat_diag = self.early_repeat_scorer.diagnostics(
                            repaired_notes,
                            role=role,
                            target=bar_target,
                            partner_notes=partner_notes,
                            config=self.config,
                        )
                        repaired_rhythm_diag = self.rhythm_scorer.score(
                            repaired_notes,
                            rhythm_target,
                            previous_cells=recent_rhythm_cells,
                        )
                        repaired_rhythm = float(repaired_rhythm_diag.get("score", 0.0))
                        repaired_base_score = (
                            repaired_development
                            + repaired_rhythm
                            + harmony_return_weight * repaired_harmony
                            + repaired_conditional
                            + repaired_dual
                            + float(repaired_early_repeat_diag.get("score", 0.0))
                        )
                        repaired_components = {
                            "base_score": float(repaired_base_score),
                            "development_score": float(repaired_development),
                            "rhythm_score": float(repaired_rhythm),
                            "harmony_score": float(repaired_harmony),
                            "harmony_weight": float(harmony_return_weight),
                            "weighted_harmony_score": float(harmony_return_weight * repaired_harmony),
                            "conditional_score": float(repaired_conditional),
                            "dual_theme_score": float(repaired_dual),
                            "raw_dual_theme_score": float(repaired_raw_dual),
                            "dual_theme_harmony_damping": float(repaired_dual_damping),
                            "early_repeat_score": float(repaired_early_repeat_diag.get("score", 0.0)),
                        }
                        repaired_reranker = self._score_with_candidate_reranker(
                            repaired_notes,
                            bar_target,
                            cluster_id,
                            source_notes=source_notes,
                            partner_notes=partner_notes,
                            score_components=repaired_components,
                            proposal_kind="adaptive_repair",
                        )
                        repaired_score = repaired_base_score + repaired_reranker.weighted
                        repaired_components.update({
                            "total_score": float(repaired_score),
                            "reranker_probability": float(repaired_reranker.probability),
                            "reranker_logit": float(repaired_reranker.logit),
                            "raw_reranker_probability": float(repaired_reranker.raw_probability),
                            "raw_reranker_logit": float(repaired_reranker.raw_logit),
                            "calibrated_reranker_probability": float(repaired_reranker.calibrated_probability),
                            "calibrated_reranker_logit": float(repaired_reranker.calibrated_logit),
                            "reranker_calibration_adjustment": float(repaired_reranker.calibration_adjustment),
                            "reranker_good_cadence_confidence": float(repaired_reranker.good_cadence_confidence),
                            "reranker_weighted_score": float(repaired_reranker.weighted),
                        })
                        if repaired_score > best_score:
                            best_score = repaired_score
                            best_notes = repaired_notes
                            best_dual_theme_diag = repaired_dual_theme_diag
                            best_early_repeat_diag = repaired_early_repeat_diag
                            best_rhythm_diag = repaired_rhythm_diag
                            best_score_components = repaired_components
                            best_candidate_index = None
                            best_proposal_kind = None
                            best_repeat_harmony_kind = None
                    self._last_dual_theme_scores[i] = {
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
                    self._last_repeat_harmony_scores[i] = {
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
                        "selected_components": (
                            best_early_repeat_diag or {"active": 0.0, "score": 0.0}
                        ),
                        "score_components": best_score_components,
                    }
                    if best_score_components is not None:
                        self._last_candidate_score_components[i] = best_score_components
                        if best_rhythm_diag is not None:
                            self._last_rhythm_scores[i] = best_rhythm_diag
                        self._last_candidate_reranker_scores[i] = {
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
                notes = best_notes if best_notes is not None else notes
            if role not in ("FREE", "FLAT") and bi == 0:
                occurrence_count[sl] = occurrence_count.get(sl, 0) + 1

            all_notes.append(notes)
            recent_rhythm_cells.append(RhythmMotifModel.cell(notes))
            recent_rhythm_cells = recent_rhythm_cells[-4:]
            previous_melody_pitch = self._last_melody_pitch(notes, previous_melody_pitch)

        # 3c. Add bass line (after transforms, so bass is unaffected)
        harmony_bass_enabled = bool(self.config.get("harmony", {}).get("enabled", False)) \
            and bool(self.config.get("harmony", {}).get("bass", {}).get("enabled", True))
        if harmony_bass_enabled:
            bl_ql = time_signature[0] * (4.0 / time_signature[1])
            for mi in range(len(all_notes)):
                harmony = harmonic_plan.get(mi)
                if harmony is not None:
                    HarmonicPlanner.add_bass_note(
                        all_notes[mi],
                        harmony,
                        bl_ql,
                        self.config,
                        np.random.RandomState(_stable_hash(base_seed, "harmony-bass", mi)),
                    )
        elif self.note_sampler._bass_enabled:
            bl_ql = time_signature[0] * (4.0 / time_signature[1])
            for mi in range(len(all_notes)):
                cid = labels[mi]
                self.note_sampler._add_bass_note(
                    all_notes[mi], cid, bl_ql,
                    np.random.RandomState(_stable_hash(base_seed, "bass", mi)),
                )

        # 3d. Rendering: clamp overlaps on every measure (melody only — bass is
        # added after transforms so it never participates in clamping).
        clamp_overlaps(all_notes, self.config)
        clamp_measure_bounds(all_notes, time_signature)
        ensure_final_bar_end(all_notes, time_signature)
        clamp_measure_bounds(all_notes, time_signature)

        # 4. Write MIDI via mido (silence = absence of note events)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_midi(all_notes, output_path, tempo, time_signature)
        log.info("Wrote MIDI to %s (%d measures).", output_path, len(all_notes))

        self._write_harmony_diagnostics(
            output_path=output_path,
            labels=labels,
            measure_context=measure_context,
            all_notes=all_notes,
            harmonic_plan=harmonic_plan,
            requested_mode=harmony_mode,
            actual_mode=actual_harmony_mode,
            has_learned_model=learned_harmony is not None,
        )

        # 5. Save structure visualization
        from structure_plotter import StructurePlotter
        plot_path = output_path.with_suffix('.png')
        StructurePlotter.plot(labels, event_log, self.model.n_clusters, plot_path)

        return labels

    def _dual_theme_harmony_damping(self, harmony_score: float) -> float:
        """Reduce dual-theme reward when a candidate is harmonically weak."""
        cfg = self.config.get("dual_theme_development", {})
        if not isinstance(cfg, dict):
            return 1.0
        scoring = cfg.get("candidate_scoring", {})
        if not isinstance(scoring, dict) or not scoring.get("harmony_damping_enabled", True):
            return 1.0
        floor = float(scoring.get("harmony_damping_floor", 0.25))
        start = float(scoring.get("harmony_damping_start", -0.10))
        full = float(scoring.get("harmony_damping_full", 0.45))
        if full <= start:
            return 1.0
        value = floor + (1.0 - floor) * ((float(harmony_score) - start) / (full - start))
        return float(np.clip(value, floor, 1.0))

    def _partner_theme_notes(
        self,
        motif_memory: Dict[str, List[List[NoteEvent]]],
        dual_theme: Optional[Dict[str, Any]],
        local_bar: int,
    ) -> Optional[List[NoteEvent]]:
        """Fetch the partner-theme bar used by dual-theme candidate scoring."""
        if not isinstance(dual_theme, dict):
            return None
        partner_label = dual_theme.get("partner_label")
        if not isinstance(partner_label, str):
            return None
        partner = motif_memory.get(partner_label)
        if not partner:
            return None
        valid = [idx for idx, notes in enumerate(partner) if notes]
        if not valid:
            return None
        idx = min(valid, key=lambda x: abs(x - local_bar))
        return self._clone_notes(partner[idx])

    def _score_with_candidate_reranker(
        self,
        notes: List[NoteEvent],
        target: BarGenerationTarget,
        cluster_id: int,
        *,
        source_notes: Optional[List[NoteEvent]] = None,
        partner_notes: Optional[List[NoteEvent]] = None,
        score_components: Optional[Dict[str, float]] = None,
        proposal_kind: Optional[str] = None,
    ):
        """Return the learned reranker contribution for one candidate."""
        reranker = getattr(self, "candidate_reranker", None)
        if reranker is None:
            reranker = CandidateReranker()
        return reranker.score_candidate(
            notes,
            target,
            cluster_id,
            self.config,
            source_notes=source_notes,
            partner_notes=partner_notes,
            score_components=score_components,
            proposal_kind=proposal_kind,
        )

    def _write_harmony_diagnostics(
        self,
        output_path: Path,
        labels: List[int],
        measure_context: List[Tuple[str, int, str, int, int]],
        all_notes: List[List[NoteEvent]],
        harmonic_plan: Dict[int, Any],
        requested_mode: str,
        actual_mode: str,
        has_learned_model: bool,
    ) -> None:
        """Write a compact JSON report for harmony debugging."""
        cfg = self.config.get("harmony", {})
        if not isinstance(cfg, dict) or not cfg.get("diagnostics", True):
            return
        bars: List[Dict[str, Any]] = []
        chord_ratios: List[float] = []
        strong_ratios: List[float] = []
        rhythm_scores: List[float] = []
        rhythm_source_similarities: List[float] = []
        rhythm_note_counts: List[float] = []
        narrative_counts: Dict[str, int] = {}
        dual_theme_active = 0
        bass_root_or_fifth = 0
        bass_count = 0
        for i, notes in enumerate(all_notes):
            harmony = harmonic_plan.get(i)
            harmony_dict = harmony.to_dict() if harmony is not None else None
            diag = HarmonicPlanner.diagnostics(notes, harmony_dict, self.config)
            chord_ratio = diag.get("chord_tone_ratio")
            strong_ratio = diag.get("strong_beat_chord_tone_ratio")
            if isinstance(chord_ratio, (int, float)):
                chord_ratios.append(float(chord_ratio))
            if isinstance(strong_ratio, (int, float)):
                strong_ratios.append(float(strong_ratio))
            if diag.get("bass_pc") is not None:
                bass_count += 1
                if diag.get("bass_is_root_or_fifth"):
                    bass_root_or_fifth += 1
            label, local_bar, role, occurrence_id, section_len = measure_context[i]
            affect = self._last_composition_plan.measure_affects.get(i, {}) if hasattr(self, "_last_composition_plan") else {}
            narrative_role = affect.get("narrative_role")
            if isinstance(narrative_role, str):
                narrative_counts[narrative_role] = narrative_counts.get(narrative_role, 0) + 1
            dual_theme = affect.get("dual_theme")
            if isinstance(dual_theme, dict):
                dual_theme_active += 1
            dual_theme_score = (
                self._last_dual_theme_scores.get(i)
                if hasattr(self, "_last_dual_theme_scores") else None
            )
            repeat_harmony_score = (
                self._last_repeat_harmony_scores.get(i)
                if hasattr(self, "_last_repeat_harmony_scores") else None
            )
            score_components = (
                self._last_candidate_score_components.get(i)
                if hasattr(self, "_last_candidate_score_components") else None
            )
            rhythm_score = (
                self._last_rhythm_scores.get(i)
                if hasattr(self, "_last_rhythm_scores") else None
            )
            if isinstance(rhythm_score, dict):
                score = rhythm_score.get("score")
                source_similarity = rhythm_score.get("source_similarity")
                note_count = rhythm_score.get("note_count")
                if isinstance(score, (int, float)):
                    rhythm_scores.append(float(score))
                if isinstance(source_similarity, (int, float)):
                    rhythm_source_similarities.append(float(source_similarity))
                if isinstance(note_count, (int, float)):
                    rhythm_note_counts.append(float(note_count))
            reranker_score = (
                self._last_candidate_reranker_scores.get(i)
                if hasattr(self, "_last_candidate_reranker_scores") else None
            )
            bars.append({
                "bar": i + 1,
                "cluster": int(labels[i]) if i < len(labels) else None,
                "section_label": label,
                "section_role": role,
                "local_bar": int(local_bar),
                "occurrence_id": int(occurrence_id),
                "section_len": int(section_len),
                "narrative_role": narrative_role,
                "narrative_tension": affect.get("narrative_tension"),
                "narrative_intensity": affect.get("narrative_intensity"),
                "dual_theme": dual_theme if isinstance(dual_theme, dict) else None,
                "dual_theme_candidate": dual_theme_score,
                "repeat_harmony_candidate": repeat_harmony_score,
                "rhythm_candidate": rhythm_score,
                "candidate_reranker": reranker_score,
                "candidate_score_components": score_components,
                "harmony": harmony_dict,
                "diagnostics": diag,
            })
        conditional_cfg = self.config.get("conditional_note_model", {})
        conditional_enabled = (
            conditional_cfg.get("enabled", True)
            if isinstance(conditional_cfg, dict)
            else True
        )
        conditional_model = getattr(self.model, "conditional_note_model", None)
        reranker_cfg = self.config.get("candidate_reranker", {})
        reranker_enabled = (
            reranker_cfg.get("enabled", True)
            if isinstance(reranker_cfg, dict)
            else True
        )
        candidate_reranker = getattr(self.model, "candidate_reranker", None)
        rhythm_candidate_prior = getattr(self.model, "rhythm_candidate_prior", None)
        dual_cfg = self.config.get("dual_theme_development", {})
        dual_scoring_cfg = dual_cfg.get("candidate_scoring", {}) if isinstance(dual_cfg, dict) else {}
        rhythm_cfg = self.config.get("rhythm_development", {})
        summary = {
            "requested_mode": requested_mode,
            "actual_mode": actual_mode,
            "has_learned_model": has_learned_model,
            "conditional_note_model_active": (
                conditional_model is not None
                and bool(conditional_enabled)
            ),
            "conditional_note_model_version": (
                getattr(conditional_model, "version", None)
            ),
            "candidate_reranker_active": (
                candidate_reranker is not None
                and bool(getattr(candidate_reranker, "available", False))
                and bool(reranker_enabled)
            ),
            "candidate_reranker_training": (
                getattr(candidate_reranker, "training_summary", None)
                if candidate_reranker is not None else None
            ),
            "dual_theme_candidate_scoring_active": (
                isinstance(dual_cfg, dict)
                and bool(dual_cfg.get("enabled", True))
                and isinstance(dual_scoring_cfg, dict)
                and bool(dual_scoring_cfg.get("enabled", True))
            ),
            "rhythm_development_active": (
                isinstance(rhythm_cfg, dict)
                and bool(rhythm_cfg.get("enabled", True))
            ),
            "rhythm_candidate_prior_active": (
                rhythm_candidate_prior is not None
                and bool(getattr(rhythm_candidate_prior, "available", False))
                and isinstance(rhythm_cfg, dict)
                and bool(rhythm_cfg.get("learned_prior_enabled", True))
                and float(rhythm_cfg.get("learned_prior_weight", 0.0)) != 0.0
            ),
            "rhythm_candidate_prior_training": (
                getattr(rhythm_candidate_prior, "training_summary", None)
                if rhythm_candidate_prior is not None else None
            ),
            "bar_count": len(all_notes),
            "narrative_role_counts": narrative_counts,
            "dual_theme_active_bars": dual_theme_active,
            "mean_rhythm_score": float(np.mean(rhythm_scores)) if rhythm_scores else None,
            "mean_rhythm_source_similarity": (
                float(np.mean(rhythm_source_similarities))
                if rhythm_source_similarities else None
            ),
            "mean_rhythm_note_count": (
                float(np.mean(rhythm_note_counts)) if rhythm_note_counts else None
            ),
            "mean_chord_tone_ratio": float(np.mean(chord_ratios)) if chord_ratios else None,
            "mean_strong_beat_chord_tone_ratio": float(np.mean(strong_ratios)) if strong_ratios else None,
            "bass_root_or_fifth_ratio": (
                bass_root_or_fifth / bass_count if bass_count else None
            ),
        }
        report = {
            "summary": summary,
            "bars": bars,
        }
        diag_path = output_path.with_suffix(".harmony.json")
        with open(diag_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info("Wrote harmony diagnostics to %s", diag_path)

    @staticmethod
    def _build_measure_context(
        event_log: List[Dict[str, Any]],
    ) -> List[Tuple[str, int, str, int, int]]:
        """Flatten event_log into per-measure section context.

        Each tuple is ``(label, bar_index, role, occurrence_id, section_len)``.
        The occurrence id prevents repeated labels such as A ... A ... A from
        being merged when calculating phrase length or cadences.
        """
        ctx: List[Tuple[str, int, str, int, int]] = []
        occurrence_id = 0
        for event in event_log:
            length = event["length"]
            if event["kind"] == "SECTION":
                for bar_in_sec in range(length):
                    ctx.append((event["label"], bar_in_sec, event["role"], occurrence_id, length))
                occurrence_id += 1
            elif event["kind"] == "FREE":
                for _ in range(length):
                    ctx.append(("FREE", 0, "FREE", occurrence_id, length))
                occurrence_id += 1
            else:
                for _ in range(length):
                    ctx.append(("FLAT", 0, "FLAT", occurrence_id, length))
                occurrence_id += 1
        return ctx

    @staticmethod
    def _compute_breathing(
        measure_context: List[Tuple[str, int, str, int, int]],
    ) -> List[bool]:
        """Mark bars that should breathe at structural boundaries.

        A boundary is detected both from the local bar index and from the next
        occurrence id.  The second check matters in product use because short
        sections, truncated templates, or mixed FREE/SECTION timelines can
        make bar-index-only cadence detection miss a phrase ending.
        """
        n = len(measure_context)
        breathing = [False] * n
        for i in range(n):
            sl, bi, role, occurrence_id, section_len = measure_context[i]
            if role not in ("FREE", "FLAT"):
                if bi == section_len - 1:
                    breathing[i] = True
            elif role in ("FREE", "FLAT") and i + 1 < n:
                if measure_context[i + 1][2] not in ("FREE", "FLAT"):
                    breathing[i] = True
            if i + 1 < n:
                next_role = measure_context[i + 1][2]
                next_occurrence_id = measure_context[i + 1][3]
                if role not in ("FREE", "FLAT") and next_occurrence_id != occurrence_id:
                    breathing[i] = True
                elif role in ("FREE", "FLAT") and next_role not in ("FREE", "FLAT"):
                    breathing[i] = True
        if breathing:
            breathing[-1] = False
        return breathing

    # ------------------------------------------------------------------
    # Structure visualization
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal: grammar-aware timeline
    # ------------------------------------------------------------------

    def _generate_with_grammar(
        self,
        target_measures: int,
        labels: List[int],
        event_log: List[Dict[str, Any]],
        template_file: Optional[Union[int, str]],
        variation_strength: float,
        rng: np.random.RandomState,
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        grammar = self.grammar

        fs = self._select_template(grammar, template_file, rng)

        label_seen: set[str] = set()
        cycle = 0
        section_labels = fs.label_sequence
        n_gaps = len(section_labels) - 1

        while len(labels) < target_measures:
            free_lengths = grammar._sample_free_lengths(n_gaps, rng)

            for i, sec_label in enumerate(section_labels):
                if len(labels) >= target_measures:
                    break

                role, vary = self._assign_role(
                    sec_label, i, cycle, section_labels,
                    label_seen, variation_strength,
                )

                content = grammar.generate_section_content(
                    sec_label, fs, vary=vary,
                    variation_strength=variation_strength,
                    seed=None if rng is None else int(rng.randint(0, 2 ** 31 - 1)),
                )

                labels.extend(content)
                event_log.append({
                    "kind": "SECTION",
                    "label": sec_label,
                    "role": role,
                    "cycle": cycle,
                    "length": len(content),
                    "labels": content,
                })

                if i < len(section_labels) - 1 and len(labels) < target_measures:
                    free_len = free_lengths[i] if i < len(free_lengths) else 4
                    free_labels = self.phrase_gen.generate(
                        free_len,
                        seed=None if rng is None else int(rng.randint(0, 2 ** 31 - 1)),
                    )
                    labels.extend(free_labels)
                    event_log.append({
                        "kind": "FREE",
                        "length": free_len,
                        "labels": free_labels,
                    })

                    # Snap to next 4-bar grid, max 3 bars padding
                    _GRID = 4
                    remainder = len(labels) % _GRID
                    if 0 < remainder <= 3:
                        pad = _GRID - remainder
                        pad_labels = self.phrase_gen.generate(
                            pad,
                            seed=None if rng is None else int(rng.randint(0, 2 ** 31 - 1)),
                        )
                        labels.extend(pad_labels)
                        event_log.append({
                            "kind": "FREE",
                            "length": pad,
                            "labels": pad_labels,
                            "grid_pad": True,
                        })

                label_seen.add(sec_label)

            cycle += 1

        self._end_with_return(labels, event_log, fs, variation_strength, rng)
        labels, event_log = self._rebalance_narrative_timeline(
            labels,
            event_log,
            fs,
            target_measures,
            variation_strength,
            rng,
        )
        return labels, event_log

    def _rebalance_narrative_timeline(
        self,
        labels: List[int],
        event_log: List[Dict[str, Any]],
        fs: Any,
        target_measures: int,
        variation_strength: float,
        rng: np.random.RandomState,
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """Increase contrast-theme presence in narrative contrast regions.

        V1 only labelled the existing timeline.  V2 gently rebalances the
        timeline by replacing selected FREE connective blocks with secondary
        section material, preserving total length and the original grammar
        skeleton.
        """
        narrative_cfg = self.config.get("narrative", {})
        if not isinstance(narrative_cfg, dict) or not narrative_cfg.get("enabled", True):
            return labels, event_log
        rebalance_cfg = narrative_cfg.get("timeline_rebalance", {})
        if not isinstance(rebalance_cfg, dict) or not rebalance_cfg.get("enabled", True):
            return labels, event_log

        primary = fs.label_sequence[0] if fs.label_sequence else "A"
        secondary_labels = [
            label for label in fs.label_sequence
            if label != primary and label in fs.prototypes
        ]
        if not secondary_labels:
            return labels, event_log
        secondary = secondary_labels[0]

        total_len = sum(int(ev.get("length", 0)) for ev in event_log)
        if total_len <= 0:
            return labels, event_log
        current_secondary = sum(
            int(ev.get("length", 0))
            for ev in event_log
            if ev.get("kind") == "SECTION" and ev.get("label") == secondary
        )
        min_ratio = float(rebalance_cfg.get("min_secondary_ratio", 0.16))
        target_secondary = int(round(total_len * min_ratio))
        needed = max(0, target_secondary - current_secondary)
        if needed <= 0:
            return labels[:target_measures], event_log

        allowed_regions = set(rebalance_cfg.get("secondary_regions", ["CONTRAST", "DEVELOPMENT", "CLIMAX"]))
        max_replaced = int(round(total_len * float(rebalance_cfg.get("max_replaced_ratio", 0.18))))
        replaced = 0
        starts = self._event_starts(event_log)
        candidates: List[Tuple[float, int]] = []
        for idx, ev in enumerate(event_log):
            if ev.get("kind") != "FREE" or ev.get("grid_pad"):
                continue
            start = starts[idx]
            length = int(ev.get("length", 0))
            if length <= 0:
                continue
            pos = (start + 0.5 * length) / max(1, total_len - 1)
            macro = NarrativePlanner._macro_role(
                pos,
                float(narrative_cfg.get("contrast_position", 0.24)),
                float(narrative_cfg.get("development_position", 0.42)),
                float(narrative_cfg.get("climax_position", 0.72)),
                float(narrative_cfg.get("recap_position", 0.84)),
                float(narrative_cfg.get("coda_position", 0.94)),
            )
            if macro in allowed_regions:
                # Prefer longer blocks near the contrast/development center.
                priority = abs(pos - 0.50) - 0.01 * length + rng.random() * 0.001
                candidates.append((priority, idx))

        for _, idx in sorted(candidates):
            if needed <= 0 or replaced >= max_replaced:
                break
            ev = event_log[idx]
            length = int(ev.get("length", 0))
            secondary_seen_before = any(
                prior.get("kind") == "SECTION" and prior.get("label") == secondary
                for prior in event_log[:idx]
            )
            content = self.grammar.generate_section_content(
                secondary,
                fs,
                vary=secondary_seen_before,
                variation_strength=variation_strength,
                seed=int(rng.randint(0, 2 ** 31 - 1)),
            )
            if not content:
                continue
            fitted: List[int] = []
            while len(fitted) < length:
                fitted.extend(content)
            fitted = fitted[:length]
            event_log[idx] = {
                "kind": "SECTION",
                "label": secondary,
                "role": "RETURN" if secondary_seen_before else "NEW",
                "cycle": ev.get("cycle", 0),
                "length": length,
                "labels": fitted,
                "narrative_rebalanced": True,
                "replaced_kind": "FREE",
            }
            needed -= length
            replaced += length

        event_log = self._trim_events_to_length(event_log, target_measures)
        return self._labels_from_events(event_log)[:target_measures], event_log

    @staticmethod
    def _trim_events_to_length(
        event_log: List[Dict[str, Any]],
        target_measures: int,
    ) -> List[Dict[str, Any]]:
        trimmed: List[Dict[str, Any]] = []
        used = 0
        for ev in event_log:
            if used >= target_measures:
                break
            length = int(ev.get("length", 0))
            keep = min(length, target_measures - used)
            if keep <= 0:
                break
            new_ev = dict(ev)
            if keep < length:
                new_ev["length"] = keep
                new_ev["labels"] = list(ev.get("labels", []))[:keep]
                new_ev["truncated"] = True
            trimmed.append(new_ev)
            used += keep
        return trimmed

    @staticmethod
    def _event_starts(event_log: List[Dict[str, Any]]) -> List[int]:
        starts: List[int] = []
        pos = 0
        for ev in event_log:
            starts.append(pos)
            pos += int(ev.get("length", 0))
        return starts

    @staticmethod
    def _labels_from_events(event_log: List[Dict[str, Any]]) -> List[int]:
        labels: List[int] = []
        for ev in event_log:
            labels.extend(int(x) for x in ev.get("labels", []))
        return labels

    @staticmethod
    def _select_template(
        grammar: Any,
        template_file: Optional[Union[int, str]],
        rng: np.random.RandomState,
    ) -> Any:
        """Pick a template file, preferring multi-family and grid-aligned."""
        if template_file is not None:
            if isinstance(template_file, int):
                return grammar.files[template_file % len(grammar.files)]
            match = next(
                (f for f in grammar.files
                 if f.filename == template_file
                 or f.filename.endswith(template_file)
                 or Path(f.filename).stem == template_file),
                None,
            )
            if match is None:
                raise KeyError(f"No file matching '{template_file}'")
            return match

        multi = [f for f in grammar.files if f.n_families >= 2]
        by_grid = {0: [], 1: [], 2: []}
        candidates = multi if multi and rng.random() < 0.7 else grammar.files
        for f in candidates:
            lengths = [len(seq) for seq in f.prototypes.values()]
            aligned = sum(1 for L in lengths if L % 4 == 0)
            if aligned == len(lengths) and lengths:
                by_grid[0].append(f)
            elif aligned >= len(lengths) // 2:
                by_grid[1].append(f)
            else:
                by_grid[2].append(f)
        pool = (by_grid[0] * 7 + by_grid[1] * 2 + by_grid[2]) or grammar.files
        return pool[rng.randint(0, len(pool))]

    @staticmethod
    def _assign_role(
        sec_label: str,
        i: int,
        cycle: int,
        section_labels: List[str],
        label_seen: set[str],
        variation_strength: float,
    ) -> Tuple[str, bool]:
        """Determine the structural role of a section occurrence."""
        if cycle == 0 and i == 0:
            return "NEW", False
        if (i > 0 and sec_label == section_labels[i - 1]) \
                or (i == 0 and sec_label == section_labels[-1]):
            return "REPEAT", False
        if sec_label in label_seen:
            return "RETURN", variation_strength > 0
        return "NEW", False

    def _end_with_return(
        self,
        labels: List[int],
        event_log: List[Dict[str, Any]],
        fs: Any,
        variation_strength: float,
        rng: np.random.RandomState,
    ) -> None:
        """Replace trailing FREE blocks with a RETURN of the primary theme."""
        if not event_log or event_log[-1]["kind"] != "FREE":
            return

        grammar = self.grammar
        primary_label = fs.label_sequence[0]

        # Count and remove trailing FREE
        free_len = 0
        while event_log and event_log[-1]["kind"] == "FREE":
            free_len += event_log[-1]["length"]
            event_log.pop()
        del labels[-free_len:]

        # Fill with RETURN of the primary section
        content: List[int] = []
        while len(content) < free_len:
            content.extend(grammar.generate_section_content(
                primary_label, fs, vary=True,
                variation_strength=variation_strength,
                seed=None if rng is None else int(rng.randint(0, 2 ** 31 - 1)),
            ))
        labels.extend(content[:free_len])
        event_log.append({
            "kind": "SECTION",
            "label": primary_label,
            "role": "RETURN",
            "length": free_len,
            "labels": content[:free_len],
        })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hierarchical Generator — three-tier music generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default="./models/corelli",
        help="Path to trained MusicModel directory.",
    )
    parser.add_argument(
        "--output", "-o",
        default="generated/hierarchical_output.mid",
        help="Output MIDI path.",
    )
    parser.add_argument(
        "--target-measures", "-n",
        type=int,
        default=120,
        help="Target number of measures.",
    )
    parser.add_argument(
        "--start-states",
        default=None,
        help="Comma-separated cluster labels for the first N bars "
        "(e.g. '2,2,2,0,0').",
    )
    parser.add_argument(
        "--template", "-t",
        default=None,
        help="Section template: file index or name stem.",
    )
    parser.add_argument(
        "--variation", "-v",
        type=float,
        default=0.3,
        help="Variation strength for RETURN sections (0–1).",
    )
    parser.add_argument(
        "--time-signature",
        default="4/4",
        help="Time signature (e.g. '4/4', '3/4').",
    )
    parser.add_argument(
        "--tempo",
        type=int,
        default=120,
        help="Tempo in BPM.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--no-variation",
        action="store_true",
        help="Disable controlled variation transforms (exact repeats only).",
    )
    parser.add_argument(
        "--no-bass",
        action="store_true",
        help="Disable bass line generation.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML overrides applied after defaults/profile.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Composer profile name from ../config/profiles/<name>.yaml.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Loading model from %s ...", args.model_dir)
    model = MusicModel.load(args.model_dir)
    print()
    print(model.summary())

    ts_parts = args.time_signature.split("/")
    time_sig = (int(ts_parts[0]), int(ts_parts[1]))

    start_states = None
    if args.start_states:
        start_states = [
            int(x.strip()) for x in args.start_states.split(",") if x.strip()
        ]

    gen = HierarchicalGenerator(
        model,
        config_path=args.config,
        composer_profile=args.profile,
    )

    log.info(
        "Generating %d measures (template=%s, variation=%.2f) ...",
        args.target_measures, args.template or "random", args.variation,
    )
    if args.no_bass:
        gen.note_sampler._bass_enabled = False

    labels = gen.generate_midi(
        output_path=args.output,
        target_measures=args.target_measures,
        start_states=start_states,
        template_file=args.template,
        variation_strength=args.variation,
        time_signature=time_sig,
        tempo=args.tempo,
        seed=args.seed,
        enable_variation=not args.no_variation,
    )

    print(f"\nGenerated {len(labels)} measures -> {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
