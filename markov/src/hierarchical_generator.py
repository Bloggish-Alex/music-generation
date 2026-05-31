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
from dataclasses import dataclass, field
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
from generation_trace import GenerationTraceRecorder
from hierarchical_note_pipeline import (
    DEFAULT_NOTE_GENERATION_MODULES,
    DEFAULT_NOTE_GENERATION_STOP_AFTER,
    HierarchicalNotePipelineMixin,
    NOTE_GENERATION_MODULES,
)
from hierarchical_planning import HierarchicalPlanningMixin
from narrative_planner import NarrativePlanner
from hierarchical_types import (
    BarGenerationTarget,
    BarSkeleton,
    CompositionPlan,
    NoteEvent,
    SectionAffect,
    StructureEdge,
)
from hierarchical_sampler import ClusterNoteSampler
from music_model import MusicModel
from phrase_generator import PhraseGenerator
from repeat_harmony_proposal import RepeatHarmonyProposal
from rhythm_development import (
    RhythmCandidateScorer,
    RhythmPhrasePlanner,
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

MIDI_GENERATION_MODULES: Tuple[Tuple[str, str], ...] = (
    ("timeline", "_run_timeline_module"),
    ("planning_context", "_run_planning_context_module"),
    ("harmony", "_run_harmony_module"),
    ("notes", "_run_note_generation_module"),
    ("bass", "_run_bass_module"),
    ("render_normalize", "_run_render_normalize_module"),
    ("write_midi", "_run_write_midi_module"),
    ("harmony_diagnostics", "_run_harmony_diagnostics_module"),
    ("structure_plot", "_run_structure_plot_module"),
)
DEFAULT_MIDI_GENERATION_MODULES: Tuple[str, ...] = tuple(
    name for name, _method_name in MIDI_GENERATION_MODULES
)

# Set to a module name to run that module and skip every later module by default.
# This is an ordered pipeline switch only; it does not infer dependencies.
DEFAULT_MIDI_GENERATION_STOP_AFTER: Optional[str] = 'notes'


@dataclass
class MidiGenerationRun:
    """Mutable state passed through the ordered MIDI generation modules."""

    output_path: Path
    target_measures: int
    start_states: Optional[List[int]]
    template_file: Optional[Union[int, str]]
    variation_strength: float
    time_signature: Tuple[int, int]
    tempo: int
    seed: Optional[int]
    enable_variation: bool
    note_modules: Optional[List[str]] = None
    note_stop_after_module: Optional[str] = None
    labels: List[int] = field(default_factory=list)
    event_log: List[Dict[str, Any]] = field(default_factory=list)
    measure_context: List[Tuple[str, int, str, int, int]] = field(default_factory=list)
    breathing: List[bool] = field(default_factory=list)
    base_seed: int = 0
    structure_graph: List[StructureEdge] = field(default_factory=list)
    composition_plan: Optional[CompositionPlan] = None
    harmonic_plan: Dict[int, Any] = field(default_factory=dict)
    harmony_mode: str = "auto"
    actual_harmony_mode: str = "disabled"
    learned_harmony: Any = None
    all_notes: List[List[NoteEvent]] = field(default_factory=list)
    trace: GenerationTraceRecorder = field(default_factory=GenerationTraceRecorder)


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


class HierarchicalGenerator(HierarchicalNotePipelineMixin, HierarchicalPlanningMixin):
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
        modules: Optional[List[str]] = None,
        stop_after_module: Optional[str] = None,
        note_modules: Optional[List[str]] = None,
        note_stop_after_module: Optional[str] = None,
    ) -> List[int]:
        """Generate a full MIDI file.

        Args:
            enable_variation: If True (default), apply controlled transforms
                to non-NEW section occurrences. Set to False for exact repeats only.
            modules: Optional ordered module list. Defaults to
                ``DEFAULT_MIDI_GENERATION_MODULES``; edit that tuple to add/remove
                a stage globally, or pass a test-specific list here.
            stop_after_module: If set, run through this module and skip all later
                modules. This does not infer dependencies; the caller controls
                the ordered module list. For a global test default, edit
                ``DEFAULT_MIDI_GENERATION_STOP_AFTER``.
            note_modules: Optional ordered stage list used inside the ``notes``
                module. Defaults to ``DEFAULT_NOTE_GENERATION_MODULES``.
            note_stop_after_module: If set, run note generation through this
                stage and skip later note stages. This does not infer dependencies.

        Returns the cluster label timeline used.
        """
        run = MidiGenerationRun(
            output_path=Path(output_path),
            target_measures=target_measures,
            start_states=start_states,
            template_file=template_file,
            variation_strength=variation_strength,
            time_signature=time_signature,
            tempo=tempo,
            seed=seed,
            enable_variation=enable_variation,
            note_modules=note_modules,
            note_stop_after_module=note_stop_after_module,
        )
        cfg = self.config.get("generation_trace", {})
        if isinstance(cfg, dict) and not cfg.get("enabled", True):
            run.trace.enabled = False
        effective_stop = (
            stop_after_module
            if stop_after_module is not None
            else DEFAULT_MIDI_GENERATION_STOP_AFTER
        )
        selected_modules = self._resolve_midi_generation_modules(modules, effective_stop)
        self._last_midi_generation_modules = selected_modules
        for module_name in selected_modules:
            self._midi_generation_module(module_name)(run)
            run.trace.record_module(run, module_name)
        run.trace.write(run, self)
        return run.labels

    def _resolve_midi_generation_modules(
        self,
        modules: Optional[List[str]],
        stop_after_module: Optional[str],
    ) -> List[str]:
        selected = list(modules or DEFAULT_MIDI_GENERATION_MODULES)
        known = set(self._midi_generation_modules())
        unknown = [name for name in selected if name not in known]
        if unknown:
            raise KeyError(f"Unknown MIDI generation module(s): {', '.join(unknown)}")
        if stop_after_module is not None:
            if stop_after_module not in selected:
                raise KeyError(f"stop_after_module '{stop_after_module}' is not in the active module list")
            selected = selected[:selected.index(stop_after_module) + 1]
        return selected

    def _midi_generation_modules(self) -> Dict[str, Any]:
        return {
            name: getattr(self, method_name)
            for name, method_name in MIDI_GENERATION_MODULES
        }

    def _midi_generation_module(self, name: str):
        return self._midi_generation_modules()[name]

    def _resolve_note_generation_modules(
        self,
        modules: Optional[List[str]],
        stop_after_module: Optional[str],
    ) -> List[str]:
        selected = list(modules or DEFAULT_NOTE_GENERATION_MODULES)
        known = set(NOTE_GENERATION_MODULES)
        unknown = [name for name in selected if name not in known]
        if unknown:
            raise KeyError(f"Unknown note generation module(s): {', '.join(unknown)}")
        if stop_after_module is not None:
            if stop_after_module not in selected:
                raise KeyError(f"note_stop_after_module '{stop_after_module}' is not in the active note module list")
            selected = selected[:selected.index(stop_after_module) + 1]
        return selected

    def _run_timeline_module(self, run: "MidiGenerationRun") -> None:
        target_measures = run.target_measures
        start_states = run.start_states
        template_file = run.template_file
        variation_strength = run.variation_strength
        seed = run.seed
        # 1. Timeline
        labels, event_log = self.generate_timeline(
            target_measures=target_measures,
            start_states=start_states,
            template_file=template_file,
            variation_strength=variation_strength,
            seed=seed,
        )

        run.labels = labels
        run.event_log = event_log

    def _run_planning_context_module(self, run: "MidiGenerationRun") -> None:
        labels = run.labels
        event_log = run.event_log
        seed = run.seed
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
        run.measure_context = measure_context
        run.breathing = breathing
        run.base_seed = base_seed
        run.structure_graph = structure_graph
        run.composition_plan = composition_plan

    def _run_harmony_module(self, run: "MidiGenerationRun") -> None:
        measure_context = run.measure_context
        composition_plan = run.composition_plan
        base_seed = run.base_seed
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

        run.harmonic_plan = harmonic_plan
        run.harmony_mode = harmony_mode
        run.actual_harmony_mode = actual_harmony_mode
        run.learned_harmony = learned_harmony

    def _run_bass_module(self, run: "MidiGenerationRun") -> None:
        labels = run.labels
        all_notes = run.all_notes
        harmonic_plan = run.harmonic_plan
        base_seed = run.base_seed
        time_signature = run.time_signature
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

    def _run_render_normalize_module(self, run: "MidiGenerationRun") -> None:
        all_notes = run.all_notes
        time_signature = run.time_signature
        # 3d. Rendering: clamp overlaps on every measure (melody only — bass is
        # added after transforms so it never participates in clamping).
        clamp_overlaps(all_notes, self.config)
        clamp_measure_bounds(all_notes, time_signature)
        ensure_final_bar_end(all_notes, time_signature)
        clamp_measure_bounds(all_notes, time_signature)

    def _run_write_midi_module(self, run: "MidiGenerationRun") -> None:
        all_notes = run.all_notes
        output_path = run.output_path
        tempo = run.tempo
        time_signature = run.time_signature
        # 4. Write MIDI via mido (silence = absence of note events)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_midi(all_notes, output_path, tempo, time_signature)
        log.info("Wrote MIDI to %s (%d measures).", output_path, len(all_notes))

        run.output_path = output_path

    def _run_harmony_diagnostics_module(self, run: "MidiGenerationRun") -> None:
        output_path = run.output_path
        labels = run.labels
        measure_context = run.measure_context
        all_notes = run.all_notes
        harmonic_plan = run.harmonic_plan
        harmony_mode = run.harmony_mode
        actual_harmony_mode = run.actual_harmony_mode
        learned_harmony = run.learned_harmony
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

    def _run_structure_plot_module(self, run: "MidiGenerationRun") -> None:
        output_path = run.output_path
        labels = run.labels
        event_log = run.event_log
        # 5. Save structure visualization
        from structure_plotter import StructurePlotter
        plot_path = output_path.with_suffix('.png')
        StructurePlotter.plot(labels, event_log, self.model.n_clusters, plot_path)

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
# CLI compatibility
# ---------------------------------------------------------------------------


def main() -> None:
    from hierarchical_cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
