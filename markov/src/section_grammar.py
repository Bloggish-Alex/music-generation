#!/usr/bin/env python3
"""
Section Grammar — learn section-level structure and generate hierarchical timelines.

Trains on a collection of SectionStructureResult dicts (from section_structure.json)
and builds models for:

- Section prototypes mapped to cluster labels (via KMeans)
- Per-file section label sequences and transition statistics
- FREE block length distribution
- Per-family state-sequence variance

Usage::

    from section_grammar import SectionGrammar

    grammar = SectionGrammar.fit("models/my_model")
    timeline = grammar.generate(target_measures=200, seed=42)
    # timeline is a list of (kind, label, cluster_labels) tuples
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("section_grammar")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FileStructure:
    """Section structure extracted from one file, with clusters mapped."""

    filename: str
    n_measures: int
    n_families: int

    # Ordered section labels from mixed_sequence (no FREE items)
    label_sequence: List[str]  # e.g. ["A", "A", "B", "A"]

    # Per-family prototype cluster labels + length
    prototypes: Dict[str, List[int]]  # "A" -> [2, 2, 0, 0, 7, 7, 1]

    # Per-family variance (mean edit distance per bar)
    variance: Dict[str, float]  # "A" -> 0.12

    # Per-family occurrence cluster sequences
    occurrences: Dict[str, List[Tuple[int, int, List[int]]]]  # "A" -> [(start, end, [2,2,0,...]), ...]

    # FREE block lengths (in bars)
    free_lengths: List[int]  # [2, 4, 1]

    # Sequence of SECTION and FREE for reconstruction
    mixed_structure: List[Dict[str, Any]]  # [{"kind": "SECTION", "label": "A", "len": 7}, ...]


@dataclass
class SectionGrammar:
    """Learned section-level grammar for hierarchical generation.

    Attributes:
        files: Per-file extracted structures.
        n_clusters: Number of KMeans clusters from the parent model.
        global_free_lengths: All FREE block lengths pooled across files.
        global_section_lengths: All section lengths pooled across files.
        global_n_families_dist: Distribution of how many families per piece.
    """

    files: List[FileStructure] = field(default_factory=list)
    n_clusters: int = 0

    # Pooled statistics (across all files)
    global_free_lengths: List[int] = field(default_factory=list)
    global_section_lengths: List[int] = field(default_factory=list)
    global_n_families_dist: Dict[int, int] = field(default_factory=dict)

    # Cached distributions for fast sampling
    _free_len_unique: np.ndarray = field(default_factory=lambda: np.array([]))
    _free_len_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    _free_len_probs: np.ndarray = field(default_factory=lambda: np.array([]))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def fit_from_results(
        cls,
        section_results: Dict[str, Dict[str, Any]],
        clusterer: Any,
        n_clusters: int,
    ) -> "SectionGrammar":
        """Train the grammar from in-memory section results and a KMeans clusterer.

        This is the integration point called by ``MusicModel.fit()`` — no
        file I/O needed.

        Args:
            section_results: Dict of filename → SectionStructureResult.to_dict().
            clusterer: Fitted MeasureClusterer with ``predict_many()``.
            n_clusters: Number of KMeans clusters.
        """
        grammar = cls(n_clusters=n_clusters)
        all_free_lengths: List[int] = []
        all_section_lengths: List[int] = []
        n_fam_counts: Dict[int, int] = {}

        for filename, result in section_results.items():
            file_struct = cls._extract_file_structure(
                filename, result, clusterer, n_clusters,
            )
            if file_struct is None:
                continue
            grammar.files.append(file_struct)
            all_free_lengths.extend(file_struct.free_lengths)
            for fam_len in (len(seq) for seq in file_struct.prototypes.values()):
                all_section_lengths.append(fam_len)

            nf = file_struct.n_families
            n_fam_counts[nf] = n_fam_counts.get(nf, 0) + 1

        grammar.global_free_lengths = all_free_lengths
        grammar.global_section_lengths = all_section_lengths
        grammar.global_n_families_dist = n_fam_counts

        if all_free_lengths:
            unique, counts = np.unique(all_free_lengths, return_counts=True)
            grammar._free_len_unique = unique
            grammar._free_len_counts = counts
            grammar._free_len_probs = counts.astype(np.float64) / counts.sum()

        log.info(
            "SectionGrammar trained: %d files, %d unique FREE lengths, "
            "%d total FREE blocks.",
            len(grammar.files),
            len(all_free_lengths),
            len(all_free_lengths),
        )
        return grammar

    @classmethod
    def fit(
        cls,
        model_dir: str | Path,
        section_structure_path: str | Path | None = None,
    ) -> "SectionGrammar":
        """Train the grammar from a trained MusicModel directory on disk.

        Requires:
          - ``section_structure.json`` in the model directory (produced by
            ``MusicModel.fit(..., analyze_sections=True)``)
          - ``clusterer.pkl`` to map prototype bar vectors → cluster labels
        """
        model_dir = Path(model_dir)

        if section_structure_path is None:
            section_structure_path = model_dir / "section_structure.json"
        else:
            section_structure_path = Path(section_structure_path)

        with open(section_structure_path) as f:
            all_results: Dict[str, Dict[str, Any]] = json.load(f)

        from measure_clustering import MeasureClusterer

        clusterer = MeasureClusterer.load(str(model_dir / "clusterer.pkl"))
        n_clusters = clusterer.centroids.shape[0] if clusterer.centroids is not None else 0

        return cls.fit_from_results(all_results, clusterer, n_clusters)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the grammar as JSON alongside a small npz for distributions."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Convert FileStructure to serialisable dicts
        files_data = []
        for fs in self.files:
            files_data.append({
                "filename": fs.filename,
                "n_measures": fs.n_measures,
                "n_families": fs.n_families,
                "label_sequence": fs.label_sequence,
                "prototypes": fs.prototypes,
                "variance": fs.variance,
                "occurrences": {
                    k: [(s, e, seq) for s, e, seq in v]
                    for k, v in fs.occurrences.items()
                },
                "free_lengths": fs.free_lengths,
                "mixed_structure": fs.mixed_structure,
            })

        with open(path / "grammar.json", "w") as f:
            json.dump({
                "n_clusters": self.n_clusters,
                "files": files_data,
                "global_n_families_dist": self.global_n_families_dist,
            }, f, indent=2)

        np.savez(
            path / "grammar_dist.npz",
            free_len_unique=self._free_len_unique,
            free_len_counts=self._free_len_counts,
            free_len_probs=self._free_len_probs,
            global_free_lengths=np.array(self.global_free_lengths, dtype=np.int64),
            global_section_lengths=np.array(self.global_section_lengths, dtype=np.int64),
        )

        log.info("Saved SectionGrammar to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SectionGrammar":
        """Load a saved grammar."""
        path = Path(path)

        with open(path / "grammar.json") as f:
            data = json.load(f)

        dist_data = np.load(path / "grammar_dist.npz")

        grammar = cls(n_clusters=data["n_clusters"])
        grammar.global_free_lengths = dist_data["global_free_lengths"].tolist()
        grammar.global_section_lengths = dist_data["global_section_lengths"].tolist()
        grammar.global_n_families_dist = {
            int(k): v for k, v in data["global_n_families_dist"].items()
        }
        grammar._free_len_unique = dist_data["free_len_unique"]
        grammar._free_len_counts = dist_data["free_len_counts"]
        grammar._free_len_probs = dist_data["free_len_probs"]

        for fd in data["files"]:
            fs = FileStructure(
                filename=fd["filename"],
                n_measures=fd["n_measures"],
                n_families=fd["n_families"],
                label_sequence=fd["label_sequence"],
                prototypes={
                    k: list(map(int, v)) for k, v in fd["prototypes"].items()
                },
                variance={
                    k: float(v) for k, v in fd["variance"].items()
                },
                occurrences={
                    k: [(int(s), int(e), list(map(int, seq))) for s, e, seq in v]
                    for k, v in fd["occurrences"].items()
                },
                free_lengths=[int(x) for x in fd["free_lengths"]],
                mixed_structure=fd["mixed_structure"],
            )
            grammar.files.append(fs)

        log.info("Loaded SectionGrammar from %s (%d files).", path, len(grammar.files))
        return grammar

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_structure(
        self,
        template_file: int | str | None = None,
        seed: int | None = None,
    ) -> Tuple[List[str], List[int], FileStructure]:
        """Generate a high-level section structure.

        Returns:
            section_labels: e.g. ["A", "A", "B", "A"]
            free_lengths:   e.g. [3, 2, 4]  (one per gap between sections)
            file_struct:    The FileStructure used as template
        """
        rng = np.random.RandomState(seed)

        # Pick a template file
        if template_file is None:
            fs = self.files[rng.randint(0, len(self.files))]
        elif isinstance(template_file, int):
            fs = self.files[template_file % len(self.files)]
        else:
            match = next(
                (f for f in self.files
                 if f.filename == template_file
                 or f.filename.endswith(template_file)
                 or Path(f.filename).stem == template_file),
                None,
            )
            if match is None:
                raise KeyError(
                    f"No file matching '{template_file}' in grammar "
                    f"({len(self.files)} files)"
                )
            fs = match

        # Sample FREE lengths from global distribution (or file-specific)
        n_gaps = len(fs.label_sequence) - 1
        free_lengths = self._sample_free_lengths(n_gaps, rng)

        return fs.label_sequence, free_lengths, fs

    def generate_section_content(
        self,
        label: str,
        file_struct: FileStructure,
        vary: bool = False,
        variation_strength: float = 0.3,
        seed: int | None = None,
    ) -> List[int]:
        """Generate a cluster label sequence for one section occurrence.

        Args:
            label: Section label ("A", "B", etc.) within file_struct.
            file_struct: The template file structure.
            vary: If True, perturb the prototype (for VARIANT / RETURN).
            variation_strength: Probability of substituting a state within
                the section (0 = prototype, 1 = random).

        Returns:
            Cluster label sequence for this section.
        """
        rng = np.random.RandomState(seed)
        prototype = file_struct.prototypes.get(label, [])

        if not vary or variation_strength <= 0 or len(prototype) == 0:
            return list(prototype)

        result = list(prototype)
        for i in range(len(result)):
            if rng.random() < variation_strength:
                # Prefer states already used by this section family.  This keeps
                # VARIANT/RETURN close to the prototype instead of jumping to an
                # arbitrary cluster with unrelated texture.
                local = set(prototype[max(0, i - 2):i] + prototype[i + 1:i + 3])
                local.update(prototype)
                candidates = [c for c in sorted(local) if c != result[i]]
                if not candidates:
                    candidates = [
                        c for c in range(self.n_clusters) if c != result[i]
                    ]
                if candidates:
                    result[i] = int(rng.choice(candidates))

        return result

    def generate_full_timeline(
        self,
        target_measures: int = 200,
        template_file: int | str | None = None,
        variation_strength: float = 0.3,
        seed: int | None = None,
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """Generate a complete cluster label timeline.

        Cycles through the template section structure (possibly multiple
        times) until *target_measures* is reached.  FREE blocks are marked
        as ``-1`` placeholders — the caller fills them with a PhraseGenerator.

        Returns:
            labels:        Full timeline of cluster labels (-1 = FREE placeholder).
            event_log:     List of events describing each segment.
        """
        rng = np.random.RandomState(seed)
        section_labels, free_lengths, fs = self.generate_structure(
            template_file=template_file,
            seed=seed,
        )

        all_labels: List[int] = []
        event_log: List[Dict[str, Any]] = []
        label_seen: set[str] = set()
        cycle = 0

        while len(all_labels) < target_measures:
            for i, label in enumerate(section_labels):
                if len(all_labels) >= target_measures:
                    break

                # Determine role
                if cycle == 0 and i == 0:
                    role = "NEW"
                    vary = False
                elif label == section_labels[i - 1] if i > 0 else (label == section_labels[-1]):
                    role = "REPEAT"
                    vary = False
                elif label in label_seen:
                    role = "RETURN"
                    vary = variation_strength > 0
                else:
                    role = "NEW"
                    vary = False

                content = self.generate_section_content(
                    label, fs, vary=vary,
                    variation_strength=variation_strength,
                    seed=None if seed is None else seed + cycle * 10000 + i * 1000,
                )

                all_labels.extend(content)
                event_log.append({
                    "kind": "SECTION",
                    "label": label,
                    "role": role,
                    "cycle": cycle,
                    "length": len(content),
                    "labels": content,
                })

                # FREE block between sections (skip after last section of each cycle)
                is_last_in_cycle = (i == len(section_labels) - 1)
                if not is_last_in_cycle:
                    free_len = free_lengths[i] if i < len(free_lengths) else 4
                    all_labels.extend([-1] * free_len)
                    event_log.append({
                        "kind": "FREE",
                        "length": free_len,
                        "labels": [-1] * free_len,
                    })

                label_seen.add(label)

            cycle += 1

        # Trim to exact target
        if len(all_labels) > target_measures:
            all_labels = all_labels[:target_measures]

        log.info(
            "Generated timeline: %d measures, %d sections, %d FREE blocks, "
            "%d cycles.",
            len(all_labels),
            sum(1 for e in event_log if e["kind"] == "SECTION"),
            sum(1 for e in event_log if e["kind"] == "FREE"),
            cycle,
        )
        return all_labels, event_log

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "SECTION GRAMMAR SUMMARY",
            "=" * 60,
            f"  Files analysed:          {len(self.files)}",
            f"  Clusters (KMeans k):     {self.n_clusters}",
            f"  Total FREE blocks:       {len(self.global_free_lengths)}",
            f"  FREE len range:          {min(self.global_free_lengths, default=0)}"
            f" – {max(self.global_free_lengths, default=0)} bars",
            f"  Section len range:       {min(self.global_section_lengths, default=0)}"
            f" – {max(self.global_section_lengths, default=0)} bars",
            "",
            "  Families per piece:",
        ]
        for nf, count in sorted(self.global_n_families_dist.items()):
            lines.append(f"    {nf} family(s): {count} piece(s)")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_file_structure(
        filename: str,
        result: Dict[str, Any],
        clusterer: Any,  # MeasureClusterer
        n_clusters: int,
    ) -> FileStructure | None:
        """Convert one file's SectionStructureResult dict to a FileStructure."""
        if not result.get("families"):
            return None

        families = result["families"]
        mixed = result["mixed_sequence"]

        # Build label sequence (SECTION items only, in order)
        label_sequence: List[str] = []
        free_lengths: List[int] = []
        mixed_structure: List[Dict[str, Any]] = []

        for item in mixed:
            if item["kind"] == "SECTION":
                label_sequence.append(item["label"])
                mixed_structure.append({
                    "kind": "SECTION",
                    "label": item["label"],
                    "role": item.get("role", "NEW"),
                    "len": item["end"] - item["start"],
                })
            elif item["kind"] == "FREE":
                length = item["end"] - item["start"]
                free_lengths.append(length)
                mixed_structure.append({
                    "kind": "FREE",
                    "len": length,
                })

        # Map prototypes to cluster labels
        prototypes: Dict[str, List[int]] = {}
        variance: Dict[str, float] = {}
        occurrences: Dict[str, List[Tuple[int, int, List[int]]]] = {}

        for fam in families:
            label = fam["label"]
            proto_tokens = fam["prototype_tokens"]

            # Convert 20-D prototype vectors → 8-D → cluster labels
            proto_clusters = SectionGrammar._vectors_to_clusters(
                proto_tokens, clusterer,
            )
            prototypes[label] = proto_clusters

            # Map each occurrence's tokens to clusters
            occ_list: List[Tuple[int, int, List[int]]] = []
            for occ in fam["occurrences"]:
                occ_start = occ["start"]
                occ_end = occ["end"]
                occ_tokens = result["tokens"][occ_start:occ_end]
                occ_clusters = SectionGrammar._vectors_to_clusters(
                    occ_tokens, clusterer,
                )
                occ_list.append((occ_start, occ_end, occ_clusters))
            occurrences[label] = occ_list

            # Compute variance as normalized edit distance between
            # prototype cluster sequence and each occurrence
            var = SectionGrammar._compute_variance(proto_clusters, occ_list)
            variance[label] = var

        return FileStructure(
            filename=filename,
            n_measures=result["sequence_length"],
            n_families=len(families),
            label_sequence=label_sequence,
            prototypes=prototypes,
            variance=variance,
            occurrences=occurrences,
            free_lengths=free_lengths,
            mixed_structure=mixed_structure,
        )

    @staticmethod
    def _vectors_to_clusters(
        tokens: List[Any],
        clusterer: Any,
    ) -> List[int]:
        """Convert bar vectors (list of lists) to cluster labels.

        Each token is a 20-D list.  We extract the first 8 dimensions
        (texture features) and predict via the KMeans clusterer.
        """
        if not tokens:
            return []

        import numpy as np
        from measure_clustering import MeasureVector

        vecs = [
            MeasureVector.from_array(
                np.array(t[:8] if isinstance(t, (list, tuple)) else t,
                        dtype=np.float64),
            )
            for t in tokens
        ]
        return clusterer.predict_many(vecs).tolist()

    @staticmethod
    def _compute_variance(
        prototype_clusters: List[int],
        occurrences: List[Tuple[int, int, List[int]]],
    ) -> float:
        """Mean normalised edit distance per bar between prototype and each occurrence."""
        if not prototype_clusters or not occurrences:
            return 0.0

        total = 0.0
        for _, _, occ_labels in occurrences:
            # Count differing positions
            mismatches = sum(
                1 for p, o in zip(prototype_clusters, occ_labels) if p != o
            )
            # Handle length mismatch
            len_diff = abs(len(prototype_clusters) - len(occ_labels))
            total += (mismatches + len_diff) / max(
                len(prototype_clusters), len(occ_labels), 1,
            )

        return total / len(occurrences)

    def _sample_free_lengths(
        self, n: int, rng: np.random.RandomState,
    ) -> List[int]:
        """Sample *n* FREE block lengths from the empirical distribution."""
        if n <= 0:
            return []

        if len(self._free_len_probs) == 0:
            # Fallback: uniform 1-8
            return [int(rng.randint(1, 9)) for _ in range(n)]

        indices = rng.choice(
            len(self._free_len_unique),
            size=n,
            p=self._free_len_probs,
        )
        return [int(self._free_len_unique[i]) for i in indices]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description="Section Grammar — train and inspect section-level structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default="./models/corelli",
        help="Path to trained MusicModel directory with section_structure.json.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save the trained grammar to a directory.",
    )
    parser.add_argument(
        "--generate",
        type=int,
        default=None,
        metavar="N_MEASURES",
        help="Generate a timeline and print the structure.",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="File index or name to use as structure template.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    grammar = SectionGrammar.fit(args.model_dir)
    print()
    print(grammar.summary())

    if args.generate:
        print(f"\n--- Generated Structure (target={args.generate} measures) ---")
        labels, events = grammar.generate_full_timeline(
            target_measures=args.generate,
            template_file=args.template,
            seed=args.seed,
        )
        for event in events:
            if event["kind"] == "SECTION":
                print(
                    f"  [{event['label']}] {event['role']:7s}  "
                    f"{event['length']:3d} bars  states={event['labels']}"
                )
            else:
                print(f"  [FREE]         {event['length']:3d} bars")

        print(f"\n  Total timeline: {len(labels)} measures "
              f"({sum(1 for l in labels if l == -1)} FREE placeholders)")

    if args.save:
        grammar.save(args.save)
        print(f"\nGrammar saved to {args.save}")


if __name__ == "__main__":
    main()
