#!/usr/bin/env python3
"""Compare encoder/codebook variants without changing the training pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from bar_clustering import GlobalCodebookClusterer
from bar_density import TokenDensityAnalyzer
from config_loader import ConfigLoader
from core_data import BarRecord, SongRecord
from edit_distance import EditDistanceCalculator
from model_analysis import MarkdownReport
from model_store import ModelBundle


DEFAULT_VARIANTS = (
    "relative_token",
    "rhythm_only",
    "contour_only",
    "pitch_only",
    "rhythm_contour_pitch",
)


@dataclass(frozen=True)
class EncoderVariantResult:
    """Summary of one encoder/codebook experiment."""

    name: str
    config: Dict[str, Any]
    total_assignments: int
    used_label_count: int
    singleton_label_count: int
    singleton_ratio: float
    max_label: Optional[str]
    max_label_count: int
    max_label_ratio: float
    entropy: float
    normalized_entropy: float
    effective_label_count: float
    actual_codebook_size: int
    assignment_distance: Dict[str, Any]
    medoid_distance_matrix: Dict[str, Any]
    top_labels: List[Dict[str, Any]]
    top_label_examples: List[Dict[str, Any]]


class EncoderVariantConfigFactory:
    """Build isolated style-config copies for each variant."""

    def __init__(self, base_config: Dict[str, Any]) -> None:
        self.base_config = base_config

    def build(self, variant: str) -> Dict[str, Any]:
        config = copy.deepcopy(self.base_config)
        distance = dict(config.get("distance_matrix", {}))
        multi = dict(distance.get("multi_channel", {}))
        distance["normalize_distance"] = bool(distance.get("normalize_distance", True))
        distance["token_strategy"] = "relative"
        distance.setdefault("context_features", {})
        self._disable_context(distance)

        if variant == "relative_token":
            distance["backend"] = "token"
        elif variant == "rhythm_only":
            distance["backend"] = "multi_channel"
            multi.update({"rhythm_weight": 1.0, "contour_weight": 0.0, "pitch_weight": 0.0})
        elif variant == "contour_only":
            distance["backend"] = "multi_channel"
            multi.update({"rhythm_weight": 0.0, "contour_weight": 1.0, "pitch_weight": 0.0})
        elif variant == "pitch_only":
            distance["backend"] = "multi_channel"
            multi.update({"rhythm_weight": 0.0, "contour_weight": 0.0, "pitch_weight": 1.0})
        elif variant == "rhythm_contour_pitch":
            distance["backend"] = "multi_channel"
            multi.update({"rhythm_weight": 0.45, "contour_weight": 0.35, "pitch_weight": 0.20})
        elif variant == "root_pitch_class_relative":
            distance["backend"] = "token"
            distance["token_strategy"] = "root_pitch_class_relative"
        else:
            raise ValueError(f"Unknown encoder variant: {variant}")

        distance["multi_channel"] = multi
        config["distance_matrix"] = distance
        return config

    def _disable_context(self, distance: Dict[str, Any]) -> None:
        context = distance.get("context_features", {})
        if not isinstance(context, dict):
            context = {}
        root = dict(context.get("root_pitch_class", {}))
        position = dict(context.get("position_bucket", {}))
        root["enabled"] = False
        position["enabled"] = False
        context["root_pitch_class"] = root
        context["position_bucket"] = position
        distance["context_features"] = context


class EncoderVariantAnalyzer:
    """Run codebook construction variants and summarize label reuse."""

    def __init__(self, base_config: Dict[str, Any], variants: Sequence[str]) -> None:
        self.base_config = base_config
        self.variants = list(variants)
        self.config_factory = EncoderVariantConfigFactory(base_config)

    def analyze(self, songs: Sequence[SongRecord]) -> List[EncoderVariantResult]:
        results: List[EncoderVariantResult] = []
        for variant in self.variants:
            variant_songs = self._clone_songs(songs)
            config = self.config_factory.build(variant)
            distance = EditDistanceCalculator.from_style_config(config)
            bars = [bar for song in variant_songs for bar in song.bars]
            distance.fit_corpus(bars)
            clusterer = GlobalCodebookClusterer.from_style_config(config, distance)
            clusterer.assign(variant_songs)
            results.append(self._summarize(variant, config, variant_songs, clusterer.diagnostics))
        return results

    def _summarize(
        self,
        variant: str,
        config: Dict[str, Any],
        songs: Sequence[SongRecord],
        diagnostics: Dict[str, Any],
    ) -> EncoderVariantResult:
        counts = Counter(
            int(bar.codebook_id)
            for song in songs
            for bar in song.bars
            if bar.codebook_id is not None
        )
        values = np.asarray([int(value) for value in counts.values() if int(value) > 0], dtype=np.float64)
        total = int(np.sum(values)) if values.size else 0
        probabilities = values / max(1.0, float(total)) if values.size else np.asarray([], dtype=np.float64)
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))) if values.size else 0.0
        used = int(len(values))
        max_label, max_count = self._max_count(counts)
        singleton = int(np.sum(values == 1)) if values.size else 0
        return EncoderVariantResult(
            name=variant,
            config={
                "distance_matrix": config.get("distance_matrix", {}),
                "global_agglomerative_clustering": config.get("global_agglomerative_clustering", {}),
            },
            total_assignments=total,
            used_label_count=used,
            singleton_label_count=singleton,
            singleton_ratio=float(singleton / max(1, used)),
            max_label=str(max_label) if max_label is not None else None,
            max_label_count=int(max_count),
            max_label_ratio=float(max_count / max(1, total)),
            entropy=entropy,
            normalized_entropy=float(entropy / math.log(used)) if used > 1 else 0.0,
            effective_label_count=float(math.exp(entropy)) if values.size else 0.0,
            actual_codebook_size=int(diagnostics.get("actual_codebook_size", 0)),
            assignment_distance=dict(diagnostics.get("assignment_distance", {})),
            medoid_distance_matrix=dict(diagnostics.get("medoid_distance_matrix", {})),
            top_labels=self._top_labels(counts, total),
            top_label_examples=self._top_label_examples(songs, counts),
        )

    def _clone_songs(self, songs: Sequence[SongRecord]) -> List[SongRecord]:
        return [SongRecord.from_dict(song.to_dict()) for song in songs]

    def _max_count(self, counts: Counter) -> tuple[Optional[int], int]:
        if not counts:
            return None, 0
        label, count = max(counts.items(), key=lambda item: int(item[1]))
        return int(label), int(count)

    def _top_labels(self, counts: Counter, total: int, limit: int = 20) -> List[Dict[str, Any]]:
        return [
            {
                "label": str(label),
                "count": int(count),
                "ratio": float(count / max(1, total)),
            }
            for label, count in counts.most_common(limit)
        ]

    def _top_label_examples(
        self,
        songs: Sequence[SongRecord],
        counts: Counter,
        limit: int = 8,
        examples_per_label: int = 4,
    ) -> List[Dict[str, Any]]:
        bars_by_label: Dict[int, List[BarRecord]] = defaultdict(list)
        for song in songs:
            for bar in song.bars:
                if bar.codebook_id is not None:
                    bars_by_label[int(bar.codebook_id)].append(bar)
        examples = []
        for label, count in counts.most_common(limit):
            label_bars = bars_by_label[int(label)]
            examples.append({
                "label": str(label),
                "count": int(count),
                "examples": [
                    {
                        "song_id": bar.song_id,
                        "bar_index": int(bar.bar_index),
                        "relative_tokens": list(bar.relative_tokens),
                        "absolute_tokens": list(bar.absolute_tokens),
                        "token_variance": round(float(bar.token_variance), 6),
                        "sharing_score": round(float(bar.sharing_score), 6),
                    }
                    for bar in label_bars[:examples_per_label]
                ],
            })
        return examples


class EncoderVariantReport:
    """Markdown renderer for encoder variant comparisons."""

    def __init__(self, results: Sequence[EncoderVariantResult], songs: Sequence[SongRecord]) -> None:
        self.results = list(results)
        self.songs = list(songs)

    def write(self, output_path: Path) -> None:
        report = MarkdownReport()
        bars = [bar for song in self.songs for bar in song.bars]
        report.heading("Encoder Variant Analysis")
        report.paragraph(
            "This report compares codebook/label reuse before HMM training. "
            "Lower singleton ratio and lower used_label_count usually mean the encoder is producing a more reusable symbolic vocabulary. "
            "These metrics must still be checked against musical cluster examples."
        )
        report.table([
            "Metric",
            "Value",
        ], [
            ["song_count", len(self.songs)],
            ["bar_count", len(bars)],
            ["variants", ", ".join(result.name for result in self.results)],
        ])
        report.heading("Variant Summary", 2)
        report.table([
            "Variant",
            "Assignments",
            "Used labels",
            "Actual codebook",
            "Singleton labels",
            "Singleton ratio",
            "Max label ratio",
            "Normalized entropy",
            "Effective labels",
            "Mean assignment distance",
        ], [
            [
                result.name,
                result.total_assignments,
                result.used_label_count,
                result.actual_codebook_size,
                result.singleton_label_count,
                result.singleton_ratio,
                result.max_label_ratio,
                result.normalized_entropy,
                result.effective_label_count,
                result.assignment_distance.get("mean", ""),
            ]
            for result in self.results
        ])
        report.heading("Interpretation Guide", 2)
        report.table([
            "Signal",
            "Meaning",
            "Typical action",
        ], [
            ["singleton_ratio high", "Many labels have only one training bar.", "Use coarser representation or reduce codebook granularity."],
            ["used_label_count close to assignments", "Encoder is almost memorizing physical bars.", "Remove surface detail from tokens or change distance weights."],
            ["max_label_ratio high", "One label absorbs too many bars.", "Inspect top examples; distance may be too coarse or rare bucket too broad."],
            ["effective labels much lower than used labels", "Many labels exist but probability mass is concentrated.", "Check both singleton tail and dominant clusters."],
            ["assignment distance high", "Bars are far from their nearest codebook representative.", "Increase codebook size or improve representation before clustering."],
        ])

        for result in self.results:
            self._write_variant(report, result)
        report.write(output_path)

    def _write_variant(self, report: MarkdownReport, result: EncoderVariantResult) -> None:
        report.heading(f"Variant: {result.name}", 2)
        distance = result.config.get("distance_matrix", {})
        clustering = result.config.get("global_agglomerative_clustering", {})
        report.table([
            "Config",
            "Value",
        ], [
            ["backend", distance.get("backend")],
            ["token_strategy", distance.get("token_strategy")],
            ["multi_channel", json.dumps(distance.get("multi_channel", {}), sort_keys=True)],
            ["codebook_size", clustering.get("codebook_size")],
            ["codebook_distance_threshold", clustering.get("codebook_distance_threshold")],
            ["assignment_distance_threshold", clustering.get("assignment_distance_threshold")],
            ["codebook_clustering_strategy", clustering.get("codebook_clustering_strategy")],
        ])
        report.heading("Top Labels", 3)
        report.table([
            "Label",
            "Count",
            "Ratio",
        ], [
            [item["label"], item["count"], item["ratio"]]
            for item in result.top_labels
        ])
        report.heading("Distance Diagnostics", 3)
        report.table([
            "Metric",
            "Value",
        ], [
            ["assignment_mean", result.assignment_distance.get("mean", "")],
            ["assignment_median", result.assignment_distance.get("median", "")],
            ["assignment_p95", result.assignment_distance.get("quantiles", {}).get("95", "")],
            ["medoid_distance_mean", result.medoid_distance_matrix.get("mean", "")],
            ["medoid_distance_median", result.medoid_distance_matrix.get("median", "")],
            ["cluster_reduction", json.dumps(result.medoid_distance_matrix.get("cluster_reduction", {}), sort_keys=True)],
        ])
        report.heading("Top Label Examples", 3)
        for label in result.top_label_examples:
            report.paragraph(f"Label `{label['label']}` count={label['count']}")
            report.table([
                "Song",
                "Bar",
                "Relative tokens",
                "Variance",
                "Sharing",
            ], [
                [
                    example["song_id"],
                    example["bar_index"],
                    example["relative_tokens"],
                    example["token_variance"],
                    example["sharing_score"],
                ]
                for example in label["examples"]
            ])


class EncoderVariantInputLoader:
    """Load SongRecord data from model, parsed JSON, or raw music directory."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def load(
        self,
        model_dir: Optional[Path],
        model_bundle: Optional[Path],
        songs_json: Optional[Path],
        music_dir: Optional[Path],
    ) -> List[SongRecord]:
        if model_dir is not None or model_bundle is not None:
            bundle = ModelBundle.load(model_dir if model_dir is not None else model_bundle.parent)
            return self._songs_from_observation_pools(bundle)
        if songs_json is not None:
            payload = json.loads(songs_json.read_text(encoding="utf-8"))
            return [SongRecord.from_dict(item) for item in payload.get("songs", [])]
        if music_dir is not None:
            from music_input import InputParser

            return InputParser.from_style_config(self.config).parse_directory(music_dir)
        raise ValueError("One input source is required.")

    def _songs_from_observation_pools(self, bundle: ModelBundle) -> List[SongRecord]:
        bars_by_song: Dict[tuple[str, str], List[BarRecord]] = defaultdict(list)
        for pool in bundle.observation_to_bars.values():
            for bar in pool:
                bars_by_song[(bar.song_id, bar.file_path)].append(bar)
        songs = []
        for (song_id, file_path), bars in sorted(bars_by_song.items(), key=lambda item: item[0]):
            ordered = sorted(bars, key=lambda bar: int(bar.bar_index))
            songs.append(SongRecord(
                song_id=song_id,
                file_path=file_path,
                genre=ordered[0].genre if ordered else None,
                form=ordered[0].form if ordered else None,
                bars=ordered,
            ))
        return songs


class EncoderVariantCLI:
    """CLI entrypoint."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Compare encoder/codebook variants on the same bar corpus.")
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--model-dir", type=Path)
        source.add_argument("--model-bundle", type=Path)
        source.add_argument("--songs-json", type=Path)
        source.add_argument("--music-dir", type=Path)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--diagnostics-output", type=Path, default=None)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        config = ConfigLoader().load(args.config)
        variants = [item.strip() for item in args.variants.split(",") if item.strip()]
        songs = EncoderVariantInputLoader(config).load(
            model_dir=args.model_dir,
            model_bundle=args.model_bundle,
            songs_json=args.songs_json,
            music_dir=args.music_dir,
        )
        results = EncoderVariantAnalyzer(config, variants).analyze(songs)
        EncoderVariantReport(results, songs).write(args.output)
        if args.diagnostics_output:
            args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
            args.diagnostics_output.write_text(
                json.dumps([self._result_to_dict(result) for result in results], indent=2),
                encoding="utf-8",
            )
        print(f"Encoder variant report -> {args.output}")

    def _result_to_dict(self, result: EncoderVariantResult) -> Dict[str, Any]:
        return {
            "name": result.name,
            "config": result.config,
            "total_assignments": result.total_assignments,
            "used_label_count": result.used_label_count,
            "singleton_label_count": result.singleton_label_count,
            "singleton_ratio": result.singleton_ratio,
            "max_label": result.max_label,
            "max_label_count": result.max_label_count,
            "max_label_ratio": result.max_label_ratio,
            "entropy": result.entropy,
            "normalized_entropy": result.normalized_entropy,
            "effective_label_count": result.effective_label_count,
            "actual_codebook_size": result.actual_codebook_size,
            "assignment_distance": result.assignment_distance,
            "medoid_distance_matrix": result.medoid_distance_matrix,
            "top_labels": result.top_labels,
            "top_label_examples": result.top_label_examples,
        }


def main() -> None:
    EncoderVariantCLI().run()


if __name__ == "__main__":
    main()
