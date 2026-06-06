#!/usr/bin/env python3
"""Generate a reviewable form.json for a music directory."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config_loader import ConfigLoader
from edit_distance import EditDistanceCalculator
from form_hmm import FormTemplate, FormTemplateLibrary
from music_input import InputParser, MUSIC_PATTERNS


log = logging.getLogger("form_json_generator")


@dataclass(frozen=True)
class FormCandidate:
    form: str
    confidence: float
    source: str
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SSMFeatures:
    bar_count: int
    average_similarity: float
    ternary_return_score: float
    binary_contrast_score: float
    rondo_return_score: float
    variation_consistency_score: float
    sonata_recap_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MusicFileDiscoverer:
    """Find supported music files under a directory."""

    def discover(self, music_dir: str | Path) -> List[Path]:
        root = Path(music_dir)
        files: List[Path] = []
        for pattern in MUSIC_PATTERNS:
            files.extend(root.rglob(pattern))
        return sorted(set(files))


class FilenameFormClassifier:
    """Infer a high-confidence form prior from file names."""

    PATTERNS = (
        (re.compile(r"\b(sonata|sonatina)\b", re.IGNORECASE), "sonata", 0.92),
        (re.compile(r"\brondo\b", re.IGNORECASE), "rondo", 0.94),
        (re.compile(r"\b(ternary|minuet|scherzo|trio)\b", re.IGNORECASE), "ternary", 0.84),
        (re.compile(r"\b(binary|allemande|courante|sarabande|gavotte|gigue)\b", re.IGNORECASE), "binary", 0.78),
        (re.compile(r"\b(variation|variations|theme[_ -]?and[_ -]?variations)\b", re.IGNORECASE), "variations", 0.82),
    )

    def classify(self, file_path: Path) -> Optional[FormCandidate]:
        searchable = f"{file_path.stem} {file_path.name}"
        for pattern, form, confidence in self.PATTERNS:
            match = pattern.search(searchable)
            if match:
                return FormCandidate(
                    form=form,
                    confidence=confidence,
                    source="filename",
                    evidence={
                        "matched_text": match.group(0),
                        "file_name": file_path.name,
                    },
                )
        return None


class SSMBuilder:
    """Build bar-level self-similarity matrices from parsed bar tokens."""

    def __init__(self, distance: EditDistanceCalculator) -> None:
        self.distance = distance

    def build_similarity(self, bars: Sequence[Any]) -> np.ndarray:
        if not bars:
            return np.zeros((0, 0), dtype=np.float64)
        distance_matrix = self.distance.build_matrix(bars)
        return np.clip(1.0 - distance_matrix, 0.0, 1.0)


class SegmentSimilarity:
    """Measure similarity between proportional regions of an SSM."""

    def score(self, ssm: np.ndarray, first: tuple[int, int], second: tuple[int, int]) -> float:
        a_start, a_end = first
        b_start, b_end = second
        if a_end <= a_start or b_end <= b_start:
            return 0.0
        block = ssm[a_start:a_end, b_start:b_end]
        if block.size == 0:
            return 0.0
        return float(np.mean(block))

    def template_ranges(self, template: FormTemplate, bar_count: int) -> List[tuple[str, int, int]]:
        total = sum(section.length for section in template.sections)
        if total <= 0 or bar_count <= 0:
            return []
        cursor = 0
        ranges: List[tuple[str, int, int]] = []
        for index, section in enumerate(template.sections):
            start = int(round(cursor * bar_count / total))
            cursor += section.length
            end = int(round(cursor * bar_count / total))
            if index == len(template.sections) - 1:
                end = bar_count
            ranges.append((section.name, max(0, start), min(bar_count, max(start + 1, end))))
        return ranges


class SSMFormClassifier:
    """Classify form candidates from recurrence patterns in an SSM."""

    def __init__(self, templates: FormTemplateLibrary) -> None:
        self.templates = templates
        self.segment = SegmentSimilarity()

    def classify(self, ssm: np.ndarray) -> List[FormCandidate]:
        features = self._features(ssm)
        raw_scores = {
            "ternary": features.ternary_return_score,
            "binary": features.binary_contrast_score,
            "rondo": features.rondo_return_score,
            "sonata": features.sonata_recap_score,
            "variations": features.variation_consistency_score,
        }
        raw_scores = {
            form: score
            for form, score in raw_scores.items()
            if form in self.templates.templates
        }
        candidates = []
        for form, score in sorted(raw_scores.items(), key=lambda item: item[1], reverse=True):
            confidence = self._confidence(score, raw_scores)
            candidates.append(FormCandidate(
                form=form,
                confidence=confidence,
                source="ssm",
                evidence={
                    "score": round(float(score), 4),
                    "features": features.to_dict(),
                    "note": "SSM classification is a review hint, not training ground truth.",
                },
            ))
        return candidates

    def _features(self, ssm: np.ndarray) -> SSMFeatures:
        n = int(ssm.shape[0])
        if n < 4:
            return SSMFeatures(n, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        avg = self._off_diagonal_mean(ssm)
        first_third = (0, max(1, n // 3))
        middle_third = (n // 3, max(n // 3 + 1, 2 * n // 3))
        last_third = (2 * n // 3, n)
        first_half = (0, max(1, n // 2))
        second_half = (n // 2, n)
        ternary = self.segment.score(ssm, first_third, last_third)
        binary_cross = self.segment.score(ssm, first_half, second_half)
        binary = max(0.0, 1.0 - binary_cross) * self._within_half_consistency(ssm)
        rondo = self._rondo_score(ssm)
        variation = self._variation_score(ssm)
        sonata = self._sonata_score(ssm)
        return SSMFeatures(
            bar_count=n,
            average_similarity=round(avg, 4),
            ternary_return_score=round(ternary, 4),
            binary_contrast_score=round(binary, 4),
            rondo_return_score=round(rondo, 4),
            variation_consistency_score=round(variation, 4),
            sonata_recap_score=round(sonata, 4),
        )

    def _off_diagonal_mean(self, ssm: np.ndarray) -> float:
        if ssm.shape[0] <= 1:
            return 0.0
        mask = ~np.eye(ssm.shape[0], dtype=bool)
        return float(np.mean(ssm[mask]))

    def _within_half_consistency(self, ssm: np.ndarray) -> float:
        n = ssm.shape[0]
        first = ssm[: n // 2, : n // 2]
        second = ssm[n // 2 :, n // 2 :]
        return float((self._off_diagonal_mean(first) + self._off_diagonal_mean(second)) / 2.0)

    def _rondo_score(self, ssm: np.ndarray) -> float:
        n = ssm.shape[0]
        anchors = [(0, max(1, n // 5)), (2 * n // 5, 3 * n // 5), (4 * n // 5, n)]
        scores = [
            self.segment.score(ssm, anchors[0], anchors[1]),
            self.segment.score(ssm, anchors[0], anchors[2]),
            self.segment.score(ssm, anchors[1], anchors[2]),
        ]
        return float(np.mean(scores))

    def _variation_score(self, ssm: np.ndarray) -> float:
        n = ssm.shape[0]
        segment_count = 4
        ranges = []
        for index in range(segment_count):
            start = int(round(index * n / segment_count))
            end = int(round((index + 1) * n / segment_count))
            ranges.append((start, max(start + 1, end)))
        scores = [
            self.segment.score(ssm, ranges[i], ranges[j])
            for i in range(segment_count)
            for j in range(i + 1, segment_count)
        ]
        return float(np.mean(scores)) if scores else 0.0

    def _sonata_score(self, ssm: np.ndarray) -> float:
        if "sonata" not in self.templates.templates:
            return 0.0
        ranges = self.segment.template_ranges(self.templates.require("sonata"), int(ssm.shape[0]))
        by_name = {name: (start, end) for name, start, end in ranges}
        pairs = [
            ("Expo_Primary", "Recap_Primary"),
            ("Expo_Secondary", "Recap_Secondary"),
            ("Expo_Codetta", "Recap_Coda"),
        ]
        scores = [
            self.segment.score(ssm, by_name[a], by_name[b])
            for a, b in pairs
            if a in by_name and b in by_name
        ]
        return float(np.mean(scores)) if scores else 0.0

    def _confidence(self, score: float, all_scores: Dict[str, float]) -> float:
        values = sorted((float(value) for value in all_scores.values()), reverse=True)
        margin = values[0] - values[1] if len(values) > 1 else values[0]
        return round(max(0.2, min(0.78, 0.35 + 0.35 * float(score) + 0.25 * margin)), 3)


class FormJsonRecordBuilder:
    """Create reviewable form.json entries from classifier candidates."""

    def __init__(self, templates: FormTemplateLibrary) -> None:
        self.templates = templates

    def build(
        self,
        file_path: Path,
        candidates: Sequence[FormCandidate],
        bar_count: Optional[int],
    ) -> Dict[str, Any]:
        best = candidates[0] if candidates else FormCandidate(
            form="ternary",
            confidence=0.2,
            source="fallback",
            evidence={"reason": "No filename or SSM evidence available."},
        )
        review_status = "auto_generated"
        selected_form = best.form
        if selected_form not in self.templates.templates:
            review_status = "needs_review"
            selected_form = "ternary"
        return {
            "form": selected_form,
            "review_status": review_status,
            "classification_source": best.source,
            "confidence": round(float(best.confidence), 3),
            "file_name": file_path.name,
            "bar_count": bar_count,
            "candidates": [candidate.to_dict() for candidate in candidates],
            "sections": self._template_sections(selected_form, bar_count),
        }

    def _template_sections(self, form_name: str, bar_count: Optional[int] = None) -> List[Dict[str, Any]]:
        template = self.templates.templates.get(form_name)
        if template is None:
            return []
        total = sum(section.length for section in template.sections)
        sections = []
        start = 0
        for index, section in enumerate(template.sections):
            if bar_count is not None and total > 0:
                scaled_start = int(round(start * bar_count / total))
                scaled_end = int(round((start + section.length) * bar_count / total))
                if index == len(template.sections) - 1:
                    scaled_end = int(bar_count)
                length = max(1, scaled_end - scaled_start)
                section_start = scaled_start
            else:
                length = section.length
                section_start = start
            payload = asdict(section)
            payload["template_length"] = section.length
            payload["start"] = section_start
            payload["length"] = length
            sections.append(payload)
            start += section.length
        return sections


class FormJsonGenerator:
    """Generate form.json metadata for a directory without training side effects."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.templates = FormTemplateLibrary.from_style_config(config)
        self.filename = FilenameFormClassifier()
        self.parser = InputParser.from_style_config(config)
        self.ssm_builder = SSMBuilder(EditDistanceCalculator.from_style_config(config))
        self.ssm_classifier = SSMFormClassifier(self.templates)
        self.record_builder = FormJsonRecordBuilder(self.templates)
        self.diagnostics: Dict[str, Any] = {"files": [], "failed_files": []}

    def generate(self, music_dir: str | Path) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for file_path in MusicFileDiscoverer().discover(music_dir):
            try:
                candidates, bar_count = self._classify_file(file_path)
                result[file_path.name] = self.record_builder.build(file_path, candidates, bar_count)
                self.diagnostics["files"].append({
                    "file_name": file_path.name,
                    "selected_form": result[file_path.name]["form"],
                    "confidence": result[file_path.name]["confidence"],
                    "source": result[file_path.name]["classification_source"],
                })
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                log.warning("Could not classify %s: %s", file_path, message)
                self.diagnostics["failed_files"].append({"file_path": str(file_path), "error": message})
                result[file_path.name] = {
                    "form": "ternary",
                    "review_status": "needs_review",
                    "classification_source": "fallback",
                    "confidence": 0.0,
                    "file_name": file_path.name,
                    "bar_count": None,
                    "candidates": [],
                    "sections": self.record_builder._template_sections("ternary"),
                    "error": message,
                }
        return result

    def _classify_file(self, file_path: Path) -> tuple[List[FormCandidate], Optional[int]]:
        filename_candidate = self.filename.classify(file_path)
        song = None
        bar_count = None
        try:
            song = self.parser.parse_file(file_path, {})
            bar_count = len(song.bars)
        except Exception:
            if filename_candidate is None:
                raise
        if filename_candidate is not None:
            return [filename_candidate], bar_count
        if song is None:
            return [], bar_count
        ssm = self.ssm_builder.build_similarity(song.bars)
        return self.ssm_classifier.classify(ssm), bar_count


class FormJsonCLI:
    """CLI for generating reviewable form.json files."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Generate a reviewable form.json for a music directory.")
        parser.add_argument("--music-dir", type=Path, required=True)
        parser.add_argument("--output", type=Path, default=None)
        parser.add_argument("--diagnostics-output", type=Path, default=None)
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--overwrite", action="store_true")
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        output = args.output or args.music_dir / "form.json"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} already exists. Use --overwrite or choose --output.")
        config = ConfigLoader().load(args.config)
        generator = FormJsonGenerator(config)
        payload = generator.generate(args.music_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if args.diagnostics_output:
            args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
            args.diagnostics_output.write_text(json.dumps(generator.diagnostics, indent=2), encoding="utf-8")
        print(f"Wrote form metadata -> {output}")


def main() -> None:
    FormJsonCLI().run()


if __name__ == "__main__":
    main()
