#!/usr/bin/env python3
"""Write a quantitative markdown quality report for a saved HMM model bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from model_analysis import (
    FormDistributionAnalysis,
    MarkdownReport,
    ModelAnalysisCharts,
    ModelBundleAnalyzer,
)


class QualityRule:
    """Score one model metric against an explicit heuristic reference range."""

    def __init__(self, name: str, value: float, good: str, status: str, note: str) -> None:
        self.name = name
        self.value = value
        self.good = good
        self.status = status
        self.note = note

    def row(self) -> List[Any]:
        return [self.name, self.value, self.good, self.status, self.note]


class ModelQualityAssessor:
    """Produce heuristic quality checks from model-bundle statistics."""

    def __init__(self, analyzer: ModelBundleAnalyzer) -> None:
        self.analyzer = analyzer

    def assess_pool(self) -> List[QualityRule]:
        pool = self.analyzer.pool_summary()
        rules = [
            self._rule(
                "observation_per_bar_ratio",
                pool["observation_per_bar_ratio"],
                "<= 0.50",
                pool["observation_per_bar_ratio"] <= 0.50,
                pool["observation_per_bar_ratio"] <= 0.80,
                "High values mean the vocabulary is too sparse for sequence learning.",
            ),
            self._rule(
                "singleton_ratio",
                pool["singleton_ratio"],
                "<= 0.30",
                pool["singleton_ratio"] <= 0.30,
                pool["singleton_ratio"] <= 0.60,
                "Many singleton observations make HMM emissions mostly smoothing-driven.",
            ),
            self._rule(
                "mean_pool_size",
                pool["mean_pool_size"],
                ">= 3.00",
                pool["mean_pool_size"] >= 3.00,
                pool["mean_pool_size"] >= 1.50,
                "Small pools reduce reliable bar realization and candidate selection.",
            ),
            self._rule(
                "max_pool_size",
                float(pool["max_pool_size"]),
                "not dominant",
                pool["max_pool_size"] <= max(10, 0.10 * max(1, pool["bar_count"])),
                pool["max_pool_size"] <= max(20, 0.25 * max(1, pool["bar_count"])),
                "Very large pools may indicate a collapsed codebook/observation bucket.",
            ),
        ]
        return rules

    def assess_layered_pools(self) -> List[QualityRule]:
        rules: List[QualityRule] = []
        for summary in self.analyzer.layered_pool_summaries():
            layer = str(summary["layer"])
            singleton_ratio = float(summary["singleton_ratio"])
            pool_per_bar_ratio = float(summary["pool_per_bar_ratio"])
            mean_pool_size = float(summary["mean_pool_size"])
            rules.extend([
                self._rule(
                    f"{layer}: singleton_ratio",
                    singleton_ratio,
                    "<= 0.30",
                    singleton_ratio <= 0.30,
                    singleton_ratio <= 0.60,
                    "Layer-specific singleton ratio; compare layers to find where sparsity is introduced.",
                ),
                self._rule(
                    f"{layer}: pool_per_bar_ratio",
                    pool_per_bar_ratio,
                    "<= 0.50",
                    pool_per_bar_ratio <= 0.50,
                    pool_per_bar_ratio <= 0.80,
                    "Layer-specific vocabulary size divided by training bars.",
                ),
                self._rule(
                    f"{layer}: mean_pool_size",
                    mean_pool_size,
                    ">= 3.00",
                    mean_pool_size >= 3.00,
                    mean_pool_size >= 1.50,
                    "Average number of bars available per pool at this layer.",
                ),
            ])
        return rules

    def assess_form(self, analysis: FormDistributionAnalysis) -> List[QualityRule]:
        row_sum_error = float(np.max(np.abs(np.sum(analysis.transmat, axis=1) - 1.0)))
        emission_row_sum_error = float(np.max(np.abs(np.sum(analysis.emissionprob, axis=1) - 1.0)))
        self_loops = np.diag(analysis.transmat)
        forward_mass = [
            float(np.sum(analysis.transmat[state, state + 1:]))
            for state in range(analysis.n_states)
        ]
        nonterminal_self_loops = self_loops[:-1] if len(self_loops) > 1 else self_loops
        nonterminal_forward_mass = forward_mass[:-1] if len(forward_mass) > 1 else forward_mass
        training_deltas = [
            float(item.get("delta", 0.0))
            for item in analysis.training_log[1:]
        ]
        negative_delta_count = sum(1 for value in training_deltas if value < -1e-6)
        final_delta = training_deltas[-1] if training_deltas else 0.0
        normalized_entropy = [
            value / np.log(max(2, analysis.n_observations))
            for value in analysis.emission_entropy
        ]
        mean_normalized_entropy = float(np.mean(normalized_entropy)) if normalized_entropy else 0.0
        return [
            self._rule(
                f"{analysis.form_name}: section-weighted TV distance",
                analysis.section_weighted_metrics.total_variation,
                "< 0.10 good, < 0.25 acceptable",
                analysis.section_weighted_metrics.total_variation < 0.10,
                analysis.section_weighted_metrics.total_variation < 0.25,
                "Difference between training observation distribution and current generation estimate.",
            ),
            self._rule(
                f"{analysis.form_name}: JS divergence",
                analysis.section_weighted_metrics.js_divergence,
                "< 0.05 good, < 0.15 acceptable",
                analysis.section_weighted_metrics.js_divergence < 0.05,
                analysis.section_weighted_metrics.js_divergence < 0.15,
                "Symmetric distribution divergence; lower is closer.",
            ),
            self._rule(
                f"{analysis.form_name}: top observation overlap",
                analysis.section_weighted_metrics.top_overlap_ratio,
                ">= 0.60",
                analysis.section_weighted_metrics.top_overlap_ratio >= 0.60,
                analysis.section_weighted_metrics.top_overlap_ratio >= 0.35,
                "Checks whether the most likely generated observations match frequent training observations.",
            ),
            self._rule(
                f"{analysis.form_name}: emission state separation",
                analysis.emission_pairwise_js["mean"],
                "0.05 - 0.35",
                0.05 <= analysis.emission_pairwise_js["mean"] <= 0.35,
                analysis.emission_pairwise_js["mean"] <= 0.50,
                "Too low means states are indistinct; too high can indicate fragmented states.",
            ),
            self._rule(
                f"{analysis.form_name}: mean normalized emission entropy",
                mean_normalized_entropy,
                "0.45 - 0.90",
                0.45 <= mean_normalized_entropy <= 0.90,
                0.30 <= mean_normalized_entropy <= 0.98,
                "Very high is too uniform; very low is too deterministic.",
            ),
            self._rule(
                f"{analysis.form_name}: transition row-sum error",
                row_sum_error,
                "< 1e-6",
                row_sum_error < 1e-6,
                row_sum_error < 1e-4,
                "Transition rows should remain normalized.",
            ),
            self._rule(
                f"{analysis.form_name}: emission row-sum error",
                emission_row_sum_error,
                "< 1e-6",
                emission_row_sum_error < 1e-6,
                emission_row_sum_error < 1e-4,
                "Emission rows should remain normalized.",
            ),
            self._rule(
                f"{analysis.form_name}: negative EM delta count",
                float(negative_delta_count),
                "0",
                negative_delta_count == 0,
                negative_delta_count <= 2,
                "Hard masking can reduce likelihood; repeated drops suggest unstable training.",
            ),
            self._rule(
                f"{analysis.form_name}: final EM delta abs",
                abs(final_delta),
                "< tol or near 0",
                abs(final_delta) < 1e-4,
                abs(final_delta) < 1e-2,
                "Large final movement means training may not have converged.",
            ),
            self._rule(
                f"{analysis.form_name}: max nonterminal self-loop",
                float(np.max(nonterminal_self_loops)) if len(nonterminal_self_loops) else 0.0,
                "context-dependent",
                float(np.max(nonterminal_self_loops)) < 0.995 if len(nonterminal_self_loops) else True,
                float(np.max(nonterminal_self_loops)) <= 1.0 if len(nonterminal_self_loops) else True,
                "A nonterminal self-loop near 1.0 can freeze transitions if generation later samples states.",
            ),
            self._rule(
                f"{analysis.form_name}: min nonterminal forward mass",
                float(np.min(nonterminal_forward_mass)) if nonterminal_forward_mass else 0.0,
                "> 0 except terminal",
                all(value > 0.001 for value in nonterminal_forward_mass),
                all(value > 0.0 for value in nonterminal_forward_mass),
                "Nonterminal states need some forward probability for left-to-right motion.",
            ),
        ]

    def problems_and_suggestions(
        self,
        pool_rules: Sequence[QualityRule],
        layered_rules: Sequence[QualityRule],
        form_rules: Sequence[QualityRule],
    ) -> List[Dict[str, str]]:
        rules = list(pool_rules) + list(layered_rules) + list(form_rules)
        problems = [rule for rule in rules if rule.status == "BAD"]
        warnings = [rule for rule in rules if rule.status == "WARN"]
        suggestions: List[Dict[str, str]] = []
        layer_summaries = {item["layer"]: item for item in self.analyzer.layered_pool_summaries()}
        codebook_singleton = float(layer_summaries.get("codebook_id", {}).get("singleton_ratio", 0.0))
        base_singleton = float(layer_summaries.get("base_composite", {}).get("singleton_ratio", 0.0))
        observation_singleton = float(layer_summaries.get("observation_id", {}).get("singleton_ratio", 0.0))
        if any("observation_per_bar_ratio" in rule.name or "singleton_ratio" in rule.name for rule in problems + warnings):
            suggestions.append({
                "problem": "Observation vocabulary is sparse.",
                "evidence": "Many observations appear once or nearly once.",
                "improvement": "Reduce codebook granularity, increase training corpus size, or avoid adding position/context dimensions until base codebook pools are stable.",
            })
        if codebook_singleton > 0.60:
            suggestions.append({
                "problem": "Sparsity is already high at codebook_id layer.",
                "evidence": "Many global codebook IDs have only one training bar before kmeans/position expansion.",
                "improvement": "Tune encoder/codebook first: reduce codebook_size, relax distance thresholds, or increase corpus size before adding composite dimensions.",
            })
        elif base_singleton - codebook_singleton > 0.20:
            suggestions.append({
                "problem": "Sparsity increases when kmeans/base composite is added.",
                "evidence": "base_composite singleton ratio is much higher than codebook_id singleton ratio.",
                "improvement": "Disable or reduce kmeans feature clustering until codebook pools have enough repeated bars.",
            })
        elif observation_singleton - base_singleton > 0.20:
            suggestions.append({
                "problem": "Sparsity increases when final observation context is added.",
                "evidence": "observation_id singleton ratio is much higher than base_composite singleton ratio.",
                "improvement": "Use fewer position/context splits or move position conditioning into the decoder instead of multiplying the observation vocabulary.",
            })
        if any("top observation overlap" in rule.name or "TV distance" in rule.name or "JS divergence" in rule.name for rule in problems):
            suggestions.append({
                "problem": "HMM emission distribution differs from the training observation distribution.",
                "evidence": "Distance metrics are outside the reference range.",
                "improvement": "Inspect per-state top emissions; tune emission_smoothing/warm_start_strength, or condition emissions on form/position only after codebook distribution is healthy.",
            })
        if any("state separation" in rule.name for rule in problems + warnings):
            suggestions.append({
                "problem": "Hidden states may not represent distinct musical roles.",
                "evidence": "Pairwise JS between emission rows is outside the reference range.",
                "improvement": "Check form labels and section lengths; if states are too similar, add stronger form supervision or richer observation features.",
            })
        if any("EM delta" in rule.name for rule in problems + warnings):
            suggestions.append({
                "problem": "HMM training convergence is weak.",
                "evidence": "Training log has negative or large final deltas.",
                "improvement": "Review hard mask constraints, increase max_iter, or lower transition/emission smoothing after confirming data size is sufficient.",
            })
        if not suggestions:
            suggestions.append({
                "problem": "No critical heuristic issue found.",
                "evidence": "All key metrics are in good or acceptable ranges.",
                "improvement": "Next improvement should come from listening tests and generation diagnostics rather than more global HMM tuning.",
            })
        return suggestions

    def _rule(
        self,
        name: str,
        value: float,
        good: str,
        is_good: bool,
        is_warn: bool,
        note: str,
    ) -> QualityRule:
        if is_good:
            status = "GOOD"
        elif is_warn:
            status = "WARN"
        else:
            status = "BAD"
        return QualityRule(name, float(value), good, status, note)


class ModelQualityReport:
    """Render full markdown model quality diagnostics."""

    def __init__(self, analyzer: ModelBundleAnalyzer, charts_dir: Path, top_n: int) -> None:
        self.analyzer = analyzer
        self.assessor = ModelQualityAssessor(analyzer)
        self.charts = ModelAnalysisCharts(analyzer, charts_dir)
        self.top_n = top_n

    def write(self, output_path: Path) -> None:
        report = MarkdownReport()
        report.heading("HMM Model Quality Report")
        report.paragraph(
            "Reference values in this report are engineering heuristics for debugging. "
            "They are not final musicological pass/fail rules."
        )
        pool = self.analyzer.pool_summary()
        report.heading("Model Summary", 2)
        report.table([
            "Metric",
            "Value",
        ], [
            ["model_bundle", self.analyzer.model_path],
            ["bar_count", pool["bar_count"]],
            ["observation_count", pool["observation_count"]],
            ["used_observation_count", pool["used_observation_count"]],
            ["forms", ", ".join(self.analyzer.form_names())],
        ])

        pool_rules = self.assessor.assess_pool()
        layered_summaries = self.analyzer.layered_pool_summaries()
        layered_rules = self.assessor.assess_layered_pools()
        report.heading("Vocabulary / Pool Quality", 2)
        report.table([
            "Metric",
            "Value",
            "Reference",
            "Status",
            "Meaning",
        ], [rule.row() for rule in pool_rules])
        report.heading("Layered Vocabulary Sparsity", 2)
        report.paragraph(
            "This table shows where sparsity enters the vocabulary. "
            "`codebook_id` is the global codebook layer, `base_composite` is codebook plus feature cluster, "
            "and `observation_id` is the final HMM symbol after any position/context expansion."
        )
        report.table([
            "Layer",
            "Pool count",
            "Bar count",
            "Singleton pools",
            "Singleton ratio",
            "Min pool",
            "Max pool",
            "Mean pool",
            "Pool/bar ratio",
        ], [
            [
                item["layer"],
                item["pool_count"],
                item["bar_count"],
                item["singleton_pool_count"],
                item["singleton_ratio"],
                item["min_pool_size"],
                item["max_pool_size"],
                item["mean_pool_size"],
                item["pool_per_bar_ratio"],
            ]
            for item in layered_summaries
        ])
        report.heading("Layered Sparsity Checks", 3)
        report.table([
            "Metric",
            "Value",
            "Reference",
            "Status",
            "Meaning",
        ], [rule.row() for rule in layered_rules])
        report.heading("Top Pools By Layer", 3)
        for item in layered_summaries:
            report.table([
                f"{item['layer']} key",
                "Count",
            ], [
                [pool["key"], pool["count"]]
                for pool in item["top_pools"]
            ])

        all_form_rules: List[QualityRule] = []
        for form_name in self.analyzer.form_names():
            analysis = self.analyzer.analyze_form(form_name)
            form_rules = self.assessor.assess_form(analysis)
            all_form_rules.extend(form_rules)
            charts = self.charts.render_form_charts(analysis, self.top_n)
            report.heading(f"Form Quality: {form_name}", 2)
            report.table([
                "Metric",
                "Value",
                "Reference",
                "Status",
                "Meaning",
            ], [rule.row() for rule in form_rules])
            report.heading("State Details", 3)
            report.table([
                "State",
                "Role",
                "Length",
                "Self-loop",
                "Forward mass",
                "Emission entropy",
                "Effective obs",
            ], [
                [
                    state,
                    analysis.state_roles.get(state, f"State_{state}"),
                    analysis.section_lengths[state] if state < len(analysis.section_lengths) else "",
                    analysis.transmat[state, state],
                    float(np.sum(analysis.transmat[state, state + 1:])),
                    analysis.emission_entropy[state],
                    analysis.emission_effective_counts[state],
                ]
                for state in range(analysis.n_states)
            ])
            report.heading("Distribution Metrics", 3)
            report.table([
                "Metric",
                "Section-weighted",
                "Transition-occupancy",
                "Reference",
            ], [
                ["Total variation", analysis.section_weighted_metrics.total_variation, analysis.transition_occupancy_metrics.total_variation, "< 0.10 good, < 0.25 acceptable"],
                ["JS divergence", analysis.section_weighted_metrics.js_divergence, analysis.transition_occupancy_metrics.js_divergence, "< 0.05 good, < 0.15 acceptable"],
                ["Cosine similarity", analysis.section_weighted_metrics.cosine_similarity, analysis.transition_occupancy_metrics.cosine_similarity, "> 0.90 good"],
                ["Pearson correlation", analysis.section_weighted_metrics.pearson_correlation, analysis.transition_occupancy_metrics.pearson_correlation, "> 0.50 useful"],
                [f"Top-{self.top_n} overlap", analysis.section_weighted_metrics.top_overlap_ratio, analysis.transition_occupancy_metrics.top_overlap_ratio, "> 0.60 good"],
            ])
            for chart_name, chart_path in charts.items():
                report.image(chart_name, chart_path, output_path)
            report.heading("Training Log", 3)
            report.table([
                "Iteration",
                "Log likelihood",
                "Delta",
            ], [
                [
                    int(item.get("iteration", index)),
                    float(item.get("log_likelihood", 0.0)),
                    float(item.get("delta", 0.0)),
                ]
                for index, item in enumerate(analysis.training_log[-20:])
            ])

        report.heading("Problem Analysis And Improvement Ideas", 2)
        suggestions = self.assessor.problems_and_suggestions(pool_rules, layered_rules, all_form_rules)
        report.table([
            "Problem",
            "Evidence",
            "Possible improvement",
        ], [
            [item["problem"], item["evidence"], item["improvement"]]
            for item in suggestions
        ])
        report.write(output_path)


class ModelQualityCLI:
    """CLI entrypoint."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Analyze quality of a saved HMM model bundle.")
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--model-dir", type=Path)
        source.add_argument("--model-bundle", type=Path)
        parser.add_argument("--output", type=Path, default=None)
        parser.add_argument("--charts-dir", type=Path, default=None)
        parser.add_argument("--top-n", type=int, default=20)
        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        args = self.build_parser().parse_args(argv)
        model_bundle = args.model_bundle or args.model_dir / "model_bundle.json"
        output = args.output or model_bundle.parent / "model_quality_report.md"
        charts_dir = args.charts_dir or output.parent / f"{output.stem}_charts"
        analyzer = ModelBundleAnalyzer.load(model_bundle)
        ModelQualityReport(analyzer, charts_dir, args.top_n).write(output)
        print(f"Model quality report -> {output}")


def main() -> None:
    ModelQualityCLI().run()


if __name__ == "__main__":
    main()
