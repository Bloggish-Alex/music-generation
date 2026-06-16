#!/usr/bin/env python3
"""Write a markdown report comparing HMM emissions with training observations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from diagnostics.model_analysis import (
    MarkdownReport,
    ModelAnalysisCharts,
    ModelBundleAnalyzer,
)


class HMMDistributionReport:
    """Render observation distribution diagnostics for saved form HMMs."""

    def __init__(self, analyzer: ModelBundleAnalyzer, charts_dir: Path, top_n: int) -> None:
        self.analyzer = analyzer
        self.charts = ModelAnalysisCharts(analyzer, charts_dir)
        self.top_n = top_n

    def write(self, output_path: Path) -> None:
        report = MarkdownReport()
        report.heading("HMM Observation Distribution Report")
        report.table([
            "Field",
            "Value",
        ], [
            ["model_bundle", self.analyzer.model_path],
            ["bar_count", self.analyzer.bar_count()],
            ["observation_count", self.analyzer.observation_count()],
            ["forms", ", ".join(self.analyzer.form_names())],
        ])
        pool = self.analyzer.pool_summary()
        report.heading("Training Observation Pool", 2)
        report.table([
            "Metric",
            "Value",
        ], [
            ["used_observation_count", pool["used_observation_count"]],
            ["singleton_observation_count", pool["singleton_observation_count"]],
            ["singleton_ratio", pool["singleton_ratio"]],
            ["min_pool_size", pool["min_pool_size"]],
            ["max_pool_size", pool["max_pool_size"]],
            ["mean_pool_size", pool["mean_pool_size"]],
            ["observation_per_bar_ratio", pool["observation_per_bar_ratio"]],
        ])

        for form_name in self.analyzer.form_names():
            analysis = self.analyzer.analyze_form(form_name)
            charts = self.charts.render_form_charts(analysis, self.top_n)
            report.heading(f"Form: {form_name}", 2)
            report.table([
                "Metric",
                "Section-weighted emission",
                "Transition-occupancy emission",
            ], [
                ["L1 distance", analysis.section_weighted_metrics.l1, analysis.transition_occupancy_metrics.l1],
                ["Total variation", analysis.section_weighted_metrics.total_variation, analysis.transition_occupancy_metrics.total_variation],
                ["JS divergence", analysis.section_weighted_metrics.js_divergence, analysis.transition_occupancy_metrics.js_divergence],
                ["KL actual->model", analysis.section_weighted_metrics.kl_actual_to_model, analysis.transition_occupancy_metrics.kl_actual_to_model],
                ["Cosine similarity", analysis.section_weighted_metrics.cosine_similarity, analysis.transition_occupancy_metrics.cosine_similarity],
                ["Pearson correlation", analysis.section_weighted_metrics.pearson_correlation, analysis.transition_occupancy_metrics.pearson_correlation],
                [f"Top-{self.top_n} overlap", analysis.section_weighted_metrics.top_overlap_ratio, analysis.transition_occupancy_metrics.top_overlap_ratio],
            ])
            report.paragraph(
                "Use section-weighted emission as the main generation estimate for the current pipeline, "
                "because generation samples observations from the fixed section state. "
                "Transition-occupancy emission is included to inspect the learned HMM transition dynamics."
            )
            for chart_name, chart_path in charts.items():
                report.image(chart_name, chart_path, output_path)

            actual = self.analyzer.actual_observation_distribution(analysis.n_observations)
            report.heading("Top Actual Observations", 3)
            report.table([
                "Rank",
                "Observation",
                "Composite",
                "Actual prob",
                "Actual count",
            ], [
                [
                    rank + 1,
                    item["observation_id"],
                    item["label"],
                    item["probability"],
                    item["count"],
                ]
                for rank, item in enumerate(self.analyzer.top_observations(actual, self.top_n))
            ])
            report.heading("Top HMM Section-Weighted Observations", 3)
            report.table([
                "Rank",
                "Observation",
                "Composite",
                "Model prob",
                "Actual count",
            ], [
                [
                    rank + 1,
                    item["observation_id"],
                    item["label"],
                    item["probability"],
                    item["count"],
                ]
                for rank, item in enumerate(
                    self.analyzer.top_observations(analysis.section_weighted_distribution, self.top_n)
                )
            ])

            report.heading("State Emission Summary", 3)
            report.table([
                "State",
                "Role",
                "Section length",
                "Emission entropy",
                "Effective observations",
                "Section weight",
                "Transition occupancy",
            ], [
                [
                    state,
                    analysis.state_roles.get(state, f"State_{state}"),
                    analysis.section_lengths[state] if state < len(analysis.section_lengths) else "",
                    analysis.emission_entropy[state],
                    analysis.emission_effective_counts[state],
                    analysis.section_state_weights[state],
                    analysis.transition_state_occupancy[state],
                ]
                for state in range(analysis.n_states)
            ])
        report.write(output_path)


class HMMDistributionCLI:
    """CLI entrypoint."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Analyze HMM emission distribution in a saved model bundle.")
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
        output = args.output or model_bundle.parent / "hmm_distribution_report.md"
        charts_dir = args.charts_dir or output.parent / f"{output.stem}_charts"
        analyzer = ModelBundleAnalyzer.load(model_bundle)
        HMMDistributionReport(analyzer, charts_dir, args.top_n).write(output)
        print(f"HMM distribution report -> {output}")


def main() -> None:
    HMMDistributionCLI().run()


if __name__ == "__main__":
    main()
