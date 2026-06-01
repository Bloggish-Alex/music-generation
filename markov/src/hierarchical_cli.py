#!/usr/bin/env python3
"""Command-line interface for hierarchical MIDI generation."""

from __future__ import annotations

import logging

from hierarchical_generator import HierarchicalGenerator, log
from music_model import MusicModel

def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hierarchical Generator — three-tier music generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default="./models/test",
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
        default=30,
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
    parser.add_argument(
        "--timeline-mode",
        choices=["section", "matrix", "matrix_mined", "hybrid"],
        default="section",
        help="Timeline source. Defaults to existing section behavior.",
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
        timeline_mode=args.timeline_mode,
    )

    executed_modules = getattr(gen, "_last_midi_generation_modules", [])
    if "write_midi" in executed_modules:
        print(f"\nGenerated {len(labels)} measures -> {args.output}")
    else:
        print(
            f"\nGenerated {len(labels)} measures in memory; no MIDI file was written "
            f"(stopped before write_midi)."
        )
    print("Done.")


if __name__ == "__main__":
    main()
