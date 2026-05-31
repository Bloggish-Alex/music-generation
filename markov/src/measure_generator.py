#!/usr/bin/env python3
"""
Measure Generator — end-to-end pipeline from model to MIDI.

Ties together::

    1. Load a trained MusicModel
    2. Generate a cluster-label timeline via PhraseGenerator
    3. Sample actual notes per measure via a pluggable strategy
    4. Write the result to a MIDI file

Usage (library)::

    from music_model import MusicModel
    from rhythmic_template_variation import RhythmicTemplateVariation
    from measure_generator import MeasureGenerator

    model = MusicModel.load("./models/my_model")
    strategy = RhythmicTemplateVariation(model.clusterer)
    gen = MeasureGenerator(model, strategy)
    gen.generate_midi("output.mid", num_measures=120, seed=42)

Usage (CLI)::

    python measure_generator.py --model ./models/my_model \\
        --strategy rhythmic --output generated/output.mid --measures 120
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from music_model import MusicModel
from phrase_generator import PhraseGenerator

log = logging.getLogger("measure_generator")

# ---------------------------------------------------------------------------
# NoteEvent — shared output type
# ---------------------------------------------------------------------------


class NoteEvent:
    """One note or rest with absolute timing for MIDI assembly.

    Using a plain class (not frozen dataclass) so it's compatible with both
    strategy modules without requiring a shared import.
    """

    __slots__ = ("pitch", "duration_ql", "velocity", "beat_offset")

    def __init__(self, pitch: int, duration_ql: float, velocity: int, beat_offset: float) -> None:
        self.pitch = pitch
        self.duration_ql = duration_ql
        self.velocity = velocity
        self.beat_offset = beat_offset

    def __repr__(self) -> str:
        return (
            f"NoteEvent(pitch={self.pitch}, dur={self.duration_ql:.2f}, "
            f"vel={self.velocity}, offset={self.beat_offset:.2f})"
        )


# ---------------------------------------------------------------------------
# MeasureGenerator
# ---------------------------------------------------------------------------


class MeasureGenerator:
    """End-to-end pipeline: model → timeline → notes → MIDI.

    Args:
        model: Trained MusicModel.
        strategy: Object with ``sample_measure(cluster_label, time_signature,
            seed, is_section_end) -> List[NoteEvent]``.
    """

    def __init__(self, model: MusicModel, strategy) -> None:
        self.model = model
        self.phrase_gen = PhraseGenerator(model)
        self._strategy = strategy

    # ------------------------------------------------------------------
    # Generate note list
    # ------------------------------------------------------------------

    def generate(
        self,
        num_measures: int,
        time_signature: Tuple[int, int] = (4, 4),
        seed: Optional[int] = None,
    ) -> List[NoteEvent]:
        """Produce a flat list of NoteEvents spanning *num_measures* bars.

        Args:
            num_measures: Target number of measures.
            time_signature: (numerator, denominator).
            seed: Base random seed (each measure gets seed + i).

        Returns:
            Flat list of NoteEvent with beat_offset adjusted to global time.
        """
        if seed is None:
            seed = np.random.randint(0, 2 ** 31 - 1)

        ts_num, ts_den = time_signature
        bar_length_ql = ts_num * (4.0 / ts_den)

        # 1. Generate cluster-label timeline
        log.info("Generating timeline (%d measures) ...", num_measures)
        labels = self.phrase_gen.generate(num_measures, seed=seed)

        # 2. Sample notes for each measure
        log.info("Sampling measures with %s ...", type(self._strategy).__name__)
        all_notes: List[NoteEvent] = []
        global_offset = 0.0  # accumulated bar start in quarterLength

        for i, cluster_label in enumerate(labels):
            measure_seed = seed + i
            notes = self._strategy.sample_measure(
                cluster_label=cluster_label,
                time_signature=time_signature,
                seed=measure_seed,
                is_section_end=False,
            )

            # Adjust beat_offset from measure-relative to global time
            for nev in notes:
                all_notes.append(NoteEvent(
                    pitch=nev.pitch,
                    duration_ql=nev.duration_ql,
                    velocity=nev.velocity,
                    beat_offset=global_offset + nev.beat_offset,
                ))

            global_offset += bar_length_ql

        log.info(
            "Generated %d notes across %d measures.",
            len(all_notes), len(labels),
        )
        return all_notes

    # ------------------------------------------------------------------
    # MIDI output
    # ------------------------------------------------------------------

    def generate_midi(
        self,
        output_path: str,
        num_measures: int = 120,
        time_signature: Tuple[int, int] = (4, 4),
        tempo: int = 120,
        seed: Optional[int] = None,
    ) -> None:
        """Generate and write a MIDI file.

        Args:
            output_path: Output .mid file path.
            num_measures: Target number of measures.
            time_signature: (numerator, denominator).
            tempo: BPM.
            seed: Random seed.
        """
        notes = self.generate(
            num_measures=num_measures,
            time_signature=time_signature,
            seed=seed,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_midi(notes, output_path, tempo, time_signature)
        log.info("Wrote MIDI to %s", output_path)

    # ------------------------------------------------------------------
    # Internal: MIDI serialisation via music21
    # ------------------------------------------------------------------

    @staticmethod
    def _write_midi(
        notes: List[NoteEvent],
        output_path: Path,
        tempo: int,
        time_signature: Tuple[int, int],
    ) -> None:
        from music21 import stream, tempo as m21tempo, meter as m21meter
        from music21 import note as m21note, instrument, midi as m21midi

        ts_num, ts_den = time_signature

        score = stream.Score()
        score.append(m21tempo.MetronomeMark(number=tempo))
        score.append(m21meter.TimeSignature(f"{ts_num}/{ts_den}"))

        part = stream.Part()
        part.append(instrument.Instrument(midiProgram=0))

        for nev in notes:
            if nev.pitch >= 0:
                n = m21note.Note(pitch=nev.pitch)
                n.duration.quarterLength = nev.duration_ql
                n.volume.velocity = nev.velocity
                part.append(n)
            else:
                r = m21note.Rest()
                r.duration.quarterLength = nev.duration_ql
                part.append(r)

        score.append(part)
        mf = m21midi.translate.music21ObjectToMidiFile(score)
        mf.open(str(output_path), "wb")
        mf.write()
        mf.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure Generator — model → timeline → notes → MIDI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        default="./models/test",
        help="Path to trained MusicModel directory.",
    )
    parser.add_argument(
        "--output", "-o",
        default="generated/output.mid",
        help="Output MIDI file path.",
    )
    parser.add_argument(
        "--measures", "-n",
        type=int,
        default=120,
        help="Target number of measures.",
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=["rhythmic", "markov"],
        default="rhythmic",
        help="Measure sampling strategy: 'rhythmic' (template variation) "
        "or 'markov' (per-cluster Markov chain).",
    )
    parser.add_argument(
        "--markov-order",
        type=int,
        default=2,
        help="Markov chain order (only for --strategy markov).",
    )
    parser.add_argument(
        "--time-signature",
        default="4/4",
        help="Time signature, e.g. '4/4' or '3/4'.",
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
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load model
    log.info("Loading model from %s ...", args.model)
    model = MusicModel.load(args.model)
    print()
    print(model.summary())

    # Select strategy
    if args.strategy == "markov":
        from cluster_markov_sampler import ClusterMarkovSampler
        strategy = ClusterMarkovSampler(model.clusterer, order=args.markov_order)
    else:
        from rhythmic_template_variation import RhythmicTemplateVariation
        strategy = RhythmicTemplateVariation(model.clusterer)

    # Parse time signature
    ts_parts = args.time_signature.split("/")
    time_sig = (int(ts_parts[0]), int(ts_parts[1]))

    # Generate
    gen = MeasureGenerator(model, strategy)
    gen.generate_midi(
        output_path=args.output,
        num_measures=args.measures,
        time_signature=time_sig,
        tempo=args.tempo,
        seed=args.seed,
    )

    print(f"\nGenerated {args.measures} measures → {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
