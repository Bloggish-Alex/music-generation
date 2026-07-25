#!/usr/bin/env python3
"""Audio-side quality analysis for generated MIDI and bar tensors."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class GenerationAudioAnalysisConfig:
    """Configuration for MIDI-to-audio diagnostics."""

    enabled: bool = True
    sample_rate: int = 22050
    n_mfcc: int = 20
    soundfont_dir: Optional[Path] = None
    soundfont_name: Optional[str] = None
    normalize_wav: bool = True
    target_peak_dbfs: float = -1.0
    max_gain_db: float = 36.0


class SoundfontResolver:
    """Find a usable soundfont for FluidSynth rendering."""

    def __init__(self, explicit_dir: Optional[str | Path] = None, explicit_name: Optional[str] = None) -> None:
        self.explicit_dir = Path(explicit_dir) if explicit_dir else None
        self.explicit_name = explicit_name

    def resolve(self, anchor: str | Path) -> Optional[Path]:
        """Return the first available .sf2/.sf3 file."""
        candidates: List[Path] = []
        if self.explicit_dir:
            candidates.append(self.explicit_dir)
        anchor_path = Path(anchor).resolve()
        project_root = self._project_root(anchor_path)
        candidates.extend([
            project_root / ".." / "soundfont",
            project_root / ".." / ".." / "soundfont",
            Path.cwd() / ".." / "soundfont",
            Path.cwd() / ".." / ".." / "soundfont",
        ])
        for directory in candidates:
            resolved = directory.resolve()
            if not resolved.exists() or not resolved.is_dir():
                continue
            if self.explicit_name:
                explicit = resolved / self.explicit_name
                if explicit.exists():
                    return explicit
            fonts = sorted([*resolved.glob("*.sf2"), *resolved.glob("*.sf3")])
            if fonts:
                return fonts[0]
        return None

    def _project_root(self, path: Path) -> Path:
        """Return mdn_transformer project root if it is in the path."""
        for parent in [path, *path.parents]:
            if parent.name == "mdn_transformer":
                return parent
        return path.parent


class MidiToWavRenderer:
    """Render MIDI to WAV using the FluidSynth CLI."""

    def __init__(self, soundfont: Path) -> None:
        self.soundfont = Path(soundfont)

    def render(
        self,
        midi_path: str | Path,
        wav_path: str | Path,
        config: Optional[GenerationAudioAnalysisConfig] = None,
    ) -> Dict[str, Any]:
        """Render a MIDI file to WAV and return diagnostics."""
        config = config or GenerationAudioAnalysisConfig()
        output = Path(wav_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        method = "fluidsynth_cli"
        fallback_reason: Optional[str] = None
        fluidsynth_diag: Dict[str, Any] = {}
        try:
            fluidsynth_diag = self._render_with_fluidsynth_cli(midi_path, output, config)
            if not output.exists() or output.stat().st_size <= 0:
                raise RuntimeError(f"FluidSynth did not create WAV output: {output}")
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            method = self._render_with_pretty_midi(midi_path, output, int(config.sample_rate))
        normalize_diag = self._normalize_wav(output, config) if config.normalize_wav else {"enabled": False}
        return {
            "wav_path": str(output),
            "soundfont": str(self.soundfont),
            "method": method,
            "fallback_reason": fallback_reason,
            "fluidsynth": fluidsynth_diag,
            "normalization": normalize_diag,
            "exists": bool(output.exists()),
            "size_bytes": int(output.stat().st_size) if output.exists() else 0,
        }

    def _render_with_fluidsynth_cli(
        self,
        midi_path: str | Path,
        wav_path: Path,
        config: GenerationAudioAnalysisConfig,
    ) -> Dict[str, Any]:
        """Render with FluidSynth; options must precede soundfont and MIDI paths."""
        executable = shutil.which("fluidsynth") or "fluidsynth"
        command = [
            executable,
            "-ni",
            "-T",
            "wav",
            "-F",
            str(wav_path),
            "-r",
            str(int(config.sample_rate)),
            str(self.soundfont),
            str(midi_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "FluidSynth failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        return {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
            "sample_rate": int(config.sample_rate),
        }

    def _render_with_pretty_midi(self, midi_path: str | Path, wav_path: Path, sample_rate: int) -> str:
        """Render with pretty_midi; prefer FluidSynth, fall back to sine synthesis."""
        import pretty_midi
        import soundfile as sf

        midi = pretty_midi.PrettyMIDI(str(midi_path))
        try:
            audio = midi.fluidsynth(fs=int(sample_rate), sf2_path=str(self.soundfont))
            method = "pretty_midi_fluidsynth"
        except Exception:
            audio = midi.synthesize(fs=int(sample_rate))
            method = "pretty_midi_synthesize"
        sf.write(str(wav_path), np.asarray(audio, dtype=np.float32), int(sample_rate))
        return method

    def _normalize_wav(self, wav_path: Path, config: GenerationAudioAnalysisConfig) -> Dict[str, Any]:
        """Peak-normalize WAV loudness without changing the MIDI file."""
        import soundfile as sf

        audio, sample_rate = sf.read(str(wav_path), always_2d=False)
        samples = np.asarray(audio, dtype=np.float32)
        if samples.size == 0:
            return {"enabled": True, "status": "skipped", "reason": "empty_audio"}

        before_peak = float(np.max(np.abs(samples)))
        before_rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        if before_peak <= 1.0e-8:
            return {
                "enabled": True,
                "status": "skipped",
                "reason": "silent_audio",
                "before_peak": before_peak,
                "before_rms": before_rms,
            }

        target_peak = float(10.0 ** (float(config.target_peak_dbfs) / 20.0))
        raw_gain = target_peak / before_peak
        max_gain = float(10.0 ** (float(config.max_gain_db) / 20.0))
        gain = min(raw_gain, max_gain)
        normalized = np.clip(samples * gain, -1.0, 1.0)
        sf.write(str(wav_path), normalized, int(sample_rate))

        after_peak = float(np.max(np.abs(normalized)))
        after_rms = float(np.sqrt(np.mean(np.square(normalized)))) if normalized.size else 0.0
        return {
            "enabled": True,
            "status": "ok",
            "target_peak_dbfs": float(config.target_peak_dbfs),
            "max_gain_db": float(config.max_gain_db),
            "before_peak": before_peak,
            "before_rms": before_rms,
            "after_peak": after_peak,
            "after_rms": after_rms,
            "applied_gain": float(gain),
            "applied_gain_db": float(20.0 * math.log10(gain)) if gain > 0 else 0.0,
            "gain_limited": bool(raw_gain > max_gain),
        }


class AudioFeatureAnalyzer:
    """Extract slot-aligned MFCC/chroma/onset diagnostics from rendered audio."""

    def __init__(self, config: GenerationAudioAnalysisConfig) -> None:
        self.config = config

    def analyze(
        self,
        wav_path: str | Path,
        bars: np.ndarray,
        output_prefix: str | Path,
        base_pitches: Optional[Sequence[int]] = None,
        fallback_base_pitch: int = 60,
        tempo_bpm: int = 100,
        bar_length_ql: float = 4.0,
    ) -> Dict[str, Any]:
        """Write JSON/NPZ/Markdown report for one generated audio file."""
        import librosa

        output = Path(output_prefix)
        output.parent.mkdir(parents=True, exist_ok=True)
        y, sr = librosa.load(str(wav_path), sr=int(self.config.sample_rate), mono=True)
        tensors = np.asarray(bars, dtype=np.float32)
        bar_count = int(tensors.shape[0])
        steps = int(tensors.shape[2])
        slot_count = bar_count * steps
        slot_duration = float(bar_length_ql) * 60.0 / float(tempo_bpm) / float(steps)

        hop_length = max(128, int(round(slot_duration * sr / 4.0)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=int(self.config.n_mfcc), hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        frame_times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=hop_length)

        slot_mfcc = self._slot_mean_matrix(mfcc.T, frame_times, slot_count, slot_duration)
        slot_audio_chroma = self._slot_mean_matrix(chroma.T, frame_times, slot_count, slot_duration)
        slot_novelty = self._slot_mean_vector(onset_env, frame_times, slot_count, slot_duration)
        slot_audio_chroma = self._normalize_rows(slot_audio_chroma)

        mfcc_delta = librosa.feature.delta(slot_mfcc.T, order=1, mode="nearest").T
        mfcc_delta2 = librosa.feature.delta(slot_mfcc.T, order=2, mode="nearest").T

        tensor_chroma = self._tensor_chroma_11(tensors)
        audio_chroma_11 = self._audio_chroma_11(slot_audio_chroma, bar_count, steps, base_pitches, fallback_base_pitch)
        chroma_similarity = self._row_cosine(audio_chroma_11, tensor_chroma)
        base_pitch_array = self._base_pitch_array(bar_count, base_pitches, fallback_base_pitch)
        tensor_target_by_track = self._tensor_chroma_11_by_track(tensors)
        tensor_pitch_by_track, tensor_active_relative, tensor_active_absolute = self._tensor_pitch_chroma_11_by_track(tensors, base_pitch_array)
        tensor_pitch_chroma = self._aggregate_track_chroma(tensor_pitch_by_track)
        midi_chroma_by_track = self._midi_symbolic_chroma_11(Path(wav_path).with_suffix(".mid"), bar_count, steps, bar_length_ql, base_pitch_array)
        midi_chroma = self._aggregate_track_chroma(midi_chroma_by_track)
        target_vs_tensor_pitch = self._row_cosine(tensor_chroma, tensor_pitch_chroma)
        tensor_pitch_vs_midi = self._row_cosine(tensor_pitch_chroma, midi_chroma)
        midi_vs_audio = self._row_cosine(midi_chroma, audio_chroma_11)

        tensor_onset = tensors[..., 2].sum(axis=1).reshape(-1)
        tensor_velocity = np.max(np.where((tensors[..., 2] > 0.5) | (tensors[..., 3] > 0.5), tensors[..., 4], 0.0), axis=1).reshape(-1)
        novelty_norm = self._normalize_vector(slot_novelty)
        onset_norm = self._normalize_vector(tensor_onset)
        velocity_norm = self._normalize_vector(tensor_velocity)

        summary = {
            "wav_path": str(wav_path),
            "sample_rate": int(sr),
            "duration_seconds": float(len(y) / max(1, sr)),
            "bar_count": int(bar_count),
            "steps_per_bar": int(steps),
            "slot_count": int(slot_count),
            "slot_duration_seconds": float(slot_duration),
            "mfcc": self._mfcc_summary(slot_mfcc, mfcc_delta, mfcc_delta2),
            "chroma_alignment": {
                "mean_cosine": float(np.mean(chroma_similarity)) if chroma_similarity.size else 0.0,
                "median_cosine": float(np.median(chroma_similarity)) if chroma_similarity.size else 0.0,
                "p10_cosine": float(np.percentile(chroma_similarity, 10)) if chroma_similarity.size else 0.0,
                "min_cosine": float(np.min(chroma_similarity)) if chroma_similarity.size else 0.0,
            },
            "chroma_diagnostics": {
                "target_vs_tensor_pitch": self._cosine_summary(target_vs_tensor_pitch),
                "tensor_pitch_vs_midi_symbolic": self._cosine_summary(tensor_pitch_vs_midi),
                "midi_symbolic_vs_audio": self._cosine_summary(midi_vs_audio),
                "target_vs_audio": self._cosine_summary(chroma_similarity),
                "bar_mean": {
                    "target_vs_tensor_pitch": self._bar_means(target_vs_tensor_pitch, steps),
                    "tensor_pitch_vs_midi_symbolic": self._bar_means(tensor_pitch_vs_midi, steps),
                    "midi_symbolic_vs_audio": self._bar_means(midi_vs_audio, steps),
                    "target_vs_audio": self._bar_means(chroma_similarity, steps),
                },
                "track_mean": self._track_chroma_summary(tensor_target_by_track, tensor_pitch_by_track, midi_chroma_by_track),
                "worst_target_vs_audio_slots": self._worst_chroma_slots(
                    target_vs_audio=chroma_similarity,
                    target_vs_tensor_pitch=target_vs_tensor_pitch,
                    tensor_pitch_vs_midi=tensor_pitch_vs_midi,
                    midi_vs_audio=midi_vs_audio,
                    tensor_target=tensor_chroma,
                    tensor_pitch=tensor_pitch_chroma,
                    midi_symbolic=midi_chroma,
                    audio=audio_chroma_11,
                    active_relative=tensor_active_relative,
                    active_absolute=tensor_active_absolute,
                    base_pitches=base_pitch_array,
                    steps=steps,
                    limit=10,
                ),
            },
            "onset_velocity_alignment": {
                "onset_novelty_cosine": self._cosine(onset_norm, novelty_norm),
                "velocity_novelty_cosine": self._cosine(velocity_norm, novelty_norm),
                "onset_novelty_corr": self._corr(onset_norm, novelty_norm),
                "velocity_novelty_corr": self._corr(velocity_norm, novelty_norm),
                "novelty_peak_ratio": float(np.mean(novelty_norm > 0.65)) if novelty_norm.size else 0.0,
            },
            "surprise": {
                "mfcc_delta_energy_mean": float(np.mean(np.linalg.norm(mfcc_delta, axis=1))) if mfcc_delta.size else 0.0,
                "mfcc_delta2_energy_mean": float(np.mean(np.linalg.norm(mfcc_delta2, axis=1))) if mfcc_delta2.size else 0.0,
                "spectral_novelty_mean": float(np.mean(slot_novelty)) if slot_novelty.size else 0.0,
                "spectral_novelty_std": float(np.std(slot_novelty)) if slot_novelty.size else 0.0,
                "spectral_novelty_cv": self._safe_ratio(float(np.std(slot_novelty)), float(np.mean(slot_novelty)))
                if slot_novelty.size
                else 0.0,
            },
        }

        npz_path = Path(f"{output}.npz")
        json_path = Path(f"{output}.json")
        md_path = Path(f"{output}.md")
        np.savez_compressed(
            npz_path,
            mfcc=slot_mfcc.astype(np.float32),
            mfcc_delta=mfcc_delta.astype(np.float32),
            mfcc_delta2=mfcc_delta2.astype(np.float32),
            audio_chroma_12=slot_audio_chroma.astype(np.float32),
            audio_chroma_11=audio_chroma_11.astype(np.float32),
            tensor_chroma_11=tensor_chroma.astype(np.float32),
            tensor_pitch_chroma_11=tensor_pitch_chroma.astype(np.float32),
            midi_symbolic_chroma_11=midi_chroma.astype(np.float32),
            tensor_track_chroma_11=tensor_target_by_track.astype(np.float32),
            tensor_track_pitch_chroma_11=tensor_pitch_by_track.astype(np.float32),
            midi_track_symbolic_chroma_11=midi_chroma_by_track.astype(np.float32),
            chroma_cosine=chroma_similarity.astype(np.float32),
            target_vs_tensor_pitch_cosine=target_vs_tensor_pitch.astype(np.float32),
            tensor_pitch_vs_midi_symbolic_cosine=tensor_pitch_vs_midi.astype(np.float32),
            midi_symbolic_vs_audio_cosine=midi_vs_audio.astype(np.float32),
            spectral_novelty=slot_novelty.astype(np.float32),
            tensor_onset=tensor_onset.astype(np.float32),
            tensor_velocity=tensor_velocity.astype(np.float32),
        )
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        md_path.write_text(self._markdown(summary), encoding="utf-8")
        return {
            **summary,
            "matrix_path": str(npz_path),
            "json_path": str(json_path),
            "report_path": str(md_path),
        }

    def _slot_mean_matrix(self, matrix: np.ndarray, frame_times: np.ndarray, slot_count: int, slot_duration: float) -> np.ndarray:
        """Aggregate frame-level feature matrix into 16-slot grid."""
        result = np.zeros((slot_count, int(matrix.shape[1])), dtype=np.float32)
        for slot in range(slot_count):
            start = slot * slot_duration
            end = (slot + 1) * slot_duration
            mask = (frame_times >= start) & (frame_times < end)
            if not bool(mask.any()):
                nearest = int(np.argmin(np.abs(frame_times - (start + end) * 0.5))) if frame_times.size else 0
                result[slot] = matrix[nearest] if matrix.size else 0.0
            else:
                result[slot] = np.mean(matrix[mask], axis=0)
        return result

    def _slot_mean_vector(self, values: np.ndarray, frame_times: np.ndarray, slot_count: int, slot_duration: float) -> np.ndarray:
        """Aggregate a frame-level vector into 16-slot grid."""
        return self._slot_mean_matrix(np.asarray(values, dtype=np.float32).reshape(-1, 1), frame_times, slot_count, slot_duration).reshape(-1)

    def _tensor_chroma_11(self, tensors: np.ndarray) -> np.ndarray:
        """Return slot-level target chroma embedding from tensor tail."""
        tail = tensors[..., -11:]
        return np.mean(tail, axis=1).reshape(-1, 11)

    def _tensor_chroma_11_by_track(self, tensors: np.ndarray) -> np.ndarray:
        """Return track-level target chroma embedding as [tracks, slots, 11]."""
        tail = np.asarray(tensors[..., -11:], dtype=np.float32)
        return np.transpose(tail, (1, 0, 2, 3)).reshape(int(tail.shape[1]), -1, 11)

    def _tensor_pitch_chroma_11_by_track(self, tensors: np.ndarray, base_pitches: np.ndarray) -> tuple[np.ndarray, List[List[float]], List[List[int]]]:
        """Infer slot-level pitch chroma directly from tensor pitch/state features."""
        arr = np.asarray(tensors, dtype=np.float32)
        bar_count = int(arr.shape[0])
        track_count = int(arr.shape[1])
        steps = int(arr.shape[2])
        chroma = np.zeros((track_count, bar_count * steps, 11), dtype=np.float32)
        active_relative_by_slot: List[List[float]] = [[] for _ in range(bar_count * steps)]
        active_absolute_by_slot: List[List[int]] = [[] for _ in range(bar_count * steps)]
        pitch_scale = 24.0
        for track in range(track_count):
            active_relative: Optional[float] = None
            active_absolute: Optional[int] = None
            for bar in range(bar_count):
                base_pitch = int(base_pitches[bar]) if bar < len(base_pitches) else 60
                for slot in range(steps):
                    slot_index = bar * steps + slot
                    features = arr[bar, track, slot]
                    state = int(np.argmax(features[1:4]))
                    if state == 1:
                        active_relative = float(features[0]) * pitch_scale
                        active_absolute = int(round(float(base_pitch) + active_relative))
                    elif state == 0:
                        active_relative = None
                        active_absolute = None
                    elif state == 2 and active_relative is None:
                        active_relative = float(features[0]) * pitch_scale
                        active_absolute = int(round(float(base_pitch) + active_relative))
                    if active_relative is None or active_absolute is None:
                        continue
                    vector = self._relative_pitch_to_chroma_11(active_relative)
                    chroma[track, slot_index] = vector
                    active_relative_by_slot[slot_index].append(float(active_relative))
                    active_absolute_by_slot[slot_index].append(int(active_absolute))
        return chroma, active_relative_by_slot, active_absolute_by_slot

    def _midi_symbolic_chroma_11(
        self,
        midi_path: Path,
        bar_count: int,
        steps: int,
        bar_length_ql: float,
        base_pitches: np.ndarray,
    ) -> np.ndarray:
        """Read MIDI notes and convert active note intervals into slot-level relative chroma."""
        track_count = 3
        result = np.zeros((track_count, bar_count * steps, 11), dtype=np.float32)
        if not midi_path.exists():
            return result
        import mido

        midi = mido.MidiFile(str(midi_path))
        ticks_per_beat = int(midi.ticks_per_beat)
        bar_ticks = int(round(float(bar_length_ql) * ticks_per_beat))
        slot_ticks = max(1, int(round(bar_ticks / max(1, steps))))
        total_ticks = int(bar_count * bar_ticks)
        for midi_track_index, track in enumerate(midi.tracks[1:1 + track_count]):
            absolute_tick = 0
            active: Dict[int, tuple[int, int]] = {}
            for message in track:
                absolute_tick += int(message.time)
                if message.type == "note_on" and int(message.velocity) > 0:
                    active[int(message.note)] = (int(absolute_tick), int(message.velocity))
                elif message.type in {"note_off", "note_on"}:
                    pitch = int(message.note)
                    if pitch not in active:
                        continue
                    start_tick, velocity = active.pop(pitch)
                    self._accumulate_midi_note_chroma(
                        result[midi_track_index],
                        pitch=pitch,
                        velocity=velocity,
                        start_tick=start_tick,
                        end_tick=int(absolute_tick),
                        slot_ticks=slot_ticks,
                        steps=steps,
                        base_pitches=base_pitches,
                    )
            for pitch, (start_tick, velocity) in active.items():
                self._accumulate_midi_note_chroma(
                    result[midi_track_index],
                    pitch=pitch,
                    velocity=velocity,
                    start_tick=start_tick,
                    end_tick=total_ticks,
                    slot_ticks=slot_ticks,
                    steps=steps,
                    base_pitches=base_pitches,
                )
        return self._normalize_track_rows(result)

    def _accumulate_midi_note_chroma(
        self,
        track_matrix: np.ndarray,
        pitch: int,
        velocity: int,
        start_tick: int,
        end_tick: int,
        slot_ticks: int,
        steps: int,
        base_pitches: np.ndarray,
    ) -> None:
        """Add one MIDI note interval into slot chroma with duration/velocity weighting."""
        if end_tick <= start_tick:
            return
        slot_count = int(track_matrix.shape[0])
        first_slot = max(0, int(start_tick // slot_ticks))
        last_slot = min(slot_count - 1, int((max(start_tick, end_tick - 1)) // slot_ticks))
        for slot in range(first_slot, last_slot + 1):
            slot_start = slot * slot_ticks
            slot_end = slot_start + slot_ticks
            overlap = max(0, min(end_tick, slot_end) - max(start_tick, slot_start))
            if overlap <= 0:
                continue
            bar = min(len(base_pitches) - 1, max(0, int(slot // max(1, steps))))
            relative_pitch = float(int(pitch) - int(base_pitches[bar]))
            track_matrix[slot] += self._relative_pitch_to_chroma_11(relative_pitch) * float(overlap) * float(max(1, velocity))

    def _relative_pitch_to_chroma_11(self, relative_pitch: float) -> np.ndarray:
        """Project one relative semitone pitch into the same 11D chroma compression as the tensor."""
        relative = np.zeros(12, dtype=np.float32)
        relative[int(round(float(relative_pitch))) % 12] = 1.0
        projected = np.zeros(11, dtype=np.float32)
        projected[:10] = relative[:10]
        projected[10] = float(relative[10] + relative[11]) * 0.5
        return projected

    def _aggregate_track_chroma(self, track_chroma: np.ndarray) -> np.ndarray:
        """Aggregate [tracks, slots, 11] chroma into [slots, 11]."""
        return self._normalize_rows(np.sum(np.asarray(track_chroma, dtype=np.float32), axis=0))

    def _normalize_track_rows(self, values: np.ndarray) -> np.ndarray:
        """L1-normalize each slot row for a track-level chroma tensor."""
        arr = np.asarray(values, dtype=np.float32)
        result = np.zeros_like(arr)
        for track in range(int(arr.shape[0])):
            result[track] = self._normalize_rows(arr[track])
        return result

    def _base_pitch_array(self, bar_count: int, base_pitches: Optional[Sequence[int]], fallback_base_pitch: int) -> np.ndarray:
        """Return one base pitch per bar."""
        result = np.full((int(bar_count),), int(fallback_base_pitch), dtype=np.int64)
        if base_pitches is None:
            return result
        for index, value in enumerate(base_pitches[:bar_count]):
            result[index] = int(value)
        return result

    def _audio_chroma_11(
        self,
        audio_chroma_12: np.ndarray,
        bar_count: int,
        steps: int,
        base_pitches: Optional[Sequence[int]],
        fallback_base_pitch: int,
    ) -> np.ndarray:
        """Rotate audio chroma to bar-relative coordinates and project to 11D."""
        projected = np.zeros((bar_count * steps, 11), dtype=np.float32)
        for bar in range(bar_count):
            base_pitch = int(base_pitches[bar]) if base_pitches is not None and bar < len(base_pitches) else int(fallback_base_pitch)
            base_pc = int(base_pitch) % 12
            for slot in range(steps):
                index = bar * steps + slot
                relative = np.zeros(12, dtype=np.float32)
                for pc in range(12):
                    relative[(pc - base_pc) % 12] = float(audio_chroma_12[index, pc])
                projected[index, :10] = relative[:10]
                projected[index, 10] = float(relative[10] + relative[11]) * 0.5
        return self._normalize_rows(projected)

    def _row_cosine(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return row-wise cosine similarity."""
        denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        result = np.zeros_like(denom, dtype=np.float32)
        np.divide(np.sum(left * right, axis=1), denom, out=result, where=denom > 1.0e-8)
        return result

    def _cosine_summary(self, values: np.ndarray) -> Dict[str, float]:
        """Return compact summary stats for a cosine sequence."""
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return {"mean": 0.0, "median": 0.0, "p10": 0.0, "min": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "min": float(np.min(arr)),
        }

    def _bar_means(self, values: np.ndarray, steps: int) -> List[float]:
        """Aggregate slot cosine values to one mean per bar."""
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0 or steps <= 0:
            return []
        usable = arr[: int(arr.size // steps) * steps]
        if usable.size == 0:
            return []
        return [float(value) for value in np.mean(usable.reshape(-1, steps), axis=1)]

    def _track_chroma_summary(self, target: np.ndarray, tensor_pitch: np.ndarray, midi_symbolic: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Summarize chroma agreement for each generated track."""
        result: Dict[str, Dict[str, float]] = {}
        track_count = min(int(target.shape[0]), int(tensor_pitch.shape[0]), int(midi_symbolic.shape[0]))
        for track in range(track_count):
            result[f"track_{track}"] = {
                "target_vs_tensor_pitch_mean": self._cosine_summary(self._row_cosine(target[track], tensor_pitch[track]))["mean"],
                "tensor_pitch_vs_midi_symbolic_mean": self._cosine_summary(self._row_cosine(tensor_pitch[track], midi_symbolic[track]))["mean"],
            }
        return result

    def _worst_chroma_slots(
        self,
        target_vs_audio: np.ndarray,
        target_vs_tensor_pitch: np.ndarray,
        tensor_pitch_vs_midi: np.ndarray,
        midi_vs_audio: np.ndarray,
        tensor_target: np.ndarray,
        tensor_pitch: np.ndarray,
        midi_symbolic: np.ndarray,
        audio: np.ndarray,
        active_relative: Sequence[Sequence[float]],
        active_absolute: Sequence[Sequence[int]],
        base_pitches: np.ndarray,
        steps: int,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Return the lowest target/audio chroma slots with layer-by-layer evidence."""
        scores = np.asarray(target_vs_audio, dtype=np.float32)
        if scores.size == 0:
            return []
        worst_indices = np.argsort(scores)[:max(0, int(limit))]
        rows: List[Dict[str, Any]] = []
        for slot_index in worst_indices:
            slot = int(slot_index)
            bar = int(slot // max(1, steps))
            rows.append({
                "slot_index": slot,
                "bar_index": bar,
                "slot_in_bar": int(slot % max(1, steps)),
                "base_pitch": int(base_pitches[bar]) if bar < len(base_pitches) else None,
                "target_vs_tensor_pitch": float(target_vs_tensor_pitch[slot]) if slot < len(target_vs_tensor_pitch) else 0.0,
                "tensor_pitch_vs_midi_symbolic": float(tensor_pitch_vs_midi[slot]) if slot < len(tensor_pitch_vs_midi) else 0.0,
                "midi_symbolic_vs_audio": float(midi_vs_audio[slot]) if slot < len(midi_vs_audio) else 0.0,
                "target_vs_audio": float(target_vs_audio[slot]) if slot < len(target_vs_audio) else 0.0,
                "target_chroma_11": self._compact_vector(tensor_target[slot]),
                "tensor_pitch_chroma_11": self._compact_vector(tensor_pitch[slot]),
                "midi_symbolic_chroma_11": self._compact_vector(midi_symbolic[slot]),
                "audio_chroma_11": self._compact_vector(audio[slot]),
                "tensor_active_relative_pitches": [round(float(value), 3) for value in active_relative[slot]] if slot < len(active_relative) else [],
                "tensor_active_absolute_pitches": [int(value) for value in active_absolute[slot]] if slot < len(active_absolute) else [],
            })
        return rows

    def _compact_vector(self, values: np.ndarray) -> List[float]:
        """Round vectors for readable JSON diagnostics."""
        return [round(float(value), 6) for value in np.asarray(values, dtype=np.float32).reshape(-1).tolist()]

    def _normalize_rows(self, values: np.ndarray) -> np.ndarray:
        """L1-normalize non-negative rows."""
        clipped = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
        denom = clipped.sum(axis=1, keepdims=True)
        result = np.zeros_like(clipped, dtype=np.float32)
        np.divide(clipped, denom, out=result, where=denom > 1.0e-8)
        return result

    def _normalize_vector(self, values: np.ndarray) -> np.ndarray:
        """Scale vector into [0, 1]."""
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        low = float(np.min(arr))
        high = float(np.max(arr))
        if high - low <= 1.0e-8:
            return np.zeros_like(arr)
        return (arr - low) / (high - low)

    def _cosine(self, left: np.ndarray, right: np.ndarray) -> float:
        """Cosine similarity for two vectors."""
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 1.0e-8:
            return 0.0
        return float(np.dot(left, right) / denom)

    def _corr(self, left: np.ndarray, right: np.ndarray) -> float:
        """Pearson correlation with zero-variance guard."""
        if left.size < 2 or right.size < 2 or float(np.std(left)) <= 1.0e-8 or float(np.std(right)) <= 1.0e-8:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        """Return a bounded diagnostic ratio with a zero-denominator guard."""
        if abs(denominator) <= 1.0e-8:
            return 0.0
        return float(numerator / denominator)

    def _mfcc_summary(self, mfcc: np.ndarray, delta: np.ndarray, delta2: np.ndarray) -> Dict[str, float]:
        """Summarize MFCC continuity and motion energy."""
        adjacent = self._row_cosine(mfcc[:-1], mfcc[1:]) if mfcc.shape[0] > 1 else np.asarray([], dtype=np.float32)
        return {
            "matrix_shape_0": int(mfcc.shape[0]),
            "matrix_shape_1": int(mfcc.shape[1]) if mfcc.ndim == 2 else 0,
            "adjacent_cosine_mean": float(np.mean(adjacent)) if adjacent.size else 0.0,
            "adjacent_cosine_p10": float(np.percentile(adjacent, 10)) if adjacent.size else 0.0,
            "change_rate_mean": 1.0 - float(np.mean(adjacent)) if adjacent.size else 0.0,
            "delta_energy_mean": float(np.mean(np.linalg.norm(delta, axis=1))) if delta.size else 0.0,
            "delta2_energy_mean": float(np.mean(np.linalg.norm(delta2, axis=1))) if delta2.size else 0.0,
        }

    def _markdown(self, summary: Dict[str, Any]) -> str:
        """Return a compact Markdown report."""
        chroma = summary["chroma_alignment"]
        chroma_diag = summary.get("chroma_diagnostics", {})
        onset = summary["onset_velocity_alignment"]
        mfcc = summary["mfcc"]
        surprise = summary["surprise"]
        target_pitch = chroma_diag.get("target_vs_tensor_pitch", {})
        pitch_midi = chroma_diag.get("tensor_pitch_vs_midi_symbolic", {})
        midi_audio = chroma_diag.get("midi_symbolic_vs_audio", {})
        return "\n".join([
            "# Generation Audio Quality Report",
            "",
            "## Files",
            f"- WAV: `{summary['wav_path']}`",
            f"- Duration: `{summary['duration_seconds']:.3f}s`",
            f"- Grid: `{summary['bar_count']} bars x {summary['steps_per_bar']} slots`",
            "",
            "## Metrics",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| MFCC adjacent cosine mean | {mfcc['adjacent_cosine_mean']:.6f} |",
            f"| MFCC change rate mean | {mfcc['change_rate_mean']:.6f} |",
            f"| MFCC delta energy mean | {mfcc['delta_energy_mean']:.6f} |",
            f"| MFCC delta2 energy mean | {mfcc['delta2_energy_mean']:.6f} |",
            f"| Chroma vs tensor mean cosine | {chroma['mean_cosine']:.6f} |",
            f"| Chroma vs tensor p10 cosine | {chroma['p10_cosine']:.6f} |",
            f"| Onset vs spectral novelty cosine | {onset['onset_novelty_cosine']:.6f} |",
            f"| Velocity vs spectral novelty cosine | {onset['velocity_novelty_cosine']:.6f} |",
            f"| Spectral novelty mean | {surprise['spectral_novelty_mean']:.6f} |",
            f"| Spectral novelty std | {surprise['spectral_novelty_std']:.6f} |",
            f"| Spectral novelty CV | {surprise['spectral_novelty_cv']:.6f} |",
            "",
            "## Chroma Layer Diagnostics",
            "| Layer Check | Mean | P10 | Min |",
            "| --- | ---: | ---: | ---: |",
            f"| Tensor target vs tensor pitch | {target_pitch.get('mean', 0.0):.6f} | {target_pitch.get('p10', 0.0):.6f} | {target_pitch.get('min', 0.0):.6f} |",
            f"| Tensor pitch vs MIDI symbolic | {pitch_midi.get('mean', 0.0):.6f} | {pitch_midi.get('p10', 0.0):.6f} | {pitch_midi.get('min', 0.0):.6f} |",
            f"| MIDI symbolic vs audio | {midi_audio.get('mean', 0.0):.6f} | {midi_audio.get('p10', 0.0):.6f} | {midi_audio.get('min', 0.0):.6f} |",
            f"| Tensor target vs audio | {chroma['mean_cosine']:.6f} | {chroma['p10_cosine']:.6f} | {chroma['min_cosine']:.6f} |",
            "",
            "## Notes",
            "- MFCC, delta MFCC, chroma, novelty, tensor onset, and tensor velocity matrices are saved in the companion `.npz` file.",
            "- Chroma comparison rotates audio chroma by each bar base pitch before projecting to the tensor's 11D chroma embedding.",
            "- Chroma layer diagnostics separate model pitch conditioning, MIDI rendering, and audio extraction effects.",
        ])


class GenerationAudioQualityAnalyzer:
    """End-to-end MIDI render plus audio feature analysis."""

    def __init__(self, config: Optional[GenerationAudioAnalysisConfig] = None) -> None:
        self.config = config or GenerationAudioAnalysisConfig()

    def run(
        self,
        midi_path: str | Path,
        bars: np.ndarray,
        base_pitches: Optional[Sequence[int]] = None,
        fallback_base_pitch: int = 60,
        tempo_bpm: Optional[int] = None,
        bar_length_ql: float = 4.0,
    ) -> Dict[str, Any]:
        """Render WAV and analyze audio side effects."""
        if not self.config.enabled:
            return {"enabled": False}
        midi = Path(midi_path)
        wav_path = midi.with_suffix(".wav")
        prefix = midi.parent / f"{midi.stem}.audio_quality"
        soundfont = SoundfontResolver(self.config.soundfont_dir, self.config.soundfont_name).resolve(midi)
        if soundfont is None:
            return {
                "enabled": True,
                "status": "skipped",
                "reason": "No .sf2/.sf3 soundfont found.",
                "searched_from": str(midi),
            }
        try:
            wav_diag = MidiToWavRenderer(soundfont).render(midi, wav_path, self.config)
            resolved_tempo = int(tempo_bpm) if tempo_bpm is not None else _infer_tempo_bpm(midi)
            analysis = AudioFeatureAnalyzer(self.config).analyze(
                wav_path=wav_path,
                bars=bars,
                output_prefix=prefix,
                base_pitches=base_pitches,
                fallback_base_pitch=fallback_base_pitch,
                tempo_bpm=resolved_tempo,
                bar_length_ql=bar_length_ql,
            )
            return {"enabled": True, "status": "ok", "render": wav_diag, "analysis": analysis}
        except Exception as exc:  # Diagnostics should not block MIDI generation.
            return {"enabled": True, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def _infer_tempo_bpm(midi_path: Path, fallback: int = 100) -> int:
    """Infer BPM from the first MIDI tempo meta event."""
    try:
        import mido

        midi = mido.MidiFile(str(midi_path))
        for track in midi.tracks:
            for message in track:
                if message.type == "set_tempo":
                    return int(round(float(mido.tempo2bpm(message.tempo))))
    except Exception:
        return int(fallback)
    return int(fallback)


def _load_bars(tensor_path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    archive = np.load(tensor_path)
    try:
        bars = archive["bars"] if "bars" in archive.files else np.stack([archive[key] for key in archive.files], axis=0)
        if "render_base_pitches" in archive.files:
            base_pitches = archive["render_base_pitches"]
        elif "source_base_pitches" in archive.files:
            base_pitches = archive["source_base_pitches"]
        else:
            base_pitches = None
        return np.asarray(bars, dtype=np.float32), None if base_pitches is None else np.asarray(base_pitches, dtype=np.int64)
    finally:
        archive.close()


def main() -> None:
    """Standalone CLI for existing generation outputs."""
    parser = argparse.ArgumentParser(description="Render generated MIDI to WAV and run audio quality diagnostics.")
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--bar-tensors", type=Path, required=True)
    parser.add_argument("--base-pitch", type=int, default=60)
    parser.add_argument("--tempo", type=int, default=None, help="BPM. Defaults to the MIDI tempo meta event when omitted.")
    parser.add_argument("--soundfont-dir", type=Path, default=None)
    parser.add_argument("--soundfont-name", type=str, default=None)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--disable-normalize-wav", action="store_true")
    parser.add_argument("--target-peak-dbfs", type=float, default=-1.0)
    parser.add_argument("--max-gain-db", type=float, default=36.0)
    args = parser.parse_args()
    bars, base_pitches = _load_bars(args.bar_tensors)
    tempo_bpm = int(args.tempo) if args.tempo is not None else _infer_tempo_bpm(args.midi)
    result = GenerationAudioQualityAnalyzer(GenerationAudioAnalysisConfig(
        sample_rate=int(args.sample_rate),
        soundfont_dir=args.soundfont_dir,
        soundfont_name=args.soundfont_name,
        normalize_wav=not bool(args.disable_normalize_wav),
        target_peak_dbfs=float(args.target_peak_dbfs),
        max_gain_db=float(args.max_gain_db),
    )).run(
        midi_path=args.midi,
        bars=bars,
        base_pitches=base_pitches,
        fallback_base_pitch=int(args.base_pitch),
        tempo_bpm=tempo_bpm,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
