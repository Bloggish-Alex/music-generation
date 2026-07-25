#!/usr/bin/env python3
"""VAR-style diagnostics for latent and explicit bar feature dynamics."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from codec.bar_feature_extractor import BAR_FEATURE_NAMES, EncodedBarFeatureStore
from codec.miditok_style_bar_encoder import MIDITOK_STYLE_FEATURE_NAMES, MidiTokStyleBarEventStore
from pipeline.latent_transformer_training_pipeline import LatentDatasetReader


@dataclass(frozen=True)
class BarDynamicsAnalysisConfig:
    """Configuration for bar dynamics diagnostics."""

    model_dir: Path
    latent_dir: Optional[Path] = None
    encoded_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    max_lag: int = 16
    validation_ratio: float = 0.2
    ridge_alpha: float = 1.0
    random_seed: int = 42
    redundancy_threshold: float = 0.95
    drift_threshold: float = 0.35
    low_lag_correlation_threshold: float = 0.05
    max_rows: Optional[int] = None
    max_songs: Optional[int] = None
    write_figures: bool = True


@dataclass(frozen=True)
class SongSequence:
    """One song-local time series."""

    song_id: str
    base_song_id: str
    indices: List[int]
    mu: np.ndarray
    features: np.ndarray
    event_features: np.ndarray
    sequence_embeddings: Optional[np.ndarray] = None


class BarDynamicsAnalyzer:
    """Analyze feature correlations, lag structure, stationarity, and VAR-style predictability."""

    def __init__(self, config: BarDynamicsAnalysisConfig) -> None:
        self.config = config

    def run(self) -> Dict[str, Any]:
        """Run analysis and write artifacts."""
        model_dir = Path(self.config.model_dir)
        latent_dir = Path(self.config.latent_dir) if self.config.latent_dir else model_dir / "latent"
        encoded_dir = Path(self.config.encoded_dir) if self.config.encoded_dir else model_dir / "encoded"
        output_dir = Path(self.config.output_dir) if self.config.output_dir else model_dir / "bar_dynamics_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        mu, rows, latent_summary = LatentDatasetReader().load(latent_dir)
        selected_indices = self._select_indices(rows)
        selected_rows = [rows[index] for index in selected_indices]
        selected_mu = mu[selected_indices].astype(np.float32)
        features, feature_source = EncodedBarFeatureStore(encoded_dir).matrix_for_rows(selected_rows)
        event_features, event_feature_source = MidiTokStyleBarEventStore(encoded_dir).matrix_for_rows(selected_rows)
        sequence_embeddings, sequence_embedding_source = self._load_miditok_sequence_embeddings(encoded_dir, selected_rows)
        songs = self._build_song_sequences(selected_mu, features, event_features, sequence_embeddings, selected_rows)
        if not songs:
            raise ValueError("No song has enough bars for dynamics analysis.")

        feature_matrix = np.concatenate([song.features for song in songs], axis=0)
        mu_matrix = np.concatenate([song.mu for song in songs], axis=0)
        delta_mu_matrix = np.concatenate([np.diff(song.mu, axis=0) for song in songs if len(song.indices) >= 2], axis=0)
        delta_feature_matrix = np.concatenate([np.diff(song.features, axis=0) for song in songs if len(song.indices) >= 2], axis=0)
        event_feature_matrix = np.concatenate([song.event_features for song in songs], axis=0)
        delta_event_feature_matrix = np.concatenate([np.diff(song.event_features, axis=0) for song in songs if len(song.indices) >= 2], axis=0)
        sequence_embedding_matrix = self._concat_optional_sequence_embeddings(songs)

        feature_redundancy = self._feature_redundancy(feature_matrix)
        event_feature_redundancy = self._named_feature_redundancy(event_feature_matrix, MIDITOK_STYLE_FEATURE_NAMES)
        zero_lag = self._zero_lag_correlation(feature_matrix, mu_matrix, delta_mu_matrix, songs)
        lag_correlation = self._lag_correlation(songs)
        event_lag_correlation = self._lag_correlation_for_matrix(
            songs=songs,
            names=MIDITOK_STYLE_FEATURE_NAMES,
            value_getter=lambda song: song.event_features,
        )
        stationarity = self._stationarity_summary(songs)
        predictive = self._predictive_probes(songs)
        recommendations = self._recommendations(
            feature_redundancy=feature_redundancy,
            lag_correlation=lag_correlation,
            stationarity=stationarity,
            predictive=predictive,
        )

        report = {
            "analysis_type": "bar_dynamics_var_style_diagnostics",
            "model_dir": str(model_dir),
            "latent_dir": str(latent_dir),
            "encoded_dir": str(encoded_dir),
            "output_dir": str(output_dir),
            "config": self._config_dict(),
            "data": {
                "latent_summary": latent_summary,
                "feature_source": feature_source,
                "event_feature_source": event_feature_source,
                "sequence_embedding_source": sequence_embedding_source,
                "song_count": int(len(songs)),
                "base_song_count": int(len({song.base_song_id for song in songs})),
                "bar_count": int(sum(len(song.indices) for song in songs)),
                "latent_dim": int(mu_matrix.shape[1]),
                "feature_dim": int(feature_matrix.shape[1]),
                "event_feature_dim": int(event_feature_matrix.shape[1]),
                "sequence_embedding_dim": int(sequence_embedding_matrix.shape[1]) if sequence_embedding_matrix is not None else 0,
                "delta_mu_rows": int(delta_mu_matrix.shape[0]),
                "delta_feature_rows": int(delta_feature_matrix.shape[0]),
                "delta_event_feature_rows": int(delta_event_feature_matrix.shape[0]),
                "song_lengths": self._numeric_summary([len(song.indices) for song in songs]),
            },
            "feature_names": list(BAR_FEATURE_NAMES),
            "event_feature_names": list(MIDITOK_STYLE_FEATURE_NAMES),
            "feature_redundancy": feature_redundancy,
            "event_feature_redundancy": event_feature_redundancy,
            "zero_lag_correlation": zero_lag,
            "lag_correlation": lag_correlation,
            "event_lag_correlation": event_lag_correlation,
            "stationarity": stationarity,
            "predictive_probes": predictive,
            "recommendations": recommendations,
        }

        self._write_outputs(output_dir, report, feature_redundancy, lag_correlation, event_feature_redundancy, event_lag_correlation)
        return report

    def _select_indices(self, rows: Sequence[Dict[str, Any]]) -> List[int]:
        """Select rows by song, preserving song-local ordering."""
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        song_ids = sorted(grouped)
        rng = np.random.default_rng(int(self.config.random_seed))
        if self.config.max_songs is not None:
            shuffled = list(song_ids)
            rng.shuffle(shuffled)
            song_ids = sorted(shuffled[: max(1, int(self.config.max_songs))])
        selected: List[int] = []
        for song_id in song_ids:
            ordered = sorted(
                grouped[song_id],
                key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx))),
            )
            selected.extend(int(index) for index in ordered)
        if self.config.max_rows is not None:
            selected = selected[: max(2, int(self.config.max_rows))]
        return selected

    def _build_song_sequences(
        self,
        mu: np.ndarray,
        features: np.ndarray,
        event_features: np.ndarray,
        sequence_embeddings: Optional[np.ndarray],
        rows: Sequence[Dict[str, Any]],
    ) -> List[SongSequence]:
        """Build song-local sequences from selected rows."""
        grouped: Dict[str, List[int]] = {}
        for local_index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(local_index)
        songs: List[SongSequence] = []
        for song_id in sorted(grouped):
            ordered = sorted(
                grouped[song_id],
                key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx))),
            )
            if len(ordered) < 3:
                continue
            songs.append(SongSequence(
                song_id=song_id,
                base_song_id=self._base_song_id(song_id),
                indices=[int(index) for index in ordered],
                mu=mu[ordered].astype(np.float32),
                features=features[ordered].astype(np.float32),
                event_features=event_features[ordered].astype(np.float32),
                sequence_embeddings=sequence_embeddings[ordered].astype(np.float32) if sequence_embeddings is not None else None,
            ))
        return songs

    def _load_miditok_sequence_embeddings(
        self,
        encoded_dir: Path,
        rows: Sequence[Dict[str, Any]],
    ) -> tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Load trained MidiTok-style sequence embeddings when available."""
        path = encoded_dir / "miditok_sequence_embeddings.npz"
        if not path.exists():
            return None, {"sequence_embedding_source": "missing", "embedding_path": str(path)}
        archive = np.load(path)
        try:
            missing = [str(row.get("tensor_key", "")) for row in rows if str(row.get("tensor_key", "")) not in archive.files]
            if missing:
                return None, {
                    "sequence_embedding_source": "cache_key_miss",
                    "embedding_path": str(path),
                    "missing_key_count": int(len(missing)),
                    "missing_key_examples": missing[:10],
                }
            matrix = np.stack([np.asarray(archive[str(row.get("tensor_key", ""))], dtype=np.float32) for row in rows], axis=0)
            return matrix.astype(np.float32), {
                "sequence_embedding_source": "cached_miditok_sequence_embeddings",
                "embedding_path": str(path),
                "shape": [int(item) for item in matrix.shape],
            }
        finally:
            archive.close()

    def _concat_optional_sequence_embeddings(self, songs: Sequence[SongSequence]) -> Optional[np.ndarray]:
        """Concatenate sequence embeddings if present."""
        if not songs or songs[0].sequence_embeddings is None:
            return None
        return np.concatenate([song.sequence_embeddings for song in songs if song.sequence_embeddings is not None], axis=0)

    def _feature_redundancy(self, features: np.ndarray) -> Dict[str, Any]:
        """Find highly correlated explicit features."""
        return self._named_feature_redundancy(features, BAR_FEATURE_NAMES)

    def _named_feature_redundancy(self, features: np.ndarray, feature_names: Sequence[str]) -> Dict[str, Any]:
        """Find highly correlated named features."""
        corr = self._corrcoef(features)
        pairs: List[Dict[str, Any]] = []
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[1]):
                value = float(corr[i, j])
                if math.isfinite(value):
                    pairs.append({
                        "feature_a": str(feature_names[i]),
                        "feature_b": str(feature_names[j]),
                        "correlation": value,
                        "abs_correlation": abs(value),
                    })
        pairs = sorted(pairs, key=lambda item: item["abs_correlation"], reverse=True)
        high_pairs = [
            item for item in pairs
            if float(item["abs_correlation"]) >= float(self.config.redundancy_threshold)
        ]
        eigenvalues = np.linalg.eigvalsh(np.nan_to_num(corr, nan=0.0))
        positive = np.clip(eigenvalues, 0.0, None)
        effective_rank = float((positive.sum() ** 2) / np.clip(np.sum(positive ** 2), 1.0e-8, None))
        return {
            "threshold": float(self.config.redundancy_threshold),
            "high_pair_count": int(len(high_pairs)),
            "top_pairs": pairs[:30],
            "high_pairs": high_pairs[:100],
            "correlation_effective_rank": effective_rank,
            "correlation_matrix": corr.tolist(),
        }

    def _zero_lag_correlation(
        self,
        features: np.ndarray,
        mu: np.ndarray,
        delta_mu: np.ndarray,
        songs: Sequence[SongSequence],
    ) -> Dict[str, Any]:
        """Measure feature relation to same-bar latent and next-step movement."""
        feature_to_mu = self._cross_corr_summary(features, mu, BAR_FEATURE_NAMES)
        aligned_features = np.concatenate([song.features[1:] for song in songs if len(song.indices) >= 2], axis=0)
        feature_to_delta_mu = self._cross_corr_summary(aligned_features, delta_mu, BAR_FEATURE_NAMES)
        event_features = np.concatenate([song.event_features for song in songs], axis=0)
        aligned_event_features = np.concatenate([song.event_features[1:] for song in songs if len(song.indices) >= 2], axis=0)
        event_to_mu = self._cross_corr_summary(event_features, mu, MIDITOK_STYLE_FEATURE_NAMES)
        event_to_delta_mu = self._cross_corr_summary(aligned_event_features, delta_mu, MIDITOK_STYLE_FEATURE_NAMES)
        return {
            "feature_to_same_bar_latent_mu": feature_to_mu,
            "feature_to_same_bar_delta_mu": feature_to_delta_mu,
            "miditok_event_to_same_bar_latent_mu": event_to_mu,
            "miditok_event_to_same_bar_delta_mu": event_to_delta_mu,
        }

    def _lag_correlation(self, songs: Sequence[SongSequence]) -> Dict[str, Any]:
        """Correlate feature lag k with current latent movement."""
        return self._lag_correlation_for_matrix(
            songs=songs,
            names=BAR_FEATURE_NAMES,
            value_getter=lambda song: song.features,
        )

    def _lag_correlation_for_matrix(
        self,
        songs: Sequence[SongSequence],
        names: Sequence[str],
        value_getter: Any,
    ) -> Dict[str, Any]:
        """Correlate lagged named feature matrix with current latent movement."""
        max_lag = max(1, int(self.config.max_lag))
        lag_rows: List[Dict[str, Any]] = []
        by_feature: Dict[str, Dict[str, Any]] = {}
        lag_matrix = np.zeros((len(names), max_lag), dtype=np.float32)
        for lag in range(1, max_lag + 1):
            x_parts: List[np.ndarray] = []
            y_parts: List[np.ndarray] = []
            for song in songs:
                if song.mu.shape[0] <= lag:
                    continue
                delta_mu = np.diff(song.mu, axis=0)
                # target movement at t is mu[t] - mu[t-1], with lagged feature at t-lag.
                values = value_getter(song)
                x_parts.append(values[: values.shape[0] - lag])
                y_parts.append(delta_mu[lag - 1:])
            if not x_parts:
                continue
            x = np.concatenate(x_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)
            corr = self._cross_corr_matrix(x, y)
            abs_corr = np.abs(corr)
            per_feature = np.nanmean(abs_corr, axis=1)
            for feature_index, score in enumerate(per_feature):
                lag_matrix[feature_index, lag - 1] = float(score) if math.isfinite(float(score)) else 0.0
            lag_rows.append({
                "lag": int(lag),
                "sample_count": int(x.shape[0]),
                "mean_abs_corr": float(np.nanmean(abs_corr)),
                "max_abs_corr": float(np.nanmax(abs_corr)),
                "top_features": self._top_named_scores(per_feature, names, top_k=10),
            })
        for feature_index, feature_name in enumerate(names):
            scores = lag_matrix[feature_index]
            best_offset = int(np.argmax(scores)) if scores.size else 0
            by_feature[str(feature_name)] = {
                "best_lag": int(best_offset + 1),
                "best_mean_abs_corr_to_delta_mu": float(scores[best_offset]) if scores.size else 0.0,
                "mean_abs_corr_across_lags": float(np.mean(scores)) if scores.size else 0.0,
            }
        return {
            "target": "delta_mu_t",
            "max_lag": int(max_lag),
            "lag_rows": lag_rows,
            "by_feature": by_feature,
            "lag_feature_matrix": lag_matrix.tolist(),
        }

    def _stationarity_summary(self, songs: Sequence[SongSequence]) -> Dict[str, Any]:
        """Summarize drift and differencing behavior for explicit features."""
        rows: List[Dict[str, Any]] = []
        for feature_index, feature_name in enumerate(BAR_FEATURE_NAMES):
            slopes: List[float] = []
            drift_values: List[float] = []
            lag1_values: List[float] = []
            diff_std_ratios: List[float] = []
            for song in songs:
                values = song.features[:, feature_index].astype(np.float64)
                if values.size < 4:
                    continue
                std = float(np.std(values))
                if std <= 1.0e-8:
                    slopes.append(0.0)
                    drift_values.append(0.0)
                    lag1_values.append(0.0)
                    diff_std_ratios.append(0.0)
                    continue
                x = np.arange(values.size, dtype=np.float64)
                slope = float(np.polyfit(x, values, deg=1)[0] * values.size / std)
                third = max(1, values.size // 3)
                start_mean = float(np.mean(values[:third]))
                end_mean = float(np.mean(values[-third:]))
                drift = abs(end_mean - start_mean) / std
                lag1 = self._safe_corr(values[:-1], values[1:])
                diff_std_ratio = float(np.std(np.diff(values)) / std)
                slopes.append(abs(slope))
                drift_values.append(float(drift))
                lag1_values.append(float(lag1))
                diff_std_ratios.append(float(diff_std_ratio))
            rows.append({
                "feature": feature_name,
                "mean_abs_normalized_slope": self._safe_mean(slopes),
                "mean_start_end_drift_std_units": self._safe_mean(drift_values),
                "mean_lag1_autocorr": self._safe_mean(lag1_values),
                "mean_diff_std_ratio": self._safe_mean(diff_std_ratios),
                "stationarity_risk": self._stationarity_risk(drift_values, slopes, lag1_values),
            })
        return {
            "method": "songwise trend, start/end drift, lag1 autocorrelation, and first-difference variance ratio",
            "drift_threshold": float(self.config.drift_threshold),
            "features": sorted(rows, key=lambda item: (
                item["stationarity_risk"] != "high",
                -float(item["mean_start_end_drift_std_units"]),
            )),
        }

    def _predictive_probes(self, songs: Sequence[SongSequence]) -> Dict[str, Any]:
        """Run VAR-style ridge probes for several representations and lag orders."""
        train_songs, val_songs, split_summary = self._split_songs(songs)
        representations = [
            ("latent_to_next_mu", "mu", "next_mu"),
            ("latent_delta_to_delta_mu", "delta_mu", "delta_mu"),
            ("features_to_delta_mu", "features", "delta_mu"),
            ("feature_delta_to_delta_mu", "delta_features", "delta_mu"),
            ("miditok_events_to_delta_mu", "event_features", "delta_mu"),
            ("miditok_event_delta_to_delta_mu", "delta_event_features", "delta_mu"),
            ("hybrid_to_next_mu", "hybrid", "next_mu"),
            ("hybrid_delta_to_delta_mu", "hybrid_delta", "delta_mu"),
            ("latent_miditok_events_to_next_mu", "latent_event_hybrid", "next_mu"),
            ("latent_miditok_event_delta_to_delta_mu", "latent_event_hybrid_delta", "delta_mu"),
        ]
        if self._has_sequence_embeddings(songs):
            representations.extend([
                ("miditok_sequence_embedding_to_delta_mu", "sequence_embeddings", "delta_mu"),
                ("miditok_sequence_embedding_delta_to_delta_mu", "delta_sequence_embeddings", "delta_mu"),
                ("latent_miditok_sequence_embedding_to_next_mu", "latent_sequence_hybrid", "next_mu"),
                ("latent_miditok_sequence_embedding_delta_to_delta_mu", "latent_sequence_hybrid_delta", "delta_mu"),
            ])
        max_lag = max(1, int(self.config.max_lag))
        rows: List[Dict[str, Any]] = []
        best_by_representation: Dict[str, Any] = {}
        for name, x_kind, y_kind in representations:
            rep_rows: List[Dict[str, Any]] = []
            for lag in range(1, max_lag + 1):
                train_x, train_y = self._lagged_supervised_arrays(train_songs, lag, x_kind, y_kind)
                val_x, val_y = self._lagged_supervised_arrays(val_songs, lag, x_kind, y_kind)
                if train_x.shape[0] < 4 or val_x.shape[0] < 1:
                    continue
                metrics = self._ridge_probe(train_x, train_y, val_x, val_y)
                item = {
                    "representation": name,
                    "x_kind": x_kind,
                    "y_kind": y_kind,
                    "lag": int(lag),
                    "train_samples": int(train_x.shape[0]),
                    "validation_samples": int(val_x.shape[0]),
                    **metrics,
                }
                rows.append(item)
                rep_rows.append(item)
            if rep_rows:
                best_by_representation[name] = min(rep_rows, key=lambda item: float(item["val_mse"]))
        return {
            "method": "VAR-style ridge probes with song-held-out validation",
            "ridge_alpha": float(self.config.ridge_alpha),
            "split": split_summary,
            "rows": rows,
            "best_by_representation": best_by_representation,
        }

    def _lagged_supervised_arrays(
        self,
        songs: Sequence[SongSequence],
        lag: int,
        x_kind: str,
        y_kind: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build lagged supervised samples from song-local sequences."""
        x_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        for song in songs:
            if song.mu.shape[0] <= lag:
                continue
            mu = song.mu.astype(np.float32)
            features = song.features.astype(np.float32)
            event_features = song.event_features.astype(np.float32)
            sequence_embeddings = song.sequence_embeddings.astype(np.float32) if song.sequence_embeddings is not None else None
            delta_mu = np.diff(mu, axis=0)
            delta_features = np.diff(features, axis=0)
            delta_event_features = np.diff(event_features, axis=0)
            delta_sequence_embeddings = np.diff(sequence_embeddings, axis=0) if sequence_embeddings is not None else None
            for t in range(lag, mu.shape[0]):
                if x_kind == "mu":
                    history = mu[t - lag:t]
                elif x_kind == "delta_mu":
                    if t < lag + 1:
                        continue
                    history = delta_mu[t - lag - 1:t - 1]
                elif x_kind == "features":
                    history = features[t - lag:t]
                elif x_kind == "delta_features":
                    if t < lag + 1:
                        continue
                    history = delta_features[t - lag - 1:t - 1]
                elif x_kind == "event_features":
                    history = event_features[t - lag:t]
                elif x_kind == "delta_event_features":
                    if t < lag + 1:
                        continue
                    history = delta_event_features[t - lag - 1:t - 1]
                elif x_kind == "sequence_embeddings":
                    if sequence_embeddings is None:
                        continue
                    history = sequence_embeddings[t - lag:t]
                elif x_kind == "delta_sequence_embeddings":
                    if sequence_embeddings is None or delta_sequence_embeddings is None or t < lag + 1:
                        continue
                    history = delta_sequence_embeddings[t - lag - 1:t - 1]
                elif x_kind == "hybrid":
                    history = np.concatenate([mu[t - lag:t], features[t - lag:t]], axis=1)
                elif x_kind == "hybrid_delta":
                    if t < lag + 1:
                        continue
                    history = np.concatenate([delta_mu[t - lag - 1:t - 1], delta_features[t - lag - 1:t - 1]], axis=1)
                elif x_kind == "latent_event_hybrid":
                    history = np.concatenate([mu[t - lag:t], event_features[t - lag:t]], axis=1)
                elif x_kind == "latent_event_hybrid_delta":
                    if t < lag + 1:
                        continue
                    history = np.concatenate([delta_mu[t - lag - 1:t - 1], delta_event_features[t - lag - 1:t - 1]], axis=1)
                elif x_kind == "latent_sequence_hybrid":
                    if sequence_embeddings is None:
                        continue
                    history = np.concatenate([mu[t - lag:t], sequence_embeddings[t - lag:t]], axis=1)
                elif x_kind == "latent_sequence_hybrid_delta":
                    if sequence_embeddings is None or delta_sequence_embeddings is None or t < lag + 1:
                        continue
                    history = np.concatenate([delta_mu[t - lag - 1:t - 1], delta_sequence_embeddings[t - lag - 1:t - 1]], axis=1)
                else:
                    raise ValueError(f"Unsupported x_kind: {x_kind}")
                if y_kind == "next_mu":
                    target = mu[t]
                elif y_kind == "delta_mu":
                    target = delta_mu[t - 1]
                else:
                    raise ValueError(f"Unsupported y_kind: {y_kind}")
                x_parts.append(history.reshape(-1))
                y_parts.append(target)
        if not x_parts:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
        return np.stack(x_parts, axis=0).astype(np.float32), np.stack(y_parts, axis=0).astype(np.float32)

    def _has_sequence_embeddings(self, songs: Sequence[SongSequence]) -> bool:
        """Return whether all songs include trained sequence embeddings."""
        return bool(songs) and all(song.sequence_embeddings is not None for song in songs)

    def _ridge_probe(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> Dict[str, float]:
        """Fit closed-form ridge regression and evaluate next-step prediction."""
        x_mean = train_x.mean(axis=0, keepdims=True)
        x_std = np.clip(train_x.std(axis=0, keepdims=True), 1.0e-6, None)
        y_mean = train_y.mean(axis=0, keepdims=True)
        train_xs = (train_x - x_mean) / x_std
        val_xs = (val_x - x_mean) / x_std
        train_yc = train_y - y_mean
        design = np.concatenate([train_xs, np.ones((train_xs.shape[0], 1), dtype=np.float32)], axis=1).astype(np.float64)
        val_design = np.concatenate([val_xs, np.ones((val_xs.shape[0], 1), dtype=np.float32)], axis=1).astype(np.float64)
        regularizer = np.eye(design.shape[1], dtype=np.float64) * float(self.config.ridge_alpha)
        regularizer[-1, -1] = 0.0
        xtx = design.T @ design + regularizer
        xty = design.T @ train_yc.astype(np.float64)
        weights = np.linalg.solve(xtx, xty)
        pred = (val_design @ weights).astype(np.float32) + y_mean.astype(np.float32)
        residual = pred - val_y
        baseline = val_y - train_y.mean(axis=0, keepdims=True)
        val_mse = float(np.mean(residual ** 2))
        baseline_mse = float(np.mean(baseline ** 2))
        r2 = float(1.0 - val_mse / max(1.0e-8, baseline_mse))
        direction = self._direction_cosine(pred, val_y)
        return {
            "val_mse": val_mse,
            "baseline_mse": baseline_mse,
            "val_r2_vs_train_mean": r2,
            "direction_cosine_mean": float(np.mean(direction)),
            "direction_cosine_median": float(np.median(direction)),
        }

    def _split_songs(self, songs: Sequence[SongSequence]) -> Tuple[List[SongSequence], List[SongSequence], Dict[str, Any]]:
        """Split by base song to reduce transposition leakage."""
        base_ids = sorted({song.base_song_id for song in songs})
        rng = np.random.default_rng(int(self.config.random_seed))
        shuffled = list(base_ids)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * float(self.config.validation_ratio)))) if len(shuffled) > 1 else 1
        validation_bases = set(shuffled[:val_count])
        train = [song for song in songs if song.base_song_id not in validation_bases]
        val = [song for song in songs if song.base_song_id in validation_bases]
        if not train and val:
            train = val[:]
        return train, val, {
            "validation_ratio": float(self.config.validation_ratio),
            "train_song_count": int(len(train)),
            "validation_song_count": int(len(val)),
            "train_base_song_count": int(len({song.base_song_id for song in train})),
            "validation_base_song_count": int(len({song.base_song_id for song in val})),
            "validation_base_song_ids": sorted(validation_bases),
        }

    def _recommendations(
        self,
        feature_redundancy: Dict[str, Any],
        lag_correlation: Dict[str, Any],
        stationarity: Dict[str, Any],
        predictive: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate data-use recommendations from diagnostics."""
        low_lag = [
            {"feature": name, **values}
            for name, values in lag_correlation["by_feature"].items()
            if float(values["best_mean_abs_corr_to_delta_mu"]) < float(self.config.low_lag_correlation_threshold)
        ]
        high_drift = [
            item for item in stationarity["features"]
            if item["stationarity_risk"] == "high"
        ]
        best = predictive.get("best_by_representation", {})
        delta_feature = best.get("feature_delta_to_delta_mu", {})
        raw_feature = best.get("features_to_delta_mu", {})
        event_feature = best.get("miditok_events_to_delta_mu", {})
        event_delta = best.get("miditok_event_delta_to_delta_mu", {})
        latent_event = best.get("latent_miditok_events_to_next_mu", {})
        sequence_feature = best.get("miditok_sequence_embedding_to_delta_mu", {})
        sequence_delta = best.get("miditok_sequence_embedding_delta_to_delta_mu", {})
        latent_sequence = best.get("latent_miditok_sequence_embedding_to_next_mu", {})
        latent_only = best.get("latent_to_next_mu", {})
        hybrid_delta = best.get("hybrid_delta_to_delta_mu", {})
        hybrid_raw = best.get("hybrid_to_next_mu", {})
        messages: List[str] = []
        if feature_redundancy["high_pair_count"] > 0:
            messages.append("Some explicit features are highly redundant; consider dropping or grouping one side of each high-correlation pair.")
        if high_drift:
            messages.append("Several explicit features show strong songwise drift; test first-difference or song-normalized versions instead of raw values.")
        if low_lag:
            messages.append("Some features have weak lag relation to latent movement; keep them as diagnostics unless ablation proves useful.")
        if delta_feature and raw_feature and float(delta_feature["val_mse"]) < float(raw_feature["val_mse"]):
            messages.append("Differenced explicit features predict latent movement better than raw explicit features in this split.")
        if hybrid_delta and hybrid_raw and float(hybrid_delta["val_mse"]) < float(hybrid_raw["val_mse"]):
            messages.append("Hybrid delta representation is more predictive than raw hybrid next-state prediction in this split.")
        if event_feature and raw_feature and float(event_feature["val_mse"]) < float(raw_feature["val_mse"]):
            messages.append("MidiTok-style event features predict latent movement better than the current 27 explicit features in this split.")
        if event_delta and delta_feature and float(event_delta["val_mse"]) < float(delta_feature["val_mse"]):
            messages.append("Differenced MidiTok-style event features outperform differenced explicit features in this split.")
        if latent_event and latent_only and float(latent_event["val_mse"]) < float(latent_only["val_mse"]):
            messages.append("Adding MidiTok-style event features to latent history improves next-latent prediction in this split.")
        if sequence_feature and raw_feature and float(sequence_feature["val_mse"]) < float(raw_feature["val_mse"]):
            messages.append("Trained MidiTok-style sequence embeddings predict latent movement better than explicit summary features in this split.")
        if sequence_delta and delta_feature and float(sequence_delta["val_mse"]) < float(delta_feature["val_mse"]):
            messages.append("Differenced trained MidiTok-style sequence embeddings outperform differenced explicit features in this split.")
        if latent_sequence and latent_only and float(latent_sequence["val_mse"]) < float(latent_only["val_mse"]):
            messages.append("Adding trained MidiTok-style sequence embeddings to latent history improves next-latent prediction in this split.")
        return {
            "summary": messages,
            "low_lag_features": sorted(low_lag, key=lambda item: float(item["best_mean_abs_corr_to_delta_mu"]))[:30],
            "high_drift_features": high_drift[:30],
            "top_redundant_pairs": feature_redundancy["top_pairs"][:20],
            "predictive_best_by_representation": best,
        }

    def _write_outputs(
        self,
        output_dir: Path,
        report: Dict[str, Any],
        feature_redundancy: Dict[str, Any],
        lag_correlation: Dict[str, Any],
        event_feature_redundancy: Dict[str, Any],
        event_lag_correlation: Dict[str, Any],
    ) -> None:
        """Write JSON, markdown, CSV, and optional figures."""
        json_path = output_dir / "bar_dynamics_diagnostics.json"
        md_path = output_dir / "bar_dynamics_report.md"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        md_path.write_text(self._markdown_report(report), encoding="utf-8")
        self._write_csv(output_dir / "feature_lag_scores.csv", lag_correlation["by_feature"])
        self._write_csv(output_dir / "miditok_event_lag_scores.csv", event_lag_correlation["by_feature"])
        self._write_csv(output_dir / "feature_stationarity.csv", {
            item["feature"]: item for item in report["stationarity"]["features"]
        })
        self._write_predictive_csv(output_dir / "predictive_probes.csv", report["predictive_probes"]["rows"])
        if self.config.write_figures:
            self._write_figures(output_dir, feature_redundancy, lag_correlation, event_feature_redundancy, event_lag_correlation)

    def _markdown_report(self, report: Dict[str, Any]) -> str:
        """Render concise markdown report."""
        data = report["data"]
        lines = [
            "# Bar Dynamics VAR-Style Diagnostics",
            "",
            "This report audits whether explicit bar features and latent states carry useful lagged information for next-bar dynamics.",
            "",
            "## Data",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Songs | {data['song_count']} |",
            f"| Base songs | {data['base_song_count']} |",
            f"| Bars | {data['bar_count']} |",
            f"| Latent dim | {data['latent_dim']} |",
            f"| Explicit feature dim | {data['feature_dim']} |",
            f"| MidiTok-style event feature dim | {data['event_feature_dim']} |",
            f"| MidiTok sequence embedding dim | {data['sequence_embedding_dim']} |",
            f"| Mean song bars | {data['song_lengths']['mean']:.3f} |",
            "",
            "## Main Recommendations",
            "",
        ]
        recommendations = report["recommendations"]["summary"]
        if recommendations:
            lines.extend(f"- {item}" for item in recommendations)
        else:
            lines.append("- No strong data-use warning was detected by the current thresholds.")
        lines.extend([
            "",
            "## Best VAR-Style Predictive Probes",
            "",
            "| Representation | Best lag | Val MSE | Baseline MSE | R2 vs mean | Direction cosine |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for name, item in report["predictive_probes"]["best_by_representation"].items():
            lines.append(
                f"| {name} | {item['lag']} | {item['val_mse']:.6f} | {item['baseline_mse']:.6f} | "
                f"{item['val_r2_vs_train_mean']:.6f} | {item['direction_cosine_mean']:.6f} |"
            )
        lines.extend([
            "",
            "## Highest Lag-Correlation Features",
            "",
            "| Feature | Best lag | Mean abs corr to delta_mu | Mean across lags |",
            "| --- | ---: | ---: | ---: |",
        ])
        by_feature = report["lag_correlation"]["by_feature"]
        for name, item in sorted(by_feature.items(), key=lambda pair: float(pair[1]["best_mean_abs_corr_to_delta_mu"]), reverse=True)[:20]:
            lines.append(
                f"| {name} | {item['best_lag']} | {item['best_mean_abs_corr_to_delta_mu']:.6f} | "
                f"{item['mean_abs_corr_across_lags']:.6f} |"
            )
        lines.extend([
            "",
            "## Highest MidiTok-Style Event Lag-Correlation Features",
            "",
            "| Feature | Best lag | Mean abs corr to delta_mu | Mean across lags |",
            "| --- | ---: | ---: | ---: |",
        ])
        by_event_feature = report["event_lag_correlation"]["by_feature"]
        for name, item in sorted(by_event_feature.items(), key=lambda pair: float(pair[1]["best_mean_abs_corr_to_delta_mu"]), reverse=True)[:20]:
            lines.append(
                f"| {name} | {item['best_lag']} | {item['best_mean_abs_corr_to_delta_mu']:.6f} | "
                f"{item['mean_abs_corr_across_lags']:.6f} |"
            )
        lines.extend([
            "",
            "## High Drift Features",
            "",
            "| Feature | Drift | Slope | Lag1 autocorr | Diff std ratio | Risk |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for item in report["stationarity"]["features"][:20]:
            lines.append(
                f"| {item['feature']} | {item['mean_start_end_drift_std_units']:.6f} | "
                f"{item['mean_abs_normalized_slope']:.6f} | {item['mean_lag1_autocorr']:.6f} | "
                f"{item['mean_diff_std_ratio']:.6f} | {item['stationarity_risk']} |"
            )
        lines.extend([
            "",
            "## Redundant Explicit Feature Pairs",
            "",
            "| Feature A | Feature B | Corr |",
            "| --- | --- | ---: |",
        ])
        for item in report["feature_redundancy"]["top_pairs"][:20]:
            lines.append(f"| {item['feature_a']} | {item['feature_b']} | {item['correlation']:.6f} |")
        lines.extend([
            "",
            "## Redundant MidiTok-Style Event Feature Pairs",
            "",
            "| Feature A | Feature B | Corr |",
            "| --- | --- | ---: |",
        ])
        for item in report["event_feature_redundancy"]["top_pairs"][:20]:
            lines.append(f"| {item['feature_a']} | {item['feature_b']} | {item['correlation']:.6f} |")
        lines.extend([
            "",
            "## Interpretation",
            "",
            "- If `feature_delta_to_delta_mu` beats `features_to_delta_mu`, explicit features should probably enter the temporal model as motion/difference signals.",
            "- If `miditok_events_to_delta_mu` beats `features_to_delta_mu`, event-level encoding is a better temporal support representation than static summary features.",
            "- If `latent_miditok_events_to_next_mu` beats `latent_to_next_mu`, MidiTok-style events should be tested as a temporal-model auxiliary input rather than only diagnostics.",
            "- If many features have high drift, raw values are likely non-stationary and should be song-normalized or differenced.",
            "- If high redundant pairs dominate, the 27D feature vector may be too wide for the amount of data and should be compressed or grouped.",
            "- If every VAR-style probe is weak, the issue is likely representation insufficiency or data scale rather than Transformer architecture alone.",
            "",
        ])
        return "\n".join(lines)

    def _write_csv(self, path: Path, mapping: Dict[str, Dict[str, Any]]) -> None:
        """Write a simple CSV from a mapping of feature diagnostics."""
        if not mapping:
            path.write_text("", encoding="utf-8")
            return
        keys = sorted({key for item in mapping.values() for key in item.keys() if key != "feature"})
        lines = ["feature," + ",".join(keys)]
        for feature, item in mapping.items():
            values = [self._csv_value(item.get(key, "")) for key in keys]
            lines.append(self._csv_value(feature) + "," + ",".join(values))
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_predictive_csv(self, path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        """Write predictive probe rows."""
        keys = ["representation", "x_kind", "y_kind", "lag", "train_samples", "validation_samples", "val_mse", "baseline_mse", "val_r2_vs_train_mean", "direction_cosine_mean", "direction_cosine_median"]
        lines = [",".join(keys)]
        for item in rows:
            lines.append(",".join(self._csv_value(item.get(key, "")) for key in keys))
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_figures(
        self,
        output_dir: Path,
        feature_redundancy: Dict[str, Any],
        lag_correlation: Dict[str, Any],
        event_feature_redundancy: Dict[str, Any],
        event_lag_correlation: Dict[str, Any],
    ) -> None:
        """Write optional diagnostic heatmaps."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        corr = np.asarray(feature_redundancy["correlation_matrix"], dtype=np.float32)
        self._heatmap(
            plt=plt,
            matrix=corr,
            title="Explicit Feature Correlation",
            path=figures_dir / "feature_correlation_heatmap.png",
            x_labels=BAR_FEATURE_NAMES,
            y_labels=BAR_FEATURE_NAMES,
        )
        lag_matrix = np.asarray(lag_correlation["lag_feature_matrix"], dtype=np.float32)
        self._heatmap(
            plt=plt,
            matrix=lag_matrix,
            title="Feature Lag Mean Abs Corr to Latent Movement",
            path=figures_dir / "feature_lag_correlation_heatmap.png",
            x_labels=[str(i) for i in range(1, lag_matrix.shape[1] + 1)],
            y_labels=BAR_FEATURE_NAMES,
        )
        event_corr = np.asarray(event_feature_redundancy["correlation_matrix"], dtype=np.float32)
        self._heatmap(
            plt=plt,
            matrix=event_corr,
            title="MidiTok-Style Event Feature Correlation",
            path=figures_dir / "miditok_event_feature_correlation_heatmap.png",
            x_labels=MIDITOK_STYLE_FEATURE_NAMES,
            y_labels=MIDITOK_STYLE_FEATURE_NAMES,
        )
        event_lag_matrix = np.asarray(event_lag_correlation["lag_feature_matrix"], dtype=np.float32)
        self._heatmap(
            plt=plt,
            matrix=event_lag_matrix,
            title="MidiTok-Style Event Feature Lag Mean Abs Corr to Latent Movement",
            path=figures_dir / "miditok_event_lag_correlation_heatmap.png",
            x_labels=[str(i) for i in range(1, event_lag_matrix.shape[1] + 1)],
            y_labels=MIDITOK_STYLE_FEATURE_NAMES,
        )

    def _heatmap(self, plt: Any, matrix: np.ndarray, title: str, path: Path, x_labels: Sequence[str], y_labels: Sequence[str]) -> None:
        """Render one heatmap."""
        fig_width = max(8.0, min(18.0, 0.38 * len(x_labels)))
        fig_height = max(8.0, min(18.0, 0.32 * len(y_labels)))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=6)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _cross_corr_summary(self, x: np.ndarray, y: np.ndarray, feature_names: Sequence[str]) -> List[Dict[str, Any]]:
        """Summarize cross correlation from x features to y dimensions."""
        corr = self._cross_corr_matrix(x, y)
        abs_corr = np.abs(corr)
        rows: List[Dict[str, Any]] = []
        for index, name in enumerate(feature_names):
            row = abs_corr[index]
            rows.append({
                "feature": str(name),
                "mean_abs_corr": float(np.nanmean(row)),
                "max_abs_corr": float(np.nanmax(row)),
                "best_target_dim": int(np.nanargmax(row)),
            })
        return sorted(rows, key=lambda item: item["mean_abs_corr"], reverse=True)

    def _cross_corr_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return correlation matrix between columns of x and columns of y."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x_centered = x - np.nanmean(x, axis=0, keepdims=True)
        y_centered = y - np.nanmean(y, axis=0, keepdims=True)
        x_std = np.nanstd(x_centered, axis=0, keepdims=True)
        y_std = np.nanstd(y_centered, axis=0, keepdims=True)
        denom = np.clip(x_std.T @ y_std * max(1, x.shape[0]), 1.0e-8, None)
        corr = (x_centered.T @ y_centered) / denom
        return np.clip(corr, -1.0, 1.0)

    def _corrcoef(self, values: np.ndarray) -> np.ndarray:
        """Return robust column correlation."""
        corr = self._cross_corr_matrix(values, values)
        np.fill_diagonal(corr, 1.0)
        return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    def _top_feature_scores(self, scores: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Return named feature scores."""
        return self._top_named_scores(scores, BAR_FEATURE_NAMES, top_k)

    def _top_named_scores(self, scores: np.ndarray, names: Sequence[str], top_k: int) -> List[Dict[str, Any]]:
        """Return named scores."""
        order = np.argsort(scores)[::-1][:top_k]
        return [
            {"feature": str(names[int(index)]), "score": float(scores[int(index)])}
            for index in order
        ]

    def _direction_cosine(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return rowwise cosine similarity."""
        numerator = np.sum(pred * target, axis=1)
        denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
        return numerator / np.clip(denominator, 1.0e-8, None)

    def _stationarity_risk(self, drift_values: Sequence[float], slopes: Sequence[float], lag1_values: Sequence[float]) -> str:
        """Classify stationarity risk from simple diagnostics."""
        drift = self._safe_mean(drift_values)
        slope = self._safe_mean(slopes)
        lag1 = self._safe_mean(lag1_values)
        if drift >= float(self.config.drift_threshold) or slope >= float(self.config.drift_threshold) or lag1 >= 0.85:
            return "high"
        if drift >= float(self.config.drift_threshold) * 0.5 or slope >= float(self.config.drift_threshold) * 0.5 or lag1 >= 0.65:
            return "medium"
        return "low"

    def _base_song_id(self, song_id: str) -> str:
        """Collapse transposition suffix."""
        return re.sub(r"_T[+-]?\d+$", "", str(song_id))

    def _numeric_summary(self, values: Sequence[float | int]) -> Dict[str, Any]:
        """Return numeric summary."""
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "mean": float(np.mean(array)) if array.size else 0.0,
            "median": float(np.median(array)) if array.size else 0.0,
            "min": float(np.min(array)) if array.size else 0.0,
            "max": float(np.max(array)) if array.size else 0.0,
        }

    def _safe_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return scalar correlation with zero-variance guard."""
        if x.size < 2 or y.size < 2:
            return 0.0
        if float(np.std(x)) <= 1.0e-8 or float(np.std(y)) <= 1.0e-8:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def _safe_mean(self, values: Sequence[float]) -> float:
        """Return safe mean."""
        if not values:
            return 0.0
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    def _csv_value(self, value: Any) -> str:
        """Escape one CSV value."""
        text = str(value)
        if any(char in text for char in [",", "\"", "\n"]):
            text = "\"" + text.replace("\"", "\"\"") + "\""
        return text

    def _config_dict(self) -> Dict[str, Any]:
        """Return JSON-friendly config."""
        result = {}
        for key, value in self.config.__dict__.items():
            result[key] = str(value) if isinstance(value, Path) else value
        return result
