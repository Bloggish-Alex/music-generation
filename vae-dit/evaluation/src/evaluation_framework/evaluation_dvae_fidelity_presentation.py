"""Report construction and presentation for DVAE fidelity."""

from __future__ import annotations

import io
from typing import Any, Mapping, Sequence

import numpy as np


def _report(inputs: Mapping[str, Any], profiles: Mapping[str, Mapping[str, Any]], missing: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "schema_version": "assessment_report.v1",
        "status": "MONITOR",
        "metrics": {"splits": profiles, "train_validation_gap": _split_gap(profiles)},
        "findings": [{"classification": "dvae_information_boundary", "text": "DVAE 重建指标按音乐信息类别分别呈现；它们用于定位表示边界，不合成为模型质量总分。"}],
        "provenance": {"input_statuses": inputs.get("splits", {})},
        "missing_inputs": list(missing),
    }


def _unavailable_report(inputs: Mapping[str, Any], missing: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "schema_version": "assessment_report.v1",
        "status": "UNAVAILABLE",
        "metrics": {},
        "findings": [],
        "provenance": {"input_availability": inputs.get("availability", {})},
        "missing_inputs": list(missing) or [{"field": "status", "reason": "No DVAE fidelity raw status artifact was provided."}],
    }


def _split_gap(profiles: Mapping[str, Mapping[str, Any]]) -> dict[str, float | None] | str:
    train, validation = profiles.get("train"), profiles.get("validation")
    if train is None or validation is None:
        return "UNAVAILABLE"
    return {
        "state_accuracy_gap": float(validation["metrics"]["state"]["accuracy"] - train["metrics"]["state"]["accuracy"]),
        "chroma_cosine_gap": _difference(validation["metrics"]["chroma"]["cosine_mean"], train["metrics"]["chroma"]["cosine_mean"]),
        "register_delta_mae_gap": _difference(validation["metrics"]["register"]["delta_mae_semitones"], train["metrics"]["register"]["delta_mae_semitones"]),
    }


def _difference(left: float | None, right: float | None) -> float | None:
    return float(left - right) if left is not None and right is not None else None


def _markdown(report: Mapping[str, Any]) -> str:
    if report["status"] == "UNAVAILABLE":
        return "# DVAE 保真度\n\n生成这份画像所需的 DVAE 重建原始观察数据不可用。\n"
    lines = [
        "# DVAE 保真度",
        "",
        "下表分别呈现同一小节经过 `source -> deterministic mu -> decoded` 后保留的音乐信息。它不是单一总分，也不直接评价后续 trajectory 生成。",
        "",
        "| Split | 小节 | 状态准确率 | 起音 F1 | Chroma cosine | 相对音高 MAE（半音） | 音区变化 MAE（半音） | 密度 MAE（active slots） |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, profile in report["metrics"]["splits"].items():
        metrics = profile["metrics"]
        lines.append(
            "| {split} | {bars} | {state:.3f} | {onset:.3f} | {chroma} | {pitch} | {register} | {density:.2f} |".format(
                split=split, bars=profile["bar_count"], state=metrics["state"]["accuracy"],
                onset=metrics["state"]["onset_f1"], chroma=_number(metrics["chroma"]["cosine_mean"], 3),
                pitch=_number(metrics["relative_pitch"]["mae_semitones"], 2),
                register=_number(metrics["register"]["delta_mae_semitones"], 2),
                density=metrics["density"]["bar_active_slot_mae"],
            )
        )
    lines.extend(["", "纹理多样性单列保留在 JSON：它比较每个 latent 最近邻与其对应 decoded tensor 的相似度。latent 彼此不同而 decoded 过度相似时，提示可能存在解码纹理塌缩；该信号需要结合重建指标阅读。", ""])
    return _pitch_error_markdown(lines, report)


def _pitch_error_markdown(lines: list[str], report: Mapping[str, Any]) -> str:
    lines.extend([
        "", "## 音高误差结构", "",
        "下表将 shared active slot 中的 `decoded relative pitch - source relative pitch` 分开呈现。Pitch-class MAE 去除最近的 12 半音倍数；octave-component MAE 只保留该倍数。它们是互补诊断，不会组成新的总分。", "",
        "| Split | 总 MAE | Pitch-class MAE | Octave-component MAE | 纯八度错位率 | +/-12/+/-24 集中率 | 其他误差率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for split, profile in report["metrics"]["splits"].items():
        pitch = profile["metrics"]["relative_pitch"]
        bands = pitch["error_bands"]
        concentration = sum(_zero_if_none(bands[name]) for name in ("plus_12_rate", "minus_12_rate", "plus_24_rate", "minus_24_rate"))
        lines.append("| {split} | {mae} | {class_mae} | {octave_mae} | {pure} | {concentration} | {other} |".format(
            split=split, mae=_number(pitch["mae_semitones"], 2), class_mae=_number(pitch["pitch_class_mae_semitones"], 2),
            octave_mae=_number(pitch["octave_component_mae_semitones"], 2), pure=_percent(pitch["octave_equivalent_error_rate"]),
            concentration=_percent(concentration if bands["sample_count"] else None), other=_percent(bands["other_error_rate"]),
        ))
    lines.extend([
        "", "若 pitch-class MAE 很低，而纯八度错位率或 +/-12/+/-24 集中率很高，说明 pitch class 大体保留，主要损失在八度或音区 placement。若其他误差率很高，则更接近一般性的音高几何重建问题。", "",
        "| Split | Melody pitch MAE | Harmony representative-pitch MAE | Bass pitch MAE |",
        "| --- | ---: | ---: | ---: |",
    ])
    for split, profile in report["metrics"]["splits"].items():
        voices = profile["metrics"]["relative_pitch"]["voice_roles"]
        lines.append("| {split} | {melody} | {harmony} | {bass} |".format(
            split=split, melody=_voice_number(voices["melody"]), harmony=_voice_number(voices["harmony"]), bass=_voice_number(voices["bass"])))
    lines.extend(["", "| Split / Voice | Slope | Signed bias (semitones) | Pearson r | Samples |", "| --- | ---: | ---: | ---: | ---: |"])
    for split, profile in report["metrics"]["splits"].items():
        for role, metric in profile["metrics"]["relative_pitch"]["voice_roles"].items():
            lines.append("| {split} / {role} | {slope} | {bias} | {correlation} | {count} |".format(
                split=split, role=role, slope=_number(metric.get("slope"), 3), bias=_number(metric.get("signed_bias_semitones"), 2),
                correlation=_number(metric.get("pearson_correlation"), 3), count=metric.get("shared_active_slot_count", "--")))
    lines.extend([
        "", "Slope 用来判断声部音高跨度是否被压缩：1 表示比例关系保留，小于 1 表示向 anchor 收缩，大于 1 表示跨度被放大。Signed bias 是 `decoded - source` 的平均值：正值整体偏高，负值整体偏低。bias 接近 0 并不表示 MAE 很小。",
        "", "## 误差与距 anchor 距离", "",
        "距离定义为 `abs(source relative pitch)` 的半音数。下表使用所有 shared active slot；JSON 还会按每个语义声部保存同一分桶，因此能够区分距离效应与 voice label 效应。", "",
        "| Split | 距 anchor | 样本数 | MAE | Signed error |", "| --- | --- | ---: | ---: | ---: |",
    ])
    for split, profile in report["metrics"]["splits"].items():
        for band in profile["metrics"]["relative_pitch"]["distance_to_anchor"]["all_shared_active_slots"]:
            lines.append("| {split} | {distance} | {count} | {mae} | {bias} |".format(
                split=split, distance=band["range_semitones"], count=band["sample_count"],
                mae=_number(band["mae_semitones"], 2), bias=_number(band["signed_error_semitones"], 2)))
    lines.extend(["", "若误差随距离稳定上升，证据更支持“较大的 relative-pitch 数值幅度难以保留”，而不只是 melody/harmony/bass 标签效应。Harmony 是 semantic codec 在 harmony track 中的代表音高，并不等同于完整和弦的所有内声部。`--` 表示当前 tensor schema 没有该语义声部，或该声部没有可比较的 shared active slot。", ""])
    return "\n".join(lines)


def _number(value: float | None, digits: int) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _percent(value: float | None) -> str:
    return "--" if value is None else f"{value * 100.0:.1f}%"


def _zero_if_none(value: float | None) -> float:
    return 0.0 if value is None else value


def _voice_number(metric: Mapping[str, Any]) -> str:
    return _number(metric.get("mae_semitones"), 2) if metric.get("status") != "UNAVAILABLE" else "--"


def _summary_png(profiles: Mapping[str, Mapping[str, Any]]) -> bytes:
    try:
        import matplotlib.pyplot as plt

        labels = list(profiles)
        figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.4))
        axes[0].bar(labels, [profiles[name]["metrics"]["state"]["accuracy"] for name in labels], color="#3B7EA1")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_title("State accuracy", loc="left")
        chroma = [profiles[name]["metrics"]["chroma"]["cosine_mean"] for name in labels]
        axes[1].bar(labels, [value if value is not None else 0.0 for value in chroma], color="#2E8B57")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Decoded chroma cosine", loc="left")
        for axis in axes:
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
            axis.set_axisbelow(True)
        figure.tight_layout()
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=160)
        plt.close(figure)
        return output.getvalue()
    except Exception:
        return bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cff0ff3f0005fe02fe8e4cacf50000000049454e44ae426082")
