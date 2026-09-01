"""Markdown and figures for teacher-forced paired trajectory reports."""

from __future__ import annotations

import io
from typing import Any, Mapping

import numpy as np


def _markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    free, teacher, gaps = metrics["free_running"], metrics["teacher_forced"], metrics["teacher_minus_free"]
    finding = report["findings"][0]
    future_rows = _future_position_markdown_rows(metrics["future_position_trajectory"])
    state_rows = _state_group_markdown_rows(metrics["future_position_trajectory"])
    semantic_rows = _semantic_markdown_rows(metrics["future_position_music_features"])
    return "\n".join([
        "# Teacher-forced 与 Free-running 轨迹对照", "",
        f"结论：**{finding['classification']}**。{finding['text']}", "",
        "| 指标 | Free-running | Teacher-forced | Teacher - free |",
        "| --- | ---: | ---: | ---: |",
        f"| 调内音比例 | {free['diatonic_fit_mean']:.3f} | {teacher['diatonic_fit_mean']:.3f} | {gaps['teacher_minus_free_diatonic_fit']:+.3f} |",
        f"| Primer 调性匹配率 | {free['key_match_ratio']:.3f} | {teacher['key_match_ratio']:.3f} | {gaps['teacher_minus_free_key_match_ratio']:+.3f} |",
        f"| 与 Primer 和声轮廓相似度 | {free['chroma_similarity_mean']:.3f} | {teacher['chroma_similarity_mean']:.3f} | {gaps['teacher_minus_free_chroma_similarity']:+.3f} |",
        f"| 平均绝对音区变化 | {free['register_delta_abs_mean']:.3f} | {teacher['register_delta_abs_mean']:.3f} | {gaps['teacher_minus_free_register_delta']:+.3f} |",
        "", "## Future-position 轨迹误差", "", *future_rows, "", *state_rows, "", *semantic_rows, "",
        "本报告只陈述配对观察结果；不将不同表示空间的差异转换为因果百分比，也不产生生成约束。", "",
    ])


def _future_position_markdown_rows(future: Mapping[str, Any]) -> list[str]:
    if future.get("status") == "UNAVAILABLE":
        return ["未提供对齐的未来位置观察，因此无法计算该项误差。"]
    arms = future.get("arms")
    if not isinstance(arms, Mapping):
        return ["未来位置观察格式无效，无法显示逐位置误差。"]
    free, teacher = arms.get("free_running"), arms.get("teacher_forced")
    if not isinstance(free, Mapping) or not isinstance(teacher, Mapping):
        return ["两臂未来位置观察不完整，无法显示逐位置误差。"]
    positions = free.get("positions")
    if positions != teacher.get("positions") or not isinstance(positions, list):
        return ["两臂 future position 未对齐，无法显示可比较的逐位置误差。"]
    free_mse, free_counts = free.get("mse_by_position"), free.get("valid_samples_by_position")
    teacher_mse, teacher_counts = teacher.get("mse_by_position"), teacher.get("valid_samples_by_position")
    if not all(isinstance(values, list) and len(values) == len(positions) for values in (free_mse, free_counts, teacher_mse, teacher_counts)):
        return ["未来位置误差数组不完整，无法显示逐位置结果。"]
    rows = [
        "MSE 在每个未来位置上分别计算；n 是具有真实源乐曲参考小节的样本数。", "",
        "| 未来位置 | Free MSE | TF MSE | Target variance | Free NMSE | TF NMSE | n |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, (position, free_error, free_count) in enumerate(zip(positions, free_mse, free_counts)):
        rows.append(f"| {int(position)} | {_number(free_error)} | {_number(teacher_mse[index])} | {_number(free['target_variance_by_position'][index])} | {_number(free['nmse_by_position'][index])} | {_number(teacher['nmse_by_position'][index])} | {int(free_count)} |")
    return rows


def _number(value: Any) -> str:
    return f"{float(value):.3f}" if value is not None else "--"


def _semantic_markdown_rows(metric: Mapping[str, Any]) -> list[str]:
    if metric.get("status") == "UNAVAILABLE":
        return ["## 音乐特征归因", "", "未提供对齐的语义 bar tensor，因此无法分解和声、节奏、力度与密度。"]
    rows = ["## 音乐特征归因", "", "下表分别显示两种 history 条件下各未来位置的特征误差；不同特征组不合成为总分。"]
    labels = {"absolute_chroma": "绝对和声轮廓", "relative_chroma": "相对和声轮廓", "relative_pitch": "相对音高", "rhythm": "节奏（休止/起音/延音）", "velocity": "力度", "density": "音符密度"}
    for arm, label in (("free_running", "Free-running"), ("teacher_forced", "Teacher-forced")):
        values = metric["arms"][arm]
        positions = len(next(iter(values.values())))
        rows.extend(["", f"### {label}", "", "| 特征组 | " + " | ".join(f"+{index}" for index in range(1, positions + 1)) + " |", "| --- | " + " | ".join("---:" for _ in range(positions)) + " |"])
        for name, values_by_position in values.items():
            rows.append("| " + labels[name] + " | " + " | ".join(_number(value) for value in values_by_position) + " |")
    return rows


def _state_group_markdown_rows(metric: Mapping[str, Any]) -> list[str]:
    if metric.get("status") == "UNAVAILABLE":
        return []
    rows = ["## Trajectory 状态分量", "", "latent 是模型内部连续表征；register delta 是相邻小节的音区变化（半音）。二者独立报告，不合成为音乐质量分。"]
    labels = {"latent_mu": "Latent trajectory", "register_delta": "音区变化（register delta）"}
    for arm, title in (("free_running", "Free-running"), ("teacher_forced", "Teacher-forced")):
        groups = metric["arms"][arm]["feature_groups_mse"]
        positions = len(next(iter(groups.values())))
        rows.extend(["", f"### {title}", "", "| 分量 | " + " | ".join(f"+{index}" for index in range(1, positions + 1)) + " |", "| --- | " + " | ".join("---:" for _ in range(positions)) + " |"])
        for name, values in groups.items():
            rows.append("| " + labels[name] + " | " + " | ".join(_number(value) for value in values) + " |")
    return rows


def _plot_png(free: Mapping[str, Any], teacher: Mapping[str, Any]) -> bytes:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(7, 4))
        labels = ("Diatonic fit", "Key match", "Chroma similarity")
        free_values = [free["diatonic_fit_mean"], free["key_match_ratio"], free["chroma_similarity_mean"]]
        teacher_values = [teacher["diatonic_fit_mean"], teacher["key_match_ratio"], teacher["chroma_similarity_mean"]]
        x = np.arange(len(labels))
        axis.bar(x - 0.18, free_values, 0.36, label="Free-running")
        axis.bar(x + 0.18, teacher_values, 0.36, label="Teacher-forced")
        axis.set_xticks(x, labels)
        axis.set_ylim(0.0, 1.0)
        axis.legend()
        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=160)
        plt.close(figure)
        return buffer.getvalue()
    except ImportError:
        return bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cff0ff3f0005fe02fe8e4cacf50000000049454e44ae426082")
