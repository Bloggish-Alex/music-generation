"""Artifact-only DVAE relative-pitch supervision and gradient diagnostics."""

from __future__ import annotations

import io
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext
from .core.artifacts import VerifiedArtifactResolver


AUDIT_POINT = "dvae_pitch_supervision_audit"
GRADIENT_POINT = "dvae_pitch_gradient_probe"
AUDIT_STATUS_SCHEMA = "dvae_pitch_supervision_audit_status.v1"
AUDIT_OBSERVATION_SCHEMA = "dvae_pitch_supervision_audit_raw.v1"
GRADIENT_STATUS_SCHEMA = "dvae_pitch_gradient_probe_status.v1"
GRADIENT_OBSERVATION_SCHEMA = "dvae_pitch_gradient_probe_raw.v1"
_SPLITS = ("train", "validation")


class DVAEPitchSupervisionAuditExporter(ArtifactExporter):
    test_point = AUDIT_POINT
    input_contract = AUDIT_STATUS_SCHEMA
    output_contract = "dvae_pitch_supervision_audit_inputs.v1"

    def export(self, context: ExportContext) -> ArtifactBundle:
        payload = _index_status(context.input_root, "dvae_pitch_supervision_audit__raw_status.v1.json", AUDIT_STATUS_SCHEMA)
        path = context.store.write_json(AUDIT_POINT, "inputs", {"schema_version": self.output_contract, "status": payload})
        return ArtifactBundle(AUDIT_POINT, {"inputs": path.name})


class DVAEPitchSupervisionAuditEvaluator(ArtifactEvaluator):
    test_point = AUDIT_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _read(context.store.run_dir / bundle.artifacts["inputs"])
        reference = inputs.get("status")
        if not reference:
            return _unavailable("DVAE 相对音高监督审计", "生成这份审计所需的相对音高监督原始观察数据不可用。", [])
        status = VerifiedArtifactResolver(context.input_root).json(reference)
        _validate_status(status, AUDIT_STATUS_SCHEMA, None)
        if status["status"] != "AVAILABLE":
            return _unavailable("DVAE 相对音高监督审计", "相对音高监督审计未运行或缺少运行时事实。", status.get("unavailable_reasons") or [])
        raw = VerifiedArtifactResolver(context.input_root).json(status["artifacts"]["observation"])
        if raw.get("schema_version") != AUDIT_OBSERVATION_SCHEMA:
            raise ValueError("Unsupported DVAE pitch supervision audit observation schema.")
        report = {"schema_version": "assessment_report.v1", "status": "MONITOR", "metrics": _audit_metrics(raw),
                  "findings": [{"classification": "wiring_observation", "text": "本报告核对相对音高是否实际进入训练损失和梯度图；它不测量重建质量，也不构成质量总分。"}],
                  "provenance": {"run": raw["run"], "raw_observation": status["artifacts"]["observation"]}, "missing_inputs": []}
        return EvaluationResult(report=report, markdown=_audit_markdown(report), figures={"summary": _audit_png(report)})


class DVAEPitchGradientProbeExporter(ArtifactExporter):
    test_point = GRADIENT_POINT
    input_contract = GRADIENT_STATUS_SCHEMA
    output_contract = "dvae_pitch_gradient_probe_inputs.v1"

    def export(self, context: ExportContext) -> ArtifactBundle:
        statuses = {split: _index_status(context.input_root, f"dvae_pitch_gradient_probe__raw_status__{split}.v1.json", GRADIENT_STATUS_SCHEMA) for split in _SPLITS}
        path = context.store.write_json(GRADIENT_POINT, "inputs", {"schema_version": self.output_contract, "splits": statuses})
        return ArtifactBundle(GRADIENT_POINT, {"inputs": path.name})


class DVAEPitchGradientProbeEvaluator(ArtifactEvaluator):
    test_point = GRADIENT_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _read(context.store.run_dir / bundle.artifacts["inputs"])
        resolver, profiles, missing = VerifiedArtifactResolver(context.input_root), {}, []
        for split, reference in dict(inputs.get("splits") or {}).items():
            if not reference:
                missing.append({"split": split, "field": "status", "reason": "raw status was not provided"})
                continue
            status = resolver.json(reference)
            _validate_status(status, GRADIENT_STATUS_SCHEMA, split)
            if status["status"] != "AVAILABLE":
                missing.extend({"split": split, **reason} for reason in status.get("unavailable_reasons") or [])
                continue
            raw = resolver.json(status["artifacts"]["observation"])
            if raw.get("schema_version") != GRADIENT_OBSERVATION_SCHEMA:
                raise ValueError("Unsupported DVAE pitch gradient probe observation schema.")
            profiles[split] = _gradient_metrics(raw)
        if not profiles:
            return _unavailable("DVAE 相对音高梯度探针", "生成这份梯度探针报告所需的原始观察数据不可用。", missing)
        report = {"schema_version": "assessment_report.v1", "status": "MONITOR", "metrics": {"splits": profiles},
                  "findings": [{"classification": "gradient_observation", "text": "梯度可用、精确为零和不可用是不同事实；本报告不以阈值判定训练是否正常。"}],
                  "provenance": {"input_splits": list(profiles)}, "missing_inputs": missing}
        return EvaluationResult(report=report, markdown=_gradient_markdown(report), figures={"summary": _gradient_png(report)})


DVAE_PITCH_SUPERVISION_AUDIT_MODULE = EvaluationModule(AUDIT_POINT, DVAEPitchSupervisionAuditExporter(), DVAEPitchSupervisionAuditEvaluator(), summary="Runtime wiring audit for DVAE relative-pitch supervision.")
DVAE_PITCH_GRADIENT_PROBE_MODULE = EvaluationModule(GRADIENT_POINT, DVAEPitchGradientProbeExporter(), DVAEPitchGradientProbeEvaluator(), summary="Opt-in artifact-only summary of DVAE relative-pitch gradients.")


def _index_status(root: Path, name: str, schema: str) -> dict[str, str] | None:
    path = root / name
    if not path.is_file():
        return None
    payload = _read(path)
    _validate_status(payload, schema, payload.get("split"))
    return {"path": path.name, "sha256": VerifiedArtifactResolver.sha256(path), "schema_version": schema}


def _validate_status(payload: Mapping[str, Any], schema: str, split: str | None) -> None:
    if payload.get("schema_version") != schema or payload.get("status") not in {"AVAILABLE", "UNAVAILABLE"}:
        raise ValueError("Unsupported or invalid DVAE pitch diagnostic status.")
    if not isinstance(payload.get("run"), Mapping):
        raise ValueError("DVAE pitch diagnostic status is missing run provenance.")
    if split is not None and payload.get("split") != split:
        raise ValueError("DVAE pitch gradient status split does not match its artifact reference.")


def _audit_metrics(raw: Mapping[str, Any]) -> dict[str, Any]:
    supervision = dict(raw["supervision"])
    return {"tensor_contract": raw["tensor_contract"], "supervision": supervision, "decoder_parameter_groups": raw["decoder_parameter_groups"], "availability": raw["availability"]}


def _gradient_metrics(raw: Mapping[str, Any]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    valid_batches = 0
    for batch in raw.get("batches") or []:
        valid = not batch.get("unavailable_reasons")
        valid_batches += int(valid)
        for group in batch.get("decoder_parameter_groups") or []:
            target = groups.setdefault(str(group["group_id"]), {"observed_batch_count": 0, "gradient_available_batch_count": 0, "gradient_unavailable_batch_count": 0, "exact_zero_gradient_batch_count": 0, "l2_norms": [], "max_abs_values": [], "nonzero_ratios": []})
            target["observed_batch_count"] += 1
            if not group.get("gradient_available"):
                target["gradient_unavailable_batch_count"] += 1
                continue
            target["gradient_available_batch_count"] += 1
            l2, maximum = float(group["gradient_l2_norm"]), float(group["gradient_max_abs"])
            target["l2_norms"].append(l2); target["max_abs_values"].append(maximum)
            elements = int(group.get("parameter_element_count") or 0)
            target["nonzero_ratios"].append(float(group["gradient_nonzero_element_count"]) / elements if elements else 0.0)
            target["exact_zero_gradient_batch_count"] += int(l2 == 0.0 and maximum == 0.0)
    summary = {}
    for name, group in groups.items():
        summary[name] = {key: value for key, value in group.items() if key not in {"l2_norms", "max_abs_values", "nonzero_ratios"}}
        summary[name].update({"l2_norm_median": _median(group["l2_norms"]), "l2_norm_min": _minimum(group["l2_norms"]), "l2_norm_max": _maximum(group["l2_norms"]), "max_abs_median": _median(group["max_abs_values"]), "nonzero_element_ratio_median": _median(group["nonzero_ratios"])})
    return {"probe": raw["probe"], "captured_valid_batch_count": valid_batches, "unavailable_batch_count": len(raw.get("batches") or []) - valid_batches, "groups": summary, "unavailable_reasons": raw.get("unavailable_reasons") or []}


def _audit_markdown(report: Mapping[str, Any]) -> str:
    if report["status"] == "UNAVAILABLE":
        return _unavailable_markdown("DVAE 相对音高监督审计", report)
    metric = report["metrics"]
    supervision, contract = metric["supervision"], metric["tensor_contract"]
    lines = ["# DVAE 相对音高监督审计", "", "这份报告回答一个实现边界问题：训练时，**相对音高**是否真的作为目标进入损失与反向传播图。它不评价音乐好坏，也不把这些事实合成为分数。", "", "| 项目 | 运行时事实 |", "| --- | --- |",
             f"| tensor 中的相对音高列 | `{contract['relative_pitch_feature_index']}` (`{contract['relative_pitch_feature_name']}`) |", f"| 归一化音高范围 | {contract['pitch_scale_semitones']:.2f} 半音 |", f"| loss 是否启用 / 权重 | {supervision['loss_enabled']} / {supervision['loss_weight']:.6g} |", f"| 归约方式 | `{supervision['reduction']}` |", f"| target 与 decoder 输出形状一致 | {supervision['target_decoder_shape_match']} |", f"| active mask 实际应用 | {supervision['active_mask_applied_to_pitch_loss']} ({supervision['active_mask_definition']}) |", f"| decoder 输出激活 | `{supervision['normalization']['decoder_output_activation']}` |", f"| target 已 detached | {supervision['gradient_path_declared']['target_detached']} |", f"| pitch loss 需要梯度 | {supervision['gradient_path_declared']['pitch_loss_requires_grad']} |", "", "## 解码器逻辑参数组", "", "| 组 | 参数张量 | 参数元素 | 当前可用 |", "| --- | ---: | ---: | --- |"]
    for group in metric["decoder_parameter_groups"]:
        lines.append(f"| {group['group_id']} | {group['parameter_tensor_count']} | {group['parameter_element_count']} | {group['available']} |")
    lines.extend(["", "下一份“梯度探针”报告才会显示这些参数组是否从实际 relative-pitch loss 收到有限梯度；两份报告需结合阅读。", ""])
    return "\n".join(lines)


def _gradient_markdown(report: Mapping[str, Any]) -> str:
    if report["status"] == "UNAVAILABLE":
        return _unavailable_markdown("DVAE 相对音高梯度探针", report)
    lines = ["# DVAE 相对音高梯度探针", "", "探针使用训练中实际构造的 relative-pitch loss，并在不写入参数 `.grad`、不执行 optimizer step 的情况下读取梯度标量。有限梯度、精确零梯度和不可用梯度是不同事实；本报告没有质量阈值。", ""]
    for split, profile in report["metrics"]["splits"].items():
        probe = profile["probe"]
        lines.extend([f"## {split}", "", f"请求 {probe['requested_batch_count']} 个可用 batch，实际保留 {profile['captured_valid_batch_count']} 个；另有 {profile['unavailable_batch_count']} 个 batch 因缺少可比较音高等原因无法测量。", "", "| 解码器组 | 梯度可用 batch | 梯度不可用 batch | 精确零梯度 batch | L2 中位数 | L2 范围 | 最大绝对值中位数 | 非零元素比例中位数 |", "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"])
        for name, group in profile["groups"].items():
            interval = f"{_number(group['l2_norm_min'], 3)} - {_number(group['l2_norm_max'], 3)}"
            lines.append(f"| {name} | {group['gradient_available_batch_count']} | {group['gradient_unavailable_batch_count']} | {group['exact_zero_gradient_batch_count']} | {_number(group['l2_norm_median'], 3)} | {interval} | {_number(group['max_abs_median'], 3)} | {_number(group['nonzero_element_ratio_median'], 3)} |")
    if report["missing_inputs"]:
        lines.extend(["", "## 不可用输入", ""] + [f"- {item.get('split', 'unknown')}: {item.get('field', 'input')} - {item.get('reason', 'unavailable')}" for item in report["missing_inputs"]])
    lines.append("")
    return "\n".join(lines)


def _unavailable(title: str, message: str, missing: Sequence[Mapping[str, Any]]) -> EvaluationResult:
    report = {"schema_version": "assessment_report.v1", "status": "UNAVAILABLE", "metrics": {}, "findings": [], "provenance": {}, "missing_inputs": list(missing)}
    markdown = _unavailable_markdown(title, report).replace("生成这份报告所需的原始音乐/训练观察数据不可用。", message)
    return EvaluationResult(report=report, markdown=markdown)


def _unavailable_markdown(title: str, report: Mapping[str, Any]) -> str:
    reasons = report.get("missing_inputs") or []
    lines = [f"# {title}", "", "生成这份报告所需的原始音乐/训练观察数据不可用。"]
    lines.extend([f"- {item.get('field', 'input')}: {item.get('reason', 'unavailable')}" for item in reasons])
    return "\n".join(lines) + "\n"


def _audit_png(report: Mapping[str, Any]) -> bytes:
    values = report["metrics"]["supervision"]
    return _boolean_png({"loss enabled": values["loss_enabled"], "shape match": values["target_decoder_shape_match"], "loss needs grad": values["gradient_path_declared"]["pitch_loss_requires_grad"]}, "Relative-pitch supervision wiring")


def _gradient_png(report: Mapping[str, Any]) -> bytes:
    values = {f"{split}:{name}": group["gradient_available_batch_count"] for split, profile in report["metrics"]["splits"].items() for name, group in profile["groups"].items()}
    return _boolean_png(values, "Available gradient observations")


def _boolean_png(values: Mapping[str, Any], title: str) -> bytes:
    try:
        import matplotlib.pyplot as plt
        labels, heights = list(values), [float(value) for value in values.values()]
        figure, axis = plt.subplots(figsize=(max(5, len(labels) * 1.4), 3.2))
        axis.bar(labels, heights, color="#3B7EA1")
        axis.set_title(title, loc="left"); axis.set_ylabel("count / true")
        axis.tick_params(axis="x", rotation=25); axis.grid(axis="y", color="#D9D9D9"); axis.set_axisbelow(True)
        figure.tight_layout(); output = io.BytesIO(); figure.savefig(output, format="png", dpi=160); plt.close(figure)
        return output.getvalue()
    except Exception:
        return bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cff0ff3f0005fe02fe8e4cacf50000000049454e44ae426082")


def _read(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))
def _median(values: Sequence[float]) -> float | None: return float(median(values)) if values else None
def _minimum(values: Sequence[float]) -> float | None: return float(min(values)) if values else None
def _maximum(values: Sequence[float]) -> float | None: return float(max(values)) if values else None
def _number(value: float | None, digits: int) -> str: return "--" if value is None else f"{value:.{digits}f}"
