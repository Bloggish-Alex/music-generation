"""Artifact-only Stage 1 physical trajectory objective assessment."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext
from .core.artifacts import VerifiedArtifactResolver
from .core.policy import (
    high_band_gate,
    low_band_gate,
    maximum_gate,
    minimum_gate,
    monitor,
    overall_status,
    probe_gate,
    r2_gate,
    warn_above,
    warn_below,
)


TEST_POINT = "physical_trajectory_objective"
STATUS_SCHEMA = "physical_trajectory_objective_raw_status.v2"
OBSERVATION_SCHEMA = "physical_trajectory_objective_raw_observation.v2"
INPUT_SCHEMA = "physical_trajectory_objective_inputs.v1"
_ARRAYS = (
    "validation_target_normalized", "validation_valid_mask", "validation_clean_reconstruction",
    "validation_denoised_reconstruction", "validation_summary_embeddings", "validation_token_embeddings",
    "probe_clean_summary_embeddings", "probe_coherent_control_summary_embeddings",
    "probe_octave_displacement_summary_embeddings", "probe_track_swap_summary_embeddings",
    "probe_boundary_shuffle_summary_embeddings", "equivariance_original_values",
    "equivariance_translated_values", "equivariance_valid_mask",
)


class PhysicalTrajectoryObjectiveExporter(ArtifactExporter):
    """Index the status-led public raw bundle without interpreting its values."""

    test_point = TEST_POINT
    input_contract = STATUS_SCHEMA
    output_contract = INPUT_SCHEMA

    def export(self, context: ExportContext) -> ArtifactBundle:
        path = context.input_root / "physical_trajectory_objective__raw_status.v2.json"
        if not path.is_file():
            raise FileNotFoundError("Missing physical trajectory objective v2 raw status artifact.")
        status = _read_json(path)
        _validate_status(status)
        inputs = {"schema_version": INPUT_SCHEMA, "status": {"path": path.name, "sha256": _sha256(path), "schema_version": STATUS_SCHEMA}}
        output = context.store.write_json(TEST_POINT, "inputs", inputs)
        return ArtifactBundle(TEST_POINT, {"inputs": output.name})


class PhysicalTrajectoryObjectiveEvaluator(ArtifactEvaluator):
    """Evaluate Stage 1 facts without loading a checkpoint or encoder implementation."""

    test_point = TEST_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _read_json(context.store.run_dir / bundle.artifacts["inputs"])
        if inputs.get("schema_version") != INPUT_SCHEMA:
            raise ValueError("Unsupported physical trajectory objective input schema.")
        resolver = VerifiedArtifactResolver(context.input_root)
        status = resolver.json(inputs["status"])
        _validate_status(status)
        if status["status"] == "UNAVAILABLE":
            report = _unavailable_report(status)
            return EvaluationResult(report=report, markdown=_markdown(report))
        observation = resolver.json(status["artifacts"]["observation"])
        _validate_observation(observation, status)
        arrays = _load_arrays(resolver, observation["arrays"])
        _validate_arrays(observation, arrays)
        policy = _load_policy()
        report = _report(observation, arrays, policy, inputs)
        marker = {
            "schema_version": "physical_trajectory_objective_freezing_marker.v1",
            "test_point": TEST_POINT,
            "validated_for_freezing": report["validated_for_freezing"],
            "assessment_status": report["status"],
            "checkpoint_sha256": observation["run"]["checkpoint_sha256"],
            "raw_observation": status["artifacts"]["observation"],
        }
        return EvaluationResult(report=report, markdown=_markdown(report), figures={"summary": _summary_png(report)}, supplementary_json={"freezing_marker": marker})


PHYSICAL_TRAJECTORY_OBJECTIVE_MODULE = EvaluationModule(
    TEST_POINT,
    PhysicalTrajectoryObjectiveExporter(),
    PhysicalTrajectoryObjectiveEvaluator(),
    summary="Artifact-only Stage 1 physical trajectory objective assessment.",
)


def _report(observation: Mapping[str, Any], arrays: Mapping[str, np.ndarray], policy: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    target = arrays["validation_target_normalized"]
    mask = arrays["validation_valid_mask"]
    clean = arrays["validation_clean_reconstruction"]
    denoised = arrays["validation_denoised_reconstruction"]
    schema = observation["trajectory_schema"]
    feature_names = list(schema["feature_names"])
    gates: list[dict[str, Any]] = []
    coverage = observation["coverage"]
    selected = coverage["selected_windows_after_limit"]
    valid_ratio = mask.mean(axis=(0, 1))
    gates.append(maximum_gate("data.degenerate_dimensions", "Degenerate physical dimensions", int(sum(schema["normalizer"]["degenerate"])), policy["data_contract"]["max_degenerate_dimensions"]))
    gates.append(minimum_gate("data.minimum_feature_valid_ratio", "Minimum feature validity ratio", float(valid_ratio.min()), policy["data_contract"]["min_feature_valid_ratio"]))
    missing_ratio = coverage["encoded_bars"]["missing_base_pitch_count"] / coverage["encoded_bars"]["count"]
    gates.append(maximum_gate("data.missing_base_pitch_ratio", "Encoded bars missing base pitch", missing_ratio, policy["data_contract"]["max_missing_base_pitch_ratio"]))
    val_songs = len({row["base_song_id"] for row in observation["validation"]["rows"]})
    gates.append(warn_below("data.validation_base_song_count", "Independent validation pieces", val_songs, policy["data_contract"]["validation_base_song_warn_below"]))
    form_counts = {str(row["form"] or "UNSPECIFIED"): row["window_count"] for row in selected["form_window_counts"]}
    dominant_form = max(form_counts, key=form_counts.get)
    form_ratio = form_counts[dominant_form] / selected["count"]
    gates.append(warn_above("data.form_dominance_ratio", "Dominant form window ratio", form_ratio, policy["data_contract"]["form_dominance_warn_above"]))

    baseline = _baseline(target, mask)
    reconstruction = {"clean": _reconstruction(clean, target, mask, baseline, schema["feature_groups"], feature_names), "denoising": _reconstruction(denoised, target, mask, baseline, schema["feature_groups"], feature_names)}
    for mode, metrics in reconstruction.items():
        gates.append(r2_gate(f"reconstruction.{mode}.overall_r2", f"{mode.title()} overall reconstruction R2", metrics["overall"]["r2"], policy["reconstruction"]))
        for group, values in metrics["groups"].items():
            gates.append(r2_gate(f"reconstruction.{mode}.{group}.r2", f"{mode.title()} {group} reconstruction R2", values["r2"], policy["reconstruction"]))

    probes = _probes(arrays, policy["probe_separation"])
    for name, values in probes["comparisons"].items():
        gates.append(probe_gate(f"probe.{name}", f"{name.replace('_', ' ')} separation", values, policy["probe_separation"]))

    embedding = {"summary": _embedding(arrays["validation_summary_embeddings"], policy["embedding_health"]), "tokens": _embedding(arrays["validation_token_embeddings"].reshape(-1, arrays["validation_token_embeddings"].shape[-1]), policy["embedding_health"])}
    for name, values in embedding.items():
        gates.append(low_band_gate(f"embedding.{name}.effective_rank_ratio", f"{name} effective-rank ratio", values["effective_rank_ratio"], policy["embedding_health"]["effective_rank_ratio_fail_below"], policy["embedding_health"]["effective_rank_ratio_warn_below"]))
        gates.append(high_band_gate(f"embedding.{name}.top_pc_ratio", f"{name} top principal-component ratio", values["top_pc_ratio"], policy["embedding_health"]["top_pc_ratio_warn_above"], policy["embedding_health"]["top_pc_ratio_fail_above"]))
    embedding["relationship"] = _embedding_relationship(arrays["validation_summary_embeddings"], target, mask, int(policy["embedding_health"]["pair_sample_count"]))
    gates.extend([monitor("embedding.physical_distance_spearman", "Embedding/physical distance rank relationship", embedding["relationship"]["physical_distance_spearman"]), monitor("embedding.knn_physical_improvement", "Embedding neighbour physical-distance improvement", embedding["relationship"]["knn_physical_improvement"])])

    equivariance = _equivariance(arrays, observation["equivariance"])
    gates.append(maximum_gate("data.octave_equivariance_error", "Projector octave-equivariance maximum error", equivariance["overall_max_absolute_error"], policy["data_contract"]["equivariance_absolute_tolerance"]))
    status = overall_status(gates)
    return {
        "schema_version": "assessment_report.v1", "status": status, "validated_for_freezing": status in {"PASS", "WARN"},
        "metrics": {"data_contract": {"encoded_bar_count": coverage["encoded_bars"]["count"], "missing_base_pitch_count": coverage["encoded_bars"]["missing_base_pitch_count"], "missing_base_pitch_ratio": missing_ratio, "feature_valid_ratio": dict(zip(feature_names, map(float, valid_ratio))), "degenerate_dimensions": [name for name, flag in zip(feature_names, schema["normalizer"]["degenerate"]) if flag], "candidate_window_count": coverage["candidate_windows_before_limit"]["count"], "selected_window_count": selected["count"], "validation_base_song_count": val_songs, "dominant_form": dominant_form, "dominant_form_ratio": form_ratio}, "reconstruction": reconstruction, "probe_separation": probes, "embedding_health": embedding, "equivariance": equivariance},
        "metric_assessments": gates,
        "findings": _findings(gates),
        "provenance": {"raw_status": inputs["status"], "run": observation["run"], "policy": {"schema_version": policy["schema_version"], "sha256": _sha256(_policy_path())}},
        "missing_inputs": [],
    }


def _baseline(target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = mask.astype(np.float64)
    mean = (target * weights).sum(axis=(0, 1)) / np.maximum(weights.sum(axis=(0, 1)), 1.0)
    return np.broadcast_to(mean, target.shape)


def _reconstruction(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, baseline: np.ndarray, groups: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    result = {"overall": _error(prediction, target, mask, baseline), "groups": {}, "features": []}
    for name, indices in groups.items(): result["groups"][name] = _error(prediction[..., indices], target[..., indices], mask[..., indices], baseline[..., indices])
    for index, name in enumerate(names): result["features"].append({"name": name, **_error(prediction[..., index:index+1], target[..., index:index+1], mask[..., index:index+1], baseline[..., index:index+1])})
    return result


def _error(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    weights = mask.astype(np.float64); count = max(float(weights.sum()), 1.0)
    mse = float((np.square(prediction-target)*weights).sum()/count); baseline_mse = float((np.square(baseline-target)*weights).sum()/count)
    return {"mse": mse, "mae": float((np.abs(prediction-target)*weights).sum()/count), "baseline_mse": baseline_mse, "r2": None if baseline_mse <= 1e-12 else float(1.0-mse/baseline_mse), "valid_value_count": int(weights.sum())}


def _probes(arrays: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> dict[str, Any]:
    clean = arrays["probe_clean_summary_embeddings"]
    distances = {name: np.sqrt(np.mean(np.square(arrays[f"probe_{name}_summary_embeddings"] - clean), axis=-1) + 1e-12) for name in ("coherent_control", "octave_displacement", "track_swap", "boundary_shuffle")}
    control = distances["coherent_control"]; comparisons: dict[str, Any] = {}
    for name in ("octave_displacement", "track_swap", "boundary_shuffle"):
        other = distances[name]; labels = np.r_[np.zeros(len(control)), np.ones(len(other))]; scores = np.r_[control, other]
        comparisons[name] = {"quantile_gap": float(np.quantile(other,.1)-np.quantile(control,.9)), "triplet_accuracy": float(np.mean(other>control)), "auroc": _auroc(labels, scores), "effect_size": _effect(other, control)}
    return {"sample_count": int(len(control)), "coherent_control": _distribution(control), "comparisons": comparisons}


def _embedding(values: np.ndarray, policy: Mapping[str, Any]) -> dict[str, Any]:
    centered = values.astype(np.float64)-values.mean(axis=0, keepdims=True); covariance = np.atleast_2d(np.cov(centered, rowvar=False)); eigen = np.maximum(np.linalg.eigvalsh(covariance), 0.0); total=float(eigen.sum())
    probabilities = eigen/total if total>1e-15 else np.zeros_like(eigen); positive=probabilities>0; rank=float(np.exp(-np.sum(probabilities[positive]*np.log(probabilities[positive])))) if np.any(positive) else 0.0
    distances = _pair_distances(values, int(policy["pair_sample_count"]))
    return {"sample_count":int(len(values)), "dimension":int(values.shape[1]), "effective_rank":rank, "effective_rank_ratio":rank/max(1,min(values.shape[1],len(values)-1)), "top_pc_ratio":float(probabilities.max()) if len(probabilities) else 1.0, "active_dimension_ratio":float(np.mean(centered.std(axis=0)>policy["active_dimension_epsilon"])), "pairwise_distance":_distribution(distances), "near_zero_distance_ratio":float(np.mean(distances<=policy["near_zero_distance_epsilon"]))}


def _embedding_relationship(embedding: np.ndarray, target: np.ndarray, mask: np.ndarray, pairs: int) -> dict[str, Any]:
    generator=np.random.default_rng(42); n=len(embedding); count=max(1,min(pairs, max(1,n*n))); left=generator.integers(0,n,count); right=generator.integers(0,n,count)
    e=np.sqrt(np.mean(np.square(embedding[left]-embedding[right]),axis=-1)+1e-12); p=_physical_distance(target,mask,left,right)
    nearest=np.argmin(np.where(np.eye(n,dtype=bool),np.inf,np.sqrt(((embedding[:,None]-embedding[None,:])**2).mean(axis=-1))),axis=1); random=generator.integers(0,n,n); random=np.where(random==np.arange(n),(random+1)%n,random)
    near_p=_physical_distance(target,mask,np.arange(n),nearest); random_p=_physical_distance(target,mask,np.arange(n),random)
    return {"pair_sample_count":count,"physical_distance_spearman":_spearman(e,p),"knn_mean_physical_distance":float(near_p.mean()),"random_mean_physical_distance":float(random_p.mean()),"knn_physical_improvement":float(1-near_p.mean()/max(random_p.mean(),1e-12))}


def _equivariance(arrays: Mapping[str, np.ndarray], definition: Mapping[str, Any]) -> dict[str, Any]:
    original, translated, valid=arrays["equivariance_original_values"],arrays["equivariance_translated_values"],arrays["equivariance_valid_mask"]
    def maximum(indices: Sequence[int], expected: float) -> float:
        selected=valid[...,indices]
        return float(np.max(np.abs((translated[...,indices]-original[...,indices])[selected]-expected))) if np.any(selected) else float("inf")
    chroma=maximum(definition["chroma_feature_indices"],0.0); density=maximum(definition["density_feature_indices"],0.0); pitch=maximum(definition["pitch_translation_feature_indices"],float(definition["octave_shift_semitones"]))
    return {"sample_count":int(original.shape[0]),"octave_shift_semitones":definition["octave_shift_semitones"],"chroma_max_absolute_error":chroma,"density_max_absolute_error":density,"pitch_translation_max_absolute_error":pitch,"overall_max_absolute_error":max(chroma,density,pitch)}


def _validate_status(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != STATUS_SCHEMA or payload.get("status") not in {"AVAILABLE","UNAVAILABLE"}: raise ValueError("Unsupported physical trajectory objective raw status.")


def _validate_observation(payload: Mapping[str, Any], status: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != OBSERVATION_SCHEMA: raise ValueError("Unsupported physical trajectory objective observation schema.")
    if payload.get("run") != status.get("run"): raise ValueError("Physical trajectory objective run provenance differs between status and observation.")
    if not all(payload["availability"].values()) or payload.get("unavailable_reasons"): raise ValueError("AVAILABLE physical trajectory observation has unavailable fields.")


def _validate_arrays(observation: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != set(_ARRAYS): raise ValueError("Physical trajectory objective NPZ has unsupported array names.")
    descriptors=observation["arrays"]["items"]
    for name in _ARRAYS:
        values=arrays[name]; expected="bool" if name.endswith("valid_mask") else "float32"
        if values.dtype.name!=expected or descriptors[name]["dtype"]!=expected or descriptors[name]["shape"]!=list(values.shape): raise ValueError(f"Invalid physical trajectory array descriptor: {name}.")
        if expected=="float32" and not np.isfinite(values).all(): raise ValueError(f"Physical trajectory array has non-finite values: {name}.")
    target=arrays["validation_target_normalized"]; mask=arrays["validation_valid_mask"]
    if target.ndim!=3 or mask.shape!=target.shape or any(arrays[name].shape!=target.shape for name in ("validation_clean_reconstruction","validation_denoised_reconstruction")): raise ValueError("Validation reconstruction arrays have incompatible shapes.")
    if arrays["validation_summary_embeddings"].shape[0]!=target.shape[0] or arrays["validation_token_embeddings"].shape[:2]!=target.shape[:2]: raise ValueError("Embedding arrays do not align with validation windows.")
    probe=arrays["probe_clean_summary_embeddings"]
    if any(arrays[f"probe_{name}_summary_embeddings"].shape!=probe.shape for name in ("coherent_control","octave_displacement","track_swap","boundary_shuffle")): raise ValueError("Probe embeddings have incompatible shapes.")


def _load_arrays(resolver: VerifiedArtifactResolver, reference: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with resolver.npz(reference) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _pair_distances(values: np.ndarray, count: int) -> np.ndarray:
    gen=np.random.default_rng(42); left=gen.integers(0,len(values),max(1,min(count,len(values)*len(values)))); right=gen.integers(0,len(values),len(left)); return np.sqrt(np.mean(np.square(values[left]-values[right]),axis=-1)+1e-12)
def _physical_distance(values: np.ndarray,mask: np.ndarray,left: np.ndarray,right: np.ndarray) -> np.ndarray:
    valid=mask[left]&mask[right]; return np.sqrt((np.square(values[left]-values[right])*valid).sum(axis=(1,2))/np.maximum(valid.sum(axis=(1,2)),1))
def _spearman(left: np.ndarray,right: np.ndarray) -> float:
    if len(left)<2 or np.std(left)==0 or np.std(right)==0:return 0.0
    return float(np.corrcoef(np.argsort(np.argsort(left)),np.argsort(np.argsort(right)))[0,1])
def _auroc(labels: np.ndarray,scores: np.ndarray) -> float:
    order=np.argsort(scores); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(scores)+1); positives=labels==1; return float((ranks[positives].sum()-positives.sum()*(positives.sum()+1)/2)/(positives.sum()*(~positives).sum()))
def _effect(left: np.ndarray,right: np.ndarray) -> float|None:
    pooled=np.sqrt((np.var(left,ddof=1)+np.var(right,ddof=1))/2) if len(left)>1 else 0.; return float((left.mean()-right.mean())/pooled) if pooled>1e-12 else None
def _distribution(values: np.ndarray) -> dict[str,float]: return {"mean":float(np.mean(values)),"p10":float(np.quantile(values,.1)),"p50":float(np.quantile(values,.5)),"p90":float(np.quantile(values,.9))}
def _findings(gates: Sequence[Mapping[str,Any]]) -> list[dict[str,str]]:
    return [{"classification":str(g["metric_id"]),"text":f"{g['label']}: {g['status']}."} for g in gates if g["status"] in {"WARN","FAIL"}]


def _unavailable_report(status: Mapping[str,Any]) -> dict[str,Any]: return {"schema_version":"assessment_report.v1","status":"UNAVAILABLE","validated_for_freezing":False,"metrics":{},"metric_assessments":[],"findings":[],"provenance":{"raw_status":status},"missing_inputs":status["unavailable_reasons"]}
def _markdown(report: Mapping[str,Any]) -> str:
    if report["status"]=="UNAVAILABLE": return "# Physical Trajectory Objective\n\nStage 1 原始观察 bundle 不可用。评估器不会加载模型或重建训练数据来填补这项证据缺口。\n"
    data=report["metrics"]["data_contract"]; rec=report["metrics"]["reconstruction"]; probes=report["metrics"]["probe_separation"]["comparisons"]; emb=report["metrics"]["embedding_health"]
    lines=["# Physical Trajectory Objective", "", "本报告检验 Stage 1 encoder 是否学习到稳定的音乐运动表征。数据覆盖、重建、对刻意制造的不自然变化的敏感度，以及 embedding 几何分别报告，不合成为单一分数。", "", f"**评估结论：** `{report['status']}`。冻结 marker：`{str(report['validated_for_freezing']).lower()}`。", "", "## 覆盖与输入", "", "| 编码小节 | 缺失 base pitch | 候选窗口 | 实际选取窗口 | 独立验证曲目 | 主导曲式占比 |", "| ---: | ---: | ---: | ---: | ---: | ---: |", f"| {data['encoded_bar_count']} | {data['missing_base_pitch_count']} ({data['missing_base_pitch_ratio']:.2%}) | {data['candidate_window_count']} | {data['selected_window_count']} | {data['validation_base_song_count']} | {data['dominant_form_ratio']:.1%} ({data['dominant_form']}) |", "", "## 重建", "", "R2 将重建结果与按特征计算的均值基线比较。接近 1 说明保留了更多已测量的轨迹信息；0 表示不优于该基线。", "", "| 模式 | 总体 R2 |", "| --- | ---: |"]
    for name, values in rec.items(): lines.append(f"| {name} | {_fmt(values['overall']['r2'],3)} |")
    lines.extend(["", "## 边界敏感度", "", "相对于小幅而连贯的对照变化，物理上不自然的变化应使 learned summary 移动得更远。", "", "| 变化 | 分离间隔 | 正确排序率 | AUROC |", "| --- | ---: | ---: | ---: |"])
    for name, value in probes.items(): lines.append(f"| {name.replace('_',' ')} | {_fmt(value['quantile_gap'],3)} | {_fmt(value['triplet_accuracy'],1,percent=True)} | {_fmt(value['auroc'],3)} |")
    lines.extend(["", "## 表征健康度", "", "| Embedding | 有效秩比例 | 首主成分占比 |", "| --- | ---: | ---: |"])
    for name in ("summary","tokens"): lines.append(f"| {name} | {_fmt(emb[name]['effective_rank_ratio'],3)} | {_fmt(emb[name]['top_pc_ratio'],1,percent=True)} |")
    lines.extend(["", "## 门禁", "", "| 检查项 | 状态 | 数值 |", "| --- | --- | --- |"])
    for gate in report["metric_assessments"]: lines.append(f"| {gate['label']} | {gate['status']} | `{gate['value']}` |")
    return "\n".join(lines)+"\n"
def _fmt(value: Any,digits: int,percent: bool=False) -> str: return "--" if value is None else (f"{value*100:.{digits}f}%" if percent else f"{value:.{digits}f}")


def _summary_png(report: Mapping[str,Any]) -> bytes:
    try:
        import matplotlib.pyplot as plt
        rec=report["metrics"]["reconstruction"]; probes=report["metrics"]["probe_separation"]["comparisons"]; emb=report["metrics"]["embedding_health"]
        fig, axes=plt.subplots(1,3,figsize=(11,3.4)); axes[0].bar(list(rec),[rec[x]["overall"]["r2"] or 0 for x in rec],color="#3B7EA1"); axes[0].set_title("Reconstruction R2",loc="left")
        axes[1].bar(list(probes),[probes[x]["triplet_accuracy"] for x in probes],color="#2E8B57"); axes[1].set_ylim(0,1); axes[1].set_title("Boundary sensitivity",loc="left")
        axes[2].bar(["summary","tokens"],[emb[x]["effective_rank_ratio"] for x in ("summary","tokens")],color="#B15D3B"); axes[2].set_ylim(0,1); axes[2].set_title("Effective-rank ratio",loc="left")
        for axis in axes: axis.grid(axis="y",color="#ddd"); axis.set_axisbelow(True); axis.tick_params(axis="x",rotation=25)
        fig.tight_layout(); output=io.BytesIO(); fig.savefig(output,format="png",dpi=160); plt.close(fig); return output.getvalue()
    except Exception: return bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cff0ff3f0005fe02fe8e4cacf50000000049454e44ae426082")


def _policy_path() -> Path: return Path(__file__).resolve().parents[3] / "contracts" / "evaluation" / "v1" / "physical_trajectory_objective__policy.v1.json"
def _load_policy() -> dict[str,Any]:
    policy=_read_json(_policy_path())
    if policy.get("schema_version")!="physical_trajectory_objective_policy.v1": raise ValueError("Unsupported physical trajectory objective policy.")
    return policy
def _read_json(path: Path) -> dict[str,Any]: return json.loads(path.read_text(encoding="utf-8"))
def _sha256(path: Path) -> str: return VerifiedArtifactResolver.sha256(path)
