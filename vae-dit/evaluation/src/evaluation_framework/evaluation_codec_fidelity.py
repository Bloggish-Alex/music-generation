"""Artifact-only codec fidelity assessment."""
from __future__ import annotations
import io, json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext
from .core.artifacts import VerifiedArtifactResolver
from .core.tensor_schema import SemanticTensorDecoder

TEST_POINT="codec_fidelity"; STATUS_SCHEMA="codec_fidelity_raw_status.v1"; INPUT_SCHEMA="codec_fidelity_inputs.v1"; V2_STATUS_SCHEMA="codec_fidelity_raw_status.v2"; V2_INPUT_SCHEMA="codec_fidelity_inputs.v2"
SPLITS=("train","validation","excluded_unpaired")

class CodecFidelityExporter(ArtifactExporter):
    test_point=TEST_POINT; input_contract=STATUS_SCHEMA; output_contract=INPUT_SCHEMA
    def export(self, context: ExportContext)->ArtifactBundle:
        refs={}; availability={}
        for split in SPLITS:
            p=context.input_root/f"codec_fidelity__raw_status__{split}.v2.json"
            if not p.is_file(): p=context.input_root/f"codec_fidelity__raw_status__{split}.v1.json"
            if not p.is_file(): availability[split]="not_provided"; continue
            status=_json(p); _status(status,split)
            refs[split]={"path":p.name,"sha256":VerifiedArtifactResolver.sha256(p)}; availability[split]=str(status["status"])
        version = V2_INPUT_SCHEMA if any(str(ref["path"]).endswith(".v2.json") for ref in refs.values()) else INPUT_SCHEMA
        path=context.store.write_json(TEST_POINT,"inputs",{"schema_version":version,"splits":refs,"availability":availability})
        return ArtifactBundle(TEST_POINT,{"inputs":path.name})

class CodecFidelityEvaluator(ArtifactEvaluator):
    test_point=TEST_POINT; required_artifacts:Sequence[str]=( "inputs", )
    def evaluate(self, context: EvaluationContext,bundle:ArtifactBundle)->EvaluationResult:
        inputs=_json(context.store.run_dir/bundle.artifacts["inputs"])
        if inputs.get("schema_version") not in {INPUT_SCHEMA, V2_INPUT_SCHEMA}: raise ValueError("Unsupported codec fidelity inputs.")
        resolver=VerifiedArtifactResolver(context.input_root); profiles={}; missing=[]
        for split,ref in inputs.get("splits",{}).items():
            status=resolver.json(ref); _status(status,split)
            if status["status"]!="AVAILABLE":
                missing.extend({"split":split,**reason} for reason in status["unavailable_reasons"]); continue
            profiles[split]=_measure_v2(resolver,status) if status.get("schema_version")==V2_STATUS_SCHEMA else _measure(resolver,status)
        if not profiles:
            report={"schema_version":"assessment_report.v1","status":"UNAVAILABLE","metrics":{},"findings":[],"provenance":{"input_availability":inputs.get("availability",{})},"missing_inputs":missing or [{"field":"status","reason":"No codec split status artifact was provided."}]}
            return EvaluationResult(report,_markdown(report))
        comparison=_gap(profiles.get("train"),profiles.get("validation"))
        report={"schema_version":"assessment_report.v1","status":"MONITOR","metrics":{"splits":profiles,"train_validation_gap":comparison},"findings":[{"classification":"codec_information_boundary","text":"可听声部与调性条件分别报告。它们描述编码边界保留的信息，不合成为模型质量总分。"}],"provenance":{"input_statuses":inputs["splits"]},"missing_inputs":missing}
        return EvaluationResult(report,_markdown(report),{"chroma_retention":_png(profiles)})

CODEC_FIDELITY_MODULE=EvaluationModule(TEST_POINT,CodecFidelityExporter(),CodecFidelityEvaluator(),summary="Source-to-bar-codec harmony, register and density fidelity.")

def _measure(resolver:VerifiedArtifactResolver,status:Mapping[str,Any])->dict[str,Any]:
    obs=resolver.json(status["artifacts"]["observation"])
    source=resolver.json(obs["source_raw"])
    tensors_archive=resolver.npz(obs["tensor"])
    try: tensors=np.asarray(tensors_archive["bar_tensors"],dtype=float)
    finally: tensors_archive.close()
    bars={(str(song["song_id"]),int(bar["bar_index"])):bar for song in source["songs"] for bar in song["bars"]}
    schema=obs["tensor_schema"]; names=list(schema["feature_names"]); decoder=SemanticTensorDecoder.from_schema(schema)
    target=[]; physical=[]; source_pitch=[]; tensor_pitch=[]; density_gap=[]
    for row in obs["alignment"]:
        i=int(row["tensor_row"]); bar=bars[(str(row["song_id"]),int(row["source_bar_index"]))]; base=float(row["base_pitch_semitones"]); notes=bar.get("notes",[])
        target.append(_source_chroma(notes,base)); t=tensors[i:i+1]; active=decoder.active_mask(t)[0]; pitches=decoder.absolute_pitch(t,np.asarray([base]))[0]
        physical.append(_tensor_chroma(pitches,active,base));
        source_pitch.extend([float(n["pitch"])-int(row["applied_transpose_semitones"]) for n in notes]); tensor_pitch.extend(pitches[active].tolist()); density_gap.append(float(active.sum())-len(notes))
    active=np.asarray([x.sum()>0 for x in target]); target=np.asarray(target); physical=np.asarray(physical)
    return {"dataset":obs["dataset"],"bar_count":len(target),"active_source_bar_count":int(active.sum()),"semantic_physical_chroma":_vectors(target,physical,active),"chroma_condition":{"status":"UNAVAILABLE","reason":"The v1 relative chroma embedding has 11 projection coordinates and no contract-defined inverse mapping to the 12 pitch classes."},"register":{"source_median":_median(source_pitch),"tensor_median":_median(tensor_pitch),"median_gap_semitones":_median(tensor_pitch)-_median(source_pitch)},"density":{"tensor_slot_minus_source_note_mean":float(np.mean(density_gap))}}

def _source_chroma(notes,base):
    base=int(round(float(base)))
    x=np.zeros(12);
    for n in notes: x[(int(n["pitch"])-base)%12]+=float(n["duration_ql"])
    return _norm(x)
def _tensor_chroma(pitches,active,base):
    x=np.zeros(12);
    for p in pitches[active]: x[int(round(float(p)-base))%12]+=1
    return _norm(x)
def _vectors(a,b,mask):
    a,b=a[mask],b[mask]
    if not len(a): return {"mse":"UNAVAILABLE","cosine_mean":"UNAVAILABLE","cosine_p10":"UNAVAILABLE"}
    cos=np.sum(a*b,1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-8)
    return {"mse":float(np.mean((a-b)**2)),"cosine_mean":float(np.mean(cos)),"cosine_p10":float(np.quantile(cos,.1))}
def _gap(a,b):
    if not a or not b:return "UNAVAILABLE"
    if a.get("schema_version")=="bar_tensor_schema.v2" or b.get("schema_version")=="bar_tensor_schema.v2":
        return {"context_chroma_cosine_gap":float(b["context_chroma"]["cosine_mean"]-a["context_chroma"]["cosine_mean"])}
    return {"semantic_physical_cosine_gap":float(b["semantic_physical_chroma"]["cosine_mean"]-a["semantic_physical_chroma"]["cosine_mean"]),"condition_cosine_gap":"UNAVAILABLE"}
def _status(x,split):
    if x.get("schema_version") not in {STATUS_SCHEMA, V2_STATUS_SCHEMA} or x.get("dataset",{}).get("split")!=split:raise ValueError("Codec status schema or split mismatch.")

def _measure_v2(resolver:VerifiedArtifactResolver,status:Mapping[str,Any])->dict[str,Any]:
    """Measure V2 lane facts without reconstructing a V1 semantic tensor."""
    obs=resolver.json(status["artifacts"]["observation"]); archive=resolver.npz(obs["arrays"])
    try:
        voices=np.asarray(archive["voice_tensors"],dtype=float); contexts=np.asarray(archive["bar_contexts"],dtype=float); bases=np.asarray(archive["base_pitches"],dtype=float); valid=np.asarray(archive["base_pitch_valid"],dtype=bool)
    finally: archive.close()
    active=(voices[...,2]>.5)|(voices[...,3]>.5); melody=active[:,0]; bass=active[:,17]; harmony=active[:,1:17]
    pitches=np.rint(bases[:,None,None]+voices[...,0]*24.0)
    harmony_counts=harmony.sum(axis=(1,2)); empty=(~active.any(axis=(1,2))).sum()
    chroma=np.asarray([_norm(row) for row in contexts])
    cosine=np.sum(chroma*chroma,axis=1)/np.maximum(np.sum(chroma*chroma,axis=1),1e-8)
    return {"dataset":obs["dataset"],"bar_count":int(len(voices)),"schema_version":"bar_tensor_schema.v2","melody":{"active_slot_count":int(melody.sum())},"bass":{"active_slot_count":int(bass.sum())},"harmony":{"active_event_count":int(harmony.sum()),"cardinality_mean":float(harmony_counts.mean()) if len(harmony_counts) else 0.,"cardinality_max":int(harmony_counts.max()) if len(harmony_counts) else 0.},"context_chroma":{"cosine_mean":float(cosine.mean()) if len(cosine) else "UNAVAILABLE"},"register":{"tensor_median":_median(pitches[active]),"anchorless_row_count":int((~valid).sum())},"counts":{"empty_row_count":int(empty),"unpaired_row_count":0}}
def _json(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def _norm(x):return x/max(float(np.sum(x)),1e-8)
def _median(x):
    values=np.asarray(x)
    return float(np.median(values)) if values.size else 0.0
def _png(profiles):
    try:
        import matplotlib.pyplot as plt
        f,a=plt.subplots(figsize=(6,3)); a.bar(list(profiles),[p["semantic_physical_chroma"]["cosine_mean"] for p in profiles.values()]); a.set_ylim(0,1); a.set_ylabel("Chroma cosine"); f.tight_layout(); o=io.BytesIO();f.savefig(o,format="png",dpi=150);plt.close(f);return o.getvalue()
    except Exception:return bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cff0ff3f0005fe02fe8e4cacf50000000049454e44ae426082")
def _markdown(r):
    if r["status"]=="UNAVAILABLE":return "# Codec 保真度\n\n所需 codec 原始观察不可用。\n"
    if any(p.get("schema_version")=="bar_tensor_schema.v2" for p in r["metrics"]["splits"].values()):
        lines=["# Codec Fidelity V2", "", "| Split | Bars | Context chroma cosine | Harmony events | Anchorless rows |", "| --- | ---: | ---: | ---: | ---: |"]
        for s,p in r["metrics"]["splits"].items(): lines.append(f"| {s} | {p['bar_count']} | {p['context_chroma']['cosine_mean']:.3f} | {p['harmony']['active_event_count']} | {p['register']['anchorless_row_count']} |")
        return "\n".join(lines)+"\n"
    lines=["# Codec Fidelity","","| Split | Bars | Audible chroma cosine | Condition chroma cosine | Register median gap (semitones) |","| --- | ---: | ---: | ---: | ---: |"]
    for s,p in r["metrics"]["splits"].items():lines.append(f"| {s} | {p['bar_count']} | {p['semantic_physical_chroma']['cosine_mean']:.3f} | 不可用 | {p['register']['median_gap_semitones']:.2f} |")
    lines.extend(["", "v1 的 relative chroma embedding 只有 11 个投影坐标，契约未提供还原为 12 个音级的方法，因此不将它伪装成调性条件 Chroma 指标。可听声部 Chroma 仍直接来自可解码的相对音高与活动状态。"])
    return "\n".join(lines)+"\n"
