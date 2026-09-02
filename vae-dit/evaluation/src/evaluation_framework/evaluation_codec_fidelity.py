"""Artifact-only codec fidelity assessment."""
from __future__ import annotations
import io, json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from codec.semantic_harmony_assignment import assign
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
    obs=resolver.json(status["artifacts"]["observation"]); source=resolver.json(obs["source_raw"]); manifest=resolver.json(obs["encoding_manifest"]); archive=resolver.npz(obs["arrays"])
    try:
        voices=np.asarray(archive["voice_tensors"],dtype=float); contexts=np.asarray(archive["bar_contexts"],dtype=float); bases=np.asarray(archive["base_pitches"],dtype=float); valid=np.asarray(archive["base_pitch_valid"],dtype=bool)
    finally: archive.close()
    epsilon=float(manifest.get("configuration",{}).get("slot_time_epsilon_ql",1e-6))
    tolerance=int(manifest.get("configuration",{}).get("melody_continuity_tolerance",7))
    bars={(str(song["song_id"]),int(bar["bar_index"])):bar for song in source["songs"] for bar in song["bars"]}
    active=(voices[...,2]>.5)|(voices[...,3]>.5); melody=active[:,0]; bass=active[:,17]; harmony=active[:,1:17]
    pitches=np.rint(bases[:,None,None]+voices[...,0]*24.0)
    harmony_counts=harmony.sum(axis=(1,2)); empty=(~active.any(axis=(1,2))).sum()
    source_register=[]; tensor_register=[]; source_context=[]; tensor_context=[]; source_lane_chroma=np.zeros(12); tensor_lane_chroma=np.zeros(12); unpaired=0; melody_exact=[]; bass_exact=[]; harmony_source=[]; harmony_tensor=[]; harmony_state_source=[]; harmony_state_tensor=[]; cardinality_errors=[]; cardinality_exact=[]; non_empty_slot_f1=[]; empty_slot_exact=[]
    for row in obs.get("alignment",[]):
        index=int(row["tensor_row"]); bar=bars.get((str(row["song_id"]), int(row["source_bar_index"])))
        if bar is None: unpaired+=1; continue
        notes=bar.get("notes", [])
        if valid[index]:
            base=int(bases[index]); expected=np.zeros(12)
            for note in notes: expected[(int(note["pitch"])-base)%12]+=max(0.,float(note.get("duration_ql",0.)))*float(note.get("velocity",0))/127.
            source_context.append(_norm(expected)); tensor_context.append(_norm(contexts[index]))
        slots=int(voices.shape[2]); slot_length=float(bar.get("bar_length_ql", 4.0))/slots; previous_source_melody=None
        for slot in range(slots):
            start,end=slot*slot_length,(slot+1)*slot_length
            slot_notes=[note for note in notes if float(note.get("onset_ql",0.))<end-epsilon and float(note.get("onset_ql",0.))+float(note.get("duration_ql",0.))>start+epsilon]
            source_melody,source_bass,source_harmony=assign(slot_notes, previous_source_melody, tolerance)
            if source_melody is not None: previous_source_melody=source_melody
            tensor_melody=int(pitches[index,0,slot]) if active[index,0,slot] else None; tensor_bass=int(pitches[index,17,slot]) if active[index,17,slot] else None
            source_melody_state="onset" if source_melody and abs(float(source_melody.get("onset_ql",0.))-start)<=epsilon else "hold"
            source_bass_state="onset" if source_bass and abs(float(source_bass.get("onset_ql",0.))-start)<=epsilon else "hold"
            tensor_melody_state="onset" if tensor_melody is not None and voices[index,0,slot,2]>.5 else "hold" if tensor_melody is not None else "rest"
            tensor_bass_state="onset" if tensor_bass is not None and voices[index,17,slot,2]>.5 else "hold" if tensor_bass is not None else "rest"
            melody_exact.append((tensor_melody,tensor_melody_state)==((int(source_melody["pitch"]),source_melody_state) if source_melody else (None,"rest"))); bass_exact.append((tensor_bass,tensor_bass_state)==((int(source_bass["pitch"]),source_bass_state) if source_bass else (None,"rest")))
            source_state=[(int(note["pitch"]), "onset" if abs(float(note.get("onset_ql",0.))-start)<=epsilon else "hold") for note in source_harmony]
            tensor_state=[(int(pitches[index,lane,slot]), "onset" if voices[index,lane,slot,2]>.5 else "hold") for lane in range(1,17) if active[index,lane,slot]]
            harmony_source.extend(item[0] for item in source_state); harmony_tensor.extend(item[0] for item in tensor_state); harmony_state_source.extend(source_state); harmony_state_tensor.extend(tensor_state)
            cardinality_errors.append(abs(len(source_state)-len(tensor_state))); cardinality_exact.append(len(source_state)==len(tensor_state))
            source_empty, tensor_empty = not source_state, not tensor_state
            empty_slot_exact.append(source_empty and tensor_empty)
            if not source_empty or not tensor_empty:
                non_empty_slot_f1.append(_multiset_f1(source_state, tensor_state)[2])
            source_assigned=[note for note in (source_melody, *source_harmony, source_bass) if note is not None]
            for note in source_assigned:
                weight=slot_length*max(0.0,min(float(note.get("velocity",0)),127.0))/127.0
                source_lane_chroma[int(note["pitch"])%12]+=weight
                source_register.append((int(note["pitch"]),weight))
            for lane in range(18):
                if active[index,lane,slot]:
                    pitch=int(pitches[index,lane,slot]); weight=slot_length*max(0.0,float(voices[index,lane,slot,4]))
                    tensor_lane_chroma[pitch%12]+=weight
                    tensor_register.append((pitch,weight))
    precision,recall,f1=_multiset_f1(harmony_source,harmony_tensor); state_precision,state_recall,state_f1=_multiset_f1(harmony_state_source,harmony_state_tensor)
    cosine=[float(np.dot(a,b)/(max(np.linalg.norm(a)*np.linalg.norm(b),1e-8))) for a,b in zip(source_context,tensor_context)]
    source_median=_weighted_median(source_register); tensor_median=_weighted_median(tensor_register)
    return {"dataset":obs["dataset"],"bar_count":int(len(voices)),"schema_version":"bar_tensor_schema.v2","melody":{"active_slot_count":int(melody.sum()),"exact_pitch_state_rate":float(np.mean(melody_exact)) if melody_exact else "UNAVAILABLE"},"bass":{"active_slot_count":int(bass.sum()),"exact_pitch_state_rate":float(np.mean(bass_exact)) if bass_exact else "UNAVAILABLE"},"harmony":{"active_event_count":int(harmony.sum()),"cardinality_mean":float(harmony_counts.mean()) if len(harmony_counts) else 0.,"cardinality_max":int(harmony_counts.max()) if len(harmony_counts) else 0.,"cardinality_exact_rate":float(np.mean(cardinality_exact)) if cardinality_exact else "UNAVAILABLE","cardinality_mae":float(np.mean(cardinality_errors)) if cardinality_errors else "UNAVAILABLE","empty_slot_exact_rate":float(np.mean(empty_slot_exact)) if empty_slot_exact else "UNAVAILABLE","non_empty_slot_macro_f1":float(np.mean(non_empty_slot_f1)) if non_empty_slot_f1 else "UNAVAILABLE","pitch_multiset_precision":precision,"pitch_multiset_recall":recall,"pitch_multiset_f1":f1,"pitch_state_multiset_precision":state_precision,"pitch_state_multiset_recall":state_recall,"pitch_state_multiset_f1":state_f1},"context_chroma":{"cosine_mean":float(np.mean(cosine)) if cosine else "UNAVAILABLE"},"lane_chroma":{"cosine":_cosine(source_lane_chroma,tensor_lane_chroma)},"register":{"source_median":source_median,"tensor_median":tensor_median,"median_gap_semitones":tensor_median-source_median if isinstance(source_median,float) and isinstance(tensor_median,float) else "UNAVAILABLE","anchorless_row_count":int((~valid).sum())},"counts":{"empty_row_count":int(empty),"unpaired_row_count":int(unpaired)}}

def _multiset_f1(source, tensor):
    from collections import Counter
    left,right=Counter(source),Counter(tensor); matched=sum((left & right).values())
    precision=matched/max(1,sum(right.values())); recall=matched/max(1,sum(left.values()))
    return float(precision),float(recall),float(2*precision*recall/max(precision+recall,1e-8))
def _json(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def _norm(x):return x/max(float(np.sum(x)),1e-8)
def _median(x):
    values=np.asarray(x)
    return float(np.median(values)) if values.size else 0.0
def _weighted_median(values):
    if not values:return "UNAVAILABLE"
    ordered=sorted((float(pitch),float(weight)) for pitch,weight in values if weight>0)
    if not ordered:return "UNAVAILABLE"
    threshold=sum(weight for _,weight in ordered)/2; total=0.0
    for pitch,weight in ordered:
        total+=weight
        if total>=threshold:return pitch
    return ordered[-1][0]
def _cosine(left,right):
    if not left.sum() and not right.sum():return 1.0
    if not left.sum() or not right.sum():return 0.0
    return float(np.dot(left,right)/(np.linalg.norm(left)*np.linalg.norm(right)))
def _png(profiles):
    try:
        import matplotlib.pyplot as plt
        f,a=plt.subplots(figsize=(6,3)); a.bar(list(profiles),[p["semantic_physical_chroma"]["cosine_mean"] for p in profiles.values()]); a.set_ylim(0,1); a.set_ylabel("Chroma cosine"); f.tight_layout(); o=io.BytesIO();f.savefig(o,format="png",dpi=150);plt.close(f);return o.getvalue()
    except Exception:return bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cff0ff3f0005fe02fe8e4cacf50000000049454e44ae426082")
def _markdown(r):
    if r["status"]=="UNAVAILABLE":return "# Codec 保真度\n\n所需 codec 原始观察不可用。\n"
    if any(p.get("schema_version")=="bar_tensor_schema.v2" for p in r["metrics"]["splits"].values()):
        lines=["# Codec Fidelity V2", "", "| Split | Bars | Context chroma cosine | Harmony events | Anchorless rows |", "| --- | ---: | ---: | ---: | ---: |"]
        for s,p in r["metrics"]["splits"].items():
            cosine=p["context_chroma"]["cosine_mean"]
            rendered=f"{cosine:.3f}" if isinstance(cosine,(int,float)) else "UNAVAILABLE"
            lines.append(f"| {s} | {p['bar_count']} | {rendered} | {p['harmony']['active_event_count']} | {p['register']['anchorless_row_count']} |")
        return "\n".join(lines)+"\n"
    lines=["# Codec Fidelity","","| Split | Bars | Audible chroma cosine | Condition chroma cosine | Register median gap (semitones) |","| --- | ---: | ---: | ---: | ---: |"]
    for s,p in r["metrics"]["splits"].items():lines.append(f"| {s} | {p['bar_count']} | {p['semantic_physical_chroma']['cosine_mean']:.3f} | 不可用 | {p['register']['median_gap_semitones']:.2f} |")
    lines.extend(["", "v1 的 relative chroma embedding 只有 11 个投影坐标，契约未提供还原为 12 个音级的方法，因此不将它伪装成调性条件 Chroma 指标。可听声部 Chroma 仍直接来自可解码的相对音高与活动状态。"])
    return "\n".join(lines)+"\n"
