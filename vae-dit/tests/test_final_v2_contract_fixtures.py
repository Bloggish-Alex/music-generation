"""Frozen static fixtures for final V2 parser and quantization contracts."""
from __future__ import annotations
import json
from pathlib import Path
from codec.slot_grid import SlotGrid
ROOT=Path(__file__).resolve().parents[1]
def load(name): return json.loads((ROOT/"tests"/"fixtures"/"final_v2_parser"/name).read_text(encoding="utf-8"))
def test_grid_capacity_and_partial_slot_policy():
    cases={case["name"]:case for case in load("measure_grid_cases.json")}
    assert cases["two_two"]["valid_slot_count"]==16
    assert cases["twelve_four"]["valid_slot_count"]==48
    assert cases["partial_51_32"]["final_slot_duration_ql"]==0.125
    assert cases["over_capacity"]["encoding_error"]=="slot_capacity_exceeded"
    for name, case in cases.items():
        if "encoding_error" in case:
            continue
        grid=SlotGrid.for_bar(case["bar_length_ql"])
        assert grid.valid_slot_count==case["valid_slot_count"], name
        assert len(grid.slot_valid_mask)==48 and len(grid.slot_durations_ql)==48
        assert sum(grid.slot_durations_ql)==case["bar_length_ql"]
    assert SlotGrid.for_bar(3.0).valid_slot_count==12
    try: SlotGrid.for_bar(cases["over_capacity"]["bar_length_ql"])
    except ValueError as error: assert str(error)=="slot_capacity_exceeded"
    else: raise AssertionError("over-capacity measure must fail")
def test_opus_track_and_quantization_policy():
    parser={case["name"]:case for case in load("opus_and_track_cases.json")}
    audit={case["name"]:case for case in load("quantization_audit_cases.json")}
    assert parser["multi_tune_opus"]["song_ids"]==["suite__tune_000","suite__tune_001"]
    assert parser["track_retention_policy"]["hard_safety_limit"]==48
    assert parser["track_retention_policy"]["default_policy"]=="error"
    assert audit["residual_monitor"]["classification"]=="MONITOR"

def test_frozen_schema_documents_lane_and_chroma_semantics():
    text=(ROOT/"contracts"/"codec"/"bar_tensor_schema.v2.md").read_text(encoding="utf-8")
    assert "melody=0" in text and "harmony_00..harmony_15=1..16" in text and "bass=17" in text
    assert "velocity_ratio =" in text and "(pitch-base_pitch) mod 12" in text

def test_grid_snaps_only_boundary_noise_and_keeps_true_partial_duration():
    assert SlotGrid.for_bar(4.0+5e-7).bar_length_ql==4.0
    assert SlotGrid.for_bar(6.375).slot_durations_ql[25]==0.125
    tiny=SlotGrid.for_bar(5e-7)
    assert tiny.valid_slot_count==1 and tiny.slot_durations_ql[0]>0
