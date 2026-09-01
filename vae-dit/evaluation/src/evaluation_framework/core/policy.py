"""Small, explicit policy-assessment builders for artifact-only evaluators."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def maximum_gate(metric_id: str, label: str, value: float, threshold: float) -> dict[str, Any]:
    return assessment(metric_id, label, value, "FAIL" if value > threshold else "PASS", True, {"max": threshold})


def minimum_gate(metric_id: str, label: str, value: float, threshold: float) -> dict[str, Any]:
    return assessment(metric_id, label, value, "FAIL" if value < threshold else "PASS", True, {"min": threshold})


def warn_below(metric_id: str, label: str, value: float, threshold: float) -> dict[str, Any]:
    return assessment(metric_id, label, value, "WARN" if value < threshold else "PASS", False, {"warn_below": threshold})


def warn_above(metric_id: str, label: str, value: float, threshold: float) -> dict[str, Any]:
    return assessment(metric_id, label, value, "WARN" if value > threshold else "PASS", False, {"warn_above": threshold})


def r2_gate(metric_id: str, label: str, value: float | None, policy: Mapping[str, Any]) -> dict[str, Any]:
    status = "FAIL" if value is None or value < policy["r2_fail_below"] else "WARN" if value < policy["r2_warn_below"] else "PASS"
    return assessment(metric_id, label, value, status, True, dict(policy))


def probe_gate(metric_id: str, label: str, value: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    status = "FAIL" if (
        value["quantile_gap"] <= policy["quantile_gap_fail_at_or_below"]
        or value["triplet_accuracy"] < policy["triplet_accuracy_fail_below"]
    ) else "WARN" if value["triplet_accuracy"] < policy["triplet_accuracy_warn_below"] else "PASS"
    return assessment(metric_id, label, value, status, True, dict(policy))


def low_band_gate(metric_id: str, label: str, value: float, fail: float, warn: float) -> dict[str, Any]:
    status = "FAIL" if value < fail else "WARN" if value < warn else "PASS"
    return assessment(metric_id, label, value, status, True, {"fail_below": fail, "warn_below": warn})


def high_band_gate(metric_id: str, label: str, value: float, warn: float, fail: float) -> dict[str, Any]:
    status = "FAIL" if value > fail else "WARN" if value > warn else "PASS"
    return assessment(metric_id, label, value, status, True, {"warn_above": warn, "fail_above": fail})


def monitor(metric_id: str, label: str, value: float) -> dict[str, Any]:
    return assessment(metric_id, label, value, "MONITOR", False, {})


def assessment(
    metric_id: str,
    label: str,
    value: Any,
    status: str,
    gate: bool,
    threshold: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "label": label,
        "value": value,
        "status": status,
        "gate": gate,
        "threshold": dict(threshold),
    }


def overall_status(assessments: Sequence[Mapping[str, Any]]) -> str:
    if any(item["status"] == "FAIL" and item["gate"] for item in assessments):
        return "FAIL"
    if any(item["status"] == "WARN" for item in assessments):
        return "WARN"
    return "PASS"
