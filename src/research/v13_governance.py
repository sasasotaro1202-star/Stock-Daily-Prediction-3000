"""Research-only v13 governance contracts.

This module does not change production routing. It provides explicit, machine-readable
experiment provenance and safety decisions so research evidence cannot be mistaken for
production approval.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def experiment_record(
    *,
    experiment_id: str,
    hypothesis: str,
    commit: str,
    dataset: str,
    feature_version: str,
    model_version: str,
    parameters: Mapping[str, Any],
    train_period: str,
    validation_period: str,
    oos_period: str,
    metrics: Mapping[str, Any] | None = None,
    decision: str = "UNDECIDED",
) -> dict[str, Any]:
    payload = {
        "experiment_id": str(experiment_id),
        "hypothesis": str(hypothesis),
        "commit": str(commit),
        "dataset": str(dataset),
        "feature_version": str(feature_version),
        "model_version": str(model_version),
        "parameters": dict(parameters),
        "train_period": str(train_period),
        "validation_period": str(validation_period),
        "oos_period": str(oos_period),
        "metrics": dict(metrics or {}),
        "decision": str(decision),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    payload["record_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return payload


def safety_governance(
    *,
    promotion_allowed: bool,
    pit_status: str,
    leakage_status: str,
    meta_leakage_status: str,
    robustness_status: str,
    reproducibility_status: str,
    production_changed: bool,
) -> dict[str, Any]:
    blockers = []
    for name, status in (
        ("PIT", pit_status),
        ("Leakage", leakage_status),
        ("Meta-Leakage", meta_leakage_status),
        ("Robustness", robustness_status),
        ("Reproducibility", reproducibility_status),
    ):
        if status != "PASS":
            blockers.append(f"{name}={status}")

    kill_switch = bool(
        pit_status in {"FAIL", "BLOCKED_NO_FULL_TIMESTAMP_LINEAGE"}
        or leakage_status == "FAIL"
        or meta_leakage_status == "FAIL"
        or production_changed
    )
    action = "ALLOW_RESEARCH_ONLY"
    fallback = "verified_baseline"
    if kill_switch:
        action = "BLOCK_NEW_V13_PATH"
        fallback = "verified_baseline"
    elif not promotion_allowed:
        action = "HOLD_PROMOTION"
        fallback = "stable_ensemble"

    return {
        "status": "EXECUTED_RESEARCH_GOVERNANCE",
        "action": action,
        "kill_switch_engaged": kill_switch,
        "fallback_target": fallback,
        "rollback_target": "previous_verified",
        "promotion_allowed": bool(promotion_allowed and not blockers),
        "blockers": blockers,
        "production_changed": bool(production_changed),
        "contracts": {
            "research_only": True,
            "no_auto_promotion": True,
            "fallback_verified_baseline": True,
            "rollback_previous_verified": True,
        },
    }


def validate_experiment_registry(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("experiment_id", "")) for row in rows]
    unique = len(ids) == len(set(ids)) and all(ids)
    required = (
        "experiment_id",
        "hypothesis",
        "commit",
        "dataset",
        "feature_version",
        "model_version",
        "parameters",
        "train_period",
        "validation_period",
        "oos_period",
        "decision",
    )
    complete = all(all(key in row for key in required) for row in rows)
    return {
        "status": "PASS" if unique and complete else "FAIL",
        "rows": len(rows),
        "unique_ids": unique,
        "complete_schema": complete,
    }
