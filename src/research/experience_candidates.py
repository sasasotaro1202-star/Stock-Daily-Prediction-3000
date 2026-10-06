from __future__ import annotations

import hashlib
from typing import Any, Mapping


SCHEMA_VERSION = 1
MAX_CANDIDATES = 20
MIN_OBSERVATIONS = 30


def _candidate_id(dimension: str, segment: str) -> str:
    raw = f"{dimension}|{segment}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"exp-{digest}"


def _candidate_for_priority(row: Mapping[str, Any]) -> dict[str, Any] | None:
    dimension = str(row.get("dimension", "")).strip()
    segment = str(row.get("segment", "")).strip()
    n = int(row.get("n", 0) or 0)
    impact = float(row.get("impact_vs_global_logloss", 0.0) or 0.0)
    if not dimension or not segment or n < MIN_OBSERVATIONS or impact <= 0.0:
        return None

    common = {
        "candidate_id": _candidate_id(dimension, segment),
        "source_dimension": dimension,
        "source_segment": segment,
        "evidence_n": n,
        "evidence_logloss": float(row.get("logloss", 0.0) or 0.0),
        "evidence_accuracy": float(row.get("accuracy", 0.0) or 0.0),
        "evidence_return_mae": float(row.get("return_mae", 0.0) or 0.0),
        "impact_vs_global_logloss": impact,
        "research_only": True,
        "production_changed": False,
        "frozen_holdout_allowed": False,
        "promotion_status": "HOLD_UNTIL_OOS",
    }

    if dimension == "by_regime":
        action = "regime_specific_model_challenge"
        hypothesis = (
            f"Evaluate whether the {segment} regime has a persistent "
            "model/error structure that supports a causal regime-specific challenger."
        )
        controls = {
            "selection_scope": "regime",
            "required_evaluation": "chronological_oos",
            "required_baseline": "current_global_or_locked_route",
        }
    elif dimension == "by_market_situation":
        action = "situation_specific_route_challenge"
        hypothesis = (
            f"Evaluate whether market situation {segment} requires a distinct "
            "model, uncertainty policy, or abstention threshold."
        )
        controls = {
            "selection_scope": "market_situation",
            "required_evaluation": "chronological_oos",
            "required_baseline": "current_locked_route",
        }
    elif dimension == "by_asset_class":
        action = "asset_specific_model_or_calibration_challenge"
        hypothesis = (
            f"Evaluate whether {segment} behaves differently enough to justify "
            "asset-specific modeling or calibration."
        )
        controls = {
            "selection_scope": "asset_class",
            "required_evaluation": "chronological_oos",
            "required_baseline": "current_global_or_asset_route",
        }
    elif dimension == "by_model_id":
        action = "model_failure_attribution_challenge"
        hypothesis = (
            f"Investigate recurring error concentration for model {segment} "
            "and test a safer alternative or routing guard."
        )
        controls = {
            "selection_scope": "model_id",
            "required_evaluation": "common_chronological_oos",
            "required_baseline": "same_fold_comparator",
        }
    elif dimension == "error_types":
        if segment == "high_confidence_wrong":
            action = "high_confidence_abstention_challenge"
            hypothesis = (
                "Test whether high-confidence wrong cases can be isolated with "
                "PIT-safe uncertainty/abstention controls without harming normal cases."
            )
        elif segment == "high_model_disagreement":
            action = "disagreement_trigger_challenge"
            hypothesis = (
                "Test whether high model disagreement predicts elevated future error "
                "and supports selective prediction or fallback."
            )
        else:
            action = "error_type_specific_failure_challenge"
            hypothesis = (
                f"Test a targeted mitigation for recurring error type {segment}."
            )
        controls = {
            "selection_scope": "error_type",
            "required_evaluation": "chronological_oos",
            "required_baseline": "no_special_handling",
        }
    elif dimension == "by_error_bucket":
        action = "return_uncertainty_challenge"
        hypothesis = (
            f"Investigate the {segment} return-error bucket for interval, "
            "uncertainty, target, or abstention improvements."
        )
        controls = {
            "selection_scope": "return_error_bucket",
            "required_evaluation": "chronological_oos",
            "required_baseline": "current_return_pipeline",
        }
    elif dimension == "by_direction_confidence":
        action = "confidence_calibration_challenge"
        hypothesis = (
            f"Test whether confidence bucket {segment} is systematically "
            "miscalibrated or better served by selective prediction."
        )
        controls = {
            "selection_scope": "direction_confidence",
            "required_evaluation": "chronological_oos_calibration",
            "required_baseline": "current_calibration",
        }
    else:
        action = "generic_failure_structure_research"
        hypothesis = (
            f"Investigate recurring performance degradation in {dimension}={segment} "
            "before considering any production policy change."
        )
        controls = {
            "selection_scope": dimension,
            "required_evaluation": "chronological_oos",
            "required_baseline": "current_pipeline",
        }

    common.update(
        {
            "action": action,
            "hypothesis": hypothesis,
            "controls": controls,
        }
    )
    return common


def build_experience_candidate_plan(
    memory: Mapping[str, Any],
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> dict[str, Any]:
    raw = list(memory.get("research_priority") or [])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for priority in raw:
        candidate = _candidate_for_priority(priority)
        if candidate is None or candidate["candidate_id"] in seen:
            continue
        seen.add(candidate["candidate_id"])
        rows.append(candidate)
        if len(rows) >= int(max_candidates):
            break

    total_resolved = int(memory.get("total_resolved", 0) or 0)
    matured = total_resolved >= MIN_OBSERVATIONS
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "READY" if matured else "WARMUP",
        "signal_status": "CANDIDATES_FOUND" if rows else (
            "NO_DEGRADATION_SIGNAL" if matured else "INSUFFICIENT_EXPERIENCE"
        ),
        "source": "experience_memory",
        "memory_updated_at": memory.get("updated_at"),
        "total_resolved": int(memory.get("total_resolved", 0) or 0),
        "candidates": rows,
        "safety_contract": {
            "research_only": True,
            "production_changed": False,
            "frozen_holdout_allowed": False,
            "promotion_requires_chronological_oos": True,
            "outcome_data_must_be_matured_before_reuse": True,
        },
    }
