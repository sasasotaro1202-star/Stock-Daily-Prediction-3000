from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _require_completed_oos(payload: dict) -> None:
    if payload.get("status") != "OOS_COMPLETE":
        raise SystemExit(
            "FAIL: research snapshot can only persist an OOS_COMPLETE metrics artifact"
        )

    results = payload.get("results")
    if not isinstance(results, dict) or not results:
        raise SystemExit("FAIL: research metrics contain no model results")

    selected_model = str(payload.get("selected_model") or "").strip()
    if not selected_model:
        raise SystemExit("FAIL: selected_model is missing from research metrics")

    selected = results.get(selected_model)
    if not isinstance(selected, dict):
        raise SystemExit(
            f"FAIL: selected_model {selected_model!r} is absent from research results"
        )
    metrics = selected.get("metrics")
    if not isinstance(metrics, dict) or "logloss" not in metrics:
        raise SystemExit(
            "FAIL: selected model is missing its OOS LogLoss evidence"
        )


def main() -> None:
    source = Path("data/research/latest_metrics.json")
    if not source.exists():
        raise SystemExit("FAIL: research metrics are absent")

    raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"FAIL: research metrics are not valid UTF-8 JSON: {type(exc).__name__}:{exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise SystemExit("FAIL: research metrics root must be an object")
    _require_completed_oos(payload)

    adaptive = payload.get("adaptive_conformal_prediction_research", {})
    group = payload.get("group_conformal_prediction_research", {})
    static = payload.get("conformal_prediction_research", {})

    # Persist a compact, auditable score surface alongside the richer OOS
    # research artifact. This keeps the research-status branch useful for
    # longitudinal comparisons without copying the full fold-level payload.
    results = payload.get("results", {}) or {}
    score_fields = (
        "logloss",
        "brier",
        "ece",
        "accuracy",
        "roc_auc",
        "rank_ic",
        "recent_logloss",
        "selection_logloss",
        "folds",
        "oos_fold_count",
        "oos_fold_signature",
        "n_test_min",
        "n_test_max",
    )
    candidate_scores = {}
    for model_name, result in results.items():
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        candidate_scores[str(model_name)] = {
            field: metrics[field]
            for field in score_fields
            if field in metrics
        }
    selected_model = str(payload.get("selected_model"))
    selected_score = candidate_scores.get(selected_model, {})

    return_oos = payload.get("return_oos", {}) or {}
    return_metrics = return_oos.get("metrics", {}) or {}
    return_score = {
        field: return_metrics[field]
        for field in (
            "mae",
            "rmse",
            "sign_accuracy",
            "rank_ic",
            "range_80_coverage",
            "folds",
        )
        if field in return_metrics
    }

    snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "workflow_sha": os.environ.get("GITHUB_SHA", ""),
        "status": payload.get("status", "UNKNOWN"),
        "selected_model": selected_model,
        "selected_model_score": selected_score,
        "candidate_scores": candidate_scores,
        "return_oos_score": return_score,
        "global_selection_evidence": payload.get("global_selection_evidence", {}),
        "calibration_method": payload.get("calibration_method"),
        "adaptive_conformal_prediction_research": adaptive,
        "group_conformal_prediction_research": group,
        "conformal_prediction_research": static,
        "evidence_scope": {
            "source": "data/research/latest_metrics.json",
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "frozen_holdout_excluded_from_selection": True,
            "research_only_layers": True,
            "snapshot_contract": "OOS_COMPLETE_selected_model_logloss_required",
        },
    }

    out = Path("artifacts/research_validation_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
