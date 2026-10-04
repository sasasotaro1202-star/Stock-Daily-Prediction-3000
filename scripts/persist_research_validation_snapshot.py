from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.validation.code_fingerprint import evidence_fingerprint_sha256


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


def _git_head_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _evidence_freshness(research_sha: str | None, current_sha: str | None, research_fingerprint: str | None = None, current_fingerprint: str | None = None) -> str:
    if research_fingerprint and current_fingerprint:
        return "FRESH" if research_fingerprint == current_fingerprint else "STALE"
    if research_sha and current_sha:
        return "FRESH" if research_sha == current_sha else "STALE"
    return "UNKNOWN"


def _evidence_freshness_basis(research_fingerprint: str | None, current_fingerprint: str | None, research_sha: str | None, current_sha: str | None) -> str:
    if research_fingerprint and current_fingerprint:
        return "evidence_code_fingerprint"
    if research_sha and current_sha:
        return "execution_sha"
    return "unknown"


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

    research_workflow_sha = (
        os.environ.get("RESEARCH_WORKFLOW_SHA", "").strip()
        or str(payload.get("workflow_sha") or "").strip()
        or None
    )
    current_main_sha = _git_head_sha()
    research_evidence_fingerprint = str(payload.get("evidence_code_fingerprint_sha256") or "").strip() or None
    current_evidence_fingerprint = evidence_fingerprint_sha256()
    evidence_freshness = _evidence_freshness(research_workflow_sha, current_main_sha, research_evidence_fingerprint, current_evidence_fingerprint)
    evidence_freshness_basis = _evidence_freshness_basis(research_evidence_fingerprint, current_evidence_fingerprint, research_workflow_sha, current_main_sha)
    snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_run_id": os.environ.get("RESEARCH_WORKFLOW_RUN_ID", "").strip()
        or os.environ.get("GITHUB_RUN_ID", ""),
        "workflow_sha": research_workflow_sha or "",
        "research_workflow_sha": research_workflow_sha,
        "status_branch_main_sha": current_main_sha,
        "research_evidence_fingerprint_sha256": research_evidence_fingerprint,
        "status_branch_evidence_fingerprint_sha256": current_evidence_fingerprint,
        "evidence_freshness": evidence_freshness,
        "evidence_freshness_basis": evidence_freshness_basis,
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
            "research_workflow_sha": research_workflow_sha,
            "status_branch_main_sha": current_main_sha,
            "research_evidence_fingerprint_sha256": research_evidence_fingerprint,
            "status_branch_evidence_fingerprint_sha256": current_evidence_fingerprint,
            "freshness": evidence_freshness,
            "freshness_basis": evidence_freshness_basis,
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
