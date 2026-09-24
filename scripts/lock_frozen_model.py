from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.validation.code_fingerprint import (
    fingerprint_sha256,
    research_fingerprint_sha256,
)


def main():
    path=Path("config/frozen_holdout.json")
    metrics=Path("data/research/latest_metrics.json")
    if not path.exists() or not metrics.exists():
        raise SystemExit(
            "DEFERRED: cutoff freeze and OOS metrics are required"
        )

    lock=json.loads(path.read_text(encoding="utf-8"))
    current_fp=fingerprint_sha256()
    current_research_fp=research_fingerprint_sha256()
    if lock.get("status")=="FROZEN":
        stored_research_fp=lock.get("research_code_fingerprint_sha256")
        if stored_research_fp != current_research_fp:
            raise SystemExit(
                "DEFERRED: frozen holdout generation is immutable; "
                "bootstrap_frozen_holdout must rotate to a new generation "
                "before locking changed research code"
            )
        if lock.get("code_fingerprint_sha256") != current_fp:
            # Workflow-only changes do not invalidate the blind holdout. Keep
            # the model selection frozen while refreshing the full provenance
            # fingerprint used by production-artifact compatibility checks.
            lock["code_fingerprint_sha256"]=current_fp
            lock["selection_locked_git_sha"]=os.getenv("GITHUB_SHA")
            path.write_text(json.dumps(lock,indent=2),encoding="utf-8")
        print("frozen-model: already locked")
        return
    if lock.get("status")!="CUTOFF_FROZEN_PENDING_MODEL":
        raise SystemExit(f"FAIL: invalid freeze status {lock.get('status')}")

    payload=json.loads(metrics.read_text(encoding="utf-8"))
    selection_evidence = payload.get("global_selection_evidence", {})
    if (
        selection_evidence.get("status") != "SUPPORTED"
        or selection_evidence.get("eligible_for_freeze") is not True
        or selection_evidence.get("selected_model") != payload.get("selected_model")
    ):
        raise SystemExit(
            "DEFERRED: global OOS model selection lacks statistically supported "
            "paired-fold evidence; frozen production selection is not permitted"
        )
    selected=payload.get("selected_model")
    if not selected:
        raise SystemExit("FAIL: OOS did not select a global model")

    lock["selected_model"]=selected
    lock["global_selection_evidence"]=selection_evidence
    training_window=payload.get("classifier_training_window_sessions", 0)
    if not isinstance(training_window,(int,float)) or int(training_window) < 0:
        raise SystemExit("FAIL: OOS classifier training window is invalid")
    lock["classifier_training_window_sessions"]=int(training_window)
    calibration_method=payload.get("calibration_method", "platt")
    if calibration_method not in {"platt", "beta", "isotonic", "temperature"}:
        raise SystemExit("FAIL: OOS did not produce a valid calibration method")
    lock["calibration_method"]=calibration_method
    rank_weight=payload.get("rank_probability_weight", 0.50)
    if not isinstance(rank_weight,(int,float)) or not 0.0 <= float(rank_weight) <= 1.0:
        raise SystemExit("FAIL: OOS did not produce a valid ranking probability weight")
    lock["rank_probability_weight"]=float(rank_weight)
    rank_uncertainty_penalty=payload.get("rank_uncertainty_penalty", 0.0)
    if not isinstance(rank_uncertainty_penalty,(int,float)) or not 0.0 <= float(rank_uncertainty_penalty) <= 1.0:
        raise SystemExit("FAIL: OOS did not produce a valid ranking uncertainty penalty")
    lock["rank_uncertainty_penalty"]=float(rank_uncertainty_penalty)
    vol_threshold_source=payload.get("regime_vol_threshold_source")
    if vol_threshold_source != "oos_fold_train_median":
        raise SystemExit("FAIL: regime volatility threshold is not OOS-train-derived")
    threshold_folds=payload.get("regime_vol_threshold_folds")
    if not isinstance(threshold_folds,(int,float)) or int(threshold_folds) < 3:
        raise SystemExit("FAIL: insufficient OOS folds for regime volatility threshold")

    vol_threshold=payload.get("regime_vol_threshold")
    if not isinstance(vol_threshold,(int,float)) or not __import__("math").isfinite(float(vol_threshold)):
        raise SystemExit("FAIL: OOS did not produce a valid regime volatility threshold")
    lock["regime_vol_threshold"]=float(vol_threshold)
    lock["regime_vol_threshold_source"]=vol_threshold_source
    lock["regime_vol_threshold_folds"]=int(threshold_folds)
    return_oos=payload.get("return_oos", {})
    return_selected=return_oos.get("selected_estimator")
    if return_selected not in {"mean", "q50", "blend_mean_q50"}:
        raise SystemExit("FAIL: OOS did not select a valid return estimator")
    lock["return_selected_estimator"]=return_selected
    lock["regime_selected_models"]=payload.get("regime_selected_models",{})
    lock["asset_class_selected_models"]=payload.get(
        "asset_class_selected_models",{}
    )
    lock["asset_regime_selected_models"]=payload.get(
        "asset_regime_selected_models",{}
    )
    lock["situation_selected_models"]=payload.get(
        "situation_selected_models",{}
    )
    lock["asset_situation_selected_models"]=payload.get(
        "asset_situation_selected_models",{}
    )
    lock["symbol_selected_models"]=payload.get(
        "symbol_selected_models",{}
    )
    lock["symbol_regime_selected_models"]=payload.get(
        "symbol_regime_selected_models",{}
    )
    lock["selection_locked_at"]=datetime.now(timezone.utc).isoformat()
    lock["selection_locked_git_sha"]=os.getenv("GITHUB_SHA")
    lock["code_fingerprint_sha256"]=current_fp
    lock["research_code_fingerprint_sha256"]=current_research_fp
    lock["selection_source"]=(
        "chronological OOS using only observations <= cutoff_date; "
        "asset/regime/situation routes inherit the same OOS-only policy"
    )
    lock["status"]="FROZEN"

    path.write_text(json.dumps(lock,indent=2),encoding="utf-8")
    print(json.dumps(lock,indent=2))


if __name__=="__main__": main()
