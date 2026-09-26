from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    metrics_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/research/latest_metrics.json")
    v6 = json.loads(metrics_path.read_text(encoding="utf-8")).get("innovative_prediction_control_v6", {})
    locked = v6.get("full_vs_baseline_locked", {})
    statistical = v6.get("statistical_validation", {})
    mandatory = {
        "OOS": v6.get("status") == "OOS_COMPLETE",
        "PIT": v6.get("locked_oos_untouched_for_tuning") is True,
        "Meta-Leakage": all(
            x.get("current_labels_used_for_meta_training") is False
            for x in v6.get("per_fold_meta_audit", [])
        ),
        "ProductionIsolation": v6.get("production_changed") is False,
        "LockedFolds": int(v6.get("locked_folds", 0)) >= 1,
        "ArtifactSource": metrics_path.exists(),
        "StatisticalValidation": all(
            key in statistical for key in
            ("delta_accuracy_ci95", "delta_logloss_ci95", "delta_brier_ci95", "delta_ece_ci95")
        ),
    }
    # Shadow/challenger have not been run by the research-only E2E path yet,
    # therefore the gate remains HOLD even when locked OOS is positive.
    decision = "HOLD"
    if (
        mandatory["OOS"]
        and mandatory["PIT"]
        and mandatory["Meta-Leakage"]
        and mandatory["ProductionIsolation"]
        and mandatory["LockedFolds"]
        and mandatory["ArtifactSource"]
        and mandatory["StatisticalValidation"]
        and locked.get("accuracy", -1.0) >= 0.03
        and locked.get("logloss", 1.0) <= 0.0
        and locked.get("brier", 1.0) <= 0.0
        and locked.get("ece", 1.0) <= 0.0
    ):
        # Candidate is allowed only as a research challenger; not production adoption.
        decision = "CANDIDATE"
    result = {
        "decision": decision,
        "production_changed": False,
        "mandatory_checks": mandatory,
        "locked_delta": locked,
        "explicit_hold_reason": (
            "shadow/challenger/fallback production verification is not executed in this research workflow"
            if decision == "CANDIDATE"
            else "locked OOS gate not fully passed"
        ),
    }
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("artifacts/v6_promotion_gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
