from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.research.ultimate_v13 import build_ultimate_intelligence

ROOT = Path("data/research")
OUT = ROOT / "v13_meta_leakage_audit.json"


def _bank(folds: int = 6):
    out = {}
    for fold in range(folds):
        y = np.array([0, 1, 0, 1, 1, 0], dtype=int)
        out[fold] = {
            "session_dates": np.array([f"2026-01-{fold + 1:02d}"] * len(y)),
            "y": y,
            "predictions": {
                "logistic": np.array([0.30, 0.70, 0.35, 0.65, 0.60, 0.40]) + fold * 0.003,
                "extra_trees": np.array([0.45, 0.55, 0.50, 0.50, 0.55, 0.45]) + fold * 0.001,
            },
            "regimes": np.array(["normal", "normal", "trend", "trend", "normal", "normal"]),
            "symbols": np.array(["A", "B", "C", "D", "E", "F"]),
            "asset_classes": np.array(["jp_stock"] * len(y)),
            "risk_context": np.column_stack([
                np.array([1, 2, 1, 2, 1, 2], dtype=float) + fold,
                np.array([10, 11, 10, 12, 11, 10], dtype=float),
                np.array([0.01, 0.00, 0.02, -0.01, 0.01, 0.00], dtype=float),
            ]),
        }
    return out


def main() -> int:
    baseline_bank = _bank()
    altered_bank = _bank()
    altered_bank[5]["y"] = np.ones(6, dtype=int)

    baseline = build_ultimate_intelligence(
        baseline_bank, out_dir=ROOT / "v13_meta_leakage_probe_a"
    )
    altered = build_ultimate_intelligence(
        altered_bank, out_dir=ROOT / "v13_meta_leakage_probe_b"
    )

    checks = {
        "prior_fold_failure_risk_immutable": (
            baseline["future_failure"]["per_model"]
            == altered["future_failure"]["per_model"]
        ),
        "prior_fold_routing_immutable": (
            baseline["fold_results"][:5] == altered["fold_results"][:5]
        ),
        "current_fold_strategy_unchanged": (
            baseline["fold_results"][-1]["chosen_strategy_counts"]
            == altered["fold_results"][-1]["chosen_strategy_counts"]
        ),
        "current_fold_action_unchanged": (
            baseline["fold_results"][-1]["chosen_action_counts"]
            == altered["fold_results"][-1]["chosen_action_counts"]
        ),
        "current_fold_routing_unchanged": (
            baseline["fold_results"][-1]["routing"]["weight_means"]
            == altered["fold_results"][-1]["routing"]["weight_means"]
        ),
        "pit_fail_closed": (
            baseline["audits"]["PIT"] == "BLOCKED_NO_FULL_TIMESTAMP_LINEAGE"
        ),
        "meta_leakage_not_claimed_pass": (
            baseline["audits"]["Meta-Leakage"] != "PASS"
        ),
        "production_isolated": (
            baseline["production_changed"] is False
            and altered["production_changed"] is False
            and baseline["promotion_allowed"] is False
            and altered["promotion_allowed"] is False
        ),
        "routing_present": (
            baseline.get("routing", {}).get("status")
            == "EXECUTED_ROW_WISE_SOFT_ROUTING"
        ),
    }

    payload = {
        "schema_version": "v13_meta_leakage_audit.1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "auditor": "v13_synthetic_outcome_perturbation",
        "scope": "research_control_plane_only",
        "production_changed": False,
        "promotion_allowed": False,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit("FAIL: v13 meta-leakage audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
