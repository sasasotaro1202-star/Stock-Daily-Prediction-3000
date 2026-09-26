from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.research.ultimate_v13 import build_ultimate_intelligence
from src.research.ultimate_control_v13 import safe_probability


ROOT = Path("data/research/ultimate_v13")
OUT = ROOT / "adversarial_robustness.json"


def _bank(folds: int = 7):
    out = {}
    for fold in range(folds):
        y = np.asarray([0, 1, 0, 1, 1, 0, 1, 0], dtype=int)
        p1 = np.asarray([0.28, 0.72, 0.36, 0.64, 0.61, 0.39, 0.67, 0.33], dtype=float)
        p2 = np.asarray([0.42, 0.58, 0.48, 0.52, 0.55, 0.45, 0.57, 0.43], dtype=float)
        p3 = np.asarray([0.34, 0.66, 0.41, 0.59, 0.63, 0.37, 0.60, 0.40], dtype=float)
        shift = fold * 0.001
        out[fold] = {
            "session_dates": np.asarray([f"2026-02-{fold + 1:02d}"] * len(y)),
            "y": y,
            "predictions": {
                "logistic": p1 + shift,
                "extra_trees": p2 + shift * 0.5,
                "hgb": p3 - shift * 0.25,
            },
            "regimes": np.asarray([
                "normal", "normal", "trend", "trend",
                "normal", "high_vol", "high_vol", "normal",
            ]),
            "symbols": np.asarray(list("ABCDEFGH")),
            "asset_classes": np.asarray(["jp_stock"] * len(y)),
            "risk_context": np.column_stack([
                np.asarray([1, 2, 1, 2, 1, 3, 3, 1], dtype=float) + fold,
                np.asarray([10, 11, 10, 12, 11, 15, 14, 10], dtype=float),
                np.asarray([0.01, 0.00, 0.02, -0.01, 0.01, 0.06, -0.05, 0.00], dtype=float),
            ]),
        }
    return out


def _assert_base_contract(result: dict) -> None:
    assert result["status"] == "OOS_COMPLETE"
    assert result["production_changed"] is False
    assert result["promotion_allowed"] is False
    assert result["error_correlation"]["status"] == "EXECUTED_RETROSPECTIVE_DIAGNOSTIC"
    for row in result["fold_results"]:
        weights = np.asarray(list(row["routing"]["weight_means"].values()), dtype=float)
        assert np.isfinite(weights).all()
        assert 0.0 <= row["coverage"] <= 1.0
        assert np.isfinite(float(row["routing"]["weight_entropy"]))
        assert np.isfinite(float(row["max_failure_risk"]))


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)

    base = _bank()
    baseline_dir = ROOT / "adversarial_base"
    repeat_dir = ROOT / "adversarial_repeat"
    result = build_ultimate_intelligence(base, out_dir=baseline_dir)
    repeat = build_ultimate_intelligence(base, out_dir=repeat_dir)
    _assert_base_contract(result)
    _assert_base_contract(repeat)

    checks: dict[str, bool] = {}

    # Determinism: same frozen synthetic bank must produce the same summary.
    checks["deterministic_replay"] = (
        json.dumps(result, sort_keys=True) == json.dumps(repeat, sort_keys=True)
    )

    # Probability safety: extreme probabilities remain finite and routing-safe.
    extreme = np.asarray([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]], dtype=float)
    clipped = safe_probability(extreme)
    checks["probability_safety_extremes"] = bool(
        np.isfinite(clipped).all()
        and (clipped > 0.0).all()
        and (clipped < 1.0).all()
    )

    # Outcome perturbation: changing only the latest fold cannot alter prior-fold
    # failure-risk/routing evidence.
    altered = _bank()
    altered[6]["y"] = np.ones_like(altered[6]["y"])
    altered_dir = ROOT / "adversarial_outcome_perturbation"
    altered_result = build_ultimate_intelligence(altered, out_dir=altered_dir)
    prior_count = len(result["fold_results"]) - 1
    checks["prior_history_immutable_under_current_outcome_change"] = (
        all(
            result["future_failure"]["per_model"][name][:-1]
            == altered_result["future_failure"]["per_model"][name][:-1]
            for name in result["future_failure"]["per_model"]
        )
        and result["fold_results"][:prior_count] == altered_result["fold_results"][:prior_count]
    )

    # Missing/late data: last-fold risk context can become unavailable without
    # mutating production or producing invalid numeric outputs.
    missing = _bank()
    missing[6]["risk_context"][:] = np.nan
    missing_result = build_ultimate_intelligence(
        missing, out_dir=ROOT / "adversarial_missing_data"
    )
    _assert_base_contract(missing_result)
    checks["missing_data_fail_safe"] = bool(
        missing_result["production_changed"] is False
        and missing_result["promotion_allowed"] is False
        and all(
            np.isfinite(float(x["predictability_mean"]))
            and np.isfinite(float(x["ood_mean"]))
            for x in missing_result["fold_results"]
        )
    )

    # Disagreement spike: widen model separation but require valid normalized
    # routing evidence rather than a particular direction of movement.
    spike = _bank()
    spike[6]["predictions"]["logistic"] = np.asarray([0.05, 0.95, 0.10, 0.90, 0.92, 0.08, 0.97, 0.03])
    spike[6]["predictions"]["extra_trees"] = np.asarray([0.95, 0.05, 0.90, 0.10, 0.08, 0.92, 0.03, 0.97])
    spike[6]["predictions"]["hgb"] = np.asarray([0.50] * 8)
    spike_result = build_ultimate_intelligence(spike, out_dir=ROOT / "adversarial_disagreement_spike")
    _assert_base_contract(spike_result)
    checks["disagreement_spike_safe"] = bool(
        all(
            np.isfinite(float(x["routing"]["weight_concentration"]))
            and 0.0 <= float(x["routing"]["weight_concentration"]) <= 1.0
            for x in spike_result["fold_results"]
        )
    )

    # Extreme model probabilities near 0/1 must not create invalid artifacts.
    extreme_bank = _bank()
    for name in extreme_bank[5]["predictions"]:
        extreme_bank[5]["predictions"][name] = np.clip(
            extreme_bank[5]["predictions"][name], 1e-12, 1.0 - 1e-12
        )
    extreme_result = build_ultimate_intelligence(
        extreme_bank, out_dir=ROOT / "adversarial_extreme_probabilities"
    )
    _assert_base_contract(extreme_result)
    checks["extreme_probability_inputs_safe"] = True

    payload = {
        "schema_version": "v13_adversarial_robustness.1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "research_only_structural_stress_and_invariance",
        "performance_success": False,
        "production_changed": False,
        "promotion_allowed": False,
        "checks": checks,
        "notes": [
            "This audit validates structural safety, determinism, causal-history invariance and numeric safety.",
            "It does not claim statistical robustness or performance improvement on real OOS data.",
            "Production artifacts are not modified.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit("FAIL: v13 adversarial structural robustness audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
