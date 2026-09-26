from __future__ import annotations

import json

import numpy as np

from src.research.ultimate_v13 import build_ultimate_intelligence


def _bank():
    out = {}
    for fold in range(4):
        y = np.array([0, 1, 0, 1, 1, 0], dtype=int)
        p1 = np.array([0.30, 0.70, 0.35, 0.65, 0.60, 0.40]) + fold * 0.005
        p2 = np.array([0.45, 0.55, 0.50, 0.50, 0.55, 0.45]) + fold * 0.002
        out[fold] = {
            "y": y,
            "predictions": {"logistic": p1, "extra_trees": p2},
            "situations": np.array(["normal", "normal", "trend", "trend", "normal", "normal"]),
            "risk_context": np.column_stack([
                np.array([1, 2, 1, 2, 1, 2], dtype=float) + fold,
                np.array([10, 11, 10, 12, 11, 10], dtype=float),
            ]),
        }
    return out


def test_v13_builds_research_only_bundle(tmp_path):
    result = build_ultimate_intelligence(_bank(), out_dir=tmp_path)
    assert result["status"] == "EVALUATED"
    assert result["production_changed"] is False
    assert result["promotion_gate"]["allowed"] is False
    assert set(result["core_layers"]) == {
        "model_disagreement",
        "predictability",
        "future_model_failure",
        "time_to_failure",
    }
    assert (tmp_path / "ultimate_summary.json").exists()
    assert json.loads((tmp_path / "artifact_manifest.json").read_text())["production_changed"] is False


def test_dynamic_router_first_fold_has_no_prior_outcomes(tmp_path):
    result = build_ultimate_intelligence(_bank(), out_dir=tmp_path)
    rows = result["routing"]["folds"]
    assert rows[0]["weights"]["logistic"] == 0.5
    assert rows[0]["weights"]["extra_trees"] == 0.5


def test_failure_risk_is_prior_only_for_each_fold(tmp_path):
    bank = _bank()
    baseline = build_ultimate_intelligence(bank, out_dir=tmp_path / "a")
    # Alter only the last fold. Earlier prior-only risks must remain identical.
    bank[3]["y"] = np.array([1, 1, 1, 1, 1, 1], dtype=int)
    changed = build_ultimate_intelligence(bank, out_dir=tmp_path / "b")
    for model in ("logistic", "extra_trees"):
        a = baseline["future_model_failure"]["risk_predictions_prior_only"][model]
        b = changed["future_model_failure"]["risk_predictions_prior_only"][model]
        assert [x["future_failure_risk"] for x in a[:3]] == [x["future_failure_risk"] for x in b[:3]]


def test_probability_safety_and_predictability_bounds(tmp_path):
    result = build_ultimate_intelligence(_bank(), out_dir=tmp_path)
    for row in result["predictability"]["rows"]:
        assert 0.0 <= row["predictability_proxy"] <= 1.0
        assert 0.0 <= row["ood_score"] <= 1.0
    for fold in result["model_disagreement"].values():
        assert 0.0 <= fold["row_probability_std"]
        assert 0.0 <= fold["row_probability_range"]
