from __future__ import annotations

import json

import numpy as np

from src.research.ultimate_v13 import build_ultimate_intelligence


def _bank(folds: int = 6):
    out = {}
    for fold in range(folds):
        y = np.array([0, 1, 0, 1, 1, 0], dtype=int)
        p1 = np.array([0.30, 0.70, 0.35, 0.65, 0.60, 0.40]) + fold * 0.003
        p2 = np.array([0.45, 0.55, 0.50, 0.50, 0.55, 0.45]) + fold * 0.001
        out[fold] = {
            "session_dates": np.array([f"2026-01-{fold + 1:02d}"] * len(y)),
            "y": y,
            "predictions": {"logistic": p1, "extra_trees": p2},
            "situations": np.array(["normal", "normal", "trend", "trend", "normal", "normal"]),
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


def test_v13_builds_and_blocks_promotion(tmp_path):
    result = build_ultimate_intelligence(_bank(), out_dir=tmp_path)
    assert result["status"] == "OOS_COMPLETE"
    assert result["production_changed"] is False
    assert result["promotion_allowed"] is False
    assert result["core_three_layers"]["model_disagreement"] == "IMPLEMENTED_EXECUTED"
    assert result["core_three_layers"]["predictability"] == "IMPLEMENTED_EXECUTED"
    assert result["core_three_layers"]["future_model_failure"] == "IMPLEMENTED_EXECUTED"
    assert (tmp_path / "ultimate_summary.json").exists()
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert manifest["production_changed"] is False
    assert manifest["promotion_allowed"] is False


def test_v13_requires_five_chronological_folds(tmp_path):
    result = build_ultimate_intelligence(_bank(4), out_dir=tmp_path)
    assert result["status"] == "BLOCKED"
    assert result["production_changed"] is False
    assert result["promotion"] == "HOLD"


def test_v13_failure_risk_is_prior_only(tmp_path):
    baseline = build_ultimate_intelligence(_bank(), out_dir=tmp_path / "a")
    altered = _bank()
    altered[5]["y"] = np.ones(6, dtype=int)
    changed = build_ultimate_intelligence(altered, out_dir=tmp_path / "b")
    for model in ("logistic", "extra_trees"):
        a = baseline["future_failure"]["per_model"][model][:4]
        b = changed["future_failure"]["per_model"][model][:4]
        assert a == b


def test_v13_artifact_is_json_safe_and_routing_has_causal_contract(tmp_path):
    result = build_ultimate_intelligence(_bank(), out_dir=tmp_path)
    payload = json.loads((tmp_path / "ultimate_summary.json").read_text())
    assert payload["audits"]["PIT"] == "BLOCKED_NO_FULL_TIMESTAMP_LINEAGE"
    assert payload["production_isolation"] is True
    assert "selection_is_outcome_free_current_fold" in payload["prediction_strategy"]
    for fold in payload["fold_results"]:
        assert 0.0 <= fold["coverage"] <= 1.0


def test_v13_exposes_failure_calibration_ttf_and_error_diversity(tmp_path):
    build_ultimate_intelligence(_bank(), out_dir=tmp_path)
    payload = json.loads((tmp_path / "ultimate_summary.json").read_text())
    for model in ("logistic", "extra_trees"):
        cal = payload["future_failure"]["calibration_1step"][model]
        assert cal["observations"] >= 1
        assert 0.0 <= cal["brier"] <= 1.0
        assert 0.0 <= cal["ece"] <= 1.0
        assert model in payload["time_to_failure"]["retrospective_evaluation"]
    assert payload["error_correlation"]["status"] == "EXECUTED_RETROSPECTIVE_DIAGNOSTIC"
    assert (tmp_path / "error_correlation.json").exists()


def test_v13_false_revision_is_revision_conditional(tmp_path):
    build_ultimate_intelligence(_bank(), out_dir=tmp_path)
    payload = json.loads((tmp_path / "ultimate_summary.json").read_text())
    value = payload["revision_metrics"]["false_revision"]
    assert value is None or 0.0 <= value <= 1.0
