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


def test_v13_prior_only_tta_scenario_contract_and_ledger(tmp_path):
    result = build_ultimate_intelligence(_bank(), out_dir=tmp_path)
    assert result["tta"]["status"] == "EXECUTED_PRIOR_ONLY_RESEARCH_ABLATION"
    assert result["tta"]["production_changed"] is False
    assert result["tta"]["promotion_allowed"] is False
    assert len(result["tta"]["folds"]) == 6
    assert result["scenarios"]["status"] == "EXECUTED_HEURISTIC_PROXY"
    assert result["prediction_contracts"]["exact_timestamp_lineage"] is False
    assert result["prediction_ledger"]["total_predictions"] == 36
    assert result["prediction_ledger"]["status"] == "EXECUTED_ROW_LEVEL_LEDGER_WITH_PIT_BLOCK"
    assert len(result["prediction_ledger"]["row_level"]) == 36
    assert all(row["pit_status"] == "BLOCKED_NO_FULL_TIMESTAMP_LINEAGE" for row in result["prediction_ledger"]["row_level"])
    assert all(row["strategy"] != "unknown" for row in result["prediction_ledger"]["row_level"])
    assert all(row["action"] != "unknown" for row in result["prediction_ledger"]["row_level"])
    assert result["router_stability"]["status"] == "EXECUTED_DESCRIPTIVE_ROUTER_MONITOR"
    assert result["active_information"]["status"] == "BLOCKED_NO_SOURCE_VALUE_OF_INFORMATION_METADATA"
    for name in (
        "tta.json",
        "scenarios.json",
        "prediction_contracts.json",
        "prediction_ledger.json",
        "router_stability.json",
        "active_information.json",
    ):
        assert (tmp_path / name).exists()


def test_v13_failure_memory_and_error_attribution_are_post_outcome_only(tmp_path):
    result = build_ultimate_intelligence(_bank(), out_dir=tmp_path)
    ledger = result["prediction_ledger"]["row_level"]
    assert all(row["result"] in (0, 1) for row in ledger)
    assert all(row["failure_type"] for row in ledger)
    assert "correct" in result["error_attribution"]["counts_all_folds"]
    memory = result["failure_memory"]
    assert memory["status"] == "EXECUTED_CAUSAL_POST_OUTCOME_MEMORY"
    assert memory["total_rows"] == 36
    assert memory["total_failures"] >= 0
    assert all(
        "prediction" in row and "failed" in row and "failure_type" in row
        for row in memory["rows"]
    )
    assert (tmp_path / "error_attribution.json").exists()
    assert (tmp_path / "failure_memory.json").exists()


def test_v13_tta_history_is_causal(tmp_path):
    baseline = build_ultimate_intelligence(_bank(), out_dir=tmp_path / "a")
    altered = _bank()
    altered[5]["y"] = np.ones(6, dtype=int)
    changed = build_ultimate_intelligence(altered, out_dir=tmp_path / "b")
    for a, b in zip(baseline["tta"]["folds"][:-1], changed["tta"]["folds"][:-1]):
        assert a["adaptation"] == b["adaptation"]
        assert a["delta_tta_minus_dynamic"] == b["delta_tta_minus_dynamic"]


def test_v13_exposes_future_failure_aware_routing_challenger(tmp_path):
    result = build_ultimate_intelligence(_bank(), out_dir=tmp_path)
    routing = result["routing"]
    assert "future_failure_aware" in routing["aggregate"]
    assert "delta_failure_aware_minus_dynamic" in routing["aggregate"]
    for fold in routing["folds"]:
        metrics = fold["routing"]["future_failure_aware"]
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["brier"] <= 1.0
        assert 0.0 <= metrics["ece"] <= 1.0
        weights = np.asarray(
            list(fold["routing"]["future_failure_weight_means"].values()),
            dtype=float,
        )
        assert np.isfinite(weights).all()
        assert np.isclose(float(weights.sum()), 1.0)


def test_v13_meta_label_is_prior_only_and_ledgered(tmp_path):
    result = build_ultimate_intelligence(_bank(), out_dir=tmp_path)
    meta = result["meta_label"]
    assert meta["status"] == "EXECUTED_PRIOR_ONLY_INDIVIDUAL_PREDICTION_META_LABEL"
    assert meta["threshold"] == 0.60
    assert 0.0 <= meta["locked_summary"]["coverage"] <= 1.0
    rows = result["prediction_ledger"]["row_level"]
    assert len(rows) == 36
    assert all(0.0 <= float(row["meta_label_probability"]) <= 1.0 for row in rows)

    altered = _bank()
    altered[5]["y"] = np.ones(6, dtype=int)
    changed = build_ultimate_intelligence(altered, out_dir=tmp_path / "altered")
    for a, b in zip(result["fold_results"][:-1], changed["fold_results"][:-1]):
        assert a["meta_label"]["scores"] == b["meta_label"]["scores"]


def test_v13_prediction_history_and_strategy_failure_are_prior_only(tmp_path):
    result = build_ultimate_intelligence(_bank(), out_dir=tmp_path)
    history = result["prediction_history"]
    strategy = result["strategy_failure"]
    assert history["status"] == "EXECUTED_PRIOR_ONLY_HISTORY_RETRIEVAL"
    assert history["current_fold_outcomes_excluded"] is True
    assert strategy["status"] == "EXECUTED_PRIOR_ONLY_STRATEGY_FAILURE_MEMORY"
    assert strategy["current_fold_outcomes_excluded"] is True
    assert len(history["rows"]) == 36
    assert len(strategy["rows"]) == 36

    altered = _bank()
    altered[5]["y"] = np.ones(6, dtype=int)
    changed = build_ultimate_intelligence(altered, out_dir=tmp_path / "altered")
    for a, b in zip(
        result["prediction_history"]["rows"][:-6],
        changed["prediction_history"]["rows"][:-6],
    ):
        for key in ("fold", "row", "regime", "strategy", "status", "hits"):
            assert a[key] == b[key]
        for key in ("failure_rate", "success_probability", "best_distance"):
            av, bv = float(a.get(key, float("nan"))), float(b.get(key, float("nan")))
            assert np.isclose(av, bv, equal_nan=True)
    for a, b in zip(
        result["strategy_failure"]["rows"][:-6],
        changed["strategy_failure"]["rows"][:-6],
    ):
        for key in ("fold", "row", "regime", "strategy"):
            assert a[key] == b[key]
        for key in ("prior_failure_rate_raw", "prior_failure_rate_smoothed"):
            assert np.isclose(
                float(a[key]), float(b[key]), equal_nan=True
            )
    assert all("historical_retrieval" in row for row in result["prediction_ledger"]["row_level"])
    assert (tmp_path / "prediction_history.json").exists()
    assert (tmp_path / "strategy_failure.json").exists()
