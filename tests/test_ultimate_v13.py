from __future__ import annotations

from src.research.case_risk_oos import analyze_case_risk
from src.research.v13_governance import (
    experiment_record,
    safety_governance,
    validate_experiment_registry,
)
from src.research.ultimate_v13_extensions import _active_information_contract


def test_experiment_record_is_deterministically_hashed():
    a = experiment_record(
        experiment_id="v13-test-001",
        hypothesis="prior-only retrieval is causal",
        commit="abc",
        dataset="synthetic",
        feature_version="f1",
        model_version="m1",
        parameters={"k": 10},
        train_period="2025",
        validation_period="2026Q1",
        oos_period="2026Q2",
    )
    b = experiment_record(
        experiment_id="v13-test-001",
        hypothesis="prior-only retrieval is causal",
        commit="abc",
        dataset="synthetic",
        feature_version="f1",
        model_version="m1",
        parameters={"k": 10},
        train_period="2025",
        validation_period="2026Q1",
        oos_period="2026Q2",
    )
    assert a == b
    assert len(a["record_sha256"]) == 64


def test_governance_blocks_when_pit_is_blocked():
    out = safety_governance(
        promotion_allowed=False,
        pit_status="BLOCKED_NO_FULL_TIMESTAMP_LINEAGE",
        leakage_status="PASS",
        meta_leakage_status="PASS",
        robustness_status="PASS",
        reproducibility_status="PASS",
        production_changed=False,
    )
    assert out["kill_switch_engaged"] is True
    assert out["action"] == "BLOCK_NEW_V13_PATH"
    assert out["fallback_target"] == "verified_baseline"
    assert out["rollback_target"] == "previous_verified"
    assert out["promotion_allowed"] is False


def test_registry_validation_rejects_duplicate_ids():
    row = experiment_record(
        experiment_id="dup",
        hypothesis="x",
        commit="c",
        dataset="d",
        feature_version="f",
        model_version="m",
        parameters={},
        train_period="t",
        validation_period="v",
        oos_period="o",
    )
    assert validate_experiment_registry([row, row])["status"] == "FAIL"
import json

import numpy as np

from src.research.ultimate_v13 import build_ultimate_intelligence


def _bank(folds: int = 6, with_lineage: bool = False):
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
        if with_lineage:
            out[fold].update({
                "pit_lineage_policy_version": "historical_scheduled_prediction_clock_v1",
                "prediction_time": np.array(
                    [f"2026-01-{fold + 1:02d}T09:17:00+00:00"] * len(y),
                    dtype=object,
                ),
                "prediction_cutoff": np.array(
                    [f"2026-01-{fold + 1:02d}T09:17:00+00:00"] * len(y),
                    dtype=object,
                ),
                "prediction_time_source": np.array(
                    ["DECLARED_CONFIG_SCHEDULE:asia_prediction_time_jst"] * len(y),
                    dtype=object,
                ),
                "prediction_time_observed": np.array([False] * len(y), dtype=bool),
                "available_at": np.array(
                    [f"2026-01-{fold + 1:02d}T07:00:00+00:00"] * len(y),
                    dtype=object,
                ),
                "published_at": np.array(
                    [f"2026-01-{fold + 1:02d}T07:30:00+00:00"] * len(y),
                    dtype=object,
                ),
                "retrieved_at": np.array(
                    [f"2026-01-{fold + 1:02d}T12:00:00+00:00"] * len(y),
                    dtype=object,
                ),
                "available_at_method": np.array(
                    ["conservative_post_close_inferred"] * len(y),
                    dtype=object,
                ),
                "source": np.array(["yfinance"] * len(y), dtype=object),
                "provider_symbol": np.array(["TEST"] * len(y), dtype=object),
                "retrieval_run_id": np.array([f"test-{fold}"] * len(y), dtype=object),
                "pit_status": np.array(["PASS"] * len(y), dtype=object),
                "lineage_sha256": np.array(["x" * 64] * len(y), dtype=object),
            })
    return out


def test_v13_embeds_expert_loss_oos_audit_and_artifact(tmp_path):
    result = build_ultimate_intelligence(
        _bank(with_lineage=True),
        out_dir=tmp_path,
    )
    audit = result["expert_loss_routing_oos"]
    assert audit["research_only"] is True
    assert audit["production_changed"] is False
    assert audit["promotion_allowed"] is False
    assert audit["frozen_holdout_used"] is False
    assert (tmp_path / "expert_loss_routing_oos.json").exists()
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert "expert_loss_routing_oos.json" in manifest["artifacts"]


def test_v13_ledger_carries_row_level_pit_lineage_and_unlocks_case_risk(tmp_path):
    result = build_ultimate_intelligence(
        _bank(with_lineage=True),
        out_dir=tmp_path,
    )
    rows = result["prediction_ledger"]["row_level"]
    assert len(rows) == 36
    assert all(row["pit_status"] == "PASS" for row in rows)
    assert all(row["prediction_time"] for row in rows)
    assert all(row["prediction_cutoff"] == row["prediction_time"] for row in rows)
    assert all(row["available_at"] for row in rows)
    assert all(row["published_at"] for row in rows)
    assert all(row["prediction_time_observed"] is False for row in rows)
    assert all(len(row["lineage_sha256"]) == 64 for row in rows)
    assert all(
        fold["prediction_timestamp_status"] == "PASS_SCHEDULED_POLICY"
        for fold in result["prediction_contracts"]["folds"]
    )
    assert result["prediction_contracts"]["scheduled_timestamp_lineage"] is True
    assert result["prediction_contracts"]["exact_prediction_time_observed"] is False
    assert result["audits"]["PIT"] == "PASS_DECLARED_SCHEDULE"
    assert result["audits"]["PIT_Row_Lineage"]["available_at_le_prediction_time"] is True

    case_risk = analyze_case_risk(rows)
    assert case_risk["status"] == "EVALUATED"
    assert case_risk["locked_rows"] == 12
    assert case_risk["scored_rows"] == 12


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


def test_v13_historical_all_missing_risk_is_safe(tmp_path):
    bank = _bank()
    for fold in bank:
        bank[fold]["risk_context"][:, 0] = np.nan

    result = build_ultimate_intelligence(bank, out_dir=tmp_path)
    assert result["status"] == "OOS_COMPLETE"
    for fold in result["fold_results"]:
        assert np.isfinite(float(fold["predictability_mean"]))
        assert np.isfinite(float(fold["ood_mean"]))
        assert np.isfinite(float(fold["routing"]["weight_concentration"]))


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
    assert all(row["pit_status"] != "PASS" for row in result["prediction_ledger"]["row_level"])
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


def test_active_information_requires_explicit_causal_timestamps():
    missing_timestamps = {
        0: {
            "information_candidates": [{
                "name": "missing",
                "pit_safe": True,
                "expected_logloss_reduction": 0.10,
                "cost": 0.01,
                "failure_risk": 0.01,
            }]
        }
    }
    assert _active_information_contract(missing_timestamps)["status"] == "BLOCKED_NO_PIT_SAFE_CANDIDATES"

    future_available = {
        0: {
            "information_candidates": [{
                "name": "future",
                "pit_safe": True,
                "available_at": "2026-01-02T09:10:00+09:00",
                "prediction_time": "2026-01-02T09:05:00+09:00",
                "expected_logloss_reduction": 0.10,
                "cost": 0.01,
                "failure_risk": 0.01,
            }]
        }
    }
    assert _active_information_contract(future_available)["status"] == "BLOCKED_NO_PIT_SAFE_CANDIDATES"

    valid = {
        0: {
            "information_candidates": [{
                "name": "valid",
                "pit_safe": True,
                "available_at": "2026-01-02T09:00:00+09:00",
                "prediction_time": "2026-01-02T09:05:00+09:00",
                "expected_logloss_reduction": 0.10,
                "cost": 0.01,
                "failure_risk": 0.01,
            }]
        }
    }
    out = _active_information_contract(valid)
    assert out["status"] == "EXECUTED_PIT_SAFE_METADATA_CONTRACT"
    assert out["accepted"][0]["available_at"] == "2026-01-02T09:00:00+09:00"
    assert out["accepted"][0]["prediction_time"] == "2026-01-02T09:05:00+09:00"


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
    for fold in result["fold_results"]:
        assert len(fold["calibrated_predictability"]) == 6
        assert len(fold["ood_by_case"]) == 6

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
            av = a.get(key)
            bv = b.get(key)
            if av is None or bv is None:
                assert av is None and bv is None
            else:
                assert np.isclose(float(av), float(bv), equal_nan=True)
    for a, b in zip(
        result["strategy_failure"]["rows"][:-6],
        changed["strategy_failure"]["rows"][:-6],
    ):
        for key in ("fold", "row", "regime", "strategy"):
            assert a[key] == b[key]
        for key in ("prior_failure_rate_raw", "prior_failure_rate_smoothed"):
            av, bv = a[key], b[key]
            if av is None or bv is None:
                assert av is None and bv is None
            else:
                assert np.isclose(float(av), float(bv), equal_nan=True)
    assert all("historical_retrieval" in row for row in result["prediction_ledger"]["row_level"])
    assert (tmp_path / "prediction_history.json").exists()
    assert (tmp_path / "strategy_failure.json").exists()

    same_fold = _bank()
    same_fold_changed = _bank()
    same_fold_changed[4]["y"][0] = 1 - same_fold_changed[4]["y"][0]
    base_h = build_ultimate_intelligence(same_fold, out_dir=tmp_path / "same_a")
    changed_h = build_ultimate_intelligence(same_fold_changed, out_dir=tmp_path / "same_b")
    base_hist = [
        r for r in base_h["prediction_history"]["rows"] if r["fold"] == 4 and r["row"] > 0
    ]
    changed_hist = [
        r for r in changed_h["prediction_history"]["rows"] if r["fold"] == 4 and r["row"] > 0
    ]
    assert base_hist == changed_hist
    base_strat = [
        r for r in base_h["strategy_failure"]["rows"] if r["fold"] == 4 and r["row"] > 0
    ]
    changed_strat = [
        r for r in changed_h["strategy_failure"]["rows"] if r["fold"] == 4 and r["row"] > 0
    ]
    for a, b in zip(base_strat, changed_strat):
        assert a["fold"] == b["fold"]
        assert a["row"] == b["row"]
        assert a["regime"] == b["regime"]
        assert a["strategy"] == b["strategy"]
        assert np.isclose(
            float(a["prior_failure_rate_raw"]),
            float(b["prior_failure_rate_raw"]),
            equal_nan=True,
        )
        assert np.isclose(
            float(a["prior_failure_rate_smoothed"]),
            float(b["prior_failure_rate_smoothed"]),
            equal_nan=True,
        )


def test_v13_build_emits_governance_and_experiment_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "test-sha")
    result = build_ultimate_intelligence(_bank(), out_dir=tmp_path)
    assert result["governance"]["status"] == "EXECUTED_RESEARCH_GOVERNANCE"
    assert result["governance"]["promotion_allowed"] is False
    assert result["governance"]["rollback_target"] == "previous_verified"
    registry = result["experiment_registry"]
    assert registry["status"] == "EXECUTED_SINGLE_EXPERIMENT_RECORD"
    assert registry["validation"]["status"] == "PASS"
    assert len(registry["rows"]) == 1
    assert len(registry["rows"][0]["record_sha256"]) == 64
    assert (tmp_path / "governance.json").exists()
    assert (tmp_path / "experiment_registry.json").exists()


def test_predictability_calibrator_is_frozen_before_locked_suffix(tmp_path):
    baseline = build_ultimate_intelligence(_bank(), out_dir=tmp_path / "a")
    altered = _bank()
    altered[5]["y"] = np.ones(6, dtype=int)
    changed = build_ultimate_intelligence(altered, out_dir=tmp_path / "b")

    cal_a = baseline["predictability"]["fold_metrics"]
    cal_b = changed["predictability"]["fold_metrics"]
    assert all(row["is_locked"] == 1.0 for row in cal_a[-2:])
    assert all(row["is_locked"] == 1.0 for row in cal_b[-2:])
    assert [row["calibrated_predictability_mean"] for row in cal_a[-2:]] == [row["calibrated_predictability_mean"] for row in cal_b[-2:]]
    assert [row["raw_predictability_mean"] for row in cal_a[-2:]] == [row["raw_predictability_mean"] for row in cal_b[-2:]]
    assert baseline["predictability"]["calibrator_frozen_before_locked"] is True
    assert baseline["predictability"]["locked_outcomes_update_calibrator"] is False


def test_predictability_calibration_remains_research_only(tmp_path):
    result = build_ultimate_intelligence(_bank(), out_dir=tmp_path)
    assert result["predictability"]["calibration_status"] == "EXECUTED_DEV_ONLY_FROZEN"
    assert result["production_changed"] is False
    assert result["promotion_allowed"] is False
    assert result["predictability"]["calibrator_frozen_before_locked"] is True


def test_v13_expert_loss_audit_fails_closed_without_pit_lineage(tmp_path):
    result = build_ultimate_intelligence(
        _bank(with_lineage=False),
        out_dir=tmp_path,
    )
    audit = result["expert_loss_routing_oos"]
    assert audit["status"] == "BLOCKED_INVALID_CASE_LINEAGE"
    assert audit["production_changed"] is False
    assert audit["promotion_allowed"] is False
    assert (tmp_path / "expert_loss_routing_oos.json").exists()
