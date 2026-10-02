from __future__ import annotations

from scripts.performance_change import build_snapshot, compare_snapshots


def test_build_snapshot_extracts_monitor_metrics_and_reporting_context():
    payload = {
        "status": "PASS",
        "evaluated": 400,
        "evaluation_start_date": "2026-09-01",
        "latest_outcome_date": "2026-09-25",
        "champion": "hgb-v3",
        "prior_champion": "logistic-v2",
        "candidate": "lgbm-v1",
        "decision": "HOLD",
        "production_status": "PRODUCTION",
        "pit_status": "PASS",
        "latest_holdout": {"status": "VERIFIED"},
        "robustness": {"status": "VERIFIED"},
        "overall": {
            "rows": 400,
            "metrics": {
                "logloss": 0.70,
                "brier": 0.24,
                "ece": 0.08,
                "accuracy": 0.66,
            },
            "return_mae": 0.031,
        },
        "recent_20_sessions": {
            "rows": 250,
            "metrics": {
                "logloss": 0.75,
                "brier": 0.26,
                "ece": 0.10,
                "accuracy": 0.64,
                "roc_auc": 0.70,
            },
            "return_rmse": 0.055,
        },
    }
    snapshot = build_snapshot(payload)
    assert snapshot["status"] == "PASS"
    assert snapshot["evaluated"] == 400
    assert snapshot["evaluation_period"] == {
        "start": "2026-09-01",
        "end": "2026-09-25",
    }
    assert snapshot["sample_size"]["evaluated"] == 400
    assert snapshot["context"]["champion"] == "hgb-v3"
    assert snapshot["context"]["prior_champion"] == "logistic-v2"
    assert snapshot["context"]["candidate"] == "lgbm-v1"
    assert snapshot["context"]["decision"] == "HOLD"
    assert snapshot["context"]["pit_status"] == "PASS"
    assert snapshot["scopes"]["overall"]["logloss"] == 0.70
    assert snapshot["scopes"]["overall"]["return_mae"] == 0.031
    assert snapshot["scopes"]["recent_20_sessions"]["roc_auc"] == 0.70
    assert snapshot["scopes"]["recent_20_sessions"]["return_rmse"] == 0.055


def test_compare_snapshots_reports_directional_changes():
    previous = {
        "scopes": {
            "overall": {
                "rows": 400,
                "logloss": 0.70,
                "brier": 0.24,
                "ece": 0.08,
                "accuracy": 0.66,
            },
            "recent_20_sessions": {
                "rows": 250,
                "logloss": 0.75,
                "accuracy": 0.64,
            },
        }
    }
    current = {
        "scopes": {
            "overall": {
                "rows": 400,
                "logloss": 0.69,
                "brier": 0.241,
                "ece": 0.09,
                "accuracy": 0.67,
            },
            "recent_20_sessions": {
                "rows": 250,
                "logloss": 0.77,
                "accuracy": 0.63,
            },
        }
    }
    changes = compare_snapshots(previous, current)
    assert len(changes) == 6
    assert {
        (row["scope"], row["metric"], row["direction"])
        for row in changes
    } == {
        ("overall", "logloss", "improved"),
        ("overall", "brier", "worsened"),
        ("overall", "ece", "worsened"),
        ("overall", "accuracy", "improved"),
        ("recent_20_sessions", "logloss", "worsened"),
        ("recent_20_sessions", "accuracy", "worsened"),
    }


def test_compare_snapshots_ignores_non_numeric_or_unchanged_values():
    previous = {
        "scopes": {
            "overall": {
                "logloss": 0.70,
                "ece": None,
                "accuracy": 0.66,
            }
        }
    }
    current = {
        "scopes": {
            "overall": {
                "logloss": 0.70,
                "ece": 0.08,
                "accuracy": 0.66,
            }
        }
    }
    assert compare_snapshots(previous, current) == []


def test_reporting_context_is_explicit_when_monitor_does_not_provide_it():
    snapshot = build_snapshot({"status": "WARMUP", "evaluated": 0})
    assert snapshot["context"]["champion"] is None
    assert snapshot["context"]["prior_champion"] is None
    assert snapshot["context"]["candidate"] is None
    assert snapshot["context"]["decision"] is None
    assert snapshot["context"]["production_status"] is None
    assert snapshot["context"]["pit_status"] is None
