from __future__ import annotations

from scripts.performance_change import build_snapshot, compare_snapshots


def test_build_snapshot_extracts_monitor_metrics():
    payload = {
        "status": "PASS",
        "evaluated": 400,
        "latest_outcome_date": "2026-09-25",
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
    assert snapshot["scopes"]["overall"]["logloss"] == 0.70
    assert snapshot["scopes"]["overall"]["return_mae"] == 0.031
    assert snapshot["scopes"]["recent_20_sessions"]["roc_auc"] == 0.70
    assert snapshot["scopes"]["recent_20_sessions"]["return_rmse"] == 0.055


def test_compare_snapshots_reports_directional_changes():
    previous = {
        "scopes": {
            "overall": {
                "logloss": 0.70,
                "brier": 0.24,
                "ece": 0.08,
                "accuracy": 0.66,
            },
            "recent_20_sessions": {
                "logloss": 0.75,
                "accuracy": 0.64,
            },
        }
    }
    current = {
        "scopes": {
            "overall": {
                "logloss": 0.69,
                "brier": 0.241,
                "ece": 0.09,
                "accuracy": 0.67,
            },
            "recent_20_sessions": {
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
