from __future__ import annotations

from scripts.run_daily_research import aggregate_group


def test_aggregate_group_records_exact_oos_fold_identity():
    rows = [
        {"fold": 0.0, "test_start_date": "2025-01-01", "test_end_date": "2025-01-21", "n_test": 100.0, "logloss": 0.70},
        {"fold": 1.0, "test_start_date": "2025-01-22", "test_end_date": "2025-02-11", "n_test": 120.0, "logloss": 0.68},
        {"fold": 3.0, "test_start_date": "2025-02-13", "test_end_date": "2025-03-05", "n_test": 110.0, "logloss": 0.69},
    ]

    result = aggregate_group(rows)

    assert result["oos_fold_count"] == 3.0
    assert result["oos_fold_signature"].startswith("oos:")


def test_aggregate_group_fold_signature_changes_when_fold_set_changes():
    base = aggregate_group([
        {"fold": 0.0, "test_start_date": "2025-01-01", "test_end_date": "2025-01-21", "n_test": 100.0, "logloss": 0.70},
        {"fold": 1.0, "test_start_date": "2025-01-22", "test_end_date": "2025-02-11", "n_test": 120.0, "logloss": 0.68},
    ])
    changed = aggregate_group([
        {"fold": 0.0, "test_start_date": "2025-01-01", "test_end_date": "2025-01-21", "n_test": 100.0, "logloss": 0.70},
        {"fold": 1.0, "test_start_date": "2025-01-23", "test_end_date": "2025-02-12", "n_test": 120.0, "logloss": 0.68},
    ])

    assert base["oos_fold_signature"] != changed["oos_fold_signature"]


def test_aggregate_group_signature_changes_when_same_fold_ids_use_different_dates():
    first = aggregate_group([
        {"fold": 0.0, "test_start_date": "2025-01-01", "test_end_date": "2025-01-21", "n_test": 100.0, "logloss": 0.70},
        {"fold": 1.0, "test_start_date": "2025-01-22", "test_end_date": "2025-02-11", "n_test": 120.0, "logloss": 0.68},
    ])
    second = aggregate_group([
        {"fold": 0.0, "test_start_date": "2025-03-01", "test_end_date": "2025-03-21", "n_test": 100.0, "logloss": 0.70},
        {"fold": 1.0, "test_start_date": "2025-03-22", "test_end_date": "2025-04-11", "n_test": 120.0, "logloss": 0.68},
    ])

    assert first["oos_fold_signature"] != second["oos_fold_signature"]


def test_aggregate_group_missing_fold_date_boundary_fails_closed_signature():
    result = aggregate_group([
        {"fold": 0.0, "n_test": 100.0, "logloss": 0.70},
        {"fold": 1.0, "test_start_date": "2025-01-22", "test_end_date": "2025-02-11", "n_test": 120.0, "logloss": 0.68},
    ])
    assert result["oos_fold_signature"] is None
