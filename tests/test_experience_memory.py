from pathlib import Path

import numpy as np
import pandas as pd

from src.research.experience_memory import compact_view, update_experience_memory


def _write_prediction(path: Path, p: float, expected: float, realized: float) -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "asset_class": ["JP", "US"],
            "session_date": ["2026-09-20", "2026-09-20"],
            "prediction_time": ["2026-09-19T10:00:00Z", "2026-09-19T10:00:00Z"],
            "p_up_1d": [p, 1 - p],
            "expected_return_1d": [expected, -expected],
            "model_id": ["model_a", "model_b"],
            "regime": ["trend", "range"],
            "market_situation": ["normal", "high_vol"],
            "model_disagreement": [0.01, 0.07],
            "model_version": ["v1", "v1"],
            "forward_return_1d": [realized, -realized],
        }
    )
    frame.to_parquet(path, index=False)


def test_experience_memory_accumulates_and_is_idempotent(tmp_path: Path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    path = pred_dir / "prediction_20260920T100000Z.parquet"
    _write_prediction(path, 0.8, 0.03, 0.01)

    observed = pd.read_parquet(path)
    observed["_prediction_file"] = path.name
    memory_path = tmp_path / "experience_memory.json"

    first = update_experience_memory(observed, pred_dir, memory_path)
    assert first["total_resolved"] == 2
    assert first["total_files_processed"] == 1

    second = update_experience_memory(observed, pred_dir, memory_path)
    assert second["total_resolved"] == 2
    assert len(second["hard_cases"]) == 2
    assert compact_view(second)["total_resolved"] == 2


def test_changed_processed_file_fails_closed(tmp_path: Path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    path = pred_dir / "prediction_20260920T100000Z.parquet"
    _write_prediction(path, 0.8, 0.03, 0.01)

    observed = pd.read_parquet(path)
    observed["_prediction_file"] = path.name
    memory_path = tmp_path / "experience_memory.json"

    first = update_experience_memory(observed, pred_dir, memory_path)
    assert first["total_resolved"] == 2

    _write_prediction(path, 0.9, 0.05, 0.01)
    observed2 = pd.read_parquet(path)
    observed2["_prediction_file"] = path.name
    second = update_experience_memory(observed2, pred_dir, memory_path)
    assert second["total_resolved"] == 2
    assert second["anomalies"][-1]["message"].startswith(
        "prediction file changed after processing"
    )



def test_incomplete_file_is_deferred_until_next_monitor_cycle(tmp_path: Path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    path = pred_dir / "prediction_20260920T100000Z.parquet"
    _write_prediction(path, 0.8, 0.03, 0.01)

    observed = pd.read_parquet(path)
    observed["_prediction_file"] = path.name
    memory_path = tmp_path / "experience_memory.json"

    deferred = update_experience_memory(
        observed,
        pred_dir,
        memory_path,
        file_complete={path.name: False},
    )
    assert deferred["total_resolved"] == 0
    assert deferred["total_files_processed"] == 0
    assert deferred["anomalies"][-1]["message"].endswith(
        "prediction_20260920T100000Z.parquet"
    )

    completed = update_experience_memory(
        observed,
        pred_dir,
        memory_path,
        file_complete={path.name: True},
    )
    assert completed["total_resolved"] == 2
    assert completed["total_files_processed"] == 1


def test_experience_memory_tracks_error_taxonomy_and_rolling_metrics(tmp_path: Path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    path = pred_dir / "prediction_20260920T100000Z.parquet"
    _write_prediction(path, 0.99, 0.20, -0.20)

    observed = pd.read_parquet(path)
    observed["_prediction_file"] = path.name
    memory_path = tmp_path / "experience_memory.json"

    memory = update_experience_memory(observed, pred_dir, memory_path)

    assert memory["version"] == 2
    assert memory["total_resolved"] == 2
    assert memory["error_types"]["high_confidence_wrong"]["n"] == 1
    assert memory["by_error_bucket"][">=20%"]["n"] == 2
    assert memory["daily"]["2026-09-20"]["n"] == 2
    assert memory["daily_by_model"]["2026-09-20"]["model_a"]["n"] == 1
    assert memory["rolling"]["5_sessions"]["n"] == 2
    assert compact_view(memory)["by_direction_confidence"]["p>=0.75"]["n"] == 1


def test_v1_memory_is_migrated_without_losing_core_history(tmp_path: Path):
    memory_path = tmp_path / "experience_memory.json"
    legacy = {
        "version": 1,
        "updated_at": "2026-09-20T00:00:00Z",
        "total_resolved": 7,
        "total_files_processed": 3,
        "by_asset_class": {},
        "by_model_id": {},
        "by_regime": {},
        "by_market_situation": {},
        "by_direction_confidence": {},
        "by_model_disagreement": {},
        "hard_cases": [],
        "processed_prediction_files": {},
        "anomalies": [],
        "research_priority": [],
    }
    memory_path.write_text(__import__("json").dumps(legacy), encoding="utf-8")
    observed = pd.DataFrame(
        columns=[
            "_prediction_file",
            "forward_return_1d",
            "p_up_1d",
            "expected_return_1d",
        ]
    )
    memory = update_experience_memory(
        observed,
        tmp_path / "predictions",
        memory_path,
    )
    assert memory["version"] == 2
    assert memory["total_resolved"] == 7


def test_compact_view_includes_latest_session_breakdown(tmp_path: Path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    path = pred_dir / "prediction_20260921T100000Z.parquet"
    _write_prediction(path, 0.8, 0.03, 0.01)

    observed = pd.read_parquet(path)
    observed["_prediction_file"] = path.name
    memory_path = tmp_path / "experience_memory.json"
    memory = update_experience_memory(observed, pred_dir, memory_path)

    latest = compact_view(memory)["latest_session"]
    assert latest["session_date"] == "2026-09-20"
    assert latest["metrics"]["n"] == 2
    assert "model_a" in latest["by_model"]


def test_repeated_no_outcome_update_does_not_change_memory_timestamp(tmp_path: Path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    path = pred_dir / "prediction_20260920T100000Z.parquet"
    _write_prediction(path, 0.8, 0.03, 0.01)

    observed = pd.read_parquet(path)
    observed["_prediction_file"] = path.name
    memory_path = tmp_path / "experience_memory.json"

    first = update_experience_memory(observed, pred_dir, memory_path)
    before = memory_path.read_text(encoding="utf-8")
    second = update_experience_memory(observed, pred_dir, memory_path)
    after = memory_path.read_text(encoding="utf-8")

    assert first["total_resolved"] == 2
    assert second["total_resolved"] == 2
    assert before == after
