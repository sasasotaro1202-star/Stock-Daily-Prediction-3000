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


