from pathlib import Path

import json
import pandas as pd

from scripts.experience_review import main


def test_experience_review_warmup_is_safe(tmp_path: Path, monkeypatch):
    memory_path = tmp_path / "experience_memory.json"
    out_path = tmp_path / "experience_review.json"
    memory_path.write_text(
        json.dumps(
            {
                "version": 2,
                "updated_at": None,
                "total_resolved": 0,
                "total_files_processed": 0,
                "by_asset_class": {},
                "by_model_id": {},
                "by_regime": {},
                "by_market_situation": {},
                "by_direction_confidence": {},
                "by_model_disagreement": {},
                "by_error_bucket": {},
                "error_types": {},
                "daily": {},
                "daily_by_model": {},
                "hard_cases": [],
                "recent_hard_cases": [],
                "processed_prediction_files": {},
                "anomalies": [],
                "total_deferred_files": 0,
                "total_anomalies": 0,
                "research_priority": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.experience_review.MEMORY", memory_path)
    monkeypatch.setattr("scripts.experience_review.OUT", out_path)

    main()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "WARMUP"
    assert payload["total_resolved"] == 0


def test_review_script_does_not_require_prediction_data(tmp_path: Path, monkeypatch):
    memory_path = tmp_path / "experience_memory.json"
    out_path = tmp_path / "experience_review.json"
    memory_path.write_text(
        json.dumps(
            {
                "version": 2,
                "updated_at": "2026-09-20T00:00:00Z",
                "total_resolved": 3,
                "total_files_processed": 1,
                "by_asset_class": {},
                "by_model_id": {},
                "by_regime": {},
                "by_market_situation": {},
                "by_direction_confidence": {},
                "by_model_disagreement": {},
                "by_error_bucket": {},
                "error_types": {},
                "daily": {
                    "2026-09-20": {
                        "n": 3,
                        "correct": 2,
                        "logloss_sum": 0.5,
                        "brier_sum": 0.2,
                        "abs_return_error_sum": 0.03,
                        "squared_return_error_sum": 0.001,
                    }
                },
                "daily_by_model": {},
                "hard_cases": [],
                "recent_hard_cases": [],
                "processed_prediction_files": {},
                "anomalies": [],
                "total_deferred_files": 0,
                "total_anomalies": 0,
                "research_priority": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.experience_review.MEMORY", memory_path)
    monkeypatch.setattr("scripts.experience_review.OUT", out_path)

    main()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "READY"
    assert payload["rolling"]["5_sessions"]["n"] == 3
