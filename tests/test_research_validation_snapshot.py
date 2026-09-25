from __future__ import annotations

import json
from pathlib import Path

from scripts.persist_research_validation_snapshot import main


def test_research_snapshot_persists_core_scores(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data" / "research"
    data_dir.mkdir(parents=True)
    payload = {
        "status": "OOS_COMPLETE",
        "selected_model": "hgb",
        "results": {
            "hgb": {
                "metrics": {
                    "logloss": 0.61,
                    "brier": 0.21,
                    "ece": 0.04,
                    "accuracy": 0.68,
                    "rank_ic": 0.09,
                    "folds": 8,
                }
            },
            "logistic": {
                "metrics": {
                    "logloss": 0.64,
                    "brier": 0.22,
                    "ece": 0.05,
                    "accuracy": 0.65,
                }
            },
        },
        "return_oos": {
            "status": "OOS_COMPLETE",
            "metrics": {
                "mae": 0.012,
                "rmse": 0.019,
                "sign_accuracy": 0.57,
                "rank_ic": 0.03,
                "range_80_coverage": 0.81,
            },
        },
        "global_selection_evidence": {"status": "SUPPORTED"},
        "calibration_method": "beta",
    }
    (data_dir / "latest_metrics.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    main()

    out = json.loads(
        (tmp_path / "artifacts" / "research_validation_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert out["selected_model"] == "hgb"
    assert out["selected_model_score"]["logloss"] == 0.61
    assert out["selected_model_score"]["brier"] == 0.21
    assert out["selected_model_score"]["ece"] == 0.04
    assert out["selected_model_score"]["accuracy"] == 0.68
    assert out["candidate_scores"]["logistic"]["logloss"] == 0.64
    assert out["return_oos_score"]["mae"] == 0.012
    assert out["return_oos_score"]["range_80_coverage"] == 0.81
    assert out["global_selection_evidence"]["status"] == "SUPPORTED"
