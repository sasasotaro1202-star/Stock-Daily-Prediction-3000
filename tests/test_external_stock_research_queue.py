from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path("scripts/build_external_stock_research_queue.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("external_stock_queue", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_queue_uses_only_verified_metadata():
    module = _load_module()
    snapshot = {
        "source_claimed_count": 1000,
        "explicit_repo_count": 3,
        "results_count": 3,
        "verified_source_count": 2,
        "results": [
            {
                "source_rank": 10,
                "source_entry": "example/direct",
                "canonical_ref": "example/direct",
                "status": "VERIFIED_SOURCE_METADATA",
                "description": "forecasting time series",
                "pushed_at": "2026-10-01T00:00:00Z",
                "stars": 1,
            },
            {
                "source_rank": 20,
                "source_entry": "example/direct",
                "canonical_ref": "example/direct",
                "status": "VERIFIED_SOURCE_METADATA",
                "description": "forecasting",
                "pushed_at": "2026-10-02T00:00:00Z",
                "stars": 2,
            },
            {
                "source_rank": 30,
                "source_entry": "missing/repo",
                "canonical_ref": "missing/repo",
                "status": "SOURCE_NOT_FOUND",
                "description": "forecasting",
            },
        ],
    }
    mechanism = {
        "candidates": [
            {
                "repo": "example/direct",
                "lane": "DIRECT_PREDICTIVE",
                "relevance": 5,
                "hypothesis": "forecasting",
            }
        ]
    }
    result = module.build_queue(snapshot, mechanism)
    assert result["status"] == "EXECUTED_RESEARCH_QUEUE"
    assert result["unique_verified_repository_count"] == 1
    assert result["blocked_source_count"] == 1
    row = result["candidates"][0]
    assert row["canonical_ref"] == "example/direct"
    assert row["source_ranks"] == [10, 20]
    assert row["mechanism_seed"] is True
    assert row["lane"] == "DIRECT_PREDICTIVE"
    assert row["priority_is_not_predictive_evidence"] is True
    assert row["predictive_performance_evidence"] == "NONE_EXTERNAL_METADATA"
    assert row["promotion_allowed"] is False
    assert row["production_changed"] is False


def test_priority_is_discovery_only():
    module = _load_module()
    snapshot = {
        "source_claimed_count": 1000,
        "explicit_repo_count": 1,
        "results_count": 1,
        "verified_source_count": 1,
        "results": [
            {
                "source_rank": 1,
                "source_entry": "example/model",
                "canonical_ref": "example/model",
                "status": "VERIFIED_SOURCE_METADATA",
                "description": "lightgbm model",
                "pushed_at": "2026-10-08T00:00:00Z",
            }
        ],
    }
    result = module.build_queue(snapshot, {"candidates": []})
    row = result["candidates"][0]
    assert row["discovery_priority_score"] >= 0
    assert row["priority_is_not_predictive_evidence"] is True
    assert row["metadata"]["stars"] is None


def test_missing_live_snapshot_is_blocked_not_promoted(tmp_path):
    module = _load_module()
    output = tmp_path / "queue.json"
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--snapshot", str(tmp_path / "missing.json"), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    text = output.read_text(encoding="utf-8")
    assert '"status": "BLOCKED_NO_LIVE_SNAPSHOT"' in text
    assert '"promotion_allowed": false' in text
