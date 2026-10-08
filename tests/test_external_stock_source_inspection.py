from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path("scripts/inspect_external_stock_research_queue.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("external_stock_inspection", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_inspection_never_executes_external_code():
    module = _load_module()
    queue = {
        "status": "EXECUTED_RESEARCH_QUEUE",
        "candidates": [
            {
                "canonical_ref": "example/research",
                "source_ranks": [1],
                "tier": "DEEP_LOCAL_REPRODUCTION",
                "lane": "DIRECT_PREDICTIVE",
                "discovery_priority_score": 99.0,
                "metadata": {"default_branch": "main"},
            }
        ],
    }

    calls = []

    def fake_fetch(repo, branch, path, token):
        calls.append((repo, branch, path))
        if path == "README.md":
            return "# forecasting\nquantile calibration\n", "ok"
        return None, "http_404"

    original = module._fetch_text
    original_latest = module._latest_commit
    module._fetch_text = fake_fetch
    module._latest_commit = lambda repo, branch, token: ("abc123", "ok")
    try:
        result = module.inspect(queue, 1, None)
    finally:
        module._fetch_text = original
        module._latest_commit = original_latest

    assert result["status"] == "EXECUTED_STATIC_SOURCE_INSPECTION"
    assert result["selected_candidate_count"] == 1
    assert result["source_inspected_count"] == 1
    row = result["candidates"][0]
    assert row["source_inspection_status"] == "SOURCE_CODE_INSPECTED"
    assert row["external_code_executed"] is False
    assert row["external_code_installed"] is False
    assert "forecast" in row["mechanism_terms_found"]
    assert row["predictive_performance_evidence"] == "NONE"
    assert row["repository_commit_sha"] == "abc123"
    assert row["content_ref"] == "abc123"
    assert row["promotion_allowed"] is False
    assert len(calls) == len(module.PATHS)


def test_inspection_queue_is_bounded():
    module = _load_module()
    queue = {
        "status": "EXECUTED_RESEARCH_QUEUE",
        "candidates": [
            {
                "canonical_ref": f"example/r{i}",
                "metadata": {"default_branch": "main"},
            }
            for i in range(5)
        ],
    }

    original = module._fetch_text
    module._fetch_text = lambda *args: (None, "http_404")
    try:
        result = module.inspect(queue, 2, None)
    finally:
        module._fetch_text = original

    assert result["selected_candidate_count"] == 2
    assert result["unverifiable_source_count"] == 2
    assert all(x["static_only"] is True for x in result["candidates"])
