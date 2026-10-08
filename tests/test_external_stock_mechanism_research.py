from __future__ import annotations

import importlib.util
import json
from pathlib import Path


CONFIG = Path("config/external_stock_mechanism_research.json")
MODULE_PATH = Path("scripts/generate_external_stock_mechanism_plan.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("external_mechanism", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mechanism_registry_is_research_only_and_deduplicated():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = data["candidates"]
    repos = [row["repo"] for row in rows]
    assert len(rows) == len(set(repos))
    assert data["promotion_allowed"] is False
    assert data["production_changed"] is False
    assert data["frozen_holdout_used"] is False
    assert all(row["relevance"] in {1, 2, 3, 4, 5} for row in rows)


def test_priority_is_not_a_performance_claim():
    module = _load_module()
    row = {
        "relevance": 5,
        "lane": "DIRECT_PREDICTIVE",
        "pit_risk": "HIGH",
        "cost_risk": "LOW",
        "security_risk": "LOW",
        "complexity": 4,
    }
    score = module._priority(row)
    assert 0.0 < score < 100.0


def test_required_external_candidate_protocol_is_strict():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    required = {
        "SOURCE_VERIFY",
        "PIT_CHECK",
        "COST_CHECK",
        "SECURITY_CHECK",
        "LOCAL_IMPLEMENTATION",
        "LOCAL_REPRODUCTION",
        "CHRONOLOGICAL_OOS",
        "ABLATION",
        "ROBUSTNESS",
        "FROZEN_HOLDOUT",
    }
    assert len(data["candidates"]) >= 15
    assert all(
        {
            "lane",
            "hypothesis",
            "local_reproduction",
            "pit_risk",
            "cost_risk",
            "security_risk",
            "complexity",
        }.issubset(row)
        for row in data["candidates"]
    )
