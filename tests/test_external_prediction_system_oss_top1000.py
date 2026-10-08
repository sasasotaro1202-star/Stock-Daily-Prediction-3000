from __future__ import annotations

import json
from pathlib import Path


CONFIG = Path("config/external_prediction_system_oss_top1000_v2.json")


def test_top1000_scope_contract():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert data["source_scope"]["count_claimed"] == 1000
    assert data["source_scope"]["ranking_basis"]
    assert data["integrity_policy"]["preserve_source_rank"] is True
    assert data["integrity_policy"]["preserve_duplicates"] is True
    assert data["integrity_policy"]["normalize_separately"] is True
    assert data["integrity_policy"]["external_star_ranking_not_performance_evidence"] is True


def test_taxonomy_covers_prediction_system_stack():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    required = {
        "DIRECT_PREDICTIVE",
        "DATA_PIT",
        "UNCERTAINTY_FAILURE",
        "RESEARCH_AUTOMATION",
        "PRODUCTION_MLOPS",
        "ANALYTICS_UI",
        "INFRASTRUCTURE",
    }
    assert required.issubset(data["taxonomy"])
    assert len(data["deep_research_seed"]) >= 40


def test_seed_contract_is_never_production():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    for row in data["deep_research_seed"]:
        assert row["repo"]
        assert row["lane"]
        assert row["reason"]
