from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path("scripts/update_external_research_candidates.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("external_candidates", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_is_research_only():
    data = json.loads(
        Path("config/external_research_candidates.json").read_text(encoding="utf-8")
    )
    assert data["evidence_contract"]["evidence_level"] == "E1"
    assert data["evidence_contract"]["research_only"] is True
    assert data["evidence_contract"]["production_changed"] is False
    assert data["evidence_contract"]["promotion_allowed"] is False
    assert all(row["decision"] == "HOLD_RESEARCH_ONLY" for row in data["candidates"])


def test_registry_contains_100_ranked_rows_and_expected_duplicate():
    data = json.loads(
        Path("config/external_research_candidates.json").read_text(encoding="utf-8")
    )
    rows = data["candidates"]
    assert len(rows) == 100
    assert [row["rank"] for row in rows] == list(range(1, 101))
    repos = [row["repo"] for row in rows]
    assert repos.count("mattpocock/skills") == 2


def test_activity_status_boundaries():
    module = _load_module()
    assert module._activity_status(0) == "RECENT_0_30D"
    assert module._activity_status(30) == "RECENT_0_30D"
    assert module._activity_status(30.1) == "ACTIVE_31_90D"
    assert module._activity_status(90) == "ACTIVE_31_90D"
    assert module._activity_status(90.1) == "AGING_91_180D"
    assert module._activity_status(180) == "AGING_91_180D"
    assert module._activity_status(180.1) == "STALE_180D_PLUS"
    assert module._activity_status(None) == "UNKNOWN"


def test_invalid_candidate_cannot_be_treated_as_safe():
    module = _load_module()
    errors = module._validate_candidate(
        {"rank": 1, "repo": "not-a-repo", "initial_score": 99, "decision": "ADOPTED"}
    )
    assert "invalid_repo_ref" in errors
    assert "unsafe_initial_decision" in errors
