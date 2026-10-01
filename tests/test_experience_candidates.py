from __future__ import annotations

from src.research.experience_candidates import build_experience_candidate_plan


def test_candidate_plan_is_research_only_and_deterministic():
    memory = {
        "updated_at": "2026-09-30T00:00:00Z",
        "total_resolved": 100,
        "research_priority": [
            {
                "dimension": "error_types",
                "segment": "high_confidence_wrong",
                "n": 120,
                "logloss": 0.90,
                "accuracy": 0.42,
                "return_mae": 0.08,
                "impact_vs_global_logloss": 0.20,
            },
            {
                "dimension": "by_regime",
                "segment": "high_vol",
                "n": 80,
                "logloss": 0.75,
                "accuracy": 0.49,
                "return_mae": 0.05,
                "impact_vs_global_logloss": 0.10,
            },
        ],
    }

    first = build_experience_candidate_plan(memory)
    second = build_experience_candidate_plan(memory)

    assert first == second
    assert first["status"] == "READY"
    assert len(first["candidates"]) == 2
    assert first["safety_contract"]["research_only"] is True
    assert first["safety_contract"]["production_changed"] is False
    assert first["safety_contract"]["frozen_holdout_allowed"] is False
    assert first["candidates"][0]["action"] == "high_confidence_abstention_challenge"
    assert first["candidates"][1]["action"] == "regime_specific_model_challenge"


def test_candidate_plan_skips_weak_evidence_and_deduplicates():
    memory = {
        "total_resolved": 20,
        "research_priority": [
            {
                "dimension": "by_regime",
                "segment": "normal",
                "n": 10,
                "impact_vs_global_logloss": 0.9,
            },
            {
                "dimension": "by_regime",
                "segment": "improving",
                "n": 100,
                "impact_vs_global_logloss": -0.2,
            },
            {
                "dimension": "by_regime",
                "segment": "high_vol",
                "n": 40,
                "impact_vs_global_logloss": 0.1,
            },
            {
                "dimension": "by_regime",
                "segment": "high_vol",
                "n": 40,
                "impact_vs_global_logloss": 0.1,
            },
        ],
    }

    plan = build_experience_candidate_plan(memory)

    assert plan["status"] == "READY"
    assert len(plan["candidates"]) == 1
    assert plan["candidates"][0]["source_segment"] == "high_vol"


def test_empty_experience_is_safe_warmup():
    plan = build_experience_candidate_plan(
        {"updated_at": None, "total_resolved": 0, "research_priority": []}
    )

    assert plan["status"] == "WARMUP"
    assert plan["candidates"] == []
    assert plan["safety_contract"]["promotion_requires_chronological_oos"] is True
