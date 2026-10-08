from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path("config/external_prediction_system_oss_top1000_v2.json")
OUTPUT = Path("data/research/external_prediction_system_oss_research_plan.json")

LANE_SCORE = {
    "DIRECT_PREDICTIVE": 1.00,
    "DATA_PIT": 0.96,
    "UNCERTAINTY_FAILURE": 0.94,
    "RESEARCH_AUTOMATION": 0.72,
    "PRODUCTION_MLOPS": 0.70,
    "ANALYTICS_UI": 0.45,
    "INFRASTRUCTURE": 0.52,
}


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    seeds = cfg.get("deep_research_seed") or []
    if not seeds:
        raise SystemExit("FAIL: deep research seed is empty")

    rows = []
    for idx, row in enumerate(seeds, start=1):
        lane = str(row["lane"])
        relevance = LANE_SCORE.get(lane, 0.50)
        if lane in {"DIRECT_PREDICTIVE", "DATA_PIT", "UNCERTAINTY_FAILURE"}:
            tier = "A_DIRECT_OOS"
        elif lane in {"RESEARCH_AUTOMATION", "PRODUCTION_MLOPS"}:
            tier = "B_SYSTEM_RESEARCH"
        else:
            tier = "C_SUPPORTING_RESEARCH"
        rows.append(
            {
                "seed_rank": idx,
                "repo": row["repo"],
                "lane": lane,
                "reason": row["reason"],
                "research_priority": round(relevance, 3),
                "tier": tier,
                "evidence_level": "E1_EXTERNAL_CLAIM",
                "status": "HOLD_RESEARCH_ONLY",
                "research_only": True,
                "production_changed": False,
                "promotion_allowed": False,
                "frozen_holdout_used": False,
                "required_protocol": [
                    "SOURCE_VERIFY",
                    "PIT_CHECK",
                    "COST_CHECK",
                    "SECURITY_CHECK",
                    "LOCAL_REPRODUCTION",
                    "CHRONOLOGICAL_OOS",
                    "ABLATION",
                    "ROBUSTNESS",
                    "FROZEN_HOLDOUT",
                ],
            }
        )

    result = {
        "schema_version": 1,
        "catalog_id": cfg["catalog_id"],
        "source_scope": cfg["source_scope"],
        "status": "EXECUTED_RESEARCH_PLAN",
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "frozen_holdout_used": False,
        "top1000_source_is_not_performance_evidence": True,
        "deep_seed_count": len(rows),
        "tier_counts": {
            "A_DIRECT_OOS": sum(r["tier"] == "A_DIRECT_OOS" for r in rows),
            "B_SYSTEM_RESEARCH": sum(r["tier"] == "B_SYSTEM_RESEARCH" for r in rows),
            "C_SUPPORTING_RESEARCH": sum(r["tier"] == "C_SUPPORTING_RESEARCH" for r in rows),
        },
        "candidates": rows,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "deep_seed_count": result["deep_seed_count"],
        "tier_counts": result["tier_counts"],
        "promotion_allowed": result["promotion_allowed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
