from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path("config/external_stock_mechanism_research.json")
OUTPUT = Path("data/research/external_stock_mechanism_plan.json")


LANE_WEIGHT = {
    "DIRECT_PREDICTIVE": 1.00,
    "DATA_PIT": 0.95,
    "INFORMATION_RETRIEVAL": 0.80,
    "PROVENANCE_MEMORY": 0.65,
    "RESEARCH_AUTOMATION": 0.60,
    "ENGINEERING_VERIFICATION": 0.50,
}
RISK_PENALTY = {"LOW": 0.00, "MEDIUM": 0.08, "HIGH": 0.18, "CRITICAL": 0.30, "UNKNOWN": 0.22}


def _priority(row: dict) -> float:
    relevance = float(row["relevance"]) / 5.0
    lane = LANE_WEIGHT.get(row["lane"], 0.50)
    risk = sum(RISK_PENALTY.get(str(row.get(key, "UNKNOWN")), 0.22) for key in ("pit_risk", "cost_risk", "security_risk")) / 3.0
    complexity = min(1.0, float(row.get("complexity", 5)) / 5.0)
    return round(100.0 * (0.60 * relevance + 0.25 * lane + 0.15 * (1.0 - risk) - 0.05 * complexity), 3)


def main() -> int:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("FAIL: external mechanism research config has no candidates")

    seen: set[str] = set()
    plan_rows = []
    for row in rows:
        repo = str(row.get("repo", "")).strip()
        if repo in seen:
            raise SystemExit(f"FAIL: duplicate mechanism candidate: {repo}")
        seen.add(repo)
        score = _priority(row)
        if score >= 75:
            tier = "DEEP_OOS_CANDIDATE"
        elif score >= 60:
            tier = "TARGETED_RESEARCH"
        else:
            tier = "WATCH"
        plan_rows.append(
            {
                **row,
                "research_priority_score": score,
                "tier": tier,
                "evidence_level": "E1_EXTERNAL_CLAIM",
                "research_only": True,
                "production_changed": False,
                "promotion_allowed": False,
                "frozen_holdout_used": False,
                "required_protocol": [
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
                ],
            }
        )

    plan_rows.sort(key=lambda x: (-x["research_priority_score"], x["rank"]))
    result = {
        "schema_version": 1,
        "plan_id": payload["plan_id"],
        "status": "EXECUTED_RESEARCH_PLAN",
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "frozen_holdout_used": False,
        "selection_source": "heuristic_mechanism_relevance_not_performance",
        "candidate_count": len(plan_rows),
        "deep_oos_candidate_count": sum(x["tier"] == "DEEP_OOS_CANDIDATE" for x in plan_rows),
        "targeted_research_count": sum(x["tier"] == "TARGETED_RESEARCH" for x in plan_rows),
        "watch_count": sum(x["tier"] == "WATCH" for x in plan_rows),
        "lane_counts": {},
        "candidates": plan_rows,
    }
    lane_counts = result["lane_counts"]
    for row in plan_rows:
        lane_counts[row["lane"]] = lane_counts.get(row["lane"], 0) + 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "candidate_count": result["candidate_count"],
        "deep_oos_candidate_count": result["deep_oos_candidate_count"],
        "targeted_research_count": result["targeted_research_count"],
        "watch_count": result["watch_count"],
        "promotion_allowed": result["promotion_allowed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
