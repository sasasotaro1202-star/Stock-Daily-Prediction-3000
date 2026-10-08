from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KEYWORDS = {
    "DIRECT_PREDICTIVE": (
        "stock", "trading", "finance", "financial", "quant", "forecast", "forecasting",
        "time series", "timeseries", "return", "price prediction", "market prediction",
        "portfolio", "factor", "alpha",
    ),
    "DATA_PIT": (
        "sec", "filing", "fundamental", "financial data", "market data", "data pipeline",
        "dataset", "provenance", "point in time", "pit", "availability",
    ),
    "UNCERTAINTY_ROBUSTNESS": (
        "uncertainty", "probabilistic", "probability", "calibration", "conformal",
        "anomaly", "outlier", "drift", "change point", "robust",
    ),
    "RESEARCH_AUTOMATION": (
        "experiment", "research", "autonomous", "agent", "benchmark", "evaluation",
        "workflow", "automation", "mlops",
    ),
    "INFRASTRUCTURE": (
        "duckdb", "polars", "pandas", "arrow", "kubernetes", "docker", "observability",
        "monitoring", "telemetry", "orchestration",
    ),
}

SEED_REPOS = {
    "microsoft/qlib",
    "google-research/timesfm",
    "shiyu-coder/Kronos",
    "karpathy/autoresearch",
    "openbq-org/OpenBB",
    "ZhuLinsen/daily_stock_analysis",
    "TauricResearch/TradingAgents",
    "freqtrade/freqtrade",
    "vnpy/vnpy",
    "Panniantong/Agent-Reach",
    "firecrawl/firecrawl",
    "microsoft/graphrag",
    "Graphify-Labs/graphify",
    "vectorize-io/hindsight",
    "mattpocock/skills",
    "obra/superpowers",
    "Fission-AI/OpenSpec",
    "tester-army/e2e",
    "666ghj/MiroFish",
    "mvanhorn/last30days-skill",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _text(row: dict[str, Any]) -> str:
    return f'{row.get("canonical_ref", "")} {row.get("source_entry", "")} {row.get("description", "")}'.casefold()


def _scores(text: str) -> dict[str, float]:
    hits: dict[str, int] = {}
    for lane, words in KEYWORDS.items():
        hits[lane] = sum(1 for word in words if re.search(r"\b" + re.escape(word) + r"\b", text))
    total = sum(hits.values())
    return {
        lane: round(min(1.0, count / 4.0), 3)
        for lane, count in hits.items()
    } | {"total_keyword_hits": total}


def _triage(row: dict[str, Any]) -> dict[str, Any]:
    text = _text(row)
    scores = _scores(text)
    rank = int(row.get("source_rank", 1000))
    rank_score = max(0.0, 1.0 - (rank - 1) / 999.0)
    active = bool(row.get("archived") is False and row.get("disabled") is False)
    activity_score = 1.0 if active else 0.0
    seed_boost = 0.20 if row.get("canonical_ref") in SEED_REPOS else 0.0
    domain = max(scores[lane] for lane in KEYWORDS)
    raw = 0.55 * domain + 0.20 * rank_score + 0.10 * activity_score + 0.15 * min(1.0, seed_boost / 0.20)
    score = round(100.0 * raw, 3)
    if score >= 70:
        tier = "DEEP_SOURCE_INSPECTION"
    elif score >= 45:
        tier = "TARGETED_SOURCE_INSPECTION"
    else:
        tier = "WATCH"
    best_lane = max(KEYWORDS, key=lambda lane: (scores[lane], -list(KEYWORDS).index(lane)))
    return {
        "source_rank": row["source_rank"],
        "source_entry": row["source_entry"],
        "canonical_ref": row["canonical_ref"],
        "metadata_status": row["status"],
        "stars": row.get("stars"),
        "forks": row.get("forks"),
        "archived": row.get("archived"),
        "disabled": row.get("disabled"),
        "default_branch": row.get("default_branch"),
        "license_spdx": row.get("license_spdx"),
        "description": row.get("description"),
        "triage_lane": best_lane,
        "keyword_scores": scores,
        "triage_score": score,
        "tier": tier,
        "is_seed_candidate": row.get("canonical_ref") in SEED_REPOS,
        "triage_is_not_performance_evidence": True,
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "frozen_holdout_used": False,
        "required_next_stage": [
            "SOURCE_CONTENT_INSPECTION",
            "METHOD_ABSTRACTION",
            "LICENSE_COST_SECURITY_CHECK",
            "PIT_FEASIBILITY_CHECK",
            "LOCAL_IMPLEMENTATION",
            "LOCAL_REPRODUCTION",
            "CHRONOLOGICAL_OOS",
            "ABLATION",
            "ROBUSTNESS",
            "FROZEN_HOLDOUT",
            "DECISION",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/external_oss_top1000_snapshot.json")
    parser.add_argument("--output", default="artifacts/external_oss_stock_research_queue.json")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = [r for r in data.get("results", []) if r.get("status") == "VERIFIED_SOURCE_METADATA"]
    if not rows:
        raise SystemExit("FAIL: no verified metadata rows available for research queue")

    queue = [_triage(row) for row in rows]
    queue.sort(key=lambda x: (-x["triage_score"], int(x["source_rank"])))

    result = {
        "schema_version": 1,
        "queue_type": "external_oss_stock_mechanism_triage",
        "observed_at": now_iso(),
        "input_snapshot": str(args.input),
        "candidate_count": len(queue),
        "deep_source_inspection_count": sum(x["tier"] == "DEEP_SOURCE_INSPECTION" for x in queue),
        "targeted_source_inspection_count": sum(x["tier"] == "TARGETED_SOURCE_INSPECTION" for x in queue),
        "watch_count": sum(x["tier"] == "WATCH" for x in queue),
        "triage_contract": {
            "popularity_is_not_performance_evidence": True,
            "source_metadata_is_discovery_evidence_only": True,
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
            "frozen_holdout_used": False,
        },
        "queue": queue,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "candidate_count": len(queue),
        "deep_source_inspection_count": result["deep_source_inspection_count"],
        "targeted_source_inspection_count": result["targeted_source_inspection_count"],
        "watch_count": result["watch_count"],
        "promotion_allowed": result["triage_contract"]["promotion_allowed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
