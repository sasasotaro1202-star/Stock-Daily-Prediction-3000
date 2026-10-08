from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SNAPSHOT = Path("data/research/external_oss_top1000_snapshot.json")
DEFAULT_CONFIG = Path("config/external_stock_mechanism_research.json")
DEFAULT_OUTPUT = Path("data/research/external_stock_research_queue.json")

LANE_RULES = {
    "DIRECT_PREDICTIVE": {
        "weight": 1.00,
        "keywords": (
            "qlib", "timesfm", "kronos", "darts", "statsforecast",
            "neuralforecast", "mlforecast", "sktime", "aeon",
            "statsmodels", "gluonts", "pmdarima", "prophet", "greykite",
            "orbit", "lag-llama", "chronos", "moirai", "patchtst",
            "autoformer", "timesnet", "n-beats", "n-its", "tabpfn",
            "tabtransformer", "ft-transformer", "forecast", "time series",
            "forecasting",
        ),
    },
    "DATA_PIT": {
        "weight": 0.96,
        "keywords": (
            "openbb", "yfinance", "datareader", "market data", "financial data",
            "sec", "filing", "fundamental", "timescale", "influxdb",
            "duckdb", "arrow", "parquet", "polars", "pandas",
        ),
    },
    "UNCERTAINTY_FAILURE": {
        "weight": 0.94,
        "keywords": (
            "pyod", "anomaly", "change point", "ruptures", "river",
            "nannyml", "evidently", "whylogs", "drift", "merlion",
            "conformal", "uncertainty", "calibration", "risk control",
            "outlier", "monitoring",
        ),
    },
    "SIMULATION_MARKET_STRUCTURE": {
        "weight": 0.90,
        "keywords": (
            "abides", "backtrader", "backtesting", "vectorbt", "zipline",
            "quantconnect", "lean", "vnpy", "freqtrade", "tensortrade",
            "finrl", "market simulator", "agent-based",
        ),
    },
    "RESEARCH_AUTOMATION": {
        "weight": 0.78,
        "keywords": (
            "autoresearch", "openhands", "autogpt", "agent", "agents",
            "agent-reach", "firecrawl", "scrapling", "crawl4ai", "browser-use",
            "playwright", "research automation", "automl", "experiment",
        ),
    },
    "PROVENANCE_MEMORY": {
        "weight": 0.68,
        "keywords": (
            "graphrag", "graphify", "hindsight", "mem0", "claude-mem",
            "memory", "provenance", "knowledge graph", "lineage", "rag",
            "vector database",
        ),
    },
    "PRODUCTION_MLOPS": {
        "weight": 0.64,
        "keywords": (
            "mlflow", "dvc", "feast", "great expectations", "pandera",
            "deepchecks", "arize", "phoenix", "langfuse", "openllmetry",
            "prometheus", "grafana", "opentelemetry", "sentry", "dagster",
            "prefect", "airflow", "kubeflow", "zenml", "clearml",
        ),
    },
    "ANALYTICS_EVALUATION": {
        "weight": 0.60,
        "keywords": (
            "quantstats", "pyfolio", "empyrical", "portfolio analytics",
            "plotly", "streamlit", "jupyter", "ranking", "evaluation",
        ),
    },
    "GENERIC_MODELING": {
        "weight": 0.56,
        "keywords": (
            "lightgbm", "xgboost", "catboost", "scikit-learn", "pytorch",
            "tensorflow", "jax", "transformers", "gradient boosting",
            "random forest", "extra trees", "tabular",
        ),
    },
}

STATUS_ALLOWLIST = {"VERIFIED_SOURCE_METADATA"}
PROMOTION_FLAGS = {
    "research_only": True,
    "production_changed": False,
    "promotion_allowed": False,
    "frozen_holdout_used": False,
}


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _freshness_score(value: str | None, now: datetime) -> float:
    dt = _parse_time(value)
    if dt is None:
        return 0.0
    days = max(0.0, (now - dt.astimezone(timezone.utc)).total_seconds() / 86400.0)
    return round(100.0 * math.exp(-days / 365.0), 3)


def _classify(text: str) -> tuple[str, float, list[str]]:
    text = text.lower()
    scores: list[tuple[float, str, list[str]]] = []
    for lane, spec in LANE_RULES.items():
        matched = [k for k in spec["keywords"] if k in text]
        if matched:
            # Saturation prevents long descriptions from dominating.
            raw = min(1.0, 0.35 + 0.15 * len(matched))
            score = raw * spec["weight"]
            scores.append((score, lane, matched))
    if not scores:
        return "UNCLASSIFIED_RESEARCH", 0.0, []
    scores.sort(reverse=True)
    best_score, best_lane, matched = scores[0]
    return best_lane, round(best_score, 4), matched[:8]


def _load_mechanism_seeds(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict] = {}
    for row in data.get("candidates", []):
        repo = str(row.get("repo", "")).strip().lower()
        if repo:
            result[repo] = row
    return result


def build_queue(snapshot: dict, mechanism_config: dict) -> dict:
    if snapshot.get("source_claimed_count") != 1000:
        raise SystemExit("FAIL: TOP1000 claimed source count is not 1000")
    if snapshot.get("results_count") != snapshot.get("explicit_repo_count"):
        raise SystemExit("FAIL: explicit repo result count mismatch")

    mechanism_rows = {
        str(row.get("repo", "")).strip().lower(): row
        for row in mechanism_config.get("candidates", [])
        if str(row.get("repo", "")).strip()
    }

    verified = []
    blocked = []
    for row in snapshot.get("results", []):
        status = row.get("status")
        if status not in STATUS_ALLOWLIST:
            blocked.append(
                {
                    "source_rank": row.get("source_rank"),
                    "source_entry": row.get("source_entry"),
                    "canonical_ref": row.get("canonical_ref"),
                    "reason": status or "UNKNOWN_STATUS",
                }
            )
            continue
        ref = str(row.get("canonical_ref") or "").strip()
        if not ref:
            blocked.append(
                {
                    "source_rank": row.get("source_rank"),
                    "source_entry": row.get("source_entry"),
                    "canonical_ref": None,
                    "reason": "VERIFIED_WITHOUT_CANONICAL_REF",
                }
            )
            continue
        verified.append(row)

    # Dedupe by canonical repository while preserving every source rank.
    grouped: dict[str, list[dict]] = {}
    for row in verified:
        grouped.setdefault(str(row["canonical_ref"]).lower(), []).append(row)

    now = datetime.now(timezone.utc)
    queue = []
    for ref, rows in grouped.items():
        rows.sort(key=lambda x: int(x.get("source_rank", 10**9)))
        primary = rows[0]
        description = str(primary.get("description") or "")
        seed = mechanism_rows.get(ref)
        seed_lane = str(seed.get("lane")) if seed else None
        seed_relevance = float(seed.get("relevance", 0)) if seed else 0.0

        lane, classifier_relevance, matched = _classify(
            " ".join(
                [
                    ref,
                    str(primary.get("source_entry") or ""),
                    description,
                    str(seed.get("hypothesis") or "") if seed else "",
                ]
            )
        )
        if seed_lane:
            lane = seed_lane
        relevance = max(seed_relevance, round(5.0 * classifier_relevance, 3))
        source_rank = int(primary["source_rank"])
        source_priority = 100.0 * (1.0 - (source_rank - 1) / 999.0)
        freshness = _freshness_score(primary.get("pushed_at"), now)

        discovery_priority = round(
            55.0 * (relevance / 5.0)
            + 30.0 * (source_priority / 100.0)
            + 15.0 * (freshness / 100.0),
            3,
        )

        if relevance >= 4.0 or seed is not None:
            tier = "DEEP_LOCAL_REPRODUCTION"
        elif relevance >= 2.5:
            tier = "TARGETED_LOCAL_REPRODUCTION"
        else:
            tier = "WATCH"

        if primary.get("archived") is True:
            tier = "WATCH"

        queue.append(
            {
                "canonical_ref": primary["canonical_ref"],
                "source_ranks": [int(x["source_rank"]) for x in rows],
                "primary_source_rank": source_rank,
                "source_entry": primary.get("source_entry"),
                "status": "READY_FOR_SOURCE_INSPECTION",
                "lane": lane,
                "matched_mechanism_terms": matched,
                "mechanism_seed": seed is not None,
                "mechanism_seed_relevance": seed_relevance,
                "research_relevance_score": round(relevance, 3),
                "source_discovery_priority": round(source_priority, 3),
                "freshness_score": freshness,
                "discovery_priority_score": discovery_priority,
                "priority_is_not_predictive_evidence": True,
                "metadata": {
                    "stars": primary.get("stars"),
                    "forks": primary.get("forks"),
                    "open_issues": primary.get("open_issues"),
                    "archived": primary.get("archived"),
                    "disabled": primary.get("disabled"),
                    "default_branch": primary.get("default_branch"),
                    "language": primary.get("language"),
                    "license_spdx": primary.get("license_spdx"),
                    "pushed_at": primary.get("pushed_at"),
                    "html_url": primary.get("html_url"),
                },
                "required_next_stage": "SOURCE_CODE_INSPECTION",
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
                "tier": tier,
                **PROMOTION_FLAGS,
                "evidence_level": "E1_EXTERNAL_CLAIM",
            }
        )

    queue.sort(
        key=lambda x: (
            -x["discovery_priority_score"],
            x["primary_source_rank"],
            x["canonical_ref"].lower(),
        )
    )

    lane_counts: dict[str, int] = {}
    tier_counts: dict[str, int] = {}
    for row in queue:
        lane_counts[row["lane"]] = lane_counts.get(row["lane"], 0) + 1
        tier_counts[row["tier"]] = tier_counts.get(row["tier"], 0) + 1

    return {
        "schema_version": 1,
        "queue_id": "external-stock-research-queue-v1",
        "status": "EXECUTED_RESEARCH_QUEUE",
        "observed_at": snapshot.get("observed_at"),
        "source_snapshot_version": snapshot.get("schema_version"),
        "source_claimed_count": snapshot.get("source_claimed_count"),
        "explicit_repo_count": snapshot.get("explicit_repo_count"),
        "verified_source_count": snapshot.get("verified_source_count"),
        "blocked_source_count": len(blocked),
        "unique_verified_repository_count": len(queue),
        "deduplication_policy": "canonical_ref_lowercase_preserve_all_source_ranks",
        "selection_policy": "mechanism_relevance_plus_source_discovery_priority_plus_metadata_freshness",
        "selection_policy_warning": (
            "Discovery priority is for research ordering only. It is not predictive "
            "performance evidence and must never select production models."
        ),
        "predictive_performance_evidence": "NONE_EXTERNAL_METADATA",
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "frozen_holdout_used": False,
        "lane_counts": lane_counts,
        "tier_counts": tier_counts,
        "blocked": blocked,
        "candidates": queue,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--mechanism-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.snapshot.exists():
        result = {
            "schema_version": 1,
            "queue_id": "external-stock-research-queue-v1",
            "status": "BLOCKED_NO_LIVE_SNAPSHOT",
            "predictive_performance_evidence": "NONE_EXTERNAL_METADATA",
            **PROMOTION_FLAGS,
            "candidates": [],
            "blocked": [{"reason": "LIVE_EXTERNAL_OSS_SNAPSHOT_MISSING"}],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"]}, ensure_ascii=False))
        return 0

    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    mechanism = json.loads(args.mechanism_config.read_text(encoding="utf-8"))
    result = build_queue(snapshot, mechanism)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "unique_verified_repository_count": result["unique_verified_repository_count"],
                "blocked_source_count": result["blocked_source_count"],
                "deep_local_reproduction": result["tier_counts"].get("DEEP_LOCAL_REPRODUCTION", 0),
                "targeted_local_reproduction": result["tier_counts"].get("TARGETED_LOCAL_REPRODUCTION", 0),
                "watch": result["tier_counts"].get("WATCH", 0),
                "promotion_allowed": result["promotion_allowed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
