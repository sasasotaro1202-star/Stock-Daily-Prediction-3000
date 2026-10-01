from __future__ import annotations

import json
from pathlib import Path

from src.research.case_risk_oos import analyze_case_risk

INPUT = Path("data/research/latest_metrics.json")
OUTPUT = Path("data/research/case_risk_oos.json")


def main() -> int:
    if not INPUT.exists():
        raise SystemExit("DEFERRED: latest_metrics.json is absent")

    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    v13 = payload.get("ultimate_v13")
    if not isinstance(v13, dict):
        raise SystemExit("FAIL: ultimate_v13 payload missing")

    ledger = v13.get("prediction_ledger")
    if not isinstance(ledger, dict):
        raise SystemExit("FAIL: v13 prediction ledger missing")

    rows = ledger.get("row_level")
    if not isinstance(rows, list):
        raise SystemExit("FAIL: v13 row-level prediction ledger missing")

    result = analyze_case_risk(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if result.get("research_only") is not True:
        raise SystemExit("FAIL: case-risk audit must remain research_only")
    if result.get("production_changed") is not False:
        raise SystemExit("FAIL: case-risk audit claims production mutation")
    if result.get("promotion_allowed") is not False:
        raise SystemExit("FAIL: case-risk audit cannot promote")

    aggregate = result.get("aggregate") or {}
    high = aggregate.get("high_risk") or {}
    low = aggregate.get("low_risk") or {}
    print("CASE_RISK_OOS_STATUS=", result.get("status"))
    print("CASE_RISK_OOS_ROWS=", result.get("scored_rows", 0))
    print("CASE_RISK_OOS_HIGH_COVERAGE=", (result.get("risk_score") or {}).get("high_risk_coverage"))
    print("CASE_RISK_OOS_HIGH_ERROR_RATE=", high.get("error_rate"))
    print("CASE_RISK_OOS_LOW_ERROR_RATE=", low.get("error_rate"))
    print("CASE_RISK_OOS_ERROR_CAPTURE=", aggregate.get("error_capture_rate"))
    print("CASE_RISK_OOS_HIGH_VS_LOW_FAILURE_LIFT=", aggregate.get("failure_rate_lift_high_vs_low"))
    print("CASE_RISK_OOS_CONTRACTS=", result.get("contracts"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
