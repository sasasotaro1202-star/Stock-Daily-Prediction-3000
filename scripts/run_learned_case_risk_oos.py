from __future__ import annotations

import json
from pathlib import Path

from src.research.learned_case_risk_oos import analyze_learned_case_risk

INPUT = Path("data/research/latest_metrics.json")
OUTPUT = Path("data/research/learned_case_risk_oos.json")


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

    result = analyze_learned_case_risk(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if result.get("research_only") is not True:
        raise SystemExit("FAIL: learned case-risk audit must remain research_only")
    if result.get("production_changed") is not False:
        raise SystemExit("FAIL: learned case-risk audit claims production mutation")
    if result.get("promotion_allowed") is not False:
        raise SystemExit("FAIL: learned case-risk audit cannot promote")
    if str(result.get("status") or "").startswith("BLOCKED_"):
        raise SystemExit(
            "FAIL: learned case-risk audit is blocked by an integrity condition: "
            + str(result.get("status"))
        )

    learned = result.get("learned_case_risk") or {}
    fixed = result.get("fixed_case_risk") or {}
    le = learned.get("evaluation") or {}
    fe = fixed.get("evaluation") or {}
    print("LEARNED_CASE_RISK_STATUS=", result.get("status"))
    print("LEARNED_CASE_RISK_DEVELOPMENT_ROWS=", result.get("development_rows", 0))
    print("LEARNED_CASE_RISK_LOCKED_ROWS=", result.get("locked_rows", 0))
    print("LEARNED_CASE_RISK_FIXED_ERROR_CAPTURE=", fe.get("error_capture_rate"))
    print("LEARNED_CASE_RISK_LEARNED_ERROR_CAPTURE=", le.get("error_capture_rate"))
    print("LEARNED_CASE_RISK_FIXED_FAILURE_LIFT=", fe.get("failure_rate_lift_high_vs_low"))
    print("LEARNED_CASE_RISK_LEARNED_FAILURE_LIFT=", le.get("failure_rate_lift_high_vs_low"))
    print("LEARNED_CASE_RISK_DELTA=", result.get("delta_learned_minus_fixed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
