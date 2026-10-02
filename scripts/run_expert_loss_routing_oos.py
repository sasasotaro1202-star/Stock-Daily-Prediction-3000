from __future__ import annotations

import json
from pathlib import Path

from src.research.expert_loss_routing_oos import analyze_expert_loss_routing

INPUT = Path("data/research/latest_metrics.json")
OUTPUT = Path("data/research/expert_loss_routing_oos.json")


def main() -> int:
    if not INPUT.exists():
        raise SystemExit("DEFERRED: latest_metrics.json is absent")

    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    v13 = payload.get("ultimate_v13")
    if not isinstance(v13, dict):
        raise SystemExit("FAIL: ultimate_v13 payload missing")

    ledger_payload = v13.get("prediction_ledger")
    if not isinstance(ledger_payload, dict):
        raise SystemExit("FAIL: v13 prediction ledger missing")
    ledger_rows = ledger_payload.get("row_level")
    if not isinstance(ledger_rows, list):
        raise SystemExit("FAIL: v13 row-level prediction ledger missing")

    fold_results = v13.get("fold_results")
    if not isinstance(fold_results, list):
        raise SystemExit("FAIL: v13 fold_results missing")

    models = v13.get("models")
    if not isinstance(models, list):
        raise SystemExit("FAIL: v13 model list missing")

    locked_folds = int(v13.get("locked_folds") or 1)
    result = analyze_expert_loss_routing(
        ledger_rows,
        fold_results,
        models=[str(model) for model in models],
        locked_folds=locked_folds,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if result.get("research_only") is not True:
        raise SystemExit("FAIL: expert-loss routing audit must remain research_only")
    if result.get("production_changed") is not False:
        raise SystemExit("FAIL: expert-loss routing claims production mutation")
    if result.get("promotion_allowed") is not False:
        raise SystemExit("FAIL: expert-loss routing cannot promote")
    if result.get("frozen_holdout_used") is not False:
        raise SystemExit("FAIL: expert-loss routing must not use frozen holdout")

    locked = result.get("locked_metrics") or {}
    candidate = locked.get("expert_loss_routing") or {}
    dynamic = locked.get("existing_dynamic_routing") or {}
    print("EXPERT_LOSS_ROUTING_STATUS=", result.get("status"))
    print("EXPERT_LOSS_ROUTING_DEVELOPMENT_ROWS=", result.get("development_rows", 0))
    print("EXPERT_LOSS_ROUTING_LOCKED_ROWS=", result.get("locked_rows", 0))
    print("EXPERT_LOSS_ROUTING_LOGLOSS=", candidate.get("logloss"))
    print("EXPERT_LOSS_ROUTING_DYNAMIC_LOGLOSS=", dynamic.get("logloss"))
    print(
        "EXPERT_LOSS_ROUTING_DELTA_LOGLOSS=",
        (locked.get("delta_candidate_minus_dynamic") or {}).get("logloss"),
    )
    print(
        "EXPERT_LOSS_ROUTING_EXPERT_LOSS_BRIER=",
        (result.get("expert_loss_prediction") or {}).get("locked_brier"),
    )
    print(
        "EXPERT_LOSS_ROUTING_WEIGHT_CONCENTRATION=",
        (result.get("routing_stability") or {}).get("locked_mean_weight_concentration"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
