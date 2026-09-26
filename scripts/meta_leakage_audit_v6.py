from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/research/latest_metrics.json")
    if not path.exists():
        print("META_LEAKAGE_AUDIT=FAIL missing_metrics")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    v6 = data.get("innovative_prediction_control_v6")
    if not isinstance(v6, dict):
        print("META_LEAKAGE_AUDIT=FAIL missing_v6_payload")
        return 1
    checks = [
        bool(v6.get("locked_oos_untouched_for_tuning")) is True,
        bool(v6.get("production_changed")) is False,
        all(row.get("current_labels_used_for_meta_training") is False for row in v6.get("per_fold_meta_audit", [])),
        all(int(row.get("prior_folds_used", 0)) <= int(row.get("fold", 0)) for row in v6.get("per_fold_meta_audit", [])),
        v6.get("promotion") in {"HOLD", "CANDIDATE"},
    ]
    if not all(checks):
        print("META_LEAKAGE_AUDIT=FAIL")
        return 1
    out = {
        "status": "PASS",
        "locked_oos_untouched_for_tuning": True,
        "current_labels_used_for_meta_training": False,
        "production_changed": False,
        "promotion": v6.get("promotion"),
        "checked_folds": len(v6.get("per_fold_meta_audit", [])),
    }
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("artifacts/v6_meta_leakage_audit.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
