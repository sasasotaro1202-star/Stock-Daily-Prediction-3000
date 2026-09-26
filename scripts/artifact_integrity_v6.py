from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    metrics = root / "data/research/latest_metrics.json"
    bank = root / "data/research/innovative_v6_oos_bank.parquet"
    required = [metrics, bank]
    missing = [str(p) for p in required if not p.exists() or p.stat().st_size == 0]
    if missing:
        print(f"ARTIFACT_INTEGRITY=FAIL missing={missing}")
        return 1

    try:
        data = json.loads(metrics.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ARTIFACT_INTEGRITY=FAIL unreadable_metrics={exc}")
        return 1

    v6 = data.get("innovative_prediction_control_v6")
    if not isinstance(v6, dict):
        print("ARTIFACT_INTEGRITY=FAIL missing_v6_payload")
        return 1
    for key in ("status", "promotion", "locked_folds", "development_folds",
                "full_vs_baseline_locked", "statistical_validation"):
        if key not in v6:
            print(f"ARTIFACT_INTEGRITY=FAIL missing_key={key}")
            return 1

    sha = hashlib.sha256(metrics.read_bytes()).hexdigest()
    out = {
        "status": "PASS",
        "metrics_bytes": metrics.stat().st_size,
        "bank_bytes": bank.stat().st_size,
        "metrics_sha256": sha,
        "v6_schema": v6.get("schema_version"),
        "oos_status": v6.get("status"),
        "locked_folds": v6.get("locked_folds"),
    }
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("artifacts/v6_artifact_integrity.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
