from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    source = Path("data/research/latest_metrics.json")
    if not source.exists():
        raise SystemExit("FAIL: research metrics are absent")

    payload = json.loads(source.read_text(encoding="utf-8"))
    adaptive = payload.get("adaptive_conformal_prediction_research", {})
    group = payload.get("group_conformal_prediction_research", {})
    static = payload.get("conformal_prediction_research", {})

    snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "workflow_sha": os.environ.get("GITHUB_SHA", ""),
        "status": payload.get("status", "UNKNOWN"),
        "selected_model": payload.get("selected_model"),
        "calibration_method": payload.get("calibration_method"),
        "adaptive_conformal_prediction_research": adaptive,
        "group_conformal_prediction_research": group,
        "conformal_prediction_research": static,
        "evidence_scope": {
            "source": "data/research/latest_metrics.json",
            "frozen_holdout_excluded_from_selection": True,
            "research_only_layers": True,
        },
    }

    out = Path("artifacts/research_validation_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
