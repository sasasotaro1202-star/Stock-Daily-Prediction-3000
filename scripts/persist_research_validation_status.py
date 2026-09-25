from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    status = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "workflow_sha": os.environ.get("GITHUB_SHA", ""),
        "job_status": os.environ.get("JOB_STATUS", "unknown"),
        "research_step": os.environ.get("RESEARCH_STEP_OUTCOME", "unknown"),
        "sec_research_step": os.environ.get("SEC_RESEARCH_STEP_OUTCOME", "unknown"),
        "sec_ablation_step": os.environ.get("SEC_ABLATION_STEP_OUTCOME", "unknown"),
        "cpcv_step": os.environ.get("CPCV_STEP_OUTCOME", "unknown"),
        "metrics_present": Path("data/research/latest_metrics.json").exists(),
        "leakage_audit_present": Path("data/research/leakage_audit.json").exists(),
        "universe_quality_present": Path("data/research/universe_quality.json").exists(),
        "research_snapshot_present": Path("artifacts/research_validation_latest.json").exists(),
    }
    out = Path("artifacts/research_validation_status.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
