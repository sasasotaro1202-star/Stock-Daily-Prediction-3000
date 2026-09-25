from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def _lookup_research_job() -> tuple[dict, str | None]:
    token = os.environ.get("GITHUB_TOKEN", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    api_base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    if not token or not repository or not run_id:
        return {}, "missing_github_api_context"
    url = f"{api_base}/repos/{repository}/actions/runs/{run_id}/jobs?per_page=100"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Stock-Daily-Prediction-3000",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        return {}, f"{type(exc).__name__}:{exc}"
    for job in payload.get("jobs", []):
        if job.get("name") == "research":
            return job, None
    return {}, "research_job_not_found"


def _step_conclusion(job: dict, needle: str) -> str:
    for step in job.get("steps", []) or []:
        if str(step.get("name", "")).strip() == needle:
            return str(step.get("conclusion") or step.get("status") or "unknown")
    return "unknown"


def main() -> int:
    job, lookup_error = _lookup_research_job()
    conclusion = str(job.get("conclusion") or job.get("status") or "unknown")
    status = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "workflow_sha": os.environ.get("GITHUB_SHA", ""),
        "job_status": conclusion,
        "research_step": _step_conclusion(job, "Chronological OOS research"),
        "sec_research_step": _step_conclusion(job, "Collect free SEC filing research inputs"),
        "sec_ablation_step": _step_conclusion(job, "Run SEC filing OOS challenger ablation"),
        "cpcv_step": _step_conclusion(job, "CPCV leakage-boundary research audit"),
        "metrics_present": Path("data/research/latest_metrics.json").exists(),
        "leakage_audit_present": Path("data/research/leakage_audit.json").exists(),
        "universe_quality_present": Path("data/research/universe_quality.json").exists(),
        "research_snapshot_present": Path("artifacts/research_validation_latest.json").exists(),
        "status_lookup_ok": lookup_error is None,
        "status_lookup_error": lookup_error,
    }
    out = Path("artifacts/research_validation_status.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    if lookup_error is not None:
        raise SystemExit(f"FAIL: research status lookup: {lookup_error}")
    return 0


if __name__ == "__main__":
    main()
