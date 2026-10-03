from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def _lookup_research_job() -> tuple[dict, str | None]:
    token = os.environ.get("GITHUB_TOKEN", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("RESEARCH_WORKFLOW_RUN_ID") or os.environ.get("GITHUB_RUN_ID", "")
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
    workflow_status = str(os.environ.get("RESEARCH_WORKFLOW_STATUS", "")).strip()
    retries = 0
    if workflow_status in {"requested", "queued", "pending", "waiting", "in_progress"}:
        retries = max(0, min(5, int(os.environ.get("RESEARCH_STATUS_JOB_LOOKUP_RETRIES", "4"))))

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.load(response)
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            return {}, f"{type(exc).__name__}:{exc}"
        for job in payload.get("jobs", []):
            if job.get("name") == "research":
                return job, None
        if attempt < retries:
            time.sleep(2)
    return {}, "research_job_not_found"


def _lookup_run_artifacts() -> tuple[list[dict], str | None]:
    token = os.environ.get("GITHUB_TOKEN", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("RESEARCH_WORKFLOW_RUN_ID") or os.environ.get("GITHUB_RUN_ID", "")
    api_base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    if not token or not repository or not run_id:
        return [], "missing_github_api_context"
    url = f"{api_base}/repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100"
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
        return [], f"{type(exc).__name__}:{exc}"
    return payload.get("artifacts", []) or [], None


def _runtime_health(job: dict, workflow_status: str = "") -> dict:
    """Summarize active research runtime without treating long duration as failure."""
    status = str(job.get("status") or workflow_status or "unknown").strip()
    started_at = str(job.get("started_at") or "").strip() or None
    active_step = next(
        (
            step
            for step in reversed(job.get("steps", []) or [])
            if str(step.get("status") or "").strip() == "in_progress"
        ),
        None,
    )
    active_step_name = (
        str(active_step.get("name") or "").strip() if active_step else None
    )
    active_step_started_at = (
        str(active_step.get("started_at") or "").strip() or None
        if active_step
        else None
    )

    now = datetime.now(timezone.utc)
    def elapsed_minutes(value: str | None) -> float | None:
        if not value:
            return None
        try:
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (now - ts).total_seconds() / 60.0)

    job_elapsed = elapsed_minutes(started_at)
    step_elapsed = elapsed_minutes(active_step_started_at)
    long_running_minutes = max(
        1,
        int(os.environ.get("RESEARCH_STATUS_LONG_RUNNING_MINUTES", "60")),
    )
    stale_risk_minutes = max(
        long_running_minutes + 1,
        int(os.environ.get("RESEARCH_STATUS_STALE_RISK_MINUTES", "240")),
    )

    if status == "in_progress" and job_elapsed is not None:
        if job_elapsed >= stale_risk_minutes:
            state = "STALE_RISK"
        elif job_elapsed >= long_running_minutes:
            state = "LONG_RUNNING"
        else:
            state = "RUNNING"
    elif status in {"requested", "queued", "pending", "waiting"}:
        state = "QUEUED"
    elif status in {"completed", "cancelled", "failure", "failed", "skipped"}:
        state = "TERMINAL"
    else:
        state = "UNKNOWN"

    return {
        "state": state,
        "job_status": status,
        "job_started_at_utc": started_at,
        "job_elapsed_minutes": job_elapsed,
        "active_step": active_step_name,
        "active_step_started_at_utc": active_step_started_at,
        "active_step_elapsed_minutes": step_elapsed,
        "long_running_threshold_minutes": long_running_minutes,
        "stale_risk_threshold_minutes": stale_risk_minutes,
    }


def _step_conclusion(job: dict, needle: str) -> str:
    for step in job.get("steps", []) or []:
        if str(step.get("name", "")).strip() == needle:
            return str(step.get("conclusion") or step.get("status") or "unknown")
    return "unknown"


def main() -> int:
    job, lookup_error = _lookup_research_job()
    artifacts, artifact_error = _lookup_run_artifacts()
    workflow_status = str(os.environ.get("RESEARCH_WORKFLOW_STATUS", "")).strip()
    workflow_conclusion = str(os.environ.get("RESEARCH_WORKFLOW_CONCLUSION", "")).strip()
    conclusion = str(job.get("conclusion") or job.get("status") or workflow_status or workflow_conclusion or "unknown")
    # A workflow can be cancelled before the research job is created. That is
    # a valid terminal state, not an API failure. For every other conclusion,
    # a missing research job remains fail-closed.
    effective_lookup_error = lookup_error
    workflow_statuses = {"requested", "queued", "pending", "waiting", "in_progress"}
    if lookup_error == "research_job_not_found" and workflow_conclusion in {"cancelled", "skipped"}:
        effective_lookup_error = None
    elif lookup_error == "research_job_not_found" and workflow_status in workflow_statuses:
        # workflow_run can emit before matrix jobs are visible
        # through the Actions API. Preserve an explicit active status rather
        # than converting a transient race into a false failure.
        effective_lookup_error = None
    evidence_name = f"research-validation-evidence-{os.environ.get('RESEARCH_WORKFLOW_RUN_ID') or os.environ.get('GITHUB_RUN_ID', '')}"
    evidence_artifact_raw_present = any(
        str(item.get("name", "")) == evidence_name and not item.get("expired", False)
        for item in artifacts
    )
    research_job_success = str(job.get("conclusion") or "").strip() == "success"
    workflow_success = workflow_conclusion == "success"
    evidence_artifact_present = (
        evidence_artifact_raw_present
        and workflow_success
        and research_job_success
    )
    if evidence_artifact_present:
        evidence_state = "COMPLETE"
    elif evidence_artifact_raw_present:
        evidence_state = "PARTIAL"
    else:
        evidence_state = "MISSING"
    status = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_run_id": os.environ.get("RESEARCH_WORKFLOW_RUN_ID") or os.environ.get("GITHUB_RUN_ID", ""),
        "workflow_sha": os.environ.get("RESEARCH_WORKFLOW_SHA") or os.environ.get("GITHUB_SHA", ""),
        "workflow_status": workflow_status or "unknown",
        "workflow_conclusion": workflow_conclusion or None,
        "job_status": conclusion,
        "research_step": _step_conclusion(job, "Chronological OOS research"),
        "sec_research_step": _step_conclusion(job, "Collect free SEC filing research inputs"),
        "sec_ablation_step": _step_conclusion(job, "Run SEC filing OOS challenger ablation"),
        "cpcv_step": _step_conclusion(job, "CPCV leakage-boundary research audit"),
        "runtime_health": _runtime_health(job, workflow_status),
        "evidence_artifact_present": evidence_artifact_present,
        "evidence_artifact_raw_present": evidence_artifact_raw_present,
        "evidence_state": evidence_state,
        "status_lookup_ok": effective_lookup_error is None and artifact_error is None,
        "status_lookup_error": effective_lookup_error,
        "artifact_lookup_error": artifact_error,
    }
    out = Path("artifacts/research_validation_status.json")
    # Workflow-run completion events may arrive out of order. Never allow an
    # older run to overwrite the status of a newer research run.
    incoming_run_id = str(
        os.environ.get("RESEARCH_WORKFLOW_RUN_ID")
        or os.environ.get("GITHUB_RUN_ID", "")
    ).strip()
    if out.exists() and incoming_run_id.isdigit():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        existing_run_id = str(existing.get("workflow_run_id", "")).strip()
        if existing_run_id.isdigit() and int(existing_run_id) > int(incoming_run_id):
            print(
                f"RESEARCH_STATUS_STALE_EVENT ignored_run={incoming_run_id} current_run={existing_run_id}",
                flush=True,
            )
            return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    if effective_lookup_error is not None or artifact_error is not None:
        raise SystemExit(f"FAIL: research status lookup: {effective_lookup_error or artifact_error}")
    return 0


if __name__ == "__main__":
    main()
