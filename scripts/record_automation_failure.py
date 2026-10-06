from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


OUT = Path("data/research/automation_failure_record.json")
FAILURE_CONCLUSIONS = frozenset({"failure", "timed_out", "startup_failure", "action_required"})


def classify_workflow(workflow_name: str) -> str:
    name = workflow_name.strip().lower()
    if name == "research validation":
        return "RESEARCH_VALIDATION_FAILURE"
    if name in {"market cycle", "u.s. close prediction", "prediction monitoring"}:
        return "PRODUCTION_WORKFLOW_FAILURE"
    if "24h research marathon" in name:
        return "RESEARCH_AUTOMATION_FAILURE"
    return "WORKFLOW_FAILURE"


def _request_json(url: str, token: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Stock-Daily-Prediction-3000",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def _load_job_snapshot(repository: str, run_id: str, token: str, api_base: str) -> list[dict]:
    if not repository or not run_id or not token:
        return []
    try:
        payload = _request_json(
            f"{api_base}/repos/{repository}/actions/runs/{run_id}/jobs?per_page=100",
            token,
        )
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return []

    jobs: list[dict] = []
    for job in payload.get("jobs", []) or []:
        failed_steps = [
            {
                "name": str(step.get("name") or ""),
                "status": str(step.get("status") or ""),
                "conclusion": str(step.get("conclusion") or ""),
            }
            for step in job.get("steps", []) or []
            if str(step.get("conclusion") or "") in FAILURE_CONCLUSIONS
        ]
        if str(job.get("conclusion") or "") in FAILURE_CONCLUSIONS or failed_steps:
            jobs.append(
                {
                    "job_id": job.get("id"),
                    "name": str(job.get("name") or ""),
                    "status": str(job.get("status") or ""),
                    "conclusion": str(job.get("conclusion") or ""),
                    "failed_steps": failed_steps,
                }
            )
    return jobs


def build_record(
    *,
    run_id: str,
    run_number: str,
    workflow_name: str,
    head_sha: str,
    head_branch: str,
    conclusion: str,
    run_attempt: str,
    created_at: str,
    updated_at: str,
    event_name: str,
    repository: str,
    jobs: list[dict],
) -> dict:
    failure_key = "|".join(
        [repository, workflow_name, run_id, run_attempt or "1", head_sha, conclusion]
    )
    failure_id = hashlib.sha256(failure_key.encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "failure_id": failure_id,
        "status": "FAILED",
        "category": classify_workflow(workflow_name),
        "workflow_name": workflow_name,
        "workflow_run_id": int(run_id) if run_id.isdigit() else run_id,
        "workflow_run_number": int(run_number) if run_number.isdigit() else run_number,
        "run_attempt": int(run_attempt) if run_attempt.isdigit() else run_attempt,
        "head_sha": head_sha,
        "head_branch": head_branch,
        "conclusion": conclusion,
        "event": event_name,
        "created_at": created_at,
        "updated_at": updated_at,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "failed_jobs": jobs,
        "recovery_policy": "bounded_watchdog_or_workflow_recovery",
        "production_mutation": False,
        "research_only": True,
    }


def main() -> int:
    conclusion = os.environ.get("WORKFLOW_RUN_CONCLUSION", "").strip()
    if conclusion not in FAILURE_CONCLUSIONS:
        raise SystemExit(f"SKIP: non-failure workflow conclusion={conclusion or 'unknown'}")

    run_id = os.environ.get("WORKFLOW_RUN_ID", "").strip()
    if not run_id:
        raise SystemExit("FAIL: workflow run id is missing")

    repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    api_base = os.environ.get("GITHUB_API_URL", "https://api.github.com").strip()
    jobs = _load_job_snapshot(repository, run_id, token, api_base)

    record = build_record(
        run_id=run_id,
        run_number=os.environ.get("WORKFLOW_RUN_NUMBER", "").strip(),
        workflow_name=os.environ.get("WORKFLOW_NAME", "").strip(),
        head_sha=os.environ.get("WORKFLOW_HEAD_SHA", "").strip(),
        head_branch=os.environ.get("WORKFLOW_HEAD_BRANCH", "").strip(),
        conclusion=conclusion,
        run_attempt=os.environ.get("WORKFLOW_RUN_ATTEMPT", "1").strip(),
        created_at=os.environ.get("WORKFLOW_CREATED_AT", "").strip(),
        updated_at=os.environ.get("WORKFLOW_UPDATED_AT", "").strip(),
        event_name=os.environ.get("WORKFLOW_EVENT", "").strip(),
        repository=repository,
        jobs=jobs,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
