from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.record_automation_failure import (
    FAILURE_CONCLUSIONS,
    _load_job_snapshot,
    _request_json,
    build_record,
)

WATCHED_WORKFLOWS = (
    "repository-verification.yml",
    "research-validation.yml",
    "market-cycle.yml",
    "us-close-prediction.yml",
    "prediction-monitoring.yml",
    "24h-research-marathon.yml",
)
MAX_RUNS_PER_WORKFLOW = 50
OUT = Path("data/research/automation_failure_records.jsonl")


def _api_context() -> tuple[str, str, str]:
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    api_base = os.environ.get("GITHUB_API_URL", "https://api.github.com").strip()
    if not repository or not token:
        raise SystemExit("FAIL: GitHub API context is missing")
    return repository, token, api_base


def discover_failure_records() -> list[dict]:
    repository, token, api_base = _api_context()
    records: list[dict] = []

    for workflow_file in WATCHED_WORKFLOWS:
        payload = _request_json(
            f"{api_base}/repos/{repository}/actions/workflows/{workflow_file}/runs"
            f"?per_page={MAX_RUNS_PER_WORKFLOW}",
            token,
        )
        for run in payload.get("workflow_runs", []) or []:
            conclusion = str(run.get("conclusion") or "").strip()
            if conclusion not in FAILURE_CONCLUSIONS:
                continue
            run_id = str(run.get("id") or "").strip()
            if not run_id:
                continue
            jobs = _load_job_snapshot(repository, run_id, token, api_base)
            records.append(
                build_record(
                    run_id=run_id,
                    run_number=str(run.get("run_number") or ""),
                    workflow_name=str(run.get("name") or workflow_file),
                    head_sha=str(run.get("head_sha") or ""),
                    head_branch=str(run.get("head_branch") or ""),
                    conclusion=conclusion,
                    run_attempt=str(run.get("run_attempt") or "1"),
                    created_at=str(run.get("created_at") or ""),
                    updated_at=str(run.get("updated_at") or ""),
                    event_name=str(run.get("event") or ""),
                    repository=repository,
                    jobs=jobs,
                )
            )

    unique: dict[str, dict] = {}
    for record in records:
        unique[str(record["failure_id"])] = record

    return sorted(
        unique.values(),
        key=lambda row: (
            str(row.get("created_at") or ""),
            str(row.get("failure_id") or ""),
        ),
    )


def main() -> int:
    records = discover_failure_records()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    print(
        json.dumps(
            {
                "status": "RECONCILED",
                "records_discovered": len(records),
                "watched_workflows": list(WATCHED_WORKFLOWS),
                "max_runs_per_workflow": MAX_RUNS_PER_WORKFLOW,
                "research_only": True,
                "production_mutation": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
