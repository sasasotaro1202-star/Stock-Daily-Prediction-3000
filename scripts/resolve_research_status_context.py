from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


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


def _write_env(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_ENV", "").strip()
    if not path:
        raise SystemExit("FAIL: GITHUB_ENV is missing")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def main() -> int:
    existing_run = os.environ.get("RESEARCH_WORKFLOW_RUN_ID", "").strip()
    if existing_run:
        for name in (
            "RESEARCH_WORKFLOW_RUN_ID",
            "RESEARCH_WORKFLOW_SHA",
            "RESEARCH_WORKFLOW_STATUS",
            "RESEARCH_WORKFLOW_CONCLUSION",
        ):
            value = os.environ.get(name, "").strip()
            if value:
                _write_env(name, value)
            elif name == "RESEARCH_WORKFLOW_CONCLUSION":
                _write_env(name, "")
        _write_env("RESEARCH_STATUS_CONTEXT_FOUND", "true")
        return 0

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
    api_base = os.environ.get("GITHUB_API_URL", "https://api.github.com").strip()
    if not token or not repository:
        raise SystemExit("FAIL: research status heartbeat GitHub context is missing")

    candidates = []
    for workflow_status in ("requested", "pending", "queued", "waiting", "in_progress"):
        try:
            payload = _request_json(
                f"{api_base}/repos/{repository}/actions/runs?status={workflow_status}&per_page=100",
                token,
            )
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise SystemExit(
                "FAIL: research status heartbeat run lookup: "
                f"{type(exc).__name__}:{exc}"
            ) from exc

        candidates.extend(
            run
            for run in payload.get("workflow_runs", []) or []
            if str(run.get("name", "")).strip() == "Research validation"
            and str(run.get("status", "")).strip() == workflow_status
        )
    if not candidates:
        _write_env("RESEARCH_STATUS_CONTEXT_FOUND", "false")
        return 0

    run = max(
        candidates,
        key=lambda item: (
            int(item.get("run_number") or 0),
            int(item.get("id") or 0),
        ),
    )
    run_id = str(run.get("id") or "").strip()
    head_sha = str(run.get("head_sha") or "").strip()
    status = str(run.get("status") or "in_progress").strip()
    if not run_id or not head_sha:
        raise SystemExit("FAIL: active Research validation run is missing id or head_sha")

    _write_env("RESEARCH_WORKFLOW_RUN_ID", run_id)
    _write_env("RESEARCH_WORKFLOW_SHA", head_sha)
    _write_env("RESEARCH_WORKFLOW_STATUS", status)
    _write_env("RESEARCH_WORKFLOW_CONCLUSION", "")
    _write_env("RESEARCH_STATUS_CONTEXT_FOUND", "true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
