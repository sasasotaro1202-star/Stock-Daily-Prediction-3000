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


ACTIVE_RESEARCH_STATUSES = frozenset(
    {"requested", "pending", "queued", "waiting", "in_progress"}
)


def _write_context(run: dict, conclusion: str = "") -> None:
    run_id = str(run.get("id") or "").strip()
    head_sha = str(run.get("head_sha") or "").strip()
    status = str(run.get("status") or "in_progress").strip()
    run_number = str(run.get("run_number") or "").strip()
    if not run_id or not head_sha:
        raise SystemExit("FAIL: Research validation run is missing id or head_sha")
    _write_env("RESEARCH_WORKFLOW_RUN_ID", run_id)
    _write_env("RESEARCH_WORKFLOW_RUN_NUMBER", run_number)
    _write_env("RESEARCH_WORKFLOW_SHA", head_sha)
    _write_env("RESEARCH_WORKFLOW_STATUS", status)
    _write_env("RESEARCH_WORKFLOW_CONCLUSION", conclusion)
    _write_env("RESEARCH_STATUS_CONTEXT_FOUND", "true")


def _run_key(run: dict) -> tuple[int, int]:
    return (int(run.get("run_number") or 0), int(run.get("id") or 0))


def _lookup_active_research_runs(token: str, repository: str, api_base: str) -> list[dict]:
    candidates: list[dict] = []
    for workflow_status in (
        "requested",
        "pending",
        "queued",
        "waiting",
        "in_progress",
    ):
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
    return candidates


def main() -> int:
    existing_run = os.environ.get("RESEARCH_WORKFLOW_RUN_ID", "").strip()
    existing_status = os.environ.get("RESEARCH_WORKFLOW_STATUS", "").strip()
    existing_run_number = os.environ.get("RESEARCH_WORKFLOW_RUN_NUMBER", "").strip()
    if existing_run and existing_status in ACTIVE_RESEARCH_STATUSES:
        _write_env("RESEARCH_WORKFLOW_RUN_ID", existing_run)
        _write_env("RESEARCH_WORKFLOW_RUN_NUMBER", existing_run_number)
        for name in (
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

    active_candidates = _lookup_active_research_runs(token, repository, api_base)

    if existing_run:
        terminal_run = {
            "id": existing_run,
            "run_number": int(existing_run_number or 0),
            "head_sha": os.environ.get("RESEARCH_WORKFLOW_SHA", "").strip(),
            "status": existing_status or "completed",
        }
        valid_terminal_key = bool(terminal_run["head_sha"]) and bool(existing_run_number)
        if active_candidates:
            newest = max(active_candidates, key=_run_key)
            if (not valid_terminal_key) or _run_key(newest) > _run_key(terminal_run):
                _write_context(newest)
                return 0
        if valid_terminal_key:
            _write_context(
                terminal_run,
                os.environ.get("RESEARCH_WORKFLOW_CONCLUSION", "").strip(),
            )
            return 0
        raise SystemExit(
            "FAIL: terminal Research validation event lacks verifiable run_number/head_sha "
            "and active run context is unavailable"
        )

    if not active_candidates:
        _write_env("RESEARCH_STATUS_CONTEXT_FOUND", "false")
        return 0

    run = max(active_candidates, key=_run_key)
    _write_context(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
