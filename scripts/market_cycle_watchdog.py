from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


API_VERSION = "2022-11-28"
JST = ZoneInfo("Asia/Tokyo")
WORKFLOW = "market-cycle.yml"
GRACE_MINUTES = 30


def api_json(url: str, token: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
        },
    )
    with urllib.request.urlopen(req, timeout=25) as response:
        return json.load(response)


def dispatch(repo: str, token: str) -> None:
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/dispatches"
    body = json.dumps({"ref": "main"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": API_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            if response.status not in (200, 201, 204):
                raise RuntimeError(f"unexpected dispatch status {response.status}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"workflow dispatch failed: HTTP {exc.code}: {detail}"
        ) from exc


def main() -> None:
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]

    payload = api_json(
        f"https://api.github.com/repos/{repo}/actions/runs?per_page=100",
        token,
    )
    runs = [
        run
        for run in payload.get("workflow_runs", [])
        if str(run.get("name")) == "Market cycle"
    ]

    now = datetime.now(timezone.utc)
    today_jst = now.astimezone(JST).date()
    todays = []
    for run in runs:
        created_raw = run.get("created_at")
        if not created_raw:
            continue
        created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        if created.astimezone(JST).date() == today_jst:
            todays.append(run)

    latest = max(todays, key=lambda r: r.get("created_at", "")) if todays else None

    if latest is not None:
        status = str(latest.get("status", ""))
        conclusion = latest.get("conclusion")
        created = datetime.fromisoformat(
            latest["created_at"].replace("Z", "+00:00")
        )
        age = now - created

        if status in {"queued", "in_progress"}:
            print(
                json.dumps(
                    {
                        "action": "NOOP",
                        "reason": "market_cycle_already_running",
                        "run_id": latest.get("id"),
                        "status": status,
                    },
                    indent=2,
                )
            )
            return

        if conclusion == "success":
            print(
                json.dumps(
                    {
                        "action": "NOOP",
                        "reason": "market_cycle_today_success",
                        "run_id": latest.get("id"),
                    },
                    indent=2,
                )
            )
            return

        if age < timedelta(minutes=GRACE_MINUTES):
            print(
                json.dumps(
                    {
                        "action": "NOOP",
                        "reason": "market_cycle_failure_within_grace_window",
                        "run_id": latest.get("id"),
                        "age_minutes": round(age.total_seconds() / 60.0, 1),
                    },
                    indent=2,
                )
            )
            return

    dispatch(repo, token)
    print(
        json.dumps(
            {
                "action": "DISPATCHED",
                "reason": (
                    "no successful market cycle today"
                    if latest is None
                    else "latest market cycle did not succeed"
                ),
                "previous_run_id": latest.get("id") if latest else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
