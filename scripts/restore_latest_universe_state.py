from __future__ import annotations

import json
import os
import tempfile

import requests
from datetime import datetime, timezone
from pathlib import Path

from src.data.paypay_collector import build_snapshot
from src.data.github_artifact import (
    download_workflow_artifact,
    validate_extracted_tree,
)


MAX_AGE_SECONDS = 7 * 24 * 60 * 60


def _validate_snapshot(path: Path, created_at: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or len(records) < 100:
        raise ValueError("restored universe has fewer than 100 records")

    retrieved_at = payload.get("retrieved_at")
    retrieved = datetime.fromisoformat(str(retrieved_at).replace("Z", "+00:00"))
    if retrieved.tzinfo is None:
        raise ValueError("restored universe retrieved_at is not timezone-aware")
    age = (
        datetime.now(timezone.utc) - retrieved.astimezone(timezone.utc)
    ).total_seconds()
    if age < -300 or age > MAX_AGE_SECONDS:
        raise ValueError(
            f"restored universe retrieved_at is outside bounded 7-day window: {age:.0f}s"
        )

    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    if created.tzinfo is None:
        raise ValueError("artifact created_at is not timezone-aware")
    artifact_age = (
        datetime.now(timezone.utc) - created.astimezone(timezone.utc)
    ).total_seconds()
    if artifact_age < -300 or artifact_age > MAX_AGE_SECONDS:
        raise ValueError(
            f"universe artifact is outside bounded 7-day window: {artifact_age:.0f}s"
        )


def _api_get(url: str, token: str) -> dict:
    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        },
        timeout=25,
    )
    response.raise_for_status()
    return response.json()


def _recent_market_cycle_universe_artifacts(repo: str, token: str) -> list[dict]:
    """Find retained universe artifacts directly from recent market-cycle runs."""
    runs = _api_get(
        f"https://api.github.com/repos/{repo}/actions/workflows/market-cycle.yml/runs?per_page=20",
        token,
    ).get("workflow_runs", [])
    now = datetime.now(timezone.utc)
    candidates: list[dict] = []
    for run in runs:
        run_id = run.get("id")
        created_at = run.get("created_at")
        if not run_id or not created_at:
            continue
        try:
            created = datetime.fromisoformat(
                str(created_at).replace("Z", "+00:00")
            )
            if created.tzinfo is None:
                continue
        except ValueError:
            continue
        age = (now - created.astimezone(timezone.utc)).total_seconds()
        if age < -300 or age > MAX_AGE_SECONDS:
            continue
        try:
            artifacts = _api_get(
                f"https://api.github.com/repos/{repo}/actions/runs/{int(run_id)}/artifacts?per_page=100",
                token,
            ).get("artifacts", [])
        except Exception:
            continue
        candidates.extend(
            artifact
            for artifact in artifacts
            if str(artifact.get("name", "")).startswith("universe-")
            and not artifact.get("expired")
            and artifact.get("created_at")
        )
    return sorted(
        candidates,
        key=lambda artifact: str(artifact.get("created_at", "")),
        reverse=True,
    )


def _find_universe_snapshot(root: Path) -> Path:
    matches = [p for p in root.rglob("latest.json") if p.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one latest.json in artifact, found {len(matches)}"
        )
    snapshot = matches[0].resolve()
    root = root.resolve()
    if snapshot != root and root not in snapshot.parents:
        raise RuntimeError("universe artifact snapshot escaped restore root")
    return snapshot


def _fresh_official_fallback() -> bool:
    if os.environ.get("ALLOW_FRESH_OFFICIAL_UNIVERSE_FALLBACK", "").lower() != "true":
        return False

    try:
        dst = Path("data/universe/latest.json")
        dst.parent.mkdir(parents=True, exist_ok=True)
        snapshot = build_snapshot(str(dst))
        _validate_snapshot(dst, str(snapshot["retrieved_at"]))
        print(
            "universe-state: fresh official fallback restored "
            f"retrieved_at={snapshot["retrieved_at"]} "
            f"record_count={snapshot["record_count"]}"
        )
        return True
    except Exception as exc:
        raise SystemExit(
            "DEFERRED: fresh official universe fallback failed; "
            f"{type(exc).__name__}:{exc}"
        ) from exc


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    candidates = _recent_market_cycle_universe_artifacts(repo, token)
    if not candidates:
        if _fresh_official_fallback():
            return
        raise SystemExit(
            "DEFERRED: no retained market-cycle universe artifact in bounded 7-day window"
        )

    errors = []
    for artifact in candidates:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                extract = Path(tmp) / "artifact"
                download_workflow_artifact(repo, token, artifact, extract)
                validate_extracted_tree(extract)
                src = _find_universe_snapshot(extract)
                _validate_snapshot(src, str(artifact["created_at"]))
                dst = Path("data/universe/latest.json")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
                print(
                    "universe-state: restored "
                    f"artifact_id={artifact['id']} "
                    f"created_at={artifact['created_at']} "
                    f"record_count={len(json.loads(dst.read_text(encoding='utf-8')).get('records', []))}"
                )
                return
        except Exception as exc:
            errors.append(
                f"artifact_id={artifact.get('id')}:{type(exc).__name__}:{exc}"
            )

    raise SystemExit(
        "DEFERRED: no valid universe artifact in bounded 7-day window; "
        + "; ".join(errors[:5])
    )


if __name__ == "__main__":
    main()
