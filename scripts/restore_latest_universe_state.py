from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

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


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    payload = __import__("requests").get(
        f"https://api.github.com/repos/{repo}/actions/artifacts?per_page=100",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        },
        timeout=25,
    )
    payload.raise_for_status()
    payload = payload.json()
    candidates = sorted(
        (
            artifact
            for artifact in payload.get("artifacts", [])
            if str(artifact.get("name", "")).startswith("universe-")
            and not artifact.get("expired")
            and artifact.get("created_at")
        ),
        key=lambda artifact: artifact.get("created_at", ""),
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            "DEFERRED: no retained successful market-cycle universe artifact"
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
