from __future__ import annotations

import json
import os
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


MAX_AGE_SECONDS = 7 * 24 * 60 * 60


def _api(url: str, token: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=25) as response:
        return json.load(response)


def _download(url: str, token: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def _validate_snapshot(path: Path, created_at: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or len(records) < 100:
        raise ValueError("restored universe has fewer than 100 records")

    retrieved_at = payload.get("retrieved_at")
    retrieved = datetime.fromisoformat(str(retrieved_at).replace("Z", "+00:00"))
    if retrieved.tzinfo is None:
        raise ValueError("restored universe retrieved_at is not timezone-aware")
    age = (datetime.now(timezone.utc) - retrieved.astimezone(timezone.utc)).total_seconds()
    if age < -300 or age > MAX_AGE_SECONDS:
        raise ValueError(
            f"restored universe retrieved_at is outside bounded 7-day window: {age:.0f}s"
        )

    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    if created.tzinfo is None:
        raise ValueError("artifact created_at is not timezone-aware")
    artifact_age = (datetime.now(timezone.utc) - created.astimezone(timezone.utc)).total_seconds()
    if artifact_age < -300 or artifact_age > MAX_AGE_SECONDS:
        raise ValueError(
            f"universe artifact is outside bounded 7-day window: {artifact_age:.0f}s"
        )


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    payload = _api(
        f"https://api.github.com/repos/{repo}/actions/artifacts?per_page=100",
        token,
    )
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
            blob = _download(artifact["archive_download_url"], token)
            with tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / "universe.zip"
                extract = Path(tmp) / "extract"
                archive.write_bytes(blob)
                extract.mkdir()
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(extract)
                src = extract / "data" / "universe" / "latest.json"
                if not src.exists():
                    raise ValueError("artifact does not contain data/universe/latest.json")
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
