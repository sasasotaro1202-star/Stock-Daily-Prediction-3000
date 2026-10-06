from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
from urllib.error import HTTPError
from pathlib import Path
import time

from src.data.github_artifact import download_workflow_artifact, validate_extracted_tree


GITHUB_API_TIMEOUT_SECONDS = int(os.getenv("GITHUB_API_TIMEOUT_SECONDS", "60"))
GITHUB_API_RETRY_ATTEMPTS = max(1, int(os.getenv("GITHUB_API_RETRY_ATTEMPTS", "3")))


def _get_json(url: str, token: str):
    last_error: Exception | None = None
    for attempt in range(GITHUB_API_RETRY_ATTEMPTS):
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(
                req,
                timeout=GITHUB_API_TIMEOUT_SECONDS,
            ) as resp:
                return json.load(resp)
        except Exception as exc:
            last_error = exc
            print(
                f"github-artifact-api-retry attempt={attempt + 1}/"
                f"{GITHUB_API_RETRY_ATTEMPTS} error={type(exc).__name__}"
            )
            if attempt < GITHUB_API_RETRY_ATTEMPTS - 1:
                time.sleep(min(15, 2 ** attempt))
    assert last_error is not None
    raise last_error


def main():
    idx = int(os.environ.get("PRICE_SHARD_INDEX", "0"))
    shard_count = max(1, int(os.environ.get("PRICE_SHARD_COUNT", "1")))
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]

    query = f"https://api.github.com/repos/{repo}/actions/artifacts?per_page=100"
    try:
        payload = _get_json(query, token)
    except HTTPError as exc:
        if exc.code in {401, 403}:
            print(
                "price-state: GitHub artifact API authentication unavailable "
                f"(HTTP {exc.code}); falling back to bounded fresh price fetch"
            )
            return
        raise
    except Exception as exc:
        destination = Path("data/prices")
        destination.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": 1,
            "status": "DEFERRED",
            "expected_shards": shard_count,
            "restored_shards": 0,
            "restored_files": 0,
            "artifact_names": [],
            "errors": [{"stage": "artifact_api", "error": repr(exc)}],
        }
        (destination / "price_state_restore.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(
            "::warning title=Price state API unavailable::"
            f"artifact lookup failed with {type(exc).__name__}; "
            "falling back to bounded fresh price fetch"
        )
        return

    artifacts = [
        a for a in payload.get("artifacts", [])
        if not a.get("expired")
    ]

    if shard_count == 1:
        shard_indices = [idx]
    else:
        shard_indices = list(range(shard_count))

    selected = []
    for shard in shard_indices:
        preferred_name = f"price-state-shard-{shard}"
        research_prefix = f"research-validation-price-{shard}-"
        candidates = [
            a for a in artifacts
            if (
                a.get("name") == preferred_name
                or str(a.get("name", "")).startswith(research_prefix)
            )
        ]
        if candidates:
            selected.append(
                max(candidates, key=lambda a: a.get("created_at", ""))
            )

    destination = Path("data/prices")
    destination.mkdir(parents=True, exist_ok=True)
    restored_shards = []
    restored_files = 0
    errors = []

    for artifact in selected:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                extract = Path(tmp) / "artifact"
                download_workflow_artifact(repo, token, artifact, extract)
                validate_extracted_tree(extract)
                matches = [p for p in extract.rglob("*.parquet") if p.is_file()]
                if not matches:
                    raise RuntimeError(
                        f"price-state artifact {artifact['id']} contains no parquet files"
                    )
                for source in matches:
                    shutil.copy2(source, destination / source.name)
                restored_files += len(matches)
                restored_shards.append(
                    str(artifact.get("name") or artifact.get("id"))
                )
        except Exception as exc:
            errors.append({
                "artifact_id": artifact.get("id"),
                "artifact_name": artifact.get("name"),
                "error": repr(exc),
            })
            print(
                "::warning title=Price state shard restore deferred::"
                f"artifact_id={artifact.get('id')} error={type(exc).__name__}"
            )

    status = (
        "PASS"
        if len(restored_shards) == len(shard_indices) and not errors
        else "DEFERRED"
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "expected_shards": len(shard_indices),
        "restored_shards": len(restored_shards),
        "restored_files": restored_files,
        "artifact_names": restored_shards,
        "errors": errors[:20],
    }
    (destination / "price_state_restore.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if not selected:
        print("price-state: no previous artifact; initial full-universe refresh may be required")
    else:
        print(
            f"price-state: restored_shards={len(selected)}/{len(shard_indices)} "
            f"files={restored_files} status={status}"
        )


if __name__ == "__main__":
    main()
