from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from src.data.github_artifact import download_workflow_artifact, validate_extracted_tree


def _get_json(url: str, token: str):
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.load(resp)


def main():
    idx = int(os.environ.get("PRICE_SHARD_INDEX", "0"))
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    name = f"price-state-shard-{idx}"

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

    candidates = [
        a for a in payload.get("artifacts", [])
        if a.get("name") == name and not a.get("expired")
    ]
    if not candidates:
        print("price-state: no previous artifact; initial 5y fetch will run")
        return

    artifact = max(candidates, key=lambda a: a.get("created_at", ""))
    with tempfile.TemporaryDirectory() as tmp:
        extract = Path(tmp) / "artifact"
        download_workflow_artifact(repo, token, artifact, extract)
        validate_extracted_tree(extract)
        matches = [p for p in extract.rglob("*.parquet") if p.is_file()]
        if not matches:
            raise SystemExit(
                f"DEFERRED: price-state artifact {artifact['id']} contains no parquet files"
            )
        destination = Path("data/prices")
        destination.mkdir(parents=True, exist_ok=True)
        for source in matches:
            shutil.copy2(source, destination / source.name)

    print(
        f"price-state: restored artifact_id={artifact['id']} "
        f"created_at={artifact['created_at']} files={len(matches)}"
    )


if __name__ == "__main__":
    main()
