from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from src.data.github_artifact import download_workflow_artifact, validate_extracted_tree


def api(url: str, token: str) -> dict:

    req=urllib.request.Request(
        url,
        headers={
            "Authorization":f"Bearer {token}",
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2026-03-10",
        },
    )
    with urllib.request.urlopen(req,timeout=30) as response:
        return json.load(response)


def main():
    repo=os.environ["GITHUB_REPOSITORY"]
    token=os.environ["GITHUB_TOKEN"]
    artifacts=api(
        f"https://api.github.com/repos/{repo}/actions/artifacts?per_page=100",
        token,
    ).get("artifacts",[])
    candidates=[
        a for a in artifacts
        if not a.get("expired")
        and str(a.get("name","")).startswith(
            ("morning-prediction-","research-cycle-")
        )
    ]
    target=Path("data/predictions")
    target.mkdir(parents=True,exist_ok=True)
    restored=0
    errors=[]

    for artifact in sorted(
        candidates,
        key=lambda a:a.get("created_at",""),
        reverse=True,
    )[:20]:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                extract = Path(tmp) / "artifact"
                download_workflow_artifact(repo, token, artifact, extract)
                validate_extracted_tree(extract)
                matches = [
                    p for p in extract.rglob("*.parquet")
                    if p.is_file() and "predictions" in p.parts
                ]
                for source in matches:
                    shutil.copy2(source, target / source.name)
                restored += len(matches)
        except Exception as exc:
            errors.append({
                "artifact_id": artifact.get("id"),
                "error": repr(exc),
            })
            print(
                "prediction_history_restore_deferred",
                artifact.get("id"),
                repr(exc),
            )

    if errors:
        raise SystemExit(
            "DEFERRED: one or more prediction-history artifacts could not be restored"
        )

    print(f"prediction-history-restored={restored}")


if __name__=="__main__":
    main()
