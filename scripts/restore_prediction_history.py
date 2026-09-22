from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


def api(url: str, token: str) -> dict:
    req=Request(
        url,
        headers={
            "Authorization":f"Bearer {token}",
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28",
        },
    )
    with urlopen(req,timeout=30) as response:
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
        req=Request(
            artifact["archive_download_url"],
            headers={
                "Authorization":f"Bearer {token}",
                "Accept":"application/vnd.github+json",
                "X-GitHub-Api-Version":"2022-11-28",
            },
        )
        try:
            with urlopen(req,timeout=60) as response:
                blob=response.read()
            with tempfile.TemporaryDirectory() as tmp:
                archive=Path(tmp)/"artifact.zip"
                archive.write_bytes(blob)
                with zipfile.ZipFile(archive) as z:
                    for name in z.namelist():
                        normalized="/"+name.lstrip("/")
                        if (
                            name.endswith(".parquet")
                            and "/data/predictions/" in normalized
                        ):
                            out=target/Path(name).name
                            with z.open(name) as src, out.open("wb") as dst:
                                dst.write(src.read())
                            restored+=1
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
