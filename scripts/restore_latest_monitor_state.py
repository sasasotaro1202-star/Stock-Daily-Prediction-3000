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
        and str(a.get("name","")).startswith("monitoring-")
    ]
    if not candidates:
        raise SystemExit("DEFERRED: no previous monitoring artifact")

    artifact=max(candidates,key=lambda a:a.get("created_at",""))
    req=Request(
        artifact["archive_download_url"],
        headers={
            "Authorization":f"Bearer {token}",
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28",
        },
    )
    with urlopen(req,timeout=60) as response:
        blob=response.read()

    out=Path("data/research")
    out.mkdir(parents=True,exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/"monitor.zip"
        archive.write_bytes(blob)
        with zipfile.ZipFile(archive) as z:
            matches=[
                name for name in z.namelist()
                if name.endswith("monitor_latest.json")
            ]
            if not matches:
                raise SystemExit(
                    "DEFERRED: monitoring artifact has no monitor_latest.json"
                )
            name=matches[0]
            with z.open(name) as src:
                payload=src.read()
            (out/"monitor_latest.json").write_bytes(payload)

    print(
        f"monitor-state-restored artifact_id={artifact['id']} "
        f"created_at={artifact['created_at']}"
    )


if __name__=="__main__":
    main()
