from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def main():
    repo=os.environ["GITHUB_REPOSITORY"]
    token=os.environ["GITHUB_TOKEN"]
    query=f"https://api.github.com/repos/{repo}/actions/artifacts?per_page=100"
    req=urllib.request.Request(
        query,
        headers={
            "Authorization":f"Bearer {token}",
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28",
        },
    )
    with urllib.request.urlopen(req,timeout=25) as resp:
        payload=json.load(resp)

    candidates=[
        a for a in payload.get("artifacts",[])
        if str(a.get("name","")).startswith("research-cycle-")
        and not a.get("expired")
    ]
    if not candidates:
        raise SystemExit("DEFERRED: no previous research-cycle artifact")

    artifact=max(candidates,key=lambda a:a.get("created_at",""))
    req=urllib.request.Request(
        artifact["archive_download_url"],
        headers={
            "Authorization":f"Bearer {token}",
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28",
        },
    )
    with urllib.request.urlopen(req,timeout=60) as resp:
        blob=resp.read()

    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/"research.zip"
        archive.write_bytes(blob)
        extract=Path(tmp)/"extract"
        extract.mkdir()
        with zipfile.ZipFile(archive) as z:
            z.extractall(extract)

        src=extract/"data"/"research"
        if not src.exists():
            raise SystemExit("DEFERRED: previous research artifact has no data/research")

        dst=Path("data/research")
        dst.mkdir(parents=True,exist_ok=True)
        for item in src.iterdir():
            target=dst/item.name
            if item.is_file():
                shutil.copy2(item,target)

    print(
        f"research-state: restored artifact_id={artifact['id']} "
        f"created_at={artifact['created_at']}"
    )


if __name__=="__main__":
    main()
