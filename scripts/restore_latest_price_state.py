from __future__ import annotations
import json, os, subprocess, tempfile, zipfile
from pathlib import Path
import urllib.request

def main():
    idx=int(os.environ.get("PRICE_SHARD_INDEX","0"))
    repo=os.environ["GITHUB_REPOSITORY"]
    token=os.environ["GITHUB_TOKEN"]
    name=f"price-state-shard-{idx}"
    query=f"https://api.github.com/repos/{repo}/actions/artifacts?per_page=100"
    req=urllib.request.Request(query,headers={
        "Authorization":f"Bearer {token}",
        "Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28",
    })
    with urllib.request.urlopen(req,timeout=25) as resp:
        payload=json.load(resp)
    candidates=[a for a in payload.get("artifacts",[]) if a.get("name")==name and not a.get("expired")]
    if not candidates:
        print("price-state: no previous artifact; initial 5y fetch will run")
        return
    artifact=max(candidates,key=lambda a:a.get("created_at",""))
    req=urllib.request.Request(artifact["archive_download_url"],headers={
        "Authorization":f"Bearer {token}",
        "Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28",
    })
    with urllib.request.urlopen(req,timeout=60) as resp:
        blob=resp.read()
    Path("data/prices").mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip",delete=False) as f:
        f.write(blob); tmp=f.name
    with zipfile.ZipFile(tmp) as z:
        z.extractall("data/prices")
    Path(tmp).unlink(missing_ok=True)
    print(f"price-state: restored artifact_id={artifact['id']} created_at={artifact['created_at']}")

if __name__=="__main__":
    main()
