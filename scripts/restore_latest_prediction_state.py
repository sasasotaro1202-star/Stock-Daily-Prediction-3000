from __future__ import annotations
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

def main():
    repo=os.environ.get("GITHUB_REPOSITORY")
    token=os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print("prediction-state: no token; no prior state")
        return
    url=f"https://api.github.com/repos/{repo}/actions/artifacts?per_page=100"
    headers={"Authorization":f"Bearer {token}","Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"}
    req=urllib.request.Request(url,headers=headers)
    with urllib.request.urlopen(req,timeout=25) as r: payload=json.load(r)
    arts=[a for a in payload.get("artifacts",[]) if a.get("name")=="prediction-state" and not a.get("expired")]
    if not arts:
        print("prediction-state: no prior artifact")
        return
    art=max(arts,key=lambda a:a.get("created_at",""))
    req=urllib.request.Request(art["archive_download_url"],headers=headers)
    with urllib.request.urlopen(req,timeout=60) as r: blob=r.read()
    Path("data/predictions").mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip",delete=False) as f:
        f.write(blob); tmp=f.name
    with zipfile.ZipFile(tmp) as z: z.extractall("data/predictions")
    Path(tmp).unlink(missing_ok=True)
    print(f"prediction-state: restored artifact_id={art['id']} created_at={art['created_at']}")

if __name__=="__main__":
    main()
