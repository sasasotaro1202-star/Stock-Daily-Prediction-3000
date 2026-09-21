from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from datetime import datetime, timezone

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    out=Path("data/research/reproducibility_manifest.json")
    rows=[]
    for root in (Path("config"),Path("src"),Path("scripts")):
        if not root.exists(): continue
        for p in sorted(root.rglob("*.py")):
            rows.append({"path":str(p),"sha256":sha256_file(p)})
        for p in sorted(root.rglob("*.yml")):
            rows.append({"path":str(p),"sha256":sha256_file(p)})
        for p in sorted(root.rglob("*.yaml")):
            rows.append({"path":str(p),"sha256":sha256_file(p)})
    universe=Path("data/universe/latest.json")
    payload={
        "created_at":datetime.now(timezone.utc).isoformat(),
        "git_sha":os.getenv("GITHUB_SHA"),
        "universe_sha256":sha256_file(universe) if universe.exists() else None,
        "files":rows,
        "status":"REPRODUCIBLE_MANIFEST_CREATED",
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
