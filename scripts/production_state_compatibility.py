from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.validation.code_fingerprint import file_fingerprint, fingerprint_sha256


ROOTS=(Path("config"),Path("src"),Path("scripts"),Path(".github/workflows"))


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def current_fingerprint() -> dict[str,str]:
    return file_fingerprint()


def main():
    manifest_path=Path("data/research/reproducibility_manifest.json")
    if not manifest_path.exists():
        raise SystemExit("DEFERRED: reproducibility manifest is absent")

    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    expected={
        str(row["path"]):str(row["sha256"])
        for row in manifest.get("files",[])
    }
    current=current_fingerprint()
    current_fp=fingerprint_sha256(current)

    mismatches=[]
    for path,digest in expected.items():
        if path not in current:
            mismatches.append(f"missing:{path}")
        elif current[path]!=digest:
            mismatches.append(f"changed:{path}")

    extra=[
        path for path in current
        if path not in expected
        and not path.startswith("data/")
    ]
    mismatches.extend(f"unrecorded:{p}" for p in extra)

    if manifest.get("code_fingerprint_sha256") != current_fp:
        mismatches.append("code_fingerprint_sha256_mismatch")

    result={
        "status":"PASS" if not mismatches else "DEFERRED",
        "manifest_git_sha":manifest.get("git_sha"),
        "manifest_code_fingerprint_sha256":manifest.get("code_fingerprint_sha256"),
        "current_code_fingerprint_sha256":current_fp,
        "mismatches":mismatches,
    }
    Path("data/research").mkdir(parents=True,exist_ok=True)
    Path("data/research/state_compatibility.json").write_text(
        json.dumps(result,indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result,indent=2))
    if mismatches:
        raise SystemExit("DEFERRED: approved research state does not match current code")


if __name__=="__main__":
    main()
