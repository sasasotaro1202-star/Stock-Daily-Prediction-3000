from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOTS=(Path("config"),Path("src"),Path("scripts"),Path(".github/workflows"))
RESEARCH_ROOTS=(Path("config"),Path("src"),Path("scripts"))


def file_hash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint_rows(roots: tuple[Path, ...]) -> dict[str,str]:
    rows={}
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in {".py",".yml",".yaml"}:
                rows[str(p)]=file_hash(p)
    return rows


def file_fingerprint() -> dict[str,str]:
    return _fingerprint_rows(ROOTS)


def research_file_fingerprint() -> dict[str,str]:
    """Fingerprint model/research code without workflow-only changes."""
    return _fingerprint_rows(RESEARCH_ROOTS)


def fingerprint_sha256(rows: dict[str,str] | None = None) -> str:
    payload=rows if rows is not None else file_fingerprint()
    raw=json.dumps(payload,sort_keys=True,separators=(",",":")).encode()
    return hashlib.sha256(raw).hexdigest()


def research_fingerprint_sha256(
    rows: dict[str,str] | None = None,
) -> str:
    payload=rows if rows is not None else research_file_fingerprint()
    return fingerprint_sha256(payload)
