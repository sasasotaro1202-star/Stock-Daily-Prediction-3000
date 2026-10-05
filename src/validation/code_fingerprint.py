from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOTS=(Path("config"),Path("src"),Path("scripts"),Path(".github/workflows"))
RESEARCH_ROOTS=(Path("config"),Path("src"),Path("scripts"))

# Workflow files that only supervise/reconcile the control plane and cannot
# change model, feature, target, PIT, scoring, routing, calibration, selection,
# adoption, or candidate identity. Keep this explicit and fail-closed for any
# new/unlisted workflow: unknown workflow changes remain evidence-affecting.
# Source-only files used to verify automation/document contracts. They cannot alter
# model, feature, target, PIT, scoring, routing, calibration, selection, or candidate
# identity, so changing them must not invalidate active OOS/frozen research evidence.
NON_EVIDENCE_CONTROL_PLANE_FILES=frozenset(
    {
        "scripts/automation_invariants.py",
        "scripts/project_source_contract.py",
    }
)

NON_EVIDENCE_CONTROL_PLANE_WORKFLOWS=frozenset(
    {
        ".github/workflows/heartbeat.yml",
        ".github/workflows/automation-supervisor.yml",
        ".github/workflows/actions-reliability-watchdog.yml",
        ".github/workflows/research-autopilot.yml",
        ".github/workflows/long-research-recovery.yml",
        ".github/workflows/automation-failure-learning.yml",
        ".github/workflows/research-validation-status.yml",
        ".github/workflows/24h-research-marathon-watchdog.yml",
        ".github/workflows/bounded-production-recovery.yml",
    }
)
EVIDENCE_ROOTS=(Path("config"),Path("src"),Path("scripts"))
EVIDENCE_FILES=(Path("pyproject.toml"),)


def file_hash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint_rows(
    roots: tuple[Path, ...],
    excluded_paths: frozenset[str] = frozenset(),
) -> dict[str,str]:
    rows={}
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            path_str = str(p)
            if path_str in excluded_paths:
                continue
            if p.is_file() and p.suffix in {".py",".yml",".yaml"}:
                rows[path_str]=file_hash(p)
    return rows


def file_fingerprint() -> dict[str,str]:
    return _fingerprint_rows(ROOTS)


def research_file_fingerprint() -> dict[str,str]:
    """Fingerprint model/research code while excluding control-plane contract scripts."""
    return _fingerprint_rows(RESEARCH_ROOTS, NON_EVIDENCE_CONTROL_PLANE_FILES)


def evidence_file_fingerprint() -> dict[str,str]:
    """Fingerprint files capable of changing research/OOS evidence.

    Control-plane-only workflow changes are deliberately excluded via an
    explicit allowlist. Any unlisted workflow remains evidence-affecting.
    """
    rows = _fingerprint_rows(EVIDENCE_ROOTS, NON_EVIDENCE_CONTROL_PLANE_FILES)
    workflows = Path(".github/workflows")
    if workflows.exists():
        for path in sorted(workflows.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".yml", ".yaml"}:
                continue
            path_str = str(path)
            if path_str in NON_EVIDENCE_CONTROL_PLANE_WORKFLOWS:
                continue
            rows[path_str] = file_hash(path)
    for path in EVIDENCE_FILES:
        if path.exists() and path.is_file():
            rows[str(path)] = file_hash(path)
    return rows


def fingerprint_sha256(rows: dict[str,str] | None = None) -> str:
    payload=rows if rows is not None else file_fingerprint()
    raw=json.dumps(payload,sort_keys=True,separators=(",",":")).encode()
    return hashlib.sha256(raw).hexdigest()


def research_fingerprint_sha256(
    rows: dict[str,str] | None = None,
) -> str:
    payload=rows if rows is not None else research_file_fingerprint()
    return fingerprint_sha256(payload)


def evidence_fingerprint_sha256(
    rows: dict[str,str] | None = None,
) -> str:
    payload = rows if rows is not None else evidence_file_fingerprint()
    return fingerprint_sha256(payload)
