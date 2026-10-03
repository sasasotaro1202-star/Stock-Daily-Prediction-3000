from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MANIFEST_PATH = Path("data/research/oos_run_manifest.json")
PROGRESS_PATH = Path("data/research/oos_progress.json")

_TRACKED_PATHS = (
    "scripts/run_daily_research.py",
    "src/research/ultimate_v13.py",
    "config/pipeline.yml",
    "config/universe.yml",
    "PROJECT_INSTRUCTIONS.md",
    "docs/PROJECT_SOURCE.md",
    "config/frozen_holdout.json",
)


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_context() -> dict[str, Any]:
    return {
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "sha": os.environ.get("GITHUB_SHA", ""),
        "ref": os.environ.get("GITHUB_REF", ""),
        "ref_name": os.environ.get("GITHUB_REF_NAME", ""),
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "runner_os": os.environ.get("RUNNER_OS", ""),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def build_manifest(
    *,
    status: str,
    stage: str,
    fold_count: int | None = None,
    fold_signature: str | None = None,
    candidate_models: list[str] | None = None,
) -> dict[str, Any]:
    tracked = {path: _sha256_file(Path(path)) for path in _TRACKED_PATHS}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "status": str(status),
        "stage": str(stage),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "runtime": _runtime_context(),
        "artifact_integrity": {
            "hash_algorithm": "sha256",
            "tracked_files": tracked,
            "missing_tracked_files": sorted(
                path for path, digest in tracked.items() if digest is None
            ),
        },
        "folds": {
            "count": int(fold_count) if fold_count is not None else None,
            "signature": fold_signature,
        },
        "candidate_models": [str(name) for name in (candidate_models or [])],
        "holdout_policy": {
            "frozen_holdout_used_for_selection": False,
            "frozen_holdout_tuning_allowed": False,
        },
    }


def write_manifest(
    *,
    status: str,
    stage: str,
    fold_count: int | None = None,
    fold_signature: str | None = None,
    candidate_models: list[str] | None = None,
) -> dict[str, Any]:
    payload = build_manifest(
        status=status,
        stage=stage,
        fold_count=fold_count,
        fold_signature=fold_signature,
        candidate_models=candidate_models,
    )
    _write_json_atomic(MANIFEST_PATH, payload)
    return payload


def write_progress(
    *,
    event: str,
    status: str = "RUNNING",
    model: str | None = None,
    model_index: int | None = None,
    total_models: int | None = None,
    fold: int | None = None,
    total_folds: int | None = None,
    completed_folds: int | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at_utc": _utc_now(),
        "status": str(status),
        "event": str(event),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "runtime": _runtime_context(),
        "progress": {
            "model": None if model is None else str(model),
            "model_index": int(model_index) if model_index is not None else None,
            "total_models": int(total_models) if total_models is not None else None,
            "fold": int(fold) if fold is not None else None,
            "total_folds": int(total_folds) if total_folds is not None else None,
            "completed_folds": (
                int(completed_folds) if completed_folds is not None else None
            ),
            "note": None if note is None else str(note),
        },
    }
    _write_json_atomic(PROGRESS_PATH, payload)
    return payload


def finalize_progress(*, outcome: str, note: str | None = None) -> dict[str, Any]:
    status = {
        "success": "COMPLETED",
        "failure": "FAILED",
        "cancelled": "CANCELLED",
        "skipped": "SKIPPED",
    }.get(str(outcome), "UNKNOWN")
    current: dict[str, Any] = {}
    if PROGRESS_PATH.exists():
        try:
            current = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
    current.update(
        {
            "schema_version": SCHEMA_VERSION,
            "updated_at_utc": _utc_now(),
            "status": status,
            "event": "run_finalized",
            "outcome": str(outcome),
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
            "runtime": _runtime_context(),
        }
    )
    if note is not None:
        progress = dict(current.get("progress") or {})
        progress["note"] = str(note)
        current["progress"] = progress
    _write_json_atomic(PROGRESS_PATH, current)
    return current


__all__ = [
    "MANIFEST_PATH",
    "PROGRESS_PATH",
    "build_manifest",
    "write_manifest",
    "write_progress",
    "finalize_progress",
]
