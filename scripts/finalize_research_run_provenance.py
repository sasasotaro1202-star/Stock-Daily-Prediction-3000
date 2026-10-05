from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from src.research.run_provenance import MANIFEST_PATH, PROGRESS_PATH, finalize_progress


def _final_status(outcome: str) -> str:
    return {
        "success": "COMPLETED",
        "failure": "FAILED",
        "cancelled": "CANCELLED",
        "skipped": "SKIPPED",
    }.get(str(outcome), "UNKNOWN")


def main() -> None:
    outcome = os.environ.get("RESEARCH_OOS_OUTCOME", "").strip() or "unknown"
    status = _final_status(outcome)

    manifest = {}
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}
    manifest.update(
        {
            "schema_version": int(manifest.get("schema_version", 1)),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "stage": "finalized",
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
        }
    )
    manifest["finalization"] = {
        "outcome": outcome,
        "note": (
            "Final status reflects the GitHub Actions research_oos outcome; "
            "no performance claim is implied."
        ),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_name(f".{MANIFEST_PATH.name}.tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(MANIFEST_PATH)

    if not PROGRESS_PATH.exists():
        # The runner normally creates progress at startup, but a missing file
        # must not prevent final status evidence from being generated.
        PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    finalize_progress(
        outcome=outcome,
        note=(
            "Final status reflects the GitHub Actions research_oos step outcome; "
            "no performance claim is implied."
        ),
    )


if __name__ == "__main__":
    main()
