from __future__ import annotations

import os

from src.research.run_provenance import finalize_progress, write_manifest


def main() -> None:
    outcome = os.environ.get("RESEARCH_OOS_OUTCOME", "").strip() or "unknown"
    status = {
        "success": "COMPLETED",
        "failure": "FAILED",
        "cancelled": "CANCELLED",
        "skipped": "SKIPPED",
    }.get(outcome, "UNKNOWN")
    write_manifest(
        status=status,
        stage="finalized",
        note=None if outcome == "success" else (
            "Final status reflects the GitHub Actions research_oos outcome; "
            "no performance claim is implied."
        ),
    )
    finalize_progress(
        outcome=outcome,
        note=(
            "Final status reflects the GitHub Actions research_oos step outcome; "
            "no performance claim is implied."
        ),
    )


if __name__ == "__main__":
    main()
