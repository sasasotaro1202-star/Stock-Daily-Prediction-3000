from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

MAX_AGE_SECONDS = 7 * 24 * 60 * 60
REFRESH_TIMEOUT_SECONDS = 240


def _validate_retained_snapshot(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit("FAIL: retained universe snapshot missing")

    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(
            f"FAIL: retained universe snapshot unreadable: {type(exc).__name__}:{exc}"
        ) from exc

    records = snapshot.get("records")
    if not isinstance(records, list) or len(records) < 100:
        raise SystemExit("FAIL: retained universe has fewer than 100 records")

    if int(snapshot.get("record_count", -1)) != len(records):
        raise SystemExit("FAIL: retained universe record_count mismatch")

    retrieved_at = snapshot.get("retrieved_at")
    if not retrieved_at:
        raise SystemExit("FAIL: retained universe lacks retrieved_at")
    try:
        retrieved = datetime.fromisoformat(str(retrieved_at).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit("FAIL: retained universe retrieved_at is invalid") from exc
    if retrieved.tzinfo is None:
        raise SystemExit("FAIL: retained universe retrieved_at is not timezone-aware")

    age_seconds = (
        datetime.now(timezone.utc) - retrieved.astimezone(timezone.utc)
    ).total_seconds()
    if age_seconds < -300 or age_seconds > MAX_AGE_SECONDS:
        raise SystemExit(
            "FAIL: retained universe retrieved_at is outside bounded 7-day window"
        )

    if not snapshot.get("source_hashes"):
        raise SystemExit("FAIL: retained universe lacks source_hashes")

    subprocess.run(
        ["python", "scripts/universe_quality_gate.py"],
        check=True,
        timeout=120,
    )
    return {
        "record_count": len(records),
        "retrieved_at": snapshot["retrieved_at"],
        "age_seconds": age_seconds,
    }


def main() -> None:
    latest = Path("data/universe/latest.json")
    try:
        subprocess.run(
            ["python", "scripts/refresh_universe.py"],
            check=True,
            timeout=REFRESH_TIMEOUT_SECONDS,
        )
        subprocess.run(
            ["python", "scripts/universe_quality_gate.py"],
            check=True,
            timeout=120,
        )
        print("UNIVERSE_REFRESH status=REFRESHED")
        return
    except subprocess.TimeoutExpired:
        retained = _validate_retained_snapshot(latest)
        print(
            "UNIVERSE_REFRESH status=DEFERRED_RETAINED_EXISTING "
            "reason=live_refresh_timeout "
            f"record_count={retained['record_count']} "
            f"retrieved_at={retained['retrieved_at']}"
        )
        return


if __name__ == "__main__":
    main()
