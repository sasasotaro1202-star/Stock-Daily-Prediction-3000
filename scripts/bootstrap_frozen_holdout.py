from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.validation.code_fingerprint import research_fingerprint_sha256
from src.validation.holdout import (
    HOLDOUT_SESSIONS,
    MIN_PRE_HOLDOUT_SESSIONS,
    automatic_holdout_lock,
    derive_rolling_holdout_dates,
    should_rotate_frozen_holdout,
)


LOCK = Path("config/frozen_holdout.json")
RESULT = Path("data/research/frozen_holdout_result.json")


def _archive_previous_generation(lock: dict[str, object]) -> Path:
    generation = int(lock.get("holdout_generation", 1))
    cutoff = str(lock.get("cutoff_date", "unknown"))
    fingerprint = str(
        lock.get("research_code_fingerprint_sha256", "unknown")
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = (
        Path("artifacts/frozen_holdout_history")
        / f"generation-{generation:04d}-{cutoff}-{stamp}-{fingerprint[:12]}"
    )
    archive.mkdir(parents=True, exist_ok=False)

    files = {
        "config_frozen_holdout.json": LOCK,
        "frozen_holdout_result.json": RESULT,
        "latest_metrics.json": Path("data/research/latest_metrics.json"),
        "release_gate.json": Path("data/research/release_gate.json"),
        "reproducibility_manifest.json": Path(
            "data/research/reproducibility_manifest.json"
        ),
    }
    for name, src in files.items():
        if src.exists():
            shutil.copy2(src, archive / name)
    return archive


def _new_generation(
    df: pd.DataFrame,
    previous: dict[str, object] | None,
) -> dict[str, object]:
    dates = sorted(pd.to_datetime(df["session_date"]).dt.date.unique())
    try:
        cutoff, holdout_start, holdout_end = derive_rolling_holdout_dates(
            dates,
            holdout_sessions=HOLDOUT_SESSIONS,
            min_pre_holdout_sessions=MIN_PRE_HOLDOUT_SESSIONS,
        )
    except ValueError as exc:
        raise SystemExit(f"DEFERRED: {exc}") from exc

    generation = int((previous or {}).get("holdout_generation", 0)) + 1
    return automatic_holdout_lock(
        cutoff=cutoff,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        generation=generation,
        research_fingerprint=research_fingerprint_sha256(),
        git_sha=os.getenv("GITHUB_SHA"),
        now=datetime.now(timezone.utc),
        selection_source=(
            "automatic rolling latest-minus-63-session cutoff; prior evaluated "
            "holdout archived and excluded from current selection"
        ),
    )


def main():
    current_research_fp = research_fingerprint_sha256()

    if LOCK.exists():
        previous = json.loads(LOCK.read_text(encoding="utf-8"))
        result_exists = RESULT.exists()
        try:
            rotate = should_rotate_frozen_holdout(
                previous,
                result_exists=result_exists,
                current_research_fingerprint=current_research_fp,
            )
        except ValueError as exc:
            raise SystemExit(f"FAIL: {exc}") from exc

        if rotate:
            prices = Path("data/prices")
            if not prices.exists():
                raise SystemExit(
                    "DEFERRED: price data is required to rotate the frozen holdout"
                )
            df = pd.read_parquet(prices)
            archive = _archive_previous_generation(previous)
            new_lock = _new_generation(df, previous)
            LOCK.write_text(
                json.dumps(new_lock, indent=2),
                encoding="utf-8",
            )
            RESULT.unlink(missing_ok=True)
            print(
                json.dumps(
                    {
                        **new_lock,
                        "rotated_from_generation": previous.get(
                            "holdout_generation"
                        ),
                        "archived_at": str(archive),
                    },
                    indent=2,
                )
            )
            return

        print("frozen-holdout: already locked")
        return

    root=Path("data/prices")
    if not root.exists():
        raise SystemExit("DEFERRED: price data is required")
    df=pd.read_parquet(root)
    lock = _new_generation(df, None)
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    LOCK.write_text(
        json.dumps(lock,indent=2),
        encoding="utf-8",
    )
    print(json.dumps(lock,indent=2))


if __name__=="__main__": main()
