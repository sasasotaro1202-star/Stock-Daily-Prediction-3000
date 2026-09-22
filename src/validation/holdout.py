from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Mapping


HOLDOUT_SESSIONS = 63
MIN_PRE_HOLDOUT_SESSIONS = 252


def should_rotate_frozen_holdout(
    lock: Mapping[str, object],
    *,
    result_exists: bool,
    current_research_fingerprint: str,
) -> bool:
    """
    Rotate only after a completed holdout exists and model/research code has
    changed. Workflow-only changes do not force a new blind holdout.
    """
    if not result_exists:
        return False
    if lock.get("status") != "FROZEN":
        raise ValueError(
            "an evaluated frozen holdout must have status FROZEN"
        )
    stored = lock.get("research_code_fingerprint_sha256")
    if not isinstance(stored, str) or not stored:
        raise ValueError(
            "frozen holdout is missing research_code_fingerprint_sha256"
        )
    return stored != current_research_fingerprint


def derive_rolling_holdout_dates(
    dates: Iterable[date],
    *,
    holdout_sessions: int = HOLDOUT_SESSIONS,
    min_pre_holdout_sessions: int = MIN_PRE_HOLDOUT_SESSIONS,
) -> tuple[date, date, date]:
    unique_dates = sorted(set(dates))
    required = min_pre_holdout_sessions + holdout_sessions + 1
    if len(unique_dates) < required:
        raise ValueError(
            f"insufficient sessions ({len(unique_dates)}; require >= {required})"
        )
    cutoff = unique_dates[-(holdout_sessions + 1)]
    holdout = unique_dates[unique_dates.index(cutoff) + 1 :]
    if len(holdout) != holdout_sessions:
        raise ValueError(
            f"unexpected holdout length {len(holdout)}; expected {holdout_sessions}"
        )
    return cutoff, holdout[0], holdout[-1]


def automatic_holdout_lock(
    *,
    cutoff: date,
    holdout_start: date,
    holdout_end: date,
    generation: int,
    research_fingerprint: str,
    git_sha: str | None,
    now: datetime,
    selection_source: str,
) -> dict[str, object]:
    return {
        "cutoff_date": str(cutoff),
        "holdout_start": str(holdout_start),
        "holdout_end": str(holdout_end),
        "cutoff_frozen_at": now.isoformat(),
        "cutoff_frozen_git_sha": git_sha,
        "status": "CUTOFF_FROZEN_PENDING_MODEL",
        "selected_model": None,
        "regime_selected_models": {},
        "holdout_generation": int(generation),
        "research_code_fingerprint_sha256": research_fingerprint,
        "selection_source": selection_source,
    }
