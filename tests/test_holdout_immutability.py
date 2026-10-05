from datetime import date

import pytest

from src.validation.holdout import (
    automatic_holdout_lock,
    derive_rolling_holdout_dates,
    should_rotate_frozen_holdout,
)


def test_holdout_rotation_requires_completed_result_and_research_change():
    lock = {"status": "FROZEN", "research_code_fingerprint_sha256": "old"}
    assert not should_rotate_frozen_holdout(
        lock,
        result_exists=False,
        current_research_fingerprint="new",
    )
    assert should_rotate_frozen_holdout(
        lock,
        result_exists=True,
        current_research_fingerprint="new",
    )
    assert not should_rotate_frozen_holdout(
        lock,
        result_exists=True,
        current_research_fingerprint="old",
    )


def test_holdout_rotation_migrates_legacy_evaluated_lock_without_fingerprint():
    assert should_rotate_frozen_holdout(
        {"status": "FROZEN"},
        result_exists=True,
        current_research_fingerprint="new",
    )


def test_holdout_rotation_rejects_non_frozen_state_with_result():
    with pytest.raises(ValueError, match="status FROZEN"):
        should_rotate_frozen_holdout(
            {"status": "CUTOFF_FROZEN_PENDING_MODEL"},
            result_exists=True,
            current_research_fingerprint="new",
        )


def test_rolling_holdout_uses_latest_63_sessions_without_shortening():
    dates = [date(2020, 1, 1)]
    dates.extend(date.fromordinal(dates[-1].toordinal() + i) for i in range(1, 252 + 63 + 1))
    cutoff, start, end = derive_rolling_holdout_dates(dates)
    assert cutoff < start <= end
    assert dates.index(cutoff) == len(dates) - 64
    assert dates.index(end) == len(dates) - 1


def test_automatic_lock_records_generation_and_research_fingerprint():
    payload = automatic_holdout_lock(
        cutoff=date(2026, 8, 1),
        holdout_start=date(2026, 8, 2),
        holdout_end=date(2026, 10, 3),
        generation=4,
        research_fingerprint="abc123",
        git_sha="deadbeef",
        now=__import__("datetime").datetime(2026, 10, 4),
        selection_source="test",
    )
    assert payload["status"] == "CUTOFF_FROZEN_PENDING_MODEL"
    assert payload["holdout_generation"] == 4
    assert payload["research_code_fingerprint_sha256"] == "abc123"
