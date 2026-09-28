from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/actions-reliability-watchdog.yml")


def test_active_research_is_not_cancelled_just_for_newer_main_sha() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index(
        '            if [ -n "$active_run_id" ]; then',
        text.index("inspect_research_validation"),
    )
    end = text.index("            local current_sha_queued=false", start)
    block = text[start:end]
    assert "superseded active run" not in block
    assert 'gh run cancel "$active_run_id"' in block
    assert "350 minutes" in block
    assert "Preserve a running chronological OOS" in block


def test_watchdog_still_cancels_only_hard_age_expired_active_research() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "has exceeded 350 minutes of active age" in text
    assert "stale active run $active_run_id cancelled by hard-age guard" in text


def test_stale_heartbeat_has_bounded_self_recovery() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "dispatch_heartbeat_recovery()" in text
    assert "for attempt in 1 2 3" in text
    assert "gh workflow run heartbeat.yml --repo" in text
    assert "bounded heartbeat dispatch retries exhausted" in text
