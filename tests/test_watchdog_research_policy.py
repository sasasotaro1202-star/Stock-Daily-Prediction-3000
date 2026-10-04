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

def test_watchdog_recovers_research_without_workflow_run_self_trigger():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'workflows: ["Research validation"]' not in text
    assert "schedule:" in text
    assert "push:" in text
    assert ".github/research_validation.trigger" in text
    assert "inspect_research_validation()" in text


def test_research_watchdog_has_bounded_inactivity_guard_without_replacing_hard_age_guard() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "research_inactive_epoch" in text
    assert "120-minute no-update window" in text
    assert "has had no Actions update for over 120 minutes" in text
    assert "stalled active run $active_run_id cancelled by inactivity guard" in text
    assert "has exceeded 350 minutes of active age" in text
    assert "stale active run $active_run_id cancelled by hard-age guard" in text


def test_superseded_queued_research_can_be_cleaned_behind_active_oos() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    research_start = text.index("inspect_research_validation")
    start = text.index(
        '            if [ -n "$active_run_id" ]; then',
        research_start,
    )
    queue_start = text.index("            local current_sha_queued=false", start)
    active_block = text[start:queue_start]
    queue_block = text[queue_start:]
    assert "superseded queued" not in active_block
    preserve = active_block[
        active_block.index(
            "              else\n                # Keep the active chronological OOS run"
        ):active_block.index(
            "              fi",
            active_block.index(
                "              else\n                # Keep the active chronological OOS run"
            ),
        )
    ]
    assert "continue into queue" in preserve
    assert "return 0" not in preserve
    assert 'if gh run cancel "$run_id" --repo "$GITHUB_REPOSITORY"; then' in queue_block
