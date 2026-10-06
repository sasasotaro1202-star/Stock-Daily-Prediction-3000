from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/actions-reliability-watchdog.yml")


def test_active_research_preserves_control_plane_only_changes() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index(
        '            if [ -n "$active_run_id" ]; then',
        text.index("inspect_research_validation"),
    )
    end = text.index("            local current_sha_queued=false", start)
    block = text[start:end]
    assert "control-plane-only main changes" in block
    assert "evidence-affecting changes invalidate the run" in block.lower()
    assert 'gh run cancel "$active_run_id"' in block
    assert "350 minutes" in block


def test_watchdog_mirrors_evidence_fingerprint_scope_for_active_run_invalidation() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "evidence_diff_state()" in text
    assert 'pyproject.toml)' in text
    assert "config/*|src/*|scripts/*" in text
    assert "scripts/automation_invariants.py|scripts/project_source_contract.py" in text
    assert "*.py|*.yml|*.yaml" in text
    assert "Full history is required" in text


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
    assert "    - cron: \"*/5 * * * *\"" in text
    assert "  push:" not in text
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


def test_daily_watchlist_can_recover_recent_superseded_failure() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'local recover_superseded_failure="${5:-false}"' in text
    assert "superseded_recent_failure_found=false" in text
    assert "recent failure on superseded SHA" in text
    assert "superseded-failure recovery dispatched on current main" in text
    assert 'inspect_workflow "daily-watchlist.yml" "Daily priority watchlist" true true true' in text


def test_daily_watchlist_missing_schedule_has_bounded_recovery() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "recover_daily_watchlist_missed_schedule()" in text
    assert 'workflow_file="daily-watchlist.yml"' in text
    assert 'weekday="$(TZ=Asia/Tokyo date +%u)"' in text
    assert "19:35:00" in text
    assert "18:40:00" in text
    assert '--argjson now "${now_epoch}"' in text
    assert "missed-schedule recovery dispatched on current main" in text
    assert "no Daily Priority Watchlist run was created" in text


def test_daily_watchlist_missing_schedule_recovery_is_invoked() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    anchor = 'inspect_workflow "daily-watchlist.yml" "Daily priority watchlist" true true true'
    assert anchor in text
    assert text.index("recover_daily_watchlist_missed_schedule()", text.index(anchor)) > text.index(anchor)


def test_watchdog_covers_short_research_collectors_and_watchlist() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    expected = (
        'inspect_workflow "daily-watchlist.yml" "Daily priority watchlist" true true',
        'inspect_workflow "free-data-source-discovery.yml" "Free Data Source Discovery" true true',
        'inspect_workflow "boj-frontier-research.yml" "BOJ Frontier Research" true true',
        'inspect_workflow "fred-cross-asset-research.yml" "FRED Cross-Asset Research" true true',
        'inspect_workflow "sec-filings-research.yml" "SEC filings research" true true',
        'inspect_workflow "treasury-rate-family-research.yml" "U.S. Treasury Rate Family Research" true true',
    )
    for needle in expected:
        assert needle in text
