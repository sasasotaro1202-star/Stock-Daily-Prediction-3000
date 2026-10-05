from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/long-research-recovery.yml"
WATCHDOG = ROOT / ".github/workflows/actions-reliability-watchdog.yml"
FAILURE_LEARNING = ROOT / ".github/workflows/automation-failure-learning.yml"
RECONCILE = ROOT / "scripts/reconcile_automation_failures.py"


def test_long_research_recovery_is_periodic_and_bounded() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert '    - cron: "*/10 * * * *"' in text
    assert "Never cancel an in-progress long collector here." in text
    assert "run_attempt // 1" in text
    assert 'gh run rerun "$run_id" --repo "$GITHUB_REPOSITORY" --failed' in text
    assert "At most one recovery action per workflow on each watchdog tick." in text


def test_long_research_recovery_dispatches_current_main_only_after_queue_cleanup() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "git/ref/heads/main" in text
    assert 'gh workflow run "$workflow_file" --repo "$GITHUB_REPOSITORY" --ref main' in text
    assert "stale_queued_ids=()" in text


def test_long_research_recovery_controller_is_self_monitored_and_learned() -> None:
    assert 'inspect_workflow "long-research-recovery.yml" "Long research recovery" true true' in WATCHDOG.read_text(encoding="utf-8")
    assert '- "Long research recovery"' in FAILURE_LEARNING.read_text(encoding="utf-8")
    assert '"long-research-recovery.yml"' in RECONCILE.read_text(encoding="utf-8")
