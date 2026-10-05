from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = ROOT / ".github/workflows/actions-reliability-watchdog.yml"


def test_controller_heartbeat_recovery_is_bounded_and_active_safe() -> None:
    text = WATCHDOG.read_text(encoding="utf-8")
    assert "recover_missing_periodic_controller() {" in text
    assert 'if [ "$active_found" = true ]; then' in text
    assert "active run exists; controller heartbeat recovery suppressed" in text
    assert "no current-main run within cadence window; dispatching bounded recovery" in text
    assert 'gh workflow run "$workflow_file" --repo "$GITHUB_REPOSITORY" --ref main' in text


def test_control_plane_controller_cadences_are_explicit() -> None:
    text = WATCHDOG.read_text(encoding="utf-8")
    expected = (
        'recover_missing_periodic_controller "research-autopilot.yml" "Research autopilot" 30 120',
        'recover_missing_periodic_controller "long-research-recovery.yml" "Long research recovery" 10 45',
        'recover_missing_periodic_controller "automation-failure-learning.yml" "Automation failure learning" 15 75',
    )
    for line in expected:
        assert line in text


def test_independent_supervisor_closes_watchdog_failure_gap() -> None:
    supervisor = ROOT / ".github/workflows/automation-supervisor.yml"
    text = supervisor.read_text(encoding="utf-8")
    assert '    - cron: "*/10 * * * *"' in text
    assert 'recover_controller "actions-reliability-watchdog.yml" "Actions reliability watchdog" 5 20' in text
    assert 'gh run rerun "$run_id" --repo "$GITHUB_REPOSITORY" --failed' in text
    assert "active run exists; no duplicate recovery" in text


def test_keepalive_recovery_is_head_sha_agnostic_after_its_own_push() -> None:
    supervisor = ROOT / ".github/workflows/automation-supervisor.yml"
    text = supervisor.read_text(encoding="utf-8")
    assert 'local head_sha_required=true' in text
    assert 'if [ "$workflow_name" = "Repository automation keepalive" ]; then' in text
    assert "head_sha_required=false" in text
    assert 'if { [ "$head_sha_required" = false ] || [ "$head_sha" = "$current_sha" ]; }' in text

    keepalive = ROOT / ".github/workflows/automation-activity-keepalive.yml"
    keepalive_text = keepalive.read_text(encoding="utf-8")
    assert "marker is fresher than 6 days; no refresh needed" in keepalive_text
    assert "do not create a self-triggering" in keepalive_text

    watchdog = WATCHDOG.read_text(encoding="utf-8")
    assert 'recover_missing_periodic_controller "automation-supervisor.yml" "Automation supervisor" 10 45' in watchdog


def test_research_autopilot_bootstraps_watchdog_for_stale_active_research() -> None:
    autopilot = ROOT / ".github/workflows/research-autopilot.yml"
    text = autopilot.read_text(encoding="utf-8")
    assert "stale_active_count=" in text
    assert "fromdateiso8601" in text
    assert "120 * 60" in text
    assert 'gh workflow run actions-reliability-watchdog.yml --repo "$GITHUB_REPOSITORY" --ref main' in text
    assert "BOOTSTRAP_STALE_ACTIVE_WATCHDOG" in text
