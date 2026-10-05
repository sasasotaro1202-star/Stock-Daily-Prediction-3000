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
