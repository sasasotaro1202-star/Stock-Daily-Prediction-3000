from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = ROOT / "scripts/actions_reliability_watchdog.sh"


def test_research_watchdog_does_not_use_run_updated_at_as_oos_progress_signal():
    text = WATCHDOG.read_text(encoding="utf-8")
    assert "research_inactive_epoch" not in text
    assert "has had no Actions update for over 120 minutes" not in text
    assert "started_epoch" in text
    assert "350 minutes" in text
    assert "updated_at field is not a reliable progress signal" in text


def test_research_watchdog_keeps_hard_age_guard():
    text = WATCHDOG.read_text(encoding="utf-8")
    assert 'research_stale_epoch="$((now_epoch - 350 * 60))"' in text
    assert 'exceeded 350 minutes of active age' in text
