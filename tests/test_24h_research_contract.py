from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARATHON = ROOT / ".github" / "workflows" / "24h-research-marathon.yml"
WATCHDOG = ROOT / ".github" / "workflows" / "24h-research-marathon-watchdog.yml"


def test_marathon_is_schedule_or_manual_only():
    text = MARATHON.read_text(encoding="utf-8")
    assert "\n  push:" not in text
    assert "workflow_dispatch:" in text
    assert '    - cron: "5 0 * * *"' in text


def test_marathon_price_shard_pid_is_real_background_pid():
    text = MARATHON.read_text(encoding="utf-8")
    assert '$$!' not in text
    assert text.count('pids+=( "$!" )') == 3


def test_marathon_final_checkpoint_is_fail_closed():
    text = MARATHON.read_text(encoding="utf-8")
    assert "Write immutable final checkpoint" in text
    assert text.count('if: always()') >= 3
    assert '"research_only":True' in text
    assert '"production_changed":False' in text
    assert 'if status != "COMPLETED":' in text
    assert 'if-no-files-found: error' in text
    for lane in ("core_oos", "finra", "treasury", "audits"):
        assert f'"{lane}"' in text


def test_watchdog_has_bounded_recovery_and_failure_cooldown():
    text = WATCHDOG.read_text(encoding="utf-8")
    assert "MAIN_SHA=" in text
    assert "SELF_RUN_ID=" in text
    assert "age >= 30" in text
    assert "age >= 1450" in text
    assert "len(recent_failures) < 3" in text
    assert "three_recent_failures_cooldown" in text
    assert "bounded dispatch retries exhausted" in text
