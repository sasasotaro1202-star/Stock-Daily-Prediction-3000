from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_automation_invariants_pass():
    result = subprocess.run(
        [sys.executable, "scripts/automation_invariants.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_automation_uses_single_daily_schedule_and_failure_only_recovery():
    market = (ROOT / ".github/workflows/market-cycle.yml").read_text(encoding="utf-8")
    monitoring = (ROOT / ".github/workflows/prediction-monitoring.yml").read_text(encoding="utf-8")
    recovery = (ROOT / ".github/workflows/bounded-production-recovery.yml").read_text(encoding="utf-8")

    assert market.count('    - cron: "37 18 * * 1-5"') == 1
    assert '    - cron: "17 18 * * 1-5"' not in market

    assert monitoring.count('    - cron: "27 9 * * 1-5"') == 1
    assert '    - cron: "17 9 * * 1-5"' not in monitoring

    assert "github.event.workflow_run.conclusion == 'failure'" in recovery
    assert "github.event.workflow_run.conclusion == 'cancelled'" not in recovery
