from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/research-autopilot.yml"


def test_research_autopilot_has_fast_missed_trigger_safety_net() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert '    - cron: "*/30 * * * *"' in text
    assert "30-minute schedule as a missed-trigger safety net" in text
    assert 'workflows: ["Repository verification"]' in text
    assert "types: [completed]" in text


def test_research_autopilot_remains_fail_closed_on_verification() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'if [ "$verification_state" != "PASS" ]; then' in text
    assert 'gh workflow run research-validation.yml --repo "$GITHUB_REPOSITORY" --ref main' in text
    assert 'if [ "$active_current" -gt 0 ]; then' in text
