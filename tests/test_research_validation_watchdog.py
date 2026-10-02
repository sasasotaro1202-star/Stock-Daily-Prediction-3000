from pathlib import Path

import yaml


def test_research_validation_watchdog_is_write_scoped_and_bounded():
    workflow = yaml.safe_load(
        Path(".github/workflows/research-validation-stale-watchdog.yml").read_text(
            encoding="utf-8"
        )
    )
    assert workflow["permissions"] == {"actions": "write"}
    assert workflow["jobs"]["cancel-stale"]["timeout-minutes"] == 5
    run = workflow["jobs"]["cancel-stale"]["steps"][0]["run"]
    assert "research-validation.yml" in run
    assert "minutes=90" in run
    assert "actions/runs/" in run
    assert '--method", "POST"' in run
