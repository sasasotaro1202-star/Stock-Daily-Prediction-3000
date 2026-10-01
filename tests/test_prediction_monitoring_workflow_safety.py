from __future__ import annotations

from pathlib import Path

import yaml


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "prediction-monitoring.yml"


def _steps() -> list[dict]:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")) or {}
    return doc["jobs"]["monitor"]["steps"]


def test_monitoring_persists_experience_without_hashfiles_gate():
    steps = _steps()
    persist = next(step for step in steps if step.get("name") == "Persist accumulated experience")
    condition = str(persist.get("if", ""))
    assert "hashFiles(" not in condition
    assert "github.ref == 'refs/heads/main'" in condition
    assert "steps.price_state.outputs.available == 'true'" in condition


def test_monitoring_artifacts_include_experience_candidates():
    steps = _steps()
    upload = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v7")
    path = str(upload.get("with", {}).get("path", ""))
    assert "data/research/experience_memory.json" in path
    assert "data/research/experience_candidates.json" in path
