from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"


def test_workflows_have_only_executable_steps() -> None:
    workflow_files = sorted(WORKFLOW_DIR.glob("*.y*ml"))
    assert workflow_files
    for path in workflow_files:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        jobs = document.get("jobs", {}) or {}
        assert isinstance(jobs, dict), f"{path} jobs must be a mapping"
        for job_name, job in jobs.items():
            if not isinstance(job, dict) or "steps" not in job:
                continue
            steps = job["steps"]
            assert isinstance(steps, list), f"{path}: job {job_name} steps must be a list"
            for index, step in enumerate(steps):
                assert isinstance(step, dict), f"{path}: job {job_name} step {index} must be a mapping"
                assert step.get("run") or step.get("uses"), (
                    f"{path}: job {job_name} step {index} must define run or uses"
                )

def test_project_instructions_preserve_adoption_experience_and_transfer_gates() -> None:
    instructions = (ROOT / "PROJECT_INSTRUCTIONS.md").read_text(encoding="utf-8")
    assert "primary relative OOS LogLoss improvement >=3%" in instructions
    assert "no worsening in >=70% of evaluation periods" in instructions
    assert "zero PIT violations" in instructions
    assert "canonical instrument-date-cutoff granularity" in instructions
    assert "DISCOVER -> ABSTRACT_MECHANISM -> COMPATIBILITY -> ADAPT -> LOCAL_PIT -> LOCAL_OOS -> LOCAL_HOLDOUT -> SHADOW -> PROMOTE" in instructions



def test_daily_watchlist_canonicalizes_restored_prices_before_prediction() -> None:
    workflow = (WORKFLOW_DIR / "daily-watchlist.yml").read_text(encoding="utf-8")
    canonical_step = workflow.index("name: Canonicalize restored price store")
    prediction_step = workflow.index(
        "name: Generate latest priority equity predictions"
    )
    assert canonical_step < prediction_step
    assert "run: python scripts/normalize_price_store.py" in workflow
