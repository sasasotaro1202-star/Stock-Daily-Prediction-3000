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
