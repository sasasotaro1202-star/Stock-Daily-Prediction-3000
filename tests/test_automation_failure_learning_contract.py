from pathlib import Path

from scripts.record_automation_failure import build_record, classify_workflow


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "automation-failure-learning.yml"
WATCHDOG = ROOT / ".github" / "workflows" / "actions-reliability-watchdog.yml"


def test_classify_workflow_categories_are_deterministic() -> None:
    assert classify_workflow("Research validation") == "RESEARCH_VALIDATION_FAILURE"
    assert classify_workflow("Market cycle") == "PRODUCTION_WORKFLOW_FAILURE"
    assert classify_workflow("U.S. close prediction") == "PRODUCTION_WORKFLOW_FAILURE"
    assert classify_workflow("Prediction monitoring") == "PRODUCTION_WORKFLOW_FAILURE"
    assert classify_workflow("24H Research Marathon") == "RESEARCH_AUTOMATION_FAILURE"
    assert classify_workflow("Other") == "WORKFLOW_FAILURE"


def test_build_record_is_research_only_and_identity_bound() -> None:
    row = build_record(
        run_id="123",
        run_number="45",
        workflow_name="Repository verification",
        head_sha="abc123",
        head_branch="main",
        conclusion="failure",
        run_attempt="2",
        created_at="2026-10-04T10:00:00Z",
        updated_at="2026-10-04T10:05:00Z",
        event_name="push",
        repository="owner/repo",
        jobs=[
            {
                "job_id": 1,
                "name": "verify",
                "status": "completed",
                "conclusion": "failure",
                "failed_steps": [{"name": "pytest -q", "status": "completed", "conclusion": "failure"}],
            }
        ],
    )
    assert row["status"] == "FAILED"
    assert row["research_only"] is True
    assert row["production_mutation"] is False
    assert row["workflow_run_id"] == 123
    assert row["run_attempt"] == 2
    assert len(row["failure_id"]) == 64
    assert row["failed_jobs"][0]["failed_steps"][0]["name"] == "pytest -q"


def test_failure_learning_is_automatically_workflow_run_driven_and_recovery_dispatchable() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "  workflow_dispatch:" in text
    assert "  workflow_run:" in text
    assert 'types: [completed]' in text
    assert "research-status-writer" in text
    assert "git push origin HEAD:research-status" in text
    assert "git push origin HEAD:main" not in text


def test_watchdog_recovers_failure_learning_and_experience_review() -> None:
    text = WATCHDOG.read_text(encoding="utf-8")
    assert 'inspect_workflow "automation-failure-learning.yml" "Automation failure learning" true true' in text
    assert 'inspect_workflow "experience-review.yml" "Experience review" true true' in text
