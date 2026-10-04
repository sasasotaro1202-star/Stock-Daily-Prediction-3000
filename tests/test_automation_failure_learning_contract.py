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

def test_failure_learning_reconciliation_safety_net() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert '    - cron: "*/15 * * * *"' in text
    assert "python scripts/reconcile_automation_failures.py" in text
    assert "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'" in text


def test_reconciliation_discovers_unique_recent_failures(monkeypatch) -> None:
    import scripts.reconcile_automation_failures as reconcile

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_API_URL", "https://api.github.test")

    def fake_request(url, token):
        return {
            "workflow_runs": [
                {
                    "id": 101,
                    "run_number": 4,
                    "name": "Repository verification",
                    "head_sha": "abc",
                    "head_branch": "main",
                    "run_attempt": 1,
                    "conclusion": "failure",
                    "created_at": "2026-10-04T10:00:00Z",
                    "updated_at": "2026-10-04T10:01:00Z",
                    "event": "push",
                },
                {
                    "id": 101,
                    "run_number": 4,
                    "name": "Repository verification",
                    "head_sha": "abc",
                    "head_branch": "main",
                    "run_attempt": 1,
                    "conclusion": "failure",
                    "created_at": "2026-10-04T10:00:00Z",
                    "updated_at": "2026-10-04T10:01:00Z",
                    "event": "push",
                },
            ]
        }

    monkeypatch.setattr(reconcile, "_request_json", fake_request)
    monkeypatch.setattr(
        reconcile,
        "_load_job_snapshot",
        lambda repository, run_id, token, api_base: [],
    )
    records = reconcile.discover_failure_records()
    assert len(records) == 1
    assert records[0]["failure_id"]
    assert records[0]["research_only"] is True

def test_failure_triage_prioritizes_pit_and_production_failures() -> None:
    import scripts.triage_automation_failures as triage

    records = [
        {
            "failure_id": "a",
            "category": "RESEARCH_VALIDATION_FAILURE",
            "workflow_name": "Research validation",
            "workflow_run_id": 1,
            "head_sha": "sha1",
            "created_at": "2026-10-04T10:00:00Z",
            "failed_jobs": [{"name": "research", "failed_steps": [{"name": "Independent PIT leakage audit", "conclusion": "failure"}]}],
        },
        {
            "failure_id": "b",
            "category": "PRODUCTION_WORKFLOW_FAILURE",
            "workflow_name": "Prediction monitoring",
            "workflow_run_id": 2,
            "head_sha": "sha2",
            "created_at": "2026-10-04T11:00:00Z",
            "failed_jobs": [{"name": "monitor", "failed_steps": [{"name": "monitor_predictions.py", "conclusion": "failure"}]}],
        },
    ]
    payload = triage.triage(records)
    assert payload["status"] == "RECONCILED"
    assert payload["production_mutation"] is False
    assert payload["promotion_allowed"] is False
    assert payload["backlog"][0]["priority_reason"] == "PIT_OR_UNIVERSE_REVIEW"
    assert payload["backlog"][0]["action"].startswith("RUN_PIT")
