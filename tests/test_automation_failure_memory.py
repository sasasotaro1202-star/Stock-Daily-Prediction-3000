from scripts.record_automation_failure import build_record, classify_workflow


def test_failure_categories_are_stable():
    assert classify_workflow("Research validation") == "RESEARCH_VALIDATION_FAILURE"
    assert classify_workflow("Market cycle") == "PRODUCTION_WORKFLOW_FAILURE"
    assert classify_workflow("U.S. close prediction") == "PRODUCTION_WORKFLOW_FAILURE"
    assert classify_workflow("Prediction monitoring") == "PRODUCTION_WORKFLOW_FAILURE"
    assert classify_workflow("24H Research Marathon") == "RESEARCH_AUTOMATION_FAILURE"
    assert classify_workflow("Other") == "WORKFLOW_FAILURE"


def test_failure_record_is_research_only_and_has_stable_id():
    record = build_record(
        run_id="123",
        run_number="9",
        workflow_name="Research validation",
        head_sha="abc123",
        head_branch="main",
        conclusion="failure",
        run_attempt="1",
        created_at="2026-10-04T10:00:00Z",
        updated_at="2026-10-04T10:03:00Z",
        event_name="workflow_dispatch",
        repository="sasasotaro1202-star/Stock-Daily-Prediction-3000",
        jobs=[
            {
                "job_id": 1,
                "name": "research",
                "status": "completed",
                "conclusion": "failure",
                "failed_steps": [
                    {
                        "name": "Chronological OOS research",
                        "status": "completed",
                        "conclusion": "failure",
                    }
                ],
            }
        ],
    )
    assert record["schema_version"] == 1
    assert record["status"] == "FAILED"
    assert record["category"] == "RESEARCH_VALIDATION_FAILURE"
    assert record["production_mutation"] is False
    assert record["research_only"] is True
    assert len(record["failure_id"]) == 64
