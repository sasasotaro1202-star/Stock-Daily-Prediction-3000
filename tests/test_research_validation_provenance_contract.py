from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/research-validation.yml")


def test_research_validation_archives_structured_oos_provenance():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "Finalize OOS provenance" in text
    assert "run: python scripts/finalize_research_run_provenance.py" in text
    assert "RESEARCH_OOS_OUTCOME: ${{ steps.research_oos.outcome }}" in text
    assert "data/research/oos_run_manifest.json" in text
    assert "data/research/oos_progress.json" in text


def test_research_provenance_changes_are_covered_by_verification_autopilot():
    validation = WORKFLOW.read_text(encoding="utf-8")
    verification = Path(".github/workflows/repository-verification.yml").read_text(encoding="utf-8")
    autopilot = Path(".github/workflows/research-autopilot.yml").read_text(encoding="utf-8")

    # Research validation is intentionally not push-triggered. Main changes
    # first enter repository verification, and only a successful current-main
    # verification may dispatch chronological OOS.
    assert "  push:" not in validation
    assert '  push:\n    branches: [main]' in verification
    assert 'workflows: ["Repository verification"]' in autopilot
    assert "github.event.workflow_run.conclusion == 'success'" in autopilot

    # Provenance execution remains part of the Research validation workflow.
    assert "run: python scripts/finalize_research_run_provenance.py" in validation
    assert "Finalize OOS provenance" in validation
