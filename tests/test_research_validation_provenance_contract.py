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


def test_research_provenance_changes_trigger_validation():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert '"src/research/run_provenance.py"' in text
    assert '"scripts/finalize_research_run_provenance.py"' in text
    assert '"tests/test_run_provenance.py"' in text
