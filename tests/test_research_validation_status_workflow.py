from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(".github/workflows/research-validation-status.yml")


def _artifact_restore_block() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("      - name: Download completed OOS evidence")
    end = text.index("      - name: Restore OOS metrics for longitudinal snapshot", start)
    return text[start:end]


def _restore_block() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("      - name: Restore OOS metrics for longitudinal snapshot")
    end = text.index("      - name: Persist research validation status", start)
    return text[start:end]


def test_completed_evidence_download_does_not_hide_failures() -> None:
    block = _artifact_restore_block()
    assert "continue-on-error: true" not in block


def test_missing_completed_evidence_fails_closed() -> None:
    block = _restore_block()
    assert 'source_file="$(find research-evidence -type f -name latest_metrics.json -print -quit 2>/dev/null || true)"' not in block
    assert 'if [ -z "${source_file}" ]; then' in block
    assert "exit 1" in block
    assert "successful Research validation produced no latest_metrics.json artifact" in block
