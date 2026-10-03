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


def test_oos_restore_skips_non_success_workflow_run_events() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("      - name: Download completed OOS evidence")
    block = text[start:text.index("      - name: Persist research validation status", start)]
    assert block.count("github.event_name == 'workflow_run'") == 2
    artifact_block = _artifact_restore_block()
    restore_block = _restore_block()
    assert artifact_block.count("github.event.workflow_run.conclusion == 'success'") == 1
    assert "env.RESEARCH_WORKFLOW_CONCLUSION == 'success'" in restore_block
    assert "github.event.workflow_run.conclusion == 'success'" not in restore_block

def test_workflow_run_context_is_captured_before_status_persistence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("      - name: Resolve Research status context")
    resolve_block = text[start:text.index("      - name: Download completed OOS evidence", start)]
    assert "RESEARCH_WORKFLOW_RUN_ID: ${{ github.event.workflow_run.id || '' }}" in resolve_block
    assert "RESEARCH_WORKFLOW_SHA: ${{ github.event.workflow_run.head_sha || '' }}" in resolve_block
    assert "RESEARCH_WORKFLOW_STATUS: ${{ github.event.workflow_run.status || '' }}" in resolve_block
    assert "RESEARCH_WORKFLOW_CONCLUSION: ${{ github.event.workflow_run.conclusion || '' }}" in resolve_block

    start = text.index("      - name: Persist research validation status")
    persist_block = text[start:text.index("      - name: Preserve generated status outputs", start)]
    assert "RESEARCH_WORKFLOW_RUN_ID: ${{ github.event.workflow_run.id }}" not in persist_block
    assert "RESEARCH_WORKFLOW_SHA: ${{ github.event.workflow_run.head_sha }}" not in persist_block
    assert "RESEARCH_WORKFLOW_STATUS: ${{ github.event.workflow_run.status }}" not in persist_block
    assert "RESEARCH_WORKFLOW_CONCLUSION: ${{ github.event.workflow_run.conclusion }}" not in persist_block


def test_status_heartbeat_tracks_pending_research_runs() -> None:
    resolver = Path("scripts/resolve_research_status_context.py").read_text(encoding="utf-8")
    assert "status=pending" in resolver
    assert "status=queued" in resolver
    assert "status=in_progress" in resolver
    assert '"pending"' in resolver


def test_status_persistence_treats_pending_as_queued_not_as_lookup_failure() -> None:
    script = Path("scripts/persist_research_validation_status.py").read_text(encoding="utf-8")
    assert '"pending"' in script
    assert 'state = "QUEUED"' in script
    status_line = script.split("workflow_statuses =", 1)[1].splitlines()[0]
    assert '"pending"' in status_line
