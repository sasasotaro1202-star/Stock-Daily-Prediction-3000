from __future__ import annotations

from pathlib import Path
import io
import json

from scripts import persist_research_validation_status as status


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

def test_status_workflow_captures_requested_research_runs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'types: [completed, in_progress, requested]' in text
    assert "requested" in text


def test_workflow_run_context_is_captured_before_status_persistence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("      - name: Resolve Research status context")
    resolve_block = text[start:text.index("      - name: Download completed OOS evidence", start)]
    assert "RESEARCH_WORKFLOW_RUN_ID: ${{ github.event.workflow_run.id || '' }}" in resolve_block
    assert "RESEARCH_WORKFLOW_RUN_NUMBER: ${{ github.event.workflow_run.run_number || '' }}" in resolve_block
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
    assert 'for workflow_status in ("requested", "pending", "queued", "waiting", "in_progress")' in resolver
    assert '?status={workflow_status}&per_page=100' in resolver
    assert '"pending"' in resolver
    assert '"queued"' in resolver
    assert '"in_progress"' in resolver


def test_status_persistence_treats_pending_as_queued_not_as_lookup_failure() -> None:
    script = Path("scripts/persist_research_validation_status.py").read_text(encoding="utf-8")
    assert '"pending"' in script
    assert 'state = "QUEUED"' in script
    assert 'workflow_status or "unknown"' in script
    status_line = script.split("workflow_statuses =", 1)[1].splitlines()[0]
    assert '"pending"' in status_line


def test_status_persistence_marks_requested_as_queued_in_runtime_health() -> None:
    script = Path("scripts/persist_research_validation_status.py").read_text(encoding="utf-8")
    assert 'status in {"requested", "queued", "pending", "waiting"}' in script
    assert 'state = "QUEUED"' in script


def test_status_persistence_retries_pending_job_creation_race(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "123")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "pending")

    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        payload = (
            {"jobs": []}
            if calls == 1
            else {"jobs": [{"name": "research", "status": "in_progress", "conclusion": None}]}
        )
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(status.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(status.time, "sleep", lambda _: None)

    job, error = status._lookup_research_job()

    assert error is None
    assert job["name"] == "research"
    assert calls == 2


def test_status_persistence_exposes_execution_sha_and_freshness_contract() -> None:
    script = Path("scripts/persist_research_validation_status.py").read_text(encoding="utf-8")
    assert '"research_workflow_sha"' in script
    assert '"status_branch_main_sha"' in script
    assert '"research_evidence_fingerprint_sha256"' in script
    assert '"evidence_freshness"' in script
    assert 'return "FRESH" if research_fingerprint == current_fingerprint else "STALE"' in script

def test_status_persistence_reads_evidence_fingerprint_from_completed_metrics() -> None:
    script = Path("scripts/persist_research_validation_status.py").read_text(encoding="utf-8")
    assert "def _load_research_evidence_fingerprint()" in script
    assert 'payload.get("evidence_code_fingerprint_sha256")' in script



def test_status_resolver_prefers_newer_active_run_over_terminal_event(monkeypatch, tmp_path) -> None:
    from scripts import resolve_research_status_context as resolver

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_API_URL", "https://api.github.com")
    monkeypatch.setenv("GITHUB_ENV", str(tmp_path / "github_env"))
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "100")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_NUMBER", "244")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "old-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "completed")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "cancelled")

    calls = 0

    def fake_request_json(url, token):
        nonlocal calls
        calls += 1
        if "status=pending" in url:
            return {
                "workflow_runs": [
                    {
                        "name": "Research validation",
                        "status": "pending",
                        "run_number": 245,
                        "id": 200,
                        "head_sha": "new-sha",
                    }
                ]
            }
        return {"workflow_runs": []}

    monkeypatch.setattr(resolver, "_request_json", fake_request_json)

    assert resolver.main() == 0

    env = (tmp_path / "github_env").read_text(encoding="utf-8")
    assert "RESEARCH_WORKFLOW_RUN_ID=200\n" in env
    assert "RESEARCH_WORKFLOW_RUN_NUMBER=245\n" in env
    assert "RESEARCH_WORKFLOW_SHA=new-sha\n" in env
    assert "RESEARCH_WORKFLOW_STATUS=pending\n" in env
    assert calls == 5