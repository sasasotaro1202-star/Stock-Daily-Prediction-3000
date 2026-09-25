from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import persist_research_validation_status as status


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "research-validation.yml"
STATUS_WORKFLOW = ROOT / ".github" / "workflows" / "research-validation-status.yml"


def _extract_run_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "run: |":
            i += 1
            continue
        parent_indent = len(lines[i]) - len(lines[i].lstrip(" "))
        block_indent = parent_indent + 2
        i += 1
        block: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip(" ")) < block_indent:
                break
            block.append(line[block_indent:] if len(line) >= block_indent else "")
            i += 1
        blocks.append("\n".join(block) + "\n")
    return blocks


def test_research_validation_bash_blocks_are_syntactically_valid() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    status_text = STATUS_WORKFLOW.read_text(encoding="utf-8")
    blocks = _extract_run_blocks(text)
    status_blocks = _extract_run_blocks(status_text)
    assert blocks, "No run: | blocks found in research-validation workflow"
    assert status_blocks, "No run: | blocks found in research-validation-status workflow"

    for index, script in enumerate(blocks + status_blocks):
        result = subprocess.run(
            ["bash", "-n"],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"research workflow run block {index} has invalid bash syntax:\n"
            f"{result.stderr}\nSCRIPT:\n{script}"
        )


def test_research_validation_has_external_status_workflow() -> None:
    text = STATUS_WORKFLOW.read_text(encoding="utf-8")
    assert 'workflows: ["Research validation"]' in text
    assert "types: [completed]" in text
    assert "python scripts/persist_research_validation_status.py" in text
    assert "RESEARCH_WORKFLOW_RUN_ID: ${{ github.event.workflow_run.id }}" in text
    assert "RESEARCH_WORKFLOW_SHA: ${{ github.event.workflow_run.head_sha }}" in text
    assert "GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}" in text
    main_text = WORKFLOW.read_text(encoding="utf-8")
    assert "  research-status:" not in main_text
    assert "research-validation-status-" not in main_text

class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_status_script_records_research_job_outcomes(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("GITHUB_SHA", "default-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "123")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "abc")
    monkeypatch.setattr(
        status.urllib.request,
        "urlopen",
        lambda request, timeout: _Response(
            {
                "jobs": [
                    {
                        "name": "research",
                        "status": "completed",
                        "conclusion": "failure",
                        "steps": [
                            {"name": "Chronological OOS research", "conclusion": "success"},
                            {"name": "Collect free SEC filing research inputs", "conclusion": "failure"},
                            {"name": "Run SEC filing OOS challenger ablation", "conclusion": "skipped"},
                            {"name": "CPCV leakage-boundary research audit", "conclusion": "skipped"},
                        ],
                    }
                ]
            }
        ),
    )

    assert status.main() == 0
    payload = json.loads(
        (Path("artifacts") / "research_validation_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status_lookup_ok"] is True
    assert payload["workflow_run_id"] == "123"
    assert payload["workflow_sha"] == "abc"
    assert payload["job_status"] == "failure"
    assert payload["research_step"] == "success"
    assert payload["evidence_artifact_present"] is False


def test_status_script_detects_evidence_artifact(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setattr(
        status.urllib.request,
        "urlopen",
        lambda request, timeout: _Response(
            {
                "artifacts": [
                    {"name": "research-validation-evidence-123", "expired": False}
                ]
            }
            if "/artifacts?" in request.full_url
            else {"jobs": [{"name": "research", "status": "completed", "conclusion": "success", "steps": []}]}
        ),
    )

    assert status.main() == 0
    payload = json.loads(
        (Path("artifacts") / "research_validation_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["evidence_artifact_present"] is True


def test_status_script_fails_closed_on_api_lookup_error(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setattr(
        status.urllib.request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(
            status.urllib.error.URLError("network failure")
        ),
    )

    with pytest.raises(SystemExit, match="research status lookup"):
        status.main()

    payload = json.loads(
        (Path("artifacts") / "research_validation_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status_lookup_ok"] is False
    assert "URLError" in payload["status_lookup_error"]
