from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
import subprocess
import textwrap
from pathlib import Path

import pytest

from scripts import persist_research_validation_status as status
from scripts import resolve_research_status_context as context


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
        blocks.append(textwrap.dedent("\n".join(block)) + "\n")
    return blocks


def _extract_python_heredocs(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() not in {"python - <<'PY'", "python3 - <<'PY'"}:
            i += 1
            continue
        i += 1
        block: list[str] = []
        while i < len(lines) and lines[i].strip() != "PY":
            block.append(lines[i])
            i += 1
        if i >= len(lines):
            raise AssertionError("unterminated Python heredoc in workflow")
        blocks.append(textwrap.dedent("\n".join(block)) + "\n")
        i += 1
    return blocks


def test_research_validation_uses_read_only_repository_permission():
    text = (ROOT / ".github" / "workflows" / "research-validation.yml").read_text(encoding="utf-8")
    assert "permissions:" in text
    assert "  actions: read" in text
    assert "  contents: read" in text
    assert "  contents: write" not in text
    assert "git push" not in text


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


    python_blocks = _extract_python_heredocs(text)
    status_python_blocks = _extract_python_heredocs(status_text)
    assert python_blocks, "No embedded Python heredocs found in research-validation workflow"
    for index, script in enumerate(python_blocks + status_python_blocks):
        try:
            compile(script, f"workflow-python-heredoc-{index}", "exec")
        except SyntaxError as exc:
            raise AssertionError(
                f"workflow embedded Python block {index} has invalid syntax: {exc}\nSCRIPT:\n{script}"
            ) from exc


def test_research_status_workflow_has_heartbeat_schedule() -> None:
    text = STATUS_WORKFLOW.read_text(encoding="utf-8")
    assert 'schedule:\n    - cron: "13,43 * * * *"' in text
    assert "workflow_dispatch:" in text
    assert "python scripts/resolve_research_status_context.py" in text
    assert "RESEARCH_STATUS_CONTEXT_FOUND" in text
    assert "if: env.RESEARCH_STATUS_CONTEXT_FOUND == 'true'" in text


def test_resolve_context_reuses_workflow_run_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "777")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "sha777")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "in_progress")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "")
    env_path = tmp_path / "github_env"
    monkeypatch.setenv("GITHUB_ENV", str(env_path))

    assert context.main() == 0
    output = env_path.read_text(encoding="utf-8")
    assert "RESEARCH_WORKFLOW_RUN_ID=777\n" in output
    assert "RESEARCH_WORKFLOW_SHA=sha777\n" in output
    assert "RESEARCH_WORKFLOW_STATUS=in_progress\n" in output
    assert "RESEARCH_WORKFLOW_CONCLUSION=\n" in output
    assert "RESEARCH_STATUS_CONTEXT_FOUND=true\n" in output


def test_resolve_context_discovers_latest_active_research(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESEARCH_WORKFLOW_RUN_ID", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_API_URL", "https://api.github.test")
    env_path = tmp_path / "github_env"
    monkeypatch.setenv("GITHUB_ENV", str(env_path))
    captured = {}

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        return _Response({
            "workflow_runs": [
                {"id": 10, "run_number": 10, "name": "Other", "status": "in_progress", "head_sha": "x"},
                {"id": 20, "run_number": 20, "name": "Research validation", "status": "in_progress", "head_sha": "old"},
                {"id": 21, "run_number": 21, "name": "Research validation", "status": "in_progress", "head_sha": "new"},
            ]
        })

    monkeypatch.setattr(context.urllib.request, "urlopen", urlopen)
    assert context.main() == 0
    assert "status=in_progress" in captured["url"]
    output = env_path.read_text(encoding="utf-8")
    assert "RESEARCH_WORKFLOW_RUN_ID=21\n" in output
    assert "RESEARCH_WORKFLOW_SHA=new\n" in output
    assert "RESEARCH_WORKFLOW_STATUS=in_progress\n" in output
    assert "RESEARCH_WORKFLOW_CONCLUSION=\n" in output
    assert "RESEARCH_STATUS_CONTEXT_FOUND=true\n" in output


def test_resolve_context_no_active_research_is_noop(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESEARCH_WORKFLOW_RUN_ID", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_API_URL", "https://api.github.test")
    env_path = tmp_path / "github_env"
    monkeypatch.setenv("GITHUB_ENV", str(env_path))

    monkeypatch.setattr(context.urllib.request, "urlopen", lambda request, timeout: _Response({"workflow_runs": []}))
    assert context.main() == 0
    assert env_path.read_text(encoding="utf-8") == "RESEARCH_STATUS_CONTEXT_FOUND=false\n"


def test_resolve_context_fails_closed_on_lookup_error(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESEARCH_WORKFLOW_RUN_ID", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    env_path = tmp_path / "github_env"
    monkeypatch.setenv("GITHUB_ENV", str(env_path))

    def fail(request, timeout):
        raise context.urllib.error.URLError("network failure")

    monkeypatch.setattr(context.urllib.request, "urlopen", fail)
    with pytest.raises(SystemExit, match="heartbeat run lookup"):
        context.main()

def test_research_validation_has_external_status_workflow() -> None:
    text = STATUS_WORKFLOW.read_text(encoding="utf-8")
    assert 'workflows: ["Research validation"]' in text
    assert "types: [completed, in_progress, requested]" in text
    assert "group: research-validation-status" in text
    assert "python scripts/persist_research_validation_status.py" in text
    assert "RESEARCH_WORKFLOW_RUN_ID: ${{ github.event.workflow_run.id || '' }}" in text
    assert "RESEARCH_WORKFLOW_SHA: ${{ github.event.workflow_run.head_sha || '' }}" in text
    assert "name: research-validation-evidence-${{ env.RESEARCH_WORKFLOW_RUN_ID }}" in text
    assert "run-id: ${{ env.RESEARCH_WORKFLOW_RUN_ID }}" in text
    assert "github.event_name == 'workflow_run'" in text
    assert "GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}" in text
    assert "ref: main" in text
    assert "github.event.workflow_run.head_sha" in text
    assert "ref: ${{ github.event.workflow_run.head_sha }}\n        with:" not in text
    assert text.index("Persist research validation status") < text.index("Preserve generated status outputs")
    assert text.index("Preserve generated status outputs") < text.index("Commit research status evidence")
    preserve_pos = text.index("      - name: Preserve generated status outputs")
    assert text.index("        if: always()", preserve_pos) < text.index("        shell: bash", preserve_pos)
    assert text.index("git checkout -B research-status origin/research-status --force") > text.index("Preserve generated status outputs")
    main_text = WORKFLOW.read_text(encoding="utf-8")
    assert "  research-status:" not in main_text
    assert "research-validation-status" not in main_text

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
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "failure")
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


def test_status_script_accepts_in_progress_workflow_before_research_job_exists(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("GITHUB_SHA", "default-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "124")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "active-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "in_progress")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "")
    monkeypatch.setenv("RESEARCH_STATUS_JOB_LOOKUP_RETRIES", "0")

    def urlopen(request, timeout):
        if "/artifacts?" in request.full_url:
            return _Response({"artifacts": []})
        return _Response({"jobs": []})

    monkeypatch.setattr(status.urllib.request, "urlopen", urlopen)

    assert status.main() == 0
    payload = json.loads(
        (Path("artifacts") / "research_validation_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status_lookup_ok"] is True
    assert payload["workflow_run_id"] == "124"
    assert payload["workflow_status"] == "in_progress"
    assert payload["job_status"] == "in_progress"
    assert payload["status_lookup_error"] is None


def test_status_script_retries_transient_research_job_visibility(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "125")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "retry-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "in_progress")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "")
    monkeypatch.setenv("RESEARCH_STATUS_JOB_LOOKUP_RETRIES", "2")
    sleeps = []
    monkeypatch.setattr(status.time, "sleep", sleeps.append)
    calls = {"jobs": 0}

    def urlopen(request, timeout):
        if "/artifacts?" in request.full_url:
            return _Response({"artifacts": []})
        calls["jobs"] += 1
        if calls["jobs"] < 3:
            return _Response({"jobs": []})
        return _Response({
            "jobs": [{
                "name": "research",
                "status": "in_progress",
                "conclusion": None,
                "steps": [{
                    "name": "Chronological OOS research",
                    "status": "in_progress",
                    "conclusion": None,
                }],
            }]
        })

    monkeypatch.setattr(status.urllib.request, "urlopen", urlopen)
    assert status.main() == 0
    payload = json.loads((Path("artifacts") / "research_validation_status.json").read_text(encoding="utf-8"))
    assert calls["jobs"] == 3
    assert sleeps == [2, 2]
    assert payload["status_lookup_ok"] is True
    assert payload["job_status"] == "in_progress"
    assert payload["research_step"] == "in_progress"

def test_status_script_does_not_overwrite_newer_research_run(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "199")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "old-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "completed")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "success")
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir()
    status_path = artifact_dir / "research_validation_status.json"
    existing = {"workflow_run_id": "200", "workflow_sha": "new-sha", "job_status": "success"}
    status_path.write_text(json.dumps(existing), encoding="utf-8")

    monkeypatch.setattr(status.urllib.request, "urlopen", lambda request, timeout: _Response({"jobs": [], "artifacts": []}))
    assert status.main() == 0
    assert json.loads(status_path.read_text(encoding="utf-8")) == existing


def test_status_script_accepts_cancelled_workflow_without_research_job(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("GITHUB_SHA", "default-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "123")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "cancelled-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "cancelled")

    def urlopen(request, timeout):
        if "/artifacts?" in request.full_url:
            return _Response({"artifacts": []})
        return _Response({"jobs": []})

    monkeypatch.setattr(status.urllib.request, "urlopen", urlopen)

    assert status.main() == 0
    payload = json.loads(
        (Path("artifacts") / "research_validation_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status_lookup_ok"] is True
    assert payload["job_status"] == "cancelled"
    assert payload["status_lookup_error"] is None


def test_status_script_records_long_running_runtime_health(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "126")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "runtime-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "in_progress")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "")
    started = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()

    def urlopen(request, timeout):
        if "/artifacts?" in request.full_url:
            return _Response({"artifacts": []})
        return _Response({
            "jobs": [{
                "name": "research",
                "status": "in_progress",
                "conclusion": None,
                "started_at": started,
                "steps": [{
                    "name": "Chronological OOS research",
                    "status": "in_progress",
                    "conclusion": None,
                    "started_at": started,
                }],
            }]
        })

    monkeypatch.setattr(status.urllib.request, "urlopen", urlopen)
    assert status.main() == 0
    payload = json.loads((Path("artifacts") / "research_validation_status.json").read_text(encoding="utf-8"))
    health = payload["runtime_health"]
    assert health["state"] == "LONG_RUNNING"
    assert health["active_step"] == "Chronological OOS research"
    assert health["job_elapsed_minutes"] >= 89.0
    assert health["active_step_elapsed_minutes"] >= 89.0
    assert health["stale_risk_threshold_minutes"] == 240


def test_status_script_detects_evidence_artifact(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "success")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "completed")
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


def test_status_script_marks_cancelled_artifact_as_partial(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "105")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "cancelled-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "completed")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "cancelled")

    def urlopen(request, timeout):
        if "/artifacts?" in request.full_url:
            return _Response({"artifacts": [{"name": "research-validation-evidence-105", "expired": False}]})
        return _Response({
            "jobs": [{"name": "research", "status": "completed", "conclusion": "cancelled", "steps": []}]
        })

    monkeypatch.setattr(status.urllib.request, "urlopen", urlopen)
    assert status.main() == 0
    payload = json.loads((Path("artifacts") / "research_validation_status.json").read_text(encoding="utf-8"))
    assert payload["evidence_artifact_raw_present"] is True
    assert payload["evidence_artifact_present"] is False
    assert payload["evidence_state"] == "PARTIAL"


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


def test_research_validation_experience_controls_are_verified_before_autopilot() -> None:
    validation = WORKFLOW.read_text(encoding="utf-8")
    verification = Path(".github/workflows/repository-verification.yml").read_text(encoding="utf-8")
    autopilot = Path(".github/workflows/research-autopilot.yml").read_text(encoding="utf-8")

    # Main changes are verified first; Research is dispatched only after
    # current-main verification passes.
    assert "  push:" not in validation
    assert '  push:\n    branches: [main]' in verification
    assert 'workflows: ["Repository verification"]' in autopilot
    assert "github.event.workflow_run.conclusion == 'success'" in autopilot

    # Research validation executes the experience-driven stages once
    # dispatched by the verified controller.
    expected_runtime_paths = (
        "scripts/generate_experience_candidates.py",
        "scripts/run_daily_research.py",
        "config/pipeline.yml",
    )
    for path in expected_runtime_paths:
        assert path in validation

import json
from datetime import datetime, timedelta, timezone
import subprocess
import textwrap
from pathlib import Path

import pytest

from scripts import persist_research_validation_status as status
from scripts import resolve_research_status_context as context


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
        blocks.append(textwrap.dedent("\n".join(block)) + "\n")
    return blocks


def _extract_python_heredocs(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() not in {"python - <<'PY'", "python3 - <<'PY'"}:
            i += 1
            continue
        i += 1
        block: list[str] = []
        while i < len(lines) and lines[i].strip() != "PY":
            block.append(lines[i])
            i += 1
        if i >= len(lines):
            raise AssertionError("unterminated Python heredoc in workflow")
        blocks.append(textwrap.dedent("\n".join(block)) + "\n")
        i += 1
    return blocks


def test_research_validation_uses_read_only_repository_permission():
    text = (ROOT / ".github" / "workflows" / "research-validation.yml").read_text(encoding="utf-8")
    assert "permissions:" in text
    assert "  actions: read" in text
    assert "  contents: read" in text
    assert "  contents: write" not in text
    assert "git push" not in text


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


    python_blocks = _extract_python_heredocs(text)
    status_python_blocks = _extract_python_heredocs(status_text)
    assert python_blocks, "No embedded Python heredocs found in research-validation workflow"
    for index, script in enumerate(python_blocks + status_python_blocks):
        try:
            compile(script, f"workflow-python-heredoc-{index}", "exec")
        except SyntaxError as exc:
            raise AssertionError(
                f"workflow embedded Python block {index} has invalid syntax: {exc}\nSCRIPT:\n{script}"
            ) from exc


def test_research_status_workflow_has_heartbeat_schedule() -> None:
    text = STATUS_WORKFLOW.read_text(encoding="utf-8")
    assert 'schedule:\n    - cron: "13,43 * * * *"' in text
    assert "workflow_dispatch:" in text
    assert "python scripts/resolve_research_status_context.py" in text
    assert "RESEARCH_STATUS_CONTEXT_FOUND" in text
    assert "if: env.RESEARCH_STATUS_CONTEXT_FOUND == 'true'" in text


def test_resolve_context_reuses_workflow_run_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "777")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "sha777")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "in_progress")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "")
    env_path = tmp_path / "github_env"
    monkeypatch.setenv("GITHUB_ENV", str(env_path))

    assert context.main() == 0
    output = env_path.read_text(encoding="utf-8")
    assert "RESEARCH_WORKFLOW_RUN_ID=777\n" in output
    assert "RESEARCH_WORKFLOW_SHA=sha777\n" in output
    assert "RESEARCH_WORKFLOW_STATUS=in_progress\n" in output
    assert "RESEARCH_WORKFLOW_CONCLUSION=\n" in output
    assert "RESEARCH_STATUS_CONTEXT_FOUND=true\n" in output


def test_resolve_context_discovers_latest_active_research(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESEARCH_WORKFLOW_RUN_ID", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_API_URL", "https://api.github.test")
    env_path = tmp_path / "github_env"
    monkeypatch.setenv("GITHUB_ENV", str(env_path))
    captured = {}

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        return _Response({
            "workflow_runs": [
                {"id": 10, "run_number": 10, "name": "Other", "status": "in_progress", "head_sha": "x"},
                {"id": 20, "run_number": 20, "name": "Research validation", "status": "in_progress", "head_sha": "old"},
                {"id": 21, "run_number": 21, "name": "Research validation", "status": "in_progress", "head_sha": "new"},
            ]
        })

    monkeypatch.setattr(context.urllib.request, "urlopen", urlopen)
    assert context.main() == 0
    assert "status=in_progress" in captured["url"]
    output = env_path.read_text(encoding="utf-8")
    assert "RESEARCH_WORKFLOW_RUN_ID=21\n" in output
    assert "RESEARCH_WORKFLOW_SHA=new\n" in output
    assert "RESEARCH_WORKFLOW_STATUS=in_progress\n" in output
    assert "RESEARCH_WORKFLOW_CONCLUSION=\n" in output
    assert "RESEARCH_STATUS_CONTEXT_FOUND=true\n" in output


def test_resolve_context_no_active_research_is_noop(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESEARCH_WORKFLOW_RUN_ID", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_API_URL", "https://api.github.test")
    env_path = tmp_path / "github_env"
    monkeypatch.setenv("GITHUB_ENV", str(env_path))

    monkeypatch.setattr(context.urllib.request, "urlopen", lambda request, timeout: _Response({"workflow_runs": []}))
    assert context.main() == 0
    assert env_path.read_text(encoding="utf-8") == "RESEARCH_STATUS_CONTEXT_FOUND=false\n"


def test_resolve_context_fails_closed_on_lookup_error(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESEARCH_WORKFLOW_RUN_ID", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    env_path = tmp_path / "github_env"
    monkeypatch.setenv("GITHUB_ENV", str(env_path))

    def fail(request, timeout):
        raise context.urllib.error.URLError("network failure")

    monkeypatch.setattr(context.urllib.request, "urlopen", fail)
    with pytest.raises(SystemExit, match="heartbeat run lookup"):
        context.main()

def test_research_validation_has_external_status_workflow() -> None:
    text = STATUS_WORKFLOW.read_text(encoding="utf-8")
    assert 'workflows: ["Research validation"]' in text
    assert "types: [completed, in_progress, requested]" in text
    assert "group: research-validation-status" in text
    assert "python scripts/persist_research_validation_status.py" in text
    assert "RESEARCH_WORKFLOW_RUN_ID: ${{ github.event.workflow_run.id || '' }}" in text
    assert "RESEARCH_WORKFLOW_SHA: ${{ github.event.workflow_run.head_sha || '' }}" in text
    assert "name: research-validation-evidence-${{ env.RESEARCH_WORKFLOW_RUN_ID }}" in text
    assert "run-id: ${{ env.RESEARCH_WORKFLOW_RUN_ID }}" in text
    assert "github.event_name == 'workflow_run'" in text
    assert "GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}" in text
    assert "ref: main" in text
    assert "github.event.workflow_run.head_sha" in text
    assert "ref: ${{ github.event.workflow_run.head_sha }}\n        with:" not in text
    assert text.index("Persist research validation status") < text.index("Preserve generated status outputs")
    assert text.index("Preserve generated status outputs") < text.index("Commit research status evidence")
    preserve_pos = text.index("      - name: Preserve generated status outputs")
    assert text.index("        if: always()", preserve_pos) < text.index("        shell: bash", preserve_pos)
    assert text.index("git checkout -B research-status origin/research-status --force") > text.index("Preserve generated status outputs")
    main_text = WORKFLOW.read_text(encoding="utf-8")
    assert "  research-status:" not in main_text
    assert "research-validation-status" not in main_text

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
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "failure")
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


def test_status_script_accepts_in_progress_workflow_before_research_job_exists(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("GITHUB_SHA", "default-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "124")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "active-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "in_progress")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "")
    monkeypatch.setenv("RESEARCH_STATUS_JOB_LOOKUP_RETRIES", "0")

    def urlopen(request, timeout):
        if "/artifacts?" in request.full_url:
            return _Response({"artifacts": []})
        return _Response({"jobs": []})

    monkeypatch.setattr(status.urllib.request, "urlopen", urlopen)

    assert status.main() == 0
    payload = json.loads(
        (Path("artifacts") / "research_validation_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status_lookup_ok"] is True
    assert payload["workflow_run_id"] == "124"
    assert payload["workflow_status"] == "in_progress"
    assert payload["job_status"] == "in_progress"
    assert payload["status_lookup_error"] is None


def test_status_script_retries_transient_research_job_visibility(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "125")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "retry-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "in_progress")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "")
    monkeypatch.setenv("RESEARCH_STATUS_JOB_LOOKUP_RETRIES", "2")
    sleeps = []
    monkeypatch.setattr(status.time, "sleep", sleeps.append)
    calls = {"jobs": 0}

    def urlopen(request, timeout):
        if "/artifacts?" in request.full_url:
            return _Response({"artifacts": []})
        calls["jobs"] += 1
        if calls["jobs"] < 3:
            return _Response({"jobs": []})
        return _Response({
            "jobs": [{
                "name": "research",
                "status": "in_progress",
                "conclusion": None,
                "steps": [{
                    "name": "Chronological OOS research",
                    "status": "in_progress",
                    "conclusion": None,
                }],
            }]
        })

    monkeypatch.setattr(status.urllib.request, "urlopen", urlopen)
    assert status.main() == 0
    payload = json.loads((Path("artifacts") / "research_validation_status.json").read_text(encoding="utf-8"))
    assert calls["jobs"] == 3
    assert sleeps == [2, 2]
    assert payload["status_lookup_ok"] is True
    assert payload["job_status"] == "in_progress"
    assert payload["research_step"] == "in_progress"

def test_status_script_does_not_overwrite_newer_research_run(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "199")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "old-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "completed")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "success")
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir()
    status_path = artifact_dir / "research_validation_status.json"
    existing = {"workflow_run_id": "200", "workflow_sha": "new-sha", "job_status": "success"}
    status_path.write_text(json.dumps(existing), encoding="utf-8")

    monkeypatch.setattr(status.urllib.request, "urlopen", lambda request, timeout: _Response({"jobs": [], "artifacts": []}))
    assert status.main() == 0
    assert json.loads(status_path.read_text(encoding="utf-8")) == existing


def test_status_script_accepts_cancelled_workflow_without_research_job(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("GITHUB_SHA", "default-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "123")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "cancelled-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "cancelled")

    def urlopen(request, timeout):
        if "/artifacts?" in request.full_url:
            return _Response({"artifacts": []})
        return _Response({"jobs": []})

    monkeypatch.setattr(status.urllib.request, "urlopen", urlopen)

    assert status.main() == 0
    payload = json.loads(
        (Path("artifacts") / "research_validation_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status_lookup_ok"] is True
    assert payload["job_status"] == "cancelled"
    assert payload["status_lookup_error"] is None


def test_status_script_records_long_running_runtime_health(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "126")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "runtime-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "in_progress")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "")
    started = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()

    def urlopen(request, timeout):
        if "/artifacts?" in request.full_url:
            return _Response({"artifacts": []})
        return _Response({
            "jobs": [{
                "name": "research",
                "status": "in_progress",
                "conclusion": None,
                "started_at": started,
                "steps": [{
                    "name": "Chronological OOS research",
                    "status": "in_progress",
                    "conclusion": None,
                    "started_at": started,
                }],
            }]
        })

    monkeypatch.setattr(status.urllib.request, "urlopen", urlopen)
    assert status.main() == 0
    payload = json.loads((Path("artifacts") / "research_validation_status.json").read_text(encoding="utf-8"))
    health = payload["runtime_health"]
    assert health["state"] == "LONG_RUNNING"
    assert health["active_step"] == "Chronological OOS research"
    assert health["job_elapsed_minutes"] >= 89.0
    assert health["active_step_elapsed_minutes"] >= 89.0
    assert health["stale_risk_threshold_minutes"] == 240


def test_status_script_detects_evidence_artifact(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "success")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "completed")
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


def test_status_script_marks_cancelled_artifact_as_partial(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("RESEARCH_WORKFLOW_RUN_ID", "105")
    monkeypatch.setenv("RESEARCH_WORKFLOW_SHA", "cancelled-sha")
    monkeypatch.setenv("RESEARCH_WORKFLOW_STATUS", "completed")
    monkeypatch.setenv("RESEARCH_WORKFLOW_CONCLUSION", "cancelled")

    def urlopen(request, timeout):
        if "/artifacts?" in request.full_url:
            return _Response({"artifacts": [{"name": "research-validation-evidence-105", "expired": False}]})
        return _Response({
            "jobs": [{"name": "research", "status": "completed", "conclusion": "cancelled", "steps": []}]
        })

    monkeypatch.setattr(status.urllib.request, "urlopen", urlopen)
    assert status.main() == 0
    payload = json.loads((Path("artifacts") / "research_validation_status.json").read_text(encoding="utf-8"))
    assert payload["evidence_artifact_raw_present"] is True
    assert payload["evidence_artifact_present"] is False
    assert payload["evidence_state"] == "PARTIAL"


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


def test_research_validation_experience_controls_flow_through_verified_autopilot() -> None:
    validation = WORKFLOW.read_text(encoding="utf-8")
    verification = (ROOT / ".github" / "workflows" / "repository-verification.yml").read_text(encoding="utf-8")
    autopilot = (ROOT / ".github" / "workflows" / "research-autopilot.yml").read_text(encoding="utf-8")

    # Research validation is intentionally not push-triggered. Every main change
    # is first checked by Repository verification, and only a successful
    # current-main verification may dispatch the Research validation workflow.
    assert "  push:" not in validation
    assert '  push:\n    branches: [main]' in verification
    assert 'workflows: ["Repository verification"]' in autopilot
    assert "github.event.workflow_run.conclusion == 'success'" in autopilot

    # The experience/research runtime remains explicitly present in the
    # Research validation execution itself.
    expected_runtime_paths = (
        "scripts/generate_experience_candidates.py",
        "scripts/run_daily_research.py",
    )
    for path in expected_runtime_paths:
        assert path in validation
