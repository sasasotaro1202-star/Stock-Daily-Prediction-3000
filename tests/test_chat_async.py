import json
from pathlib import Path

from scripts.validate_chat_request import validate
from scripts.run_chat_task import TASK_COMMANDS


def test_chat_request_schema_is_strict_and_allowlisted(tmp_path):
    path = tmp_path / "demo.request.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "demo-001",
                "task_type": "research_smoke",
                "requested_at": "2026-10-02T07:00:00+00:00",
                "mode": "async",
                "max_runtime_minutes": 50,
                "arguments": {},
            }
        ),
        encoding="utf-8",
    )
    data = validate(path)
    assert data["task_type"] == "research_smoke"


def test_chat_task_commands_are_fixed_not_shell_strings():
    for task_type, commands in TASK_COMMANDS.items():
        assert task_type in {
            "repository_verification",
            "research_smoke",
            "daily_research",
            "v13_audit",
            "experience_review",
        }
        assert all(isinstance(command, list) for command in commands)
        assert all(
            command and (command[0].endswith("python") or command[0] == "python")
            for command in commands
        )


def test_async_chat_worker_has_total_deadline_and_artifact_read_scope():
    runner = Path("scripts/run_chat_task.py").read_text(encoding="utf-8")
    workflow = __import__("yaml").safe_load(
        Path(".github/workflows/chat-request-worker.yml").read_text(encoding="utf-8")
    )
    assert "deadline = time.monotonic()" in runner
    assert "timeout=remaining" in runner
    assert workflow["permissions"] == {"actions": "read", "contents": "read"}
    assert workflow["jobs"]["execute"]["timeout-minutes"] == 55
