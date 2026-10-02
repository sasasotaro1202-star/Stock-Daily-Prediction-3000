from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ALLOWED_TASK_TYPES = {
    "repository_verification",
    "research_smoke",
    "daily_research",
    "v13_audit",
    "experience_review",
}
TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def fail(message: str) -> None:
    raise SystemExit(f"INVALID_CHAT_REQUEST: {message}")


def validate(path: Path) -> dict:
    if path.suffixes[-2:] != [".request", ".json"]:
        fail("request file must end with .request.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        fail("request must be a JSON object")
    if data.get("schema_version") != 1:
        fail("schema_version must be 1")
    task_id = data.get("task_id")
    if not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id):
        fail("task_id is missing or invalid")
    task_type = data.get("task_type")
    if task_type not in ALLOWED_TASK_TYPES:
        fail(f"task_type must be one of {sorted(ALLOWED_TASK_TYPES)}")
    requested_at = data.get("requested_at")
    if not isinstance(requested_at, str):
        fail("requested_at is required")
    try:
        datetime.fromisoformat(requested_at.replace("Z", "+00:00"))
    except ValueError:
        fail("requested_at must be ISO-8601")
    mode = data.get("mode", "async")
    if mode != "async":
        fail("mode must be async")
    max_runtime = data.get("max_runtime_minutes", 50)
    if not isinstance(max_runtime, int) or not 1 <= max_runtime <= 50:
        fail("max_runtime_minutes must be an integer between 1 and 50")
    arguments = data.get("arguments", {})
    if arguments != {}:
        fail("arguments must be an empty object; task commands are fixed allowlist entries")
    return data


if __name__ == "__main__":
    if len(sys.argv) != 2:
        fail("usage: validate_chat_request.py <request.json>")
    request = validate(Path(sys.argv[1]))
    print(json.dumps(request, ensure_ascii=False, sort_keys=True))
