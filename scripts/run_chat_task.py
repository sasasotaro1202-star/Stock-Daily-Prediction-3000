from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

TASK_COMMANDS = {
    "repository_verification": [
        [sys.executable, "-m", "compileall", "-q", "src", "scripts"],
        [sys.executable, "-c", (
            "from pathlib import Path; import yaml; "
            "docs=[yaml.safe_load(p.read_text(encoding='utf-8')) "
            "for p in Path('.github/workflows').glob('*.yml')]; "
            "assert all(isinstance(d, dict) and 'jobs' in d for d in docs); "
            "print('workflow_yaml_valid', len(docs))"
        )],
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "scripts/research_smoke.py"],
        [sys.executable, "scripts/automation_invariants.py"],
        [sys.executable, "scripts/production_invariants.py"],
    ],
    "research_smoke": [[sys.executable, "scripts/research_smoke.py"]],
    "daily_research": [[sys.executable, "scripts/run_daily_research.py"]],
    "v13_audit": [
        [sys.executable, "scripts/run_v13_meta_leakage_audit.py"],
        [sys.executable, "scripts/run_v13_adversarial_robustness.py"],
    ],
    "experience_review": [
        [sys.executable, "scripts/generate_experience_candidates.py"],
        [sys.executable, "scripts/evaluate_experience_candidates.py"],
    ],
}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_chat_task.py <request.json>")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    task_id = request["task_id"]
    task_type = request["task_type"]
    max_minutes = min(int(request.get("max_runtime_minutes", 50)), 50)
    commands = TASK_COMMANDS.get(task_type)
    if not commands:
        raise SystemExit(f"UNSUPPORTED_CHAT_TASK: {task_type}")

    deadline = time.monotonic() + max_minutes * 60
    log_dir = Path("chat_task_results")
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "task_id": task_id,
        "task_type": task_type,
        "status": "RUNNING",
        "max_runtime_minutes": max_minutes,
        "commands": [shlex.join(cmd) for cmd in commands],
        "results": [],
    }
    (log_dir / "status.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for index, command in enumerate(commands, start=1):
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            summary["status"] = "TIMEOUT"
            break
        started = time.monotonic()
        print(f"CHAT_TASK_START task_id={task_id} step={index} command={shlex.join(command)}", flush=True)
        try:
            completed = subprocess.run(
                command,
                check=False,
                timeout=remaining,
                text=True,
            )
            rc = int(completed.returncode)
            status = "PASS" if rc == 0 else "FAILED"
        except subprocess.TimeoutExpired:
            rc = 124
            status = "TIMEOUT"
        result = {
            "step": index,
            "command": shlex.join(command),
            "returncode": rc,
            "status": status,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        summary["results"].append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        if status != "PASS":
            summary["status"] = status
            (log_dir / "status.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return rc or 1

    if summary["status"] == "RUNNING":
        summary["status"] = "COMPLETED"
    (log_dir / "status.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"CHAT_TASK_COMPLETE task_id={task_id} status=COMPLETED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
