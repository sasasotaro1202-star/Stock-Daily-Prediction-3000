from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

INPUT = Path("data/research/automation_failure_records.jsonl")
OUTPUT = Path("data/research/automation_failure_triage.json")
SCHEMA_VERSION = 1


def _step_names(record: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for job in record.get("failed_jobs") or []:
        for step in job.get("failed_steps") or []:
            name = str(step.get("name") or "").strip()
            if name:
                names.append(name)
        if not names:
            name = str(job.get("name") or "").strip()
            if name:
                names.append(name)
    return tuple(sorted(set(names)))


def _priority(category: str, steps: tuple[str, ...]) -> tuple[int, str]:
    step_text = " ".join(steps).lower()
    if any(token in step_text for token in ("pit", "leakage", "meta-leakage", "survivorship", "universe")):
        return 100, "PIT_OR_UNIVERSE_REVIEW"
    if category == "PRODUCTION_WORKFLOW_FAILURE":
        return 90, "PRODUCTION_RECOVERY_REVIEW"
    if category == "RESEARCH_VALIDATION_FAILURE":
        return 80, "RESEARCH_VALIDATION_REPAIR"
    if category == "RESEARCH_AUTOMATION_FAILURE":
        return 70, "RESEARCH_AUTOMATION_REPAIR"
    if category == "WORKFLOW_FAILURE":
        return 60, "AUTOMATION_REPAIR"
    return 50, "REVIEW"


def _action(category: str, steps: tuple[str, ...]) -> str:
    step_text = " ".join(steps).lower()
    if any(token in step_text for token in ("pit", "leakage", "meta-leakage", "survivorship", "universe")):
        return "RUN_PIT_UNIVERSE_TRIAGE_AND_BLOCK_PRODUCTION_EVIDENCE"
    if category == "PRODUCTION_WORKFLOW_FAILURE":
        return "RUN_BOUNDED_RECOVERY_THEN_RECONCILE"
    if category == "RESEARCH_VALIDATION_FAILURE":
        return "REPAIR_VALIDATION_BEFORE_NEW_OOS"
    if category == "RESEARCH_AUTOMATION_FAILURE":
        return "REPAIR_RESEARCH_CONTROL_PLANE"
    return "REPAIR_AUTOMATION_AND_REVERIFY"


def triage(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    category_counts = Counter()
    for record in records:
        category = str(record.get("category") or "UNKNOWN")
        steps = _step_names(record)
        signature = "|".join([
            str(record.get("workflow_name") or "UNKNOWN"),
            category,
            ",".join(steps) or "UNKNOWN_STEP",
        ])
        item = grouped.setdefault(
            signature,
            {
                "signature": signature,
                "workflow_name": str(record.get("workflow_name") or "UNKNOWN"),
                "category": category,
                "step_names": list(steps),
                "count": 0,
                "run_ids": [],
                "latest_created_at": "",
                "head_shas": [],
            },
        )
        item["count"] += 1
        run_id = str(record.get("workflow_run_id") or "")
        if run_id and run_id not in item["run_ids"]:
            item["run_ids"].append(run_id)
        created = str(record.get("created_at") or "")
        if created > str(item["latest_created_at"]):
            item["latest_created_at"] = created
        sha = str(record.get("head_sha") or "")
        if sha and sha not in item["head_shas"]:
            item["head_shas"].append(sha)
        category_counts[category] += 1

    backlog: list[dict[str, Any]] = []
    for item in grouped.values():
        rank, reason = _priority(item["category"], tuple(item["step_names"]))
        backlog.append({
            **item,
            "priority_score": rank + min(int(item["count"]), 20),
            "priority_reason": reason,
            "action": _action(item["category"], tuple(item["step_names"])),
            "state": "OPEN_REVIEW",
            "research_only": True,
            "production_mutation": False,
            "promotion_allowed": False,
            "resolution_note": "Failure records do not prove the runtime failure remains unresolved; reconcile against later executions before closing.",
        })
    backlog.sort(
        key=lambda row: (
            -int(row["priority_score"]),
            -int(row["count"]),
            str(row["latest_created_at"]),
            str(row["signature"]),
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "RECONCILED",
        "source": str(INPUT),
        "records_discovered": len(records),
        "unique_failure_signatures": len(backlog),
        "category_counts": dict(sorted(category_counts.items())),
        "backlog": backlog[:100],
        "research_only": True,
        "production_mutation": False,
        "promotion_allowed": False,
    }


def main() -> int:
    if INPUT.exists():
        records = [
            json.loads(line)
            for line in INPUT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        records = []
    payload = triage(records)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
