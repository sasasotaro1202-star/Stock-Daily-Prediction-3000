from __future__ import annotations

import json
from pathlib import Path

from src.research.experience_memory import compact_view, load_experience_memory

MEMORY = Path("data/research/experience_memory.json")
OUT = Path("data/research/experience_review.json")


def main() -> None:
    memory = load_experience_memory(MEMORY)
    view = compact_view(memory)
    priority = view.get("research_priority", [])
    errors = view.get("error_types", {})
    high_conf = errors.get("high_confidence_wrong", {})
    high_disagreement = view.get("by_model_disagreement", {}).get(
        "disagreement>=0.06", {}
    )

    payload = {
        "status": "READY" if int(view.get("total_resolved", 0)) > 0 else "WARMUP",
        "generated_from": str(MEMORY),
        "total_resolved": int(view.get("total_resolved", 0)),
        "total_files_processed": int(view.get("total_files_processed", 0)),
        "total_deferred_files": int(view.get("total_deferred_files", 0)),
        "rolling": view.get("rolling", {}),
        "latest_session": view.get("latest_session"),
        "top_research_priority": priority[:30],
        "high_confidence_wrong": high_conf,
        "high_model_disagreement": high_disagreement,
        "top_hard_cases": view.get("top_hard_cases", [])[:50],
        "recent_hard_cases": view.get("recent_hard_cases", [])[:50],
        "anomalies": view.get("anomalies", [])[-20:],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
