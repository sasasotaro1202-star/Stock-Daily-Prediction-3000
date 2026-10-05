from __future__ import annotations

import json
from pathlib import Path

from src.research.experience_candidates import build_experience_candidate_plan
from src.research.experience_memory import load_experience_memory


MEMORY = Path("data/research/experience_memory.json")
OUT = Path("data/research/experience_candidates.json")


def main() -> int:
    memory = load_experience_memory(MEMORY)
    plan = build_experience_candidate_plan(memory)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(plan, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
