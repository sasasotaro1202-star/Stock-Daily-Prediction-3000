from __future__ import annotations

import json
from pathlib import Path

from src.research.experience_candidate_evaluator import evaluate_experience_candidates

CANDIDATES = Path("data/research/experience_candidates.json")
METRICS = Path("data/research/latest_metrics.json")
OUT = Path("data/research/experience_candidate_evaluation.json")


def main() -> None:
    if not CANDIDATES.exists():
        raise SystemExit("DEFERRED: experience candidate plan is unavailable")
    if not METRICS.exists():
        raise SystemExit("DEFERRED: chronological OOS metrics are unavailable")

    candidate_plan = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    result = evaluate_experience_candidates(candidate_plan, metrics)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
