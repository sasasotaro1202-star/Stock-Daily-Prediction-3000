from __future__ import annotations

import json
import re
from pathlib import Path

SOURCE = Path("research_sources/external_prediction_system_oss_top1000_source.tsv")
OUTPUT = Path("config/external_prediction_system_oss_top1000_registry.json")

OWNER_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def main() -> int:
    rows = []
    seen = {}
    for raw in SOURCE.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#") or raw.startswith("RANK"):
            continue
        rank_s, entry = raw.split("\t", 1)
        rank = int(rank_s)
        entry = entry.strip()
        if OWNER_REPO.fullmatch(entry):
            entry_type = "EXPLICIT_REPOSITORY_REF"
            canonical_ref = entry
        else:
            entry_type = "NONCANONICAL_OR_ABSTRACT"
            canonical_ref = None

        duplicate_of = seen.get(entry)
        if duplicate_of is None:
            seen[entry] = rank

        rows.append(
            {
                "source_rank": rank,
                "source_entry": entry,
                "entry_type": entry_type,
                "canonical_ref": canonical_ref,
                "duplicate_of_rank": duplicate_of,
                "evidence_level": "E1_EXTERNAL_CLAIM",
                "status": "HOLD_RESEARCH_ONLY",
                "research_only": True,
                "production_changed": False,
                "promotion_allowed": False,
                "frozen_holdout_used": False,
            }
        )

    ranks = [r["source_rank"] for r in rows]
    if len(rows) != 1000:
        raise SystemExit(f"FAIL: expected 1000 rows, got {len(rows)}")
    if sorted(ranks) != list(range(1, 1001)):
        raise SystemExit("FAIL: source ranks are not exactly 1..1000")

    explicit = sum(r["entry_type"] == "EXPLICIT_REPOSITORY_REF" for r in rows)
    abstract = len(rows) - explicit
    duplicates = sum(r["duplicate_of_rank"] is not None for r in rows)

    result = {
        "schema_version": 1,
        "registry_id": "external-prediction-system-oss-top1000-source-registry-v1",
        "source_file": str(SOURCE),
        "source_rank_integrity": {
            "expected_count": 1000,
            "actual_count": len(rows),
            "unique_rank_count": len(set(ranks)),
            "min_rank": min(ranks),
            "max_rank": max(ranks),
            "complete": True,
        },
        "entry_summary": {
            "explicit_repository_refs": explicit,
            "noncanonical_or_abstract": abstract,
            "duplicate_entry_rows": duplicates,
            "unique_entry_strings": len(seen),
        },
        "safety_contract": {
            "external_popularity_is_not_predictive_evidence": True,
            "source_metadata_must_be_verified": True,
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
            "frozen_holdout_used": False,
        },
        "rows": rows,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "PASS",
        "rows": len(rows),
        "explicit_repository_refs": explicit,
        "noncanonical_or_abstract": abstract,
        "duplicate_entry_rows": duplicates,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
