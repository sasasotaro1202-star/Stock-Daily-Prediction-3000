from __future__ import annotations

import json
from pathlib import Path
import os

import yaml

from src.research.pit_contract import audit_pit_rows, audit_source_config

CONFIG = Path("config/research_data_sources.yml")
LEDGER_CANDIDATES = (
    Path("data/research/ultimate_v13/prediction_ledger.json"),
    Path("data/research/prediction_ledger.json"),
)
OUT = Path("data/research/research_pit_contract_audit.json")


def _load_ledger() -> tuple[str, list[dict]]:
    for path in LEDGER_CANDIDATES:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("row_level")
        if not isinstance(rows, list):
            raise SystemExit(f"FAIL: prediction ledger row_level is not a list: {path}")
        return str(path), rows
    return "", []


def main() -> int:
    if not CONFIG.exists():
        raise SystemExit("FAIL: research data-source config is missing")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise SystemExit("FAIL: research data-source config is not a mapping")

    source_audit = audit_source_config(config)
    ledger_path, rows = _load_ledger()
    require_ledger = os.environ.get("REQUIRE_PIT_LEDGER", "0").strip() == "1"

    if rows:
        ledger_audit = audit_pit_rows(rows)
    else:
        ledger_audit = {
            "status": "NO_LEDGER_EVIDENCE",
            "rows": 0,
            "valid_rows": 0,
            "violation_count": 0,
            "violations": [],
        }
        if require_ledger:
            ledger_audit = {
                "status": "FAIL",
                "rows": 0,
                "valid_rows": 0,
                "violation_count": 1,
                "violations": [
                    {
                        "row_key": None,
                        "violations": ["NO_LEDGER_EVIDENCE"],
                        "reason": (
                            "A successful post-OOS PIT audit requires a non-empty "
                            "prediction ledger."
                        ),
                    }
                ],
            }

    payload = {
        "schema_version": 1,
        "status": (
            "FAIL"
            if source_audit["status"] == "FAIL" or ledger_audit["status"] == "FAIL"
            else "RESEARCH_ONLY_UNVERIFIED"
        ),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "source_config": source_audit,
        "prediction_ledger": {
            "path": ledger_path or None,
            **ledger_audit,
        },
        "contracts": {
            "nonempty_prediction_ledger_required_when_enforced": require_ledger,
            "unknown_or_unverifiable_availability_is_not_pit_ready": True,
            "available_at_must_not_exceed_prediction_cutoff": True,
            "published_at_must_not_exceed_available_at": True,
            "retrieved_at_must_not_precede_available_at_when_both_are_present": True,
            "pit_status_must_be_pass_for_row_level_readiness": True,
            "duplicate_canonical_cases_are_blocking": True,
            "research_sources_never_promote_automatically": True,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if payload["status"] == "FAIL":
        raise SystemExit("FAIL: research PIT contract audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
