from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "PROJECT_SOURCE.md"
OUT_PATH = ROOT / "data" / "research" / "project_source_contract.json"

REQUIRED_TERMS = (
    "available_at <= prediction_cutoff",
    "random split禁止",
    "Chronological OOS/WFO",
    "survivorship",
    "frozen holdout",
    "NO-FAKE-SUCCESS",
    "cost不明 = HOLD / UNCONFIRMED。",
    "PIT > Apparent Backtest Gain",
    "Future Generalization",
    "Safe Degradation > Forced Prediction",
)

def validate_source_text(text: str) -> list[str]:
    errors: list[str] = []
    if "=== COPY START ===" in text or "=== COPY END ===" in text:
        errors.append("copy_wrapper_present")
    if "Stock-Daily-Prediction-3000" not in text:
        errors.append("project_name_missing")
    if "https://github.com/sasasotaro1202-star/Stock-Daily-Prediction-3000" not in text:
        errors.append("target_repository_missing")
    section_numbers = [int(match.group(1)) for match in re.finditer(r"(?m)^(\d+)\.\s+[A-Z][A-Z0-9 /&._-]*$", text)]
    expected = list(range(1, 99))
    if section_numbers != expected:
        errors.append("section_sequence_invalid:" + ",".join(map(str, section_numbers[:110])))
    for term in REQUIRED_TERMS:
        if term not in text:
            errors.append(f"required_term_missing:{term}")
    if "UNKNOWN\nまたは\nUNVERIFIABLE" not in text:
        errors.append("unknown_unverifiable_fail_closed_rule_missing")
    return errors

def build_contract() -> dict[str, Any]:
    if not SOURCE_PATH.is_file():
        raise SystemExit(f"FAIL: missing canonical project source: {SOURCE_PATH}")
    text = SOURCE_PATH.read_text(encoding="utf-8")
    errors = validate_source_text(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAILED",
        "source": str(SOURCE_PATH.relative_to(ROOT)),
        "sha256": digest,
        "line_count": len(text.splitlines()),
        "section_count": len(re.findall(r"(?m)^\d+\.\s+[A-Z][A-Z0-9 /&._-]*$", text)),
        "required_terms": list(REQUIRED_TERMS),
        "validation_errors": errors,
        "git_sha": os.environ.get("GITHUB_SHA"),
        "repository": os.environ.get("GITHUB_REPOSITORY", "sasasotaro1202-star/Stock-Daily-Prediction-3000"),
    }

def main() -> int:
    contract = build_contract()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    return 0 if contract["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
