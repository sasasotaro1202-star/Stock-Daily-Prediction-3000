from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping

TIMESTAMP_FIELDS = (
    "prediction_time",
    "prediction_cutoff",
)
AVAILABLE_FIELD = "available_at"
PUBLISHED_FIELD = "published_at"
RETRIEVED_FIELD = "retrieved_at"


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _first_timestamp(row: Mapping[str, Any]) -> tuple[str | None, datetime | None]:
    for field in TIMESTAMP_FIELDS:
        parsed = _parse_timestamp(row.get(field))
        if parsed is not None:
            return field, parsed
    return None, None


def audit_pit_row(
    row: Mapping[str, Any],
    *,
    row_key: str | None = None,
) -> dict[str, Any]:
    violations: list[str] = []
    prediction_field, prediction_time = _first_timestamp(row)

    if prediction_field is None:
        violations.append("missing_or_invalid_prediction_time")
    available_time = _parse_timestamp(row.get(AVAILABLE_FIELD))
    if available_time is None:
        violations.append("missing_or_invalid_available_at")

    published_time = None
    if row.get(PUBLISHED_FIELD) is not None:
        published_time = _parse_timestamp(row.get(PUBLISHED_FIELD))
        if published_time is None:
            violations.append("invalid_published_at")

    retrieved_time = None
    if row.get(RETRIEVED_FIELD) is not None:
        retrieved_time = _parse_timestamp(row.get(RETRIEVED_FIELD))
        if retrieved_time is None:
            violations.append("invalid_retrieved_at")

    if prediction_time is not None and available_time is not None:
        if available_time > prediction_time:
            violations.append("available_at_after_prediction_cutoff")

    if published_time is not None and available_time is not None:
        if published_time > available_time:
            violations.append("published_at_after_available_at")

    if retrieved_time is not None and available_time is not None:
        if retrieved_time < available_time:
            violations.append("retrieved_at_before_available_at")

    if row.get("pit_status") != "PASS":
        violations.append("pit_status_not_pass")

    return {
        "ok": not violations,
        "row_key": row_key,
        "prediction_field": prediction_field,
        "prediction_time": (
            prediction_time.isoformat() if prediction_time is not None else None
        ),
        "available_at": (
            available_time.isoformat() if available_time is not None else None
        ),
        "published_at": (
            published_time.isoformat() if published_time is not None else None
        ),
        "retrieved_at": (
            retrieved_time.isoformat() if retrieved_time is not None else None
        ),
        "violations": violations,
    }


def canonical_case_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    scope = str(row.get("asset_class") or row.get("product_family") or "")
    symbol = str(
        row.get("provider_symbol")
        or row.get("symbol")
        or row.get("instrument")
        or ""
    )
    session = str(
        row.get("session_date")
        or row.get("date")
        or row.get("target_date")
        or ""
    )
    cutoff = str(
        row.get("prediction_time")
        or row.get("prediction_cutoff")
        or ""
    )
    return scope, symbol, session, cutoff


def audit_pit_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    violations: list[dict[str, Any]] = []
    duplicate_keys: list[list[Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for index, row in enumerate(rows):
        key = canonical_case_key(row)
        if key in seen:
            duplicate_keys.append([*key])
        else:
            seen.add(key)

        result = audit_pit_row(row, row_key=f"row:{index}")
        if not result["ok"]:
            violations.append(result)

    if duplicate_keys:
        violations.append(
            {
                "row_key": None,
                "violations": ["duplicate_canonical_case_key"],
                "duplicate_keys": duplicate_keys,
            }
        )

    return {
        "status": "PASS" if not violations else "FAIL",
        "rows": len(rows),
        "valid_rows": len(rows) - sum(
            1
            for row in violations
            if "violations" in row and row.get("row_key") is not None
        ),
        "violation_count": len(violations),
        "violations": violations,
    }


def audit_source_config(config: Mapping[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    for name, raw in config.items():
        if name == "policy" or not isinstance(raw, Mapping):
            continue
        if raw.get("enabled") is not True:
            continue

        pit_policy = raw.get("pit_policy")
        source_result = {
            "name": str(name),
            "enabled": True,
            "research_only_until_pit_validation": False,
            "availability_verified": False,
            "status": "UNVERIFIED",
        }
        if not isinstance(pit_policy, Mapping):
            violations.append(
                {"source": str(name), "reason": "missing_pit_policy"}
            )
            source_result["status"] = "BLOCKED"
            sources.append(source_result)
            continue

        research_only = (
            pit_policy.get("research_use_only_until_pit_validation") is True
        )
        source_result["research_only_until_pit_validation"] = research_only
        if not research_only:
            violations.append(
                {
                    "source": str(name),
                    "reason": "research_only_until_pit_validation_not_true",
                }
            )

        availability_flags = (
            "available_at_verified",
            "row_level_available_at_verified",
            "source_available_at_verified",
        )
        verified_flags = [flag for flag in availability_flags if pit_policy.get(flag) is True]
        source_result["availability_verified"] = bool(verified_flags)

        if verified_flags and not str(pit_policy.get("verification_evidence") or "").strip():
            violations.append(
                {
                    "source": str(name),
                    "reason": "availability_claim_has_no_verification_evidence",
                    "flags": verified_flags,
                }
            )
            source_result["status"] = "BLOCKED"
        elif pit_policy.get("source_available_at_inferred") is True:
            violations.append(
                {
                    "source": str(name),
                    "reason": "historical_available_at_must_not_be_inferred",
                }
            )
            source_result["status"] = "BLOCKED"
        elif research_only:
            source_result["status"] = "UNVERIFIED"

        sources.append(source_result)

    return {
        "status": "PASS" if not violations else "FAIL",
        "enabled_sources": len(sources),
        "unverified_sources": sum(row["status"] == "UNVERIFIED" for row in sources),
        "pit_ready_sources": sum(row["availability_verified"] for row in sources),
        "violations": violations,
        "sources": sources,
    }
