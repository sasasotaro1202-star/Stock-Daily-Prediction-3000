from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


FORBIDDEN_TOKENS = (
    "target_",
    "future_",
    "fwd_",
    "label_",
    "next_close",
    "next_open",
    "lead_",
    "shift_-",
    "tomorrow_",
)

PRICE_REQUIRED = {
    "symbol",
    "asset_class",
    "session_date",
    "available_at",
    "source",
    "provider_symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
}

CONTEXT_REQUIRED = {
    "family",
    "session_date",
    "available_at",
}


@dataclass(frozen=True)
class IndependentAuditResult:
    ok: bool
    violations: tuple[str, ...]
    checks: dict[str, int]


def _forbidden(columns: list[str]) -> list[str]:
    return sorted(
        f"forbidden_column_name:{name}"
        for name in columns
        if any(token in name.lower() for token in FORBIDDEN_TOKENS)
    )


def _audit_temporal_frame(
    frame: pd.DataFrame,
    *,
    key_columns: list[str],
    required: set[str],
    prefix: str,
    now: pd.Timestamp,
) -> tuple[list[str], dict[str, int]]:
    violations: list[str] = []
    checks: dict[str, int] = {}

    missing = sorted(required - set(frame.columns))
    if missing:
        violations.append(f"{prefix}:missing_columns:{missing}")
        return violations, {"rows": int(len(frame)), "duplicates": 0}

    if frame.empty:
        violations.append(f"{prefix}:empty")
        return violations, {"rows": 0, "duplicates": 0}

    session = pd.to_datetime(frame["session_date"], errors="coerce")
    available = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")

    checks["rows"] = int(len(frame))
    checks["invalid_session_date"] = int(session.isna().sum())
    checks["invalid_available_at"] = int(available.isna().sum())

    duplicates = int(frame.duplicated(key_columns).sum())
    checks["duplicates"] = duplicates

    if checks["invalid_session_date"]:
        violations.append(
            f"{prefix}:invalid_session_date:{checks['invalid_session_date']}"
        )
    if checks["invalid_available_at"]:
        violations.append(
            f"{prefix}:invalid_available_at:{checks['invalid_available_at']}"
        )
    if duplicates:
        violations.append(f"{prefix}:duplicates:{duplicates}")

    max_session = session.dropna().max()
    if pd.notna(max_session):
        today_local = now.tz_convert(ZoneInfo("Asia/Tokyo")).date()
        future_sessions = int(
            (
                session.notna()
                & session.dt.date.gt(today_local)
            ).sum()
        )
        checks["future_sessions"] = future_sessions
        if future_sessions:
            violations.append(
                f"{prefix}:future_session_date:{future_sessions}"
            )

    available_after_now = int(
        (available.notna() & available.gt(now)).sum()
    )
    checks["available_after_now"] = available_after_now
    if available_after_now:
        violations.append(
            f"{prefix}:available_at_after_audit_time:{available_after_now}"
        )

    retrieved_missing = 0
    retrieved_after_now = 0
    available_after_retrieved = 0
    if "retrieved_at" in frame.columns:
        retrieved = pd.to_datetime(
            frame["retrieved_at"], utc=True, errors="coerce"
        )
        retrieved_missing = int(retrieved.isna().sum())
        retrieved_after_now = int(
            (
                retrieved.notna()
                & retrieved.gt(now + pd.Timedelta(minutes=5))
            ).sum()
        )
        available_after_retrieved = int(
            (
                available.notna()
                & retrieved.notna()
                & available.gt(retrieved)
            ).sum()
        )
        checks["retrieved_missing"] = retrieved_missing
        checks["retrieved_after_now"] = retrieved_after_now
        checks["available_after_retrieved"] = available_after_retrieved

        if retrieved_after_now:
            violations.append(
                f"{prefix}:retrieved_at_in_future:{retrieved_after_now}"
            )
        if available_after_retrieved:
            violations.append(
                f"{prefix}:available_at_after_retrieved_at:{available_after_retrieved}"
            )

    # Enforce chronological integrity independently of any feature implementation.
    if len(key_columns) >= 2 and set(key_columns).issubset(frame.columns):
        observed = frame.assign(_audit_session=session)
        for _, group in observed.groupby(
            key_columns[:-1], sort=False, dropna=False
        ):
            values = group["_audit_session"]
            if values.notna().any() and not values.is_monotonic_increasing:
                violations.append(
                    f"{prefix}:non_monotonic_session_order:1"
                )
                break

    return violations, checks


def audit_raw_inputs(
    prices: pd.DataFrame,
    market_context: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
) -> IndependentAuditResult:
    now = now or pd.Timestamp.now(tz="UTC")
    violations: list[str] = []

    violations.extend(_forbidden(list(prices.columns)))
    violations.extend(_forbidden(list(market_context.columns)))

    price_violations, price_checks = _audit_temporal_frame(
        prices,
        key_columns=["asset_class", "symbol", "session_date"],
        required=PRICE_REQUIRED,
        prefix="prices",
        now=now,
    )
    context_violations, context_checks = _audit_temporal_frame(
        market_context,
        key_columns=["family", "session_date"],
        required=CONTEXT_REQUIRED,
        prefix="market_context",
        now=now,
    )
    violations.extend(price_violations)
    violations.extend(context_violations)

    # Raw input must not contain target/label columns. Targets are created later.
    raw_target_columns = [
        c for c in list(prices.columns) + list(market_context.columns)
        if c.lower().startswith(("target_", "label_", "future_", "fwd_"))
    ]
    if raw_target_columns:
        violations.extend(
            f"raw_target_column_present:{c}"
            for c in sorted(set(raw_target_columns))
        )

    # Check that the primary PIT timestamp is never earlier than the dated
    # observation's UTC midnight. This catches impossible timestamps without
    # making assumptions about exchange-local close times.
    if {"session_date", "available_at"}.issubset(prices.columns):
        session_utc = pd.to_datetime(
            prices["session_date"], errors="coerce", utc=True
        )
        available_utc = pd.to_datetime(
            prices["available_at"], errors="coerce", utc=True
        )
        impossible = int(
            available_utc.notna()
            & session_utc.notna()
            & available_utc.lt(session_utc)
        )
        price_checks["available_before_session_date"] = impossible
        if impossible:
            violations.append(
                f"prices:available_at_before_session_date:{impossible}"
            )

    checks = {
        "prices_rows": int(price_checks.get("rows", 0)),
        "market_context_rows": int(context_checks.get("rows", 0)),
        "prices_duplicates": int(price_checks.get("duplicates", 0)),
        "market_context_duplicates": int(context_checks.get("duplicates", 0)),
        "prices_available_after_now": int(
            price_checks.get("available_after_now", 0)
        ),
        "context_available_after_now": int(
            context_checks.get("available_after_now", 0)
        ),
        "prices_available_after_retrieved": int(
            price_checks.get("available_after_retrieved", 0)
        ),
        "prices_available_before_session_date": int(
            price_checks.get("available_before_session_date", 0)
        ),
    }

    return IndependentAuditResult(
        ok=not violations,
        violations=tuple(sorted(set(violations))),
        checks=checks,
    )
