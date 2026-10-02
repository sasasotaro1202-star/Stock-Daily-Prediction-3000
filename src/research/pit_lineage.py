from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timedelta
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd


_POLICY_VERSION = "historical_scheduled_prediction_clock_v1"


def _parse_hhmm(value: Any, *, key: str) -> time:
    if not isinstance(value, str):
        raise ValueError(f"missing_or_invalid_schedule:{key}")
    raw = value.strip()
    try:
        parsed = time.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid_schedule_time:{key}") from exc
    if parsed.second != 0 or parsed.microsecond != 0:
        raise ValueError(f"schedule_time_must_be_minute_precision:{key}")
    return parsed


def _next_weekday(day: date) -> date:
    candidate = day
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def scheduled_prediction_time(
    session_date: Any,
    asset_class: Any,
    pipeline_cfg: Mapping[str, Any],
) -> tuple[pd.Timestamp | None, str]:
    """Return the deterministic historical PIT prediction clock.

    This is the declared prediction schedule used for chronological OOS, not an
    observed GitHub runner wall-clock. It is intentionally conservative for PIT:
    research timestamps are never inferred from future outcomes or retrieval time.
    """
    if isinstance(session_date, pd.Timestamp):
        session_day = session_date.date()
    elif isinstance(session_date, datetime):
        session_day = session_date.date()
    elif isinstance(session_date, date):
        session_day = session_date
    else:
        session_day = pd.Timestamp(session_date).date()

    asset = str(asset_class)
    if asset.startswith("jp_"):
        schedule_key = "asia_prediction_time_jst"
        clock = _parse_hhmm(pipeline_cfg.get(schedule_key), key=schedule_key)
        local = datetime.combine(
            session_day,
            clock,
            tzinfo=ZoneInfo("Asia/Tokyo"),
        )
    elif asset.startswith("us_"):
        schedule_key = "us_prediction_time_jst"
        clock = _parse_hhmm(pipeline_cfg.get(schedule_key), key=schedule_key)
        # U.S. daily bars close on the prior U.S. session and become the input
        # to the next configured weekday's 07:17 JST prediction run.
        prediction_day = _next_weekday(session_day + timedelta(days=1))
        local = datetime.combine(
            prediction_day,
            clock,
            tzinfo=ZoneInfo("Asia/Tokyo"),
        )
    else:
        return None, "BLOCKED_UNKNOWN_ASSET_CLASS"

    return pd.Timestamp(local).tz_convert("UTC"), f"DECLARED_CONFIG_SCHEDULE:{schedule_key}"


def build_research_pit_lineage(
    frame: pd.DataFrame,
    pipeline_cfg: Mapping[str, Any],
) -> dict[str, list[Any]]:
    """Build row-level PIT/provenance fields for a chronological OOS test slice."""
    required = {"session_date", "asset_class", "available_at"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing_pit_lineage_columns:{','.join(missing)}")

    available = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    retrieved = (
        pd.to_datetime(frame["retrieved_at"], utc=True, errors="coerce")
        if "retrieved_at" in frame.columns
        else pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    )

    prediction_times: list[str | None] = []
    prediction_sources: list[str] = []
    available_values: list[str | None] = []
    retrieved_values: list[str | None] = []
    statuses: list[str] = []
    lineage_hashes: list[str | None] = []
    available_methods: list[str | None] = []
    sources: list[str | None] = []
    provider_symbols: list[str | None] = []
    retrieval_run_ids: list[str | None] = []

    for idx in frame.index:
        prediction, prediction_source = scheduled_prediction_time(
            frame.at[idx, "session_date"],
            frame.at[idx, "asset_class"],
            pipeline_cfg,
        )
        available_at = available.loc[idx]
        retrieved_at = retrieved.loc[idx]

        available_iso = available_at.isoformat() if pd.notna(available_at) else None
        retrieved_iso = retrieved_at.isoformat() if pd.notna(retrieved_at) else None
        prediction_iso = prediction.isoformat() if prediction is not None else None

        method = frame.at[idx, "available_at_method"] if "available_at_method" in frame.columns else None
        source = frame.at[idx, "source"] if "source" in frame.columns else None
        provider = frame.at[idx, "provider_symbol"] if "provider_symbol" in frame.columns else None
        run_id = frame.at[idx, "retrieval_run_id"] if "retrieval_run_id" in frame.columns else None

        if prediction is None:
            status = prediction_source
        elif pd.isna(available_at):
            status = "BLOCKED_INVALID_AVAILABLE_AT"
        elif pd.isna(retrieved_at):
            status = "BLOCKED_INVALID_RETRIEVED_AT"
        elif available_at > prediction:
            status = "BLOCKED_AVAILABLE_AFTER_PREDICTION"
        elif not str(method or "").strip():
            status = "BLOCKED_MISSING_AVAILABLE_AT_METHOD"
        elif not str(source or "").strip():
            status = "BLOCKED_MISSING_SOURCE"
        elif not str(provider or "").strip():
            status = "BLOCKED_MISSING_PROVIDER_SYMBOL"
        else:
            status = "PASS"

        lineage_payload = {
            "policy_version": _POLICY_VERSION,
            "session_date": str(frame.at[idx, "session_date"]),
            "asset_class": str(frame.at[idx, "asset_class"]),
            "prediction_time": prediction_iso,
            "prediction_time_source": prediction_source,
            "available_at": available_iso,
            "retrieved_at": retrieved_iso,
            "available_at_method": str(method) if method is not None else None,
            "source": str(source) if source is not None else None,
            "provider_symbol": str(provider) if provider is not None else None,
            "retrieval_run_id": str(run_id) if run_id is not None else None,
            "pit_status": status,
        }
        digest = hashlib.sha256(
            json.dumps(
                lineage_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

        prediction_times.append(prediction_iso)
        prediction_sources.append(prediction_source)
        available_values.append(available_iso)
        retrieved_values.append(retrieved_iso)
        statuses.append(status)
        lineage_hashes.append(digest)
        available_methods.append(
            str(method) if method is not None else None
        )
        sources.append(str(source) if source is not None else None)
        provider_symbols.append(
            str(provider) if provider is not None else None
        )
        retrieval_run_ids.append(
            str(run_id) if run_id is not None else None
        )

    return {
        "policy_version": _POLICY_VERSION,
        "prediction_time": prediction_times,
        "prediction_time_source": prediction_sources,
        "prediction_time_observed": [False] * len(frame),
        "available_at": available_values,
        "retrieved_at": retrieved_values,
        "available_at_method": available_methods,
        "source": sources,
        "provider_symbol": provider_symbols,
        "retrieval_run_id": retrieval_run_ids,
        "pit_status": statuses,
        "lineage_sha256": lineage_hashes,
    }
