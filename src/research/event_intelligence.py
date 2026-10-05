from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

EVENT_INTELLIGENCE_FEATURE_COLUMNS = (
    "event_data_available",
    "event_days_since_latest",
    "event_count_5d",
    "event_count_30d",
    "event_count_90d",
    "event_current_report_5d",
    "event_current_report_30d",
    "event_financial_report_180d",
    "event_ownership_180d",
    "event_proxy_180d",
    "event_burst_5d_vs_30d",
    "event_rate_change_30d_vs_prev180d",
    "event_latest_is_current_report",
    "event_latest_is_financial_report",
    "event_latest_is_ownership",
    "event_latest_is_proxy",
)

_FORM_FAMILY = {
    "8-K": "current_report",
    "6-K": "current_report",
    "10-Q": "financial_report",
    "10-K": "financial_report",
    "20-F": "financial_report",
    "40-F": "financial_report",
    "SC 13D": "ownership",
    "SC 13G": "ownership",
    "DEF 14A": "proxy",
    "S-1": "offering",
    "S-3": "offering",
    "S-4": "offering",
    "424B2": "offering",
}

_NS_PER_DAY = 86_400_000_000_000


def _event_id(row: pd.Series) -> str:
    accession = str(row.get("accession_number") or "").strip()
    if accession:
        return f"{str(row.get('symbol', ''))}:{accession}"
    payload = "|".join(
        [
            str(row.get("symbol", "")),
            str(row.get("available_at", "")),
            str(row.get("form", "")),
            str(row.get("primary_document", "")),
        ]
    )
    return f"{str(row.get('symbol', ''))}:anon-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def build_event_ledger(filings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize source events into a deterministic PIT-ready event ledger.

    Events without a known available_at are excluded from the usable ledger.
    Callers must treat any pit_invalid_rows > 0 as DEFERRED.
    """
    required = {"symbol", "asset_class", "available_at", "form"}
    if filings is None or filings.empty:
        return pd.DataFrame(), {
            "status": "DEFERRED",
            "reason": "empty_input",
            "input_rows": 0,
            "pit_invalid_rows": 0,
            "deduplicated_rows": 0,
            "usable_rows": 0,
            "symbols": 0,
            "families": [],
        }
    if not required.issubset(filings.columns):
        missing = sorted(required - set(filings.columns))
        return pd.DataFrame(), {
            "status": "DEFERRED",
            "reason": "missing_columns:" + ",".join(missing),
            "input_rows": int(len(filings)),
            "pit_invalid_rows": 0,
            "deduplicated_rows": 0,
            "usable_rows": 0,
            "symbols": 0,
            "families": [],
        }

    data = filings.copy()
    data["symbol"] = data["symbol"].astype(str).str.strip()
    data["asset_class"] = data["asset_class"].astype(str).str.strip()
    data["form"] = data["form"].astype(str).str.upper().str.strip()
    data["available_at"] = pd.to_datetime(data["available_at"], utc=True, errors="coerce")
    pit_invalid = int(data["available_at"].isna().sum())

    data = data[
        data["asset_class"].eq("us_stock")
        & data["symbol"].ne("")
        & data["available_at"].notna()
        & data["form"].ne("")
    ].copy()

    if data.empty:
        return pd.DataFrame(), {
            "status": "DEFERRED" if pit_invalid else "EMPTY",
            "reason": "no_pit_safe_events",
            "input_rows": int(len(filings)),
            "pit_invalid_rows": pit_invalid,
            "deduplicated_rows": 0,
            "usable_rows": 0,
            "symbols": 0,
            "families": [],
        }

    for col in (
        "accession_number",
        "primary_document",
        "source",
        "source_url",
        "collected_at",
    ):
        if col not in data.columns:
            data[col] = ""

    data["event_id"] = data.apply(_event_id, axis=1)
    data["event_family"] = data["form"].map(_FORM_FAMILY).fillna("other")

    before = len(data)
    data = (
        data.sort_values(["symbol", "available_at", "event_id"])
        .drop_duplicates(["symbol", "event_id"], keep="last")
        .reset_index(drop=True)
    )
    deduplicated = before - len(data)

    cols = [
        "event_id",
        "symbol",
        "asset_class",
        "available_at",
        "form",
        "event_family",
        "accession_number",
        "primary_document",
        "source",
        "source_url",
        "collected_at",
    ]
    data = data[cols].copy()
    status = "DEFERRED" if pit_invalid else "PASS"
    return data, {
        "status": status,
        "reason": None if status == "PASS" else "pit_unknown_available_at",
        "input_rows": int(len(filings)),
        "usable_rows": int(len(data)),
        "pit_invalid_rows": pit_invalid,
        "deduplicated_rows": int(deduplicated),
        "symbols": int(data["symbol"].nunique()),
        "families": sorted(data["event_family"].unique().tolist()),
    }


def _count_window(
    event_ns: np.ndarray,
    query_ns: np.ndarray,
    lower_days: int,
    upper_days: int,
    flags: np.ndarray | None = None,
) -> np.ndarray:
    right = np.searchsorted(event_ns, query_ns - lower_days * _NS_PER_DAY, side="right")
    left = np.searchsorted(event_ns, query_ns - upper_days * _NS_PER_DAY, side="left")
    values = np.ones(event_ns.size, dtype=float) if flags is None else flags.astype(float)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    return cumulative[right] - cumulative[left]


def add_event_intelligence_features(
    prices: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Attach PIT-safe event signatures derived from a normalized event ledger."""
    out = prices.copy()
    for col in EVENT_INTELLIGENCE_FEATURE_COLUMNS:
        out[col] = 0.0

    if events is None or events.empty:
        return out

    required = {"symbol", "asset_class", "available_at", "form", "event_family"}
    if not required.issubset(events.columns):
        return out

    out["available_at"] = pd.to_datetime(out["available_at"], utc=True, errors="coerce")
    ev = events.copy()
    ev["symbol"] = ev["symbol"].astype(str)
    ev["asset_class"] = ev["asset_class"].astype(str)
    ev["form"] = ev["form"].astype(str).str.upper()
    ev["event_family"] = ev["event_family"].astype(str)
    ev["available_at"] = pd.to_datetime(ev["available_at"], utc=True, errors="coerce")
    ev = ev[
        ev["asset_class"].eq("us_stock")
        & ev["available_at"].notna()
        & ev["symbol"].ne("")
    ].copy()
    if ev.empty:
        return out

    sort_cols = ["symbol", "available_at"]
    if "event_id" in ev.columns:
        sort_cols.append("event_id")
    ev = ev.sort_values(sort_cols).reset_index(drop=True)

    us_mask = out["asset_class"].astype(str).eq("us_stock")
    for symbol, idx_values in out.loc[us_mask].groupby(
        out.loc[us_mask, "symbol"].astype(str)
    ).groups.items():
        idx = np.asarray(list(idx_values), dtype=int)
        query = out.loc[idx].sort_values("available_at")
        valid_q = query["available_at"].notna().to_numpy()
        if not valid_q.any():
            continue

        current = ev[ev["symbol"].eq(symbol)].sort_values("available_at")
        if current.empty:
            continue

        event_ns = current["available_at"].astype("int64").to_numpy()
        query_ns_all = query["available_at"].astype("int64").to_numpy()
        families = current["event_family"].to_numpy()

        valid_query_ns = query_ns_all[valid_q]
        right = np.searchsorted(event_ns, valid_query_ns, side="right")
        has_event = right > 0
        latest_idx = np.maximum(right - 1, 0)

        age_days = np.zeros(valid_query_ns.size, dtype=float)
        age_days[has_event] = (
            valid_query_ns[has_event] - event_ns[latest_idx[has_event]]
        ) / _NS_PER_DAY

        count_5d = _count_window(event_ns, valid_query_ns, 0, 5)
        count_30d = _count_window(event_ns, valid_query_ns, 0, 30)
        count_90d = _count_window(event_ns, valid_query_ns, 0, 90)
        current_5d = _count_window(
            event_ns, valid_query_ns, 0, 5, np.isin(families, ["current_report"])
        )
        current_30d = _count_window(
            event_ns, valid_query_ns, 0, 30, np.isin(families, ["current_report"])
        )
        financial_180d = _count_window(
            event_ns, valid_query_ns, 0, 180, np.isin(families, ["financial_report"])
        )
        ownership_180d = _count_window(
            event_ns, valid_query_ns, 0, 180, np.isin(families, ["ownership"])
        )
        proxy_180d = _count_window(
            event_ns, valid_query_ns, 0, 180, np.isin(families, ["proxy"])
        )
        prior_180d = _count_window(event_ns, valid_query_ns, 30, 210)

        burst = np.zeros_like(count_30d)
        np.divide(
            6.0 * count_5d,
            np.maximum(count_30d, 1.0),
            out=burst,
            where=count_30d > 0,
        )
        rate_change = np.where(
            prior_180d > 0,
            count_30d / np.maximum(prior_180d / 6.0, 1e-9),
            count_30d,
        )

        latest_family = np.full(valid_query_ns.size, "", dtype=object)
        latest_family[has_event] = families[latest_idx[has_event]]
        values = {
            "event_data_available": has_event.astype(float),
            "event_days_since_latest": np.maximum(age_days, 0.0),
            "event_count_5d": count_5d,
            "event_count_30d": count_30d,
            "event_count_90d": count_90d,
            "event_current_report_5d": current_5d,
            "event_current_report_30d": current_30d,
            "event_financial_report_180d": financial_180d,
            "event_ownership_180d": ownership_180d,
            "event_proxy_180d": proxy_180d,
            "event_burst_5d_vs_30d": burst,
            "event_rate_change_30d_vs_prev180d": rate_change,
            "event_latest_is_current_report": (latest_family == "current_report").astype(float),
            "event_latest_is_financial_report": (latest_family == "financial_report").astype(float),
            "event_latest_is_ownership": (latest_family == "ownership").astype(float),
            "event_latest_is_proxy": (latest_family == "proxy").astype(float),
        }

        valid_positions = np.flatnonzero(valid_q)
        target_idx = query.index.to_numpy()[valid_positions]
        order = np.argsort(target_idx)
        target_idx = target_idx[order]
        for col, value in values.items():
            out.loc[target_idx, col] = np.asarray(value)[order]

    for col in EVENT_INTELLIGENCE_FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out
