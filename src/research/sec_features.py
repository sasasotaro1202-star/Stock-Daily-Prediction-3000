from __future__ import annotations

import numpy as np
import pandas as pd

SEC_FEATURE_COLUMNS = (
    "sec_data_available",
    "sec_days_since_filing",
    "sec_filings_30d",
    "sec_filings_90d",
    "sec_8k_30d",
    "sec_10q_180d",
    "sec_10k_365d",
    "sec_proxy_90d",
)


def _counts_before(
    event_ns: np.ndarray,
    event_is_8k: np.ndarray,
    event_is_10q: np.ndarray,
    event_is_10k: np.ndarray,
    event_is_proxy: np.ndarray,
    query_ns: np.ndarray,
    window_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if event_ns.size == 0:
        z = np.zeros(query_ns.size, dtype=float)
        return z, z.copy(), z.copy(), z.copy(), z.copy()
    idx_now = np.searchsorted(event_ns, query_ns, side="right")
    idx_old = np.searchsorted(
        event_ns,
        query_ns - int(window_days) * 86_400_000_000_000,
        side="left",
    )
    cumulative = np.arange(event_ns.size, dtype=float) + 1.0
    total = cumulative[np.maximum(idx_now - 1, 0)]
    total[idx_now == 0] = 0.0

    def window_count(flags: np.ndarray) -> np.ndarray:
        c = np.cumsum(flags.astype(float))
        now = c[np.maximum(idx_now - 1, 0)]
        now[idx_now == 0] = 0.0
        old = c[np.maximum(idx_old - 1, 0)]
        old[idx_old == 0] = 0.0
        return now - old

    return (
        total - np.where(
            idx_old > 0,
            cumulative[np.maximum(idx_old - 1, 0)],
            0.0,
        ),
        window_count(event_is_8k),
        window_count(event_is_10q),
        window_count(event_is_10k),
        window_count(event_is_proxy),
    )


def add_sec_filing_features(
    prices: pd.DataFrame,
    filings: pd.DataFrame,
) -> pd.DataFrame:
    """Attach PIT-safe SEC event features to US stock price observations.

    Every filing is admitted only when its SEC acceptance/available timestamp
    is <= the price observation's available_at. Missing SEC history is kept
    explicit through sec_data_available=0 rather than treated as observed
    absence.
    """
    out = prices.copy()
    for col in SEC_FEATURE_COLUMNS:
        out[col] = 0.0

    if filings is None or filings.empty:
        return out

    required = {"symbol", "asset_class", "available_at", "form"}
    if not required.issubset(filings.columns):
        return out

    out["available_at"] = pd.to_datetime(
        out["available_at"], utc=True, errors="coerce"
    )
    sec = filings.copy()
    sec["available_at"] = pd.to_datetime(
        sec["available_at"], utc=True, errors="coerce"
    )
    sec = sec[
        sec["asset_class"].astype(str).eq("us_stock")
        & sec["available_at"].notna()
    ].copy()
    if sec.empty:
        return out

    sec["symbol"] = sec["symbol"].astype(str)
    sec["form"] = sec["form"].astype(str).str.upper()
    sec = sec.sort_values(["symbol", "available_at"]).reset_index(drop=True)

    us_mask = out["asset_class"].astype(str).eq("us_stock")
    for symbol, idx in out.loc[us_mask].groupby(out.loc[us_mask, "symbol"].astype(str)).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        query = out.loc[idx].sort_values("available_at")
        valid_q = query["available_at"].notna().to_numpy()
        if not valid_q.any():
            continue

        events = sec[sec["symbol"].eq(symbol)].sort_values("available_at")
        if events.empty:
            continue

        event_ns = events["available_at"].astype("int64").to_numpy()
        query_ns = query["available_at"].astype("int64").to_numpy()
        forms = events["form"].to_numpy()
        is_8k = np.isin(forms, ["8-K", "6-K"])
        is_10q = np.isin(forms, ["10-Q"])
        is_10k = np.isin(forms, ["10-K", "20-F", "40-F"])
        is_proxy = np.isin(forms, ["DEF 14A"])

        valid_query_ns = query_ns[valid_q]
        pos_now = np.searchsorted(event_ns, valid_query_ns, side="right")
        has_prev = pos_now > 0
        latest_idx = np.maximum(pos_now - 1, 0)
        age_days = np.full(valid_query_ns.size, np.nan, dtype=float)
        age_days[has_prev] = (
            valid_query_ns[has_prev] - event_ns[latest_idx[has_prev]]
        ) / 86_400_000_000_000.0

        total_30, k_30, q_180, k_365, proxy_90 = _counts_before(
            event_ns,
            is_8k,
            is_10q,
            is_10k,
            is_proxy,
            valid_query_ns,
            30,
        )

        # _counts_before returns total event count for the selected window;
        # obtain the requested longer-window totals separately.
        _, _, q_180b, _, _ = _counts_before(
            event_ns, is_8k, is_10q, is_10k, is_proxy, valid_query_ns, 180
        )
        _, _, _, k_365b, _ = _counts_before(
            event_ns, is_8k, is_10q, is_10k, is_10k, valid_query_ns, 365
        )
        _, _, _, _, proxy_90b = _counts_before(
            event_ns, is_8k, is_10q, is_10k, is_proxy, valid_query_ns, 90
        )

        # Correct the total to a 90/30-day pair used by the public feature schema.
        total_90, _, _, _, _ = _counts_before(
            event_ns, is_8k, is_10q, is_10k, is_proxy, valid_query_ns, 90
        )

        valid_positions = np.flatnonzero(valid_q)
        # query retains the original out index labels, even after sorting by
        # available_at. Preserve those labels explicitly so a non-monotone
        # input row order can never redirect SEC features to another security.
        target_idx = query.index.to_numpy()[valid_positions]
        order = np.argsort(target_idx)
        target_idx = target_idx[order]

        out.loc[target_idx, "sec_data_available"] = (
            has_prev[order].astype(float)
        )
        out.loc[target_idx, "sec_days_since_filing"] = (
            np.maximum(age_days[order], 0.0)
        )
        out.loc[target_idx, "sec_filings_30d"] = total_30[order]
        out.loc[target_idx, "sec_filings_90d"] = total_90[order]
        out.loc[target_idx, "sec_8k_30d"] = k_30[order]
        out.loc[target_idx, "sec_10q_180d"] = q_180b[order]
        out.loc[target_idx, "sec_10k_365d"] = k_365b[order]
        out.loc[target_idx, "sec_proxy_90d"] = proxy_90b[order]

    # Keep explicit neutral values for non-US assets and missing SEC history.
    out["sec_days_since_filing"] = out["sec_days_since_filing"].replace(
        [np.inf, -np.inf], np.nan
    )
    out["sec_days_since_filing"] = out["sec_days_since_filing"].fillna(0.0)
    for col in SEC_FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out
