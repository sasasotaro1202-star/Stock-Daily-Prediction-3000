from __future__ import annotations

import pandas as pd

TREASURY_FEATURE_COLUMNS = [
    "treasury_3m", "treasury_6m", "treasury_1y", "treasury_2y",
    "treasury_5y", "treasury_10y", "treasury_30y",
    "treasury_2s10s", "treasury_3m10y", "treasury_curvature_2y5y10y",
    "treasury_2s10s_chg1d", "treasury_3m10y_chg1d", "treasury_10y_chg1d",
    "treasury_2y_chg1d", "treasury_5y_chg1d", "treasury_10y_chg5d",
]

def add_treasury_context(df: pd.DataFrame, treasury: pd.DataFrame) -> pd.DataFrame:
    if "available_at" not in df.columns:
        raise ValueError("price frame missing available_at")
    if treasury.empty or not {"available_at", "session_date", "source"} <= set(treasury.columns):
        raise ValueError("Treasury frame missing required columns")
    price = df.copy()
    # merge_asof sorts by the time key; retain and restore the caller's row order
    # so feature enrichment never silently reorders the prediction dataset.
    price["__treasury_input_order"] = range(len(price))
    price["available_at"] = pd.to_datetime(price["available_at"], utc=True, errors="coerce")
    curve = treasury.copy()
    curve["available_at"] = pd.to_datetime(curve["available_at"], utc=True, errors="coerce")
    curve = curve.dropna(subset=["available_at"]).sort_values("available_at")
    columns = ["available_at"] + [c for c in TREASURY_FEATURE_COLUMNS if c in curve.columns]
    merged = pd.merge_asof(
        price.sort_values("available_at", kind="stable"),
        curve[columns],
        on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.sort_values("__treasury_input_order", kind="stable")
    merged = merged.drop(columns="__treasury_input_order")
    merged.index = df.index
    return merged
