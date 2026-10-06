from __future__ import annotations

import numpy as np
import pandas as pd


BOJ_FEATURE_COLUMNS = [
    "boj_call_rate_level",
    "boj_call_rate_high",
    "boj_call_rate_low",
    "boj_call_rate_change",
    "boj_call_rate_range",
    "boj_call_rate_level_z20",
]


def add_boj_context(
    df: pd.DataFrame,
    boj: pd.DataFrame,
    *,
    min_history: int = 20,
) -> pd.DataFrame:
    out = df.copy()
    required_price = {"available_at"}
    required_boj = {
        "available_at",
        "series_code",
        "observation_date",
        "value",
        "source",
    }
    if not required_price.issubset(out.columns):
        raise ValueError("price frame missing available_at")
    if boj.empty or not required_boj.issubset(boj.columns):
        raise ValueError("BOJ frame missing required columns")

    price = out.copy()
    price["available_at"] = pd.to_datetime(
        price["available_at"], utc=True, errors="coerce"
    )
    price = price.sort_values("available_at")

    macro = boj.copy()
    macro["available_at"] = pd.to_datetime(
        macro["available_at"], utc=True, errors="coerce"
    )
    macro["observation_date"] = pd.to_datetime(
        macro["observation_date"], errors="coerce"
    ).dt.date
    macro["value"] = pd.to_numeric(macro["value"], errors="coerce")
    macro = macro.dropna(
        subset=["available_at", "observation_date", "value"]
    )
    if macro["source"].astype(str).ne("boj_timeseries").any():
        raise ValueError("FAIL: non-BOJ rows supplied to BOJ feature layer")

    piv = (
        macro.pivot_table(
            index=["available_at", "observation_date"],
            columns="series_code",
            values="value",
            aggfunc="last",
        )
        .sort_index()
    )
    required_codes = {"STRDCLUCON", "STRDCLUCONH", "STRDCLUCONL"}
    missing_codes = required_codes - set(piv.columns)
    if missing_codes:
        raise ValueError(
            "FAIL: BOJ feature layer missing series: "
            + ",".join(sorted(missing_codes))
        )

    series = pd.DataFrame(index=piv.index)
    series["boj_call_rate_level"] = piv["STRDCLUCON"]
    series["boj_call_rate_high"] = piv["STRDCLUCONH"]
    series["boj_call_rate_low"] = piv["STRDCLUCONL"]
    series = series.sort_values("observation_date")
    series["boj_call_rate_change"] = series["boj_call_rate_level"].diff()
    series["boj_call_rate_range"] = (
        series["boj_call_rate_high"] - series["boj_call_rate_low"]
    )
    # The z-score is computed only on the BOJ-observation history. Because
    # merge_asof below is time-causal, the resulting feature is PIT-safe.
    level = series["boj_call_rate_level"]
    rolling_mean = level.rolling(min_history, min_periods=min_history).mean()
    rolling_std = level.rolling(min_history, min_periods=min_history).std()
    series["boj_call_rate_level_z20"] = (
        (level - rolling_mean) / rolling_std.replace(0, np.nan)
    )
    series = series.reset_index()
    series["available_at"] = pd.to_datetime(
        series["available_at"], utc=True, errors="coerce"
    )

    merged = pd.merge_asof(
        price.sort_values("available_at"),
        series[
            ["available_at"] + BOJ_FEATURE_COLUMNS
        ].sort_values("available_at"),
        on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    # All rows here are sourced from a strictly earlier/equal BOJ availability
    # timestamp. Preserve the source timestamp for independent auditability.
    return merged.sort_index()
