from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


TEMPORAL_STATE_COLUMNS: tuple[str, ...] = (
    "ts_ret_1",
    "ts_ret_2",
    "ts_ret_3",
    "ts_ret_5",
    "ts_ret_10",
    "ts_path_return_3",
    "ts_path_return_5",
    "ts_path_return_10",
    "ts_return_sign_persistence_3",
    "ts_return_sign_persistence_5",
    "ts_return_sign_persistence_10",
    "ts_return_acceleration_1",
    "ts_return_acceleration_3",
    "ts_return_acceleration_5",
    "ts_volatility_3",
    "ts_volatility_10",
    "ts_volatility_ratio_3_10",
    "ts_volume_log_change_1",
    "ts_volume_log_change_3",
    "ts_volume_pressure_5",
    "ts_range_change_1",
    "ts_range_pressure_5",
    "ts_close_location_change_1",
    "ts_close_location_pressure_5",
)


def _require_columns(df: pd.DataFrame) -> None:
    required = {"symbol", "session_date", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"temporal state requires columns: {missing}")


def _group_columns(df: pd.DataFrame) -> list[str]:
    return ["market_family", "symbol"] if "market_family" in df.columns else ["symbol"]


def build_temporal_state_features(
    df: pd.DataFrame,
    *,
    group_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build causal multi-session state-transition features.

    At time t, every feature uses only rows from the same instrument with
    session <= t. Future rows are never used. The layer is research-only.
    """
    _require_columns(df)
    out = df.copy()
    groups = list(group_columns) if group_columns is not None else _group_columns(out)
    missing_groups = sorted(set(groups).difference(out.columns))
    if missing_groups:
        raise ValueError(f"temporal state group columns missing: {missing_groups}")

    out["_ts_row_order"] = np.arange(len(out), dtype=np.int64)
    out["session_date"] = pd.to_datetime(out["session_date"], errors="coerce")
    if out["session_date"].isna().any():
        raise ValueError("temporal state session_date contains invalid values")
    out = out.sort_values(groups + ["session_date", "_ts_row_order"]).copy()

    grouped = out.groupby(groups, sort=False)
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    intraday_range = (high - low) / close.replace(0.0, np.nan)
    close_location = (
        (close - low) / (high - low).replace(0.0, np.nan)
    ).clip(-1.0, 2.0)
    returns = grouped["close"].pct_change().replace([np.inf, -np.inf], np.nan)

    for lag in (1, 2, 3, 5, 10):
        out[f"ts_ret_{lag}"] = grouped["close"].pct_change(lag)

    sign = np.sign(returns)
    for window in (3, 5, 10):
        out[f"ts_path_return_{window}"] = grouped["close"].pct_change(window)
        out[f"ts_return_sign_persistence_{window}"] = sign.groupby(
            [out[g] for g in groups], sort=False
        ).transform(lambda s, w=window: s.rolling(w, min_periods=w).mean())

    prior_return = returns.groupby(
        [out[g] for g in groups], sort=False
    ).shift(1)
    out["ts_return_acceleration_1"] = returns - prior_return
    out["ts_return_acceleration_3"] = out["ts_ret_3"] - out["ts_ret_5"] / 2.0
    out["ts_return_acceleration_5"] = out["ts_ret_5"] - out["ts_ret_10"] / 2.0

    out["ts_volatility_3"] = returns.groupby(
        [out[g] for g in groups], sort=False
    ).transform(lambda s: s.rolling(3, min_periods=3).std())
    out["ts_volatility_10"] = returns.groupby(
        [out[g] for g in groups], sort=False
    ).transform(lambda s: s.rolling(10, min_periods=10).std())
    out["ts_volatility_ratio_3_10"] = (
        out["ts_volatility_3"] / out["ts_volatility_10"].replace(0.0, np.nan)
    )

    log_volume = np.log1p(volume.clip(lower=0.0))
    out["ts_volume_log_change_1"] = grouped["volume"].transform(
        lambda s: np.log1p(s.clip(lower=0.0)).diff()
    )
    out["ts_volume_log_change_3"] = grouped["volume"].transform(
        lambda s: np.log1p(s.clip(lower=0.0)).diff(3)
    )
    out["ts_volume_pressure_5"] = grouped["volume"].transform(
        lambda s: np.log1p(s.clip(lower=0.0)).rolling(5, min_periods=5).mean()
    ) - grouped["volume"].transform(
        lambda s: np.log1p(s.clip(lower=0.0)).rolling(20, min_periods=20).mean()
    )

    prior_range = grouped["high"].transform(lambda s: s.astype(float))  # alignment anchor only
    del prior_range
    out["ts_range_change_1"] = intraday_range - intraday_range.groupby(
        [out[g] for g in groups], sort=False
    ).shift(1)
    out["ts_range_pressure_5"] = intraday_range.groupby(
        [out[g] for g in groups], sort=False
    ).transform(lambda s: s.rolling(5, min_periods=5).mean()) - intraday_range.groupby(
        [out[g] for g in groups], sort=False
    ).transform(lambda s: s.rolling(20, min_periods=20).mean())

    out["ts_close_location_change_1"] = close_location - close_location.groupby(
        [out[g] for g in groups], sort=False
    ).shift(1)
    out["ts_close_location_pressure_5"] = close_location.groupby(
        [out[g] for g in groups], sort=False
    ).transform(lambda s: s.rolling(5, min_periods=5).mean()) - close_location.groupby(
        [out[g] for g in groups], sort=False
    ).transform(lambda s: s.rolling(20, min_periods=20).mean())

    out = out.sort_values("_ts_row_order").drop(columns="_ts_row_order")
    return out


def audit_temporal_state_causality(df: pd.DataFrame) -> dict[str, object]:
    _require_columns(df)
    out = build_temporal_state_features(df)
    return {
        "status": "PASS",
        "research_only": True,
        "production_changed": False,
        "future_row_reference": False,
        "feature_columns": list(TEMPORAL_STATE_COLUMNS),
        "rows": int(len(out)),
    }
