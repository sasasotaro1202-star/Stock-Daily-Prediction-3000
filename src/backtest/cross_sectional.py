from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_predictions(
    predictions: pd.DataFrame,
    bars: pd.DataFrame,
    top_quantile: float = 0.1,
    cost_bps: float = 5.0,
) -> dict:
    p = predictions.copy()
    b = bars.copy()

    required_pred = {
        "symbol",
        "session_date",
        "prediction_date",
        "expected_return_1d",
    }
    required_bar = {"symbol", "session_date", "close", "available_at"}
    if not required_pred.issubset(p.columns):
        return {
            "status": "DEFERRED",
            "rows": 0,
            "reason": "prediction_schema_missing",
        }
    if not required_bar.issubset(b.columns):
        return {
            "status": "DEFERRED",
            "rows": 0,
            "reason": "bar_schema_missing",
        }

    pred_has_asset = "asset_class" in p.columns
    bar_has_asset = "asset_class" in b.columns
    if pred_has_asset != bar_has_asset:
        return {
            "status": "DEFERRED",
            "rows": 0,
            "reason": "asset_class_schema_mismatch",
        }
    asset_aware = pred_has_asset and bar_has_asset

    p["session_date"] = pd.to_datetime(
        p["session_date"], errors="coerce"
    ).dt.date
    p["prediction_date"] = pd.to_datetime(
        p["prediction_date"], errors="coerce"
    ).dt.date
    b["session_date"] = pd.to_datetime(
        b["session_date"], errors="coerce"
    ).dt.date

    series_keys = ["asset_class", "symbol"] if asset_aware else ["symbol"]
    b = b.sort_values(series_keys + ["session_date"])

    # The prediction is made from the last known session bar. Joining on that
    # bar keeps the backtest correct for Japan/U.S. market-clock differences.
    b["forward_return_1d"] = (
        b.groupby(series_keys)["close"].shift(-1) / b["close"] - 1.0
    )
    if "stock_splits" in b.columns:
        current_split = b["stock_splits"].fillna(0).ne(0)
        next_split = (
            b.groupby(series_keys)["stock_splits"].shift(-1).fillna(0).ne(0)
        )
        b.loc[current_split | next_split, "forward_return_1d"] = np.nan

    outcome_cols = series_keys + [
        "session_date",
        "forward_return_1d",
        "available_at",
    ]
    outcome = b[outcome_cols].rename(
        columns={"available_at": "outcome_available_at"}
    )

    m = p.merge(
        outcome,
        on=series_keys + ["session_date"],
        how="inner",
        suffixes=("_prediction", "_outcome"),
    )
    if "prediction_time" in m:
        m["prediction_time"] = pd.to_datetime(
            m["prediction_time"], utc=True, errors="coerce"
        )
        m["outcome_available_at"] = pd.to_datetime(
            m["outcome_available_at"], utc=True, errors="coerce"
        )
        m = m[m["outcome_available_at"].gt(m["prediction_time"])].copy()

    rows = []
    group_cols = ["prediction_date"] + (["asset_class"] if asset_aware else [])
    for group_key, g in m.groupby(group_cols, dropna=False):
        if asset_aware:
            date, asset_class = group_key
        else:
            date, asset_class = group_key, None
        g = g.dropna(subset=["expected_return_1d", "forward_return_1d"])
        if len(g) < 20:
            continue

        q = g["expected_return_1d"].rank(pct=True)
        long = g.loc[
            q >= 1 - top_quantile, "forward_return_1d"
        ].mean()
        short = g.loc[
            q <= top_quantile, "forward_return_1d"
        ].mean()
        spread = long - short - (2 * cost_bps / 10000)
        rows.append(
            {
                "date": str(date),
                "asset_class": str(asset_class) if asset_aware else None,
                "n": len(g),
                "long_return": float(long),
                "short_return": float(short),
                "long_short_return": float(spread),
            }
        )

    if not rows:
        return {
            "status": "DEFERRED",
            "rows": 0,
            "reason": "no_valid_outcomes",
        }

    r = pd.DataFrame(rows).sort_values(
        ["date"] + (["asset_class"] if asset_aware else [])
    )
    # Treat each market/asset class as an equal-capital sleeve, then average
    # simultaneous sleeves before compounding. This mirrors the production
    # ranking boundary and avoids fabricating cross-market capital allocation.
    daily = r.groupby("date", sort=True)["long_short_return"].mean()
    equity = (1 + daily.fillna(0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1

    result = {
        "status": "PASS",
        "rows": int(len(r)),
        "days": int(len(daily)),
        "asset_classes": (
            sorted(r["asset_class"].dropna().unique().tolist())
            if asset_aware
            else []
        ),
        "mean_daily_spread": float(daily.mean()),
        "vol_daily_spread": float(daily.std(ddof=0)),
        "cumulative_spread": float(equity.iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "cost_bps_per_side": float(cost_bps),
    }
    if asset_aware:
        result["asset_class_summary"] = {
            str(asset): {
                "group_days": int(group["date"].nunique()),
                "mean_spread": float(group["long_short_return"].mean()),
            }
            for asset, group in r.groupby("asset_class", dropna=False)
        }
    return result
