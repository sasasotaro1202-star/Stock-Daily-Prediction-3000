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

    # A one-day forward return is only observable when the *next* session
    # close becomes available. Carry that next-session availability timestamp
    # with the forward return so the PIT test does not reject valid outcomes.
    b["outcome_available_at"] = (
        b.groupby(series_keys)["available_at"].shift(-1)
    )
    outcome_cols = series_keys + [
        "session_date",
        "forward_return_1d",
        "outcome_available_at",
    ]
    outcome = b[outcome_cols]

    m = p.merge(
        outcome,
        on=series_keys + ["session_date"],
        how="inner",
        suffixes=("_prediction", "_outcome"),
    )
    if "rank_score" not in m.columns:
        rank_groups = ["prediction_date"] + (
            ["asset_class"] if asset_aware else []
        )
        m["rank_score"] = m.groupby(rank_groups)["expected_return_1d"].rank(
            method="average", ascending=False, pct=True
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

        q = g["rank_score"].rank(pct=True)
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
                "gross_long_short_return": float(long - short),
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

    cost_grid = (0.0, 5.0, 10.0, 20.0)
    sensitivity = {}
    gross_by_day = r.groupby("date", sort=True)["gross_long_short_return"].mean()
    for grid_cost in cost_grid:
        net_by_day = gross_by_day - (2.0 * grid_cost / 10000.0)
        net_equity = (1.0 + net_by_day.fillna(0.0)).cumprod()
        net_peak = net_equity.cummax()
        net_dd = net_equity / net_peak - 1.0
        sensitivity[str(int(grid_cost))] = {
            "cost_bps_per_side": float(grid_cost),
            "mean_daily_spread": float(net_by_day.mean()),
            "cumulative_spread": float(net_equity.iloc[-1] - 1.0),
            "max_drawdown": float(net_dd.min()),
            "positive_days": float((net_by_day > 0).mean()),
        }

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
        "cost_sensitivity": sensitivity,
        "_daily_rows": r.to_dict(orient="records"),
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
