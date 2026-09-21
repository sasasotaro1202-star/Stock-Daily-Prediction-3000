from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.features.context import add_cross_sectional_context, add_market_context
from src.features.technical import FEATURE_COLUMNS, add_technical_features
from src.prediction.regression import make_return_model
from src.prediction.targets import add_targets
from src.ranking.cross_sectional import cross_sectional_rank
from src.research.router import Regime, regime_for_row, route_plan
from src.validation.calibration import PlattCalibrator

PRICE = Path("data/prices")
METRICS = Path("data/research/latest_metrics.json")
GATE = Path("data/research/release_gate.json")
FROZEN = Path("config/frozen_holdout.json")
OUT = Path("data/predictions/latest.parquet")


def models():
    return {
        "logistic": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            LogisticRegression(max_iter=1000, C=0.5),
        ),
        "extra_trees": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        "hgb": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.04,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    }


def production_eligible(df: pd.DataFrame) -> pd.DataFrame:
    if not FROZEN.exists():
        return df
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        return df

    dates = pd.to_datetime(df["session_date"], errors="coerce").dt.date
    cutoff = pd.Timestamp(frozen["cutoff_date"]).date()
    holdout_end = pd.Timestamp(frozen["holdout_end"]).date()
    # The acceptance holdout is immutable. Fresh observations after that
    # one-time holdout may be used for production updates.
    keep = (dates <= cutoff) | (dates > holdout_end)
    return df.loc[keep].copy()


def split_train_cal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(df["session_date"]).dt.date.unique())
    if len(dates) < 40:
        raise ValueError("insufficient chronological dates for calibration split")
    cal_n = max(20, int(len(dates) * 0.2))
    if len(dates) - cal_n < 20:
        raise ValueError("insufficient core dates after calibration split")
    core = df[df.session_date.isin(set(dates[:-cal_n]))]
    cal = df[df.session_date.isin(set(dates[-cal_n:]))]
    if len(core) < 200 or len(cal) < 100:
        raise ValueError("insufficient rows for stable production fit")
    if core.target_up_1d.nunique() < 2 or cal.target_up_1d.nunique() < 2:
        raise ValueError("production calibration split lacks both target classes")
    return core, cal


def regime_series(df: pd.DataFrame, threshold: float) -> pd.Series:
    return df.apply(
        lambda x: regime_for_row(
            float(x["volatility_20"]),
            float(x["price_vs_sma60"]),
            threshold,
        ).value,
        axis=1,
    )


def fit_scoped_model(
    name: str,
    scope: str,
    labeled: pd.DataFrame,
    threshold: float,
    cache: dict[tuple[str, str], tuple[object, PlattCalibrator, str]],
):
    key = (name, scope)
    if key in cache:
        return cache[key]

    subset = labeled
    actual_scope = scope

    if scope.startswith("asset:"):
        asset = scope.split(":", 1)[1]
        subset = labeled[labeled["asset_class"].eq(asset)]
    elif "::" in scope and not scope.startswith("regime:"):
        asset, reg = scope.split("::", 1)
        subset = labeled[labeled["asset_class"].eq(asset)].copy()
        subset = subset[regime_series(subset, threshold).eq(reg)]
    elif scope.startswith("regime:"):
        reg = scope.split(":", 1)[1]
        subset = labeled[regime_series(labeled, threshold).eq(reg)]

    if len(subset) < 500 or subset.target_up_1d.nunique() < 2:
        subset = labeled
        actual_scope = "global"

    try:
        core, cal = split_train_cal(subset)
    except ValueError:
        subset = labeled
        actual_scope = "global"
        core, cal = split_train_cal(subset)

    model = models()[name]()
    model.fit(core[FEATURE_COLUMNS], core.target_up_1d.astype(int))
    cal_p = model.predict_proba(cal[FEATURE_COLUMNS])[:, 1]
    calibrator = PlattCalibrator().fit(
        cal_p, cal.target_up_1d.astype(int)
    )
    cache[key] = (model, calibrator, actual_scope)
    return cache[key]


def main():
    if not PRICE.exists() or not METRICS.exists() or not GATE.exists():
        raise SystemExit("DEFERRED: research/release artifacts are missing")

    gate = json.loads(GATE.read_text(encoding="utf-8"))
    if not gate.get("approved", False):
        raise SystemExit(
            f"DEFERRED: release gate not approved: {gate.get('reasons', [])}"
        )

    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    df = pd.read_parquet(PRICE)
    context_path = Path("data/market_context.parquet")
    if not context_path.exists():
        raise SystemExit("DEFERRED: market context is absent")
    market_context = pd.read_parquet(context_path)
    asset_filter = [
        x.strip()
        for x in os.environ.get("PREDICT_ASSET_CLASSES", "").split(",")
        if x.strip()
    ]
    df["available_at"] = pd.to_datetime(
        df["available_at"], utc=True, errors="coerce"
    )
    prediction_time = pd.Timestamp(datetime.now(timezone.utc))

    # End-to-end PIT gate for the production snapshot.
    df = df[df["available_at"].le(prediction_time)].copy()
    if df.empty:
        raise SystemExit("DEFERRED: no price rows satisfy available_at <= prediction_time")

    df = add_technical_features(df)
    df = add_market_context(df, market_context)
    df = add_cross_sectional_context(df)
    labeled = add_targets(df).dropna(
        subset=FEATURE_COLUMNS + ["target_up_1d", "target_ret_1d"]
    ).copy()
    labeled = production_eligible(labeled)

    if len(labeled) < 5000:
        raise SystemExit("DEFERRED: insufficient PIT-safe production training data")

    # Pick the latest actually-known session independently for each market family.
    latest = (
        df.sort_values(["asset_class", "session_date"])
        .groupby("asset_class", group_keys=False)
        .tail(1)
        .copy()
    )
    if asset_filter:
        latest = latest[latest["asset_class"].isin(asset_filter)].copy()
    if latest.empty:
        raise SystemExit("DEFERRED: no latest PIT-safe session rows")

    threshold = (
        float(labeled["volatility_20"].dropna().quantile(0.75))
        if labeled["volatility_20"].notna().any()
        else 0.02
    )

    metric_payload = payload
    frozen_routes = {}
    if FROZEN.exists():
        frozen_routes = json.loads(FROZEN.read_text(encoding="utf-8"))
    locked_mode = frozen_routes.get("status") == "FROZEN"
    regime_metrics = metric_payload.get("regime_metrics", {})
    asset_metrics = metric_payload.get("asset_class_metrics", {})
    asset_regime_metrics = metric_payload.get("asset_regime_metrics", {})
    global_selected = metric_payload.get("selected_model", "hgb")

    # Always retain three global challenger probabilities as a lightweight
    # uncertainty signal, while routing the production probability through
    # the OOS-selected scoped model.
    global_cache: dict[tuple[str, str], tuple[object, PlattCalibrator, str]] = {}
    for name in models():
        fit_scoped_model(name, "global", labeled, threshold, global_cache)

    route_cache: dict[tuple[str, str], tuple[object, PlattCalibrator, str]] = {}
    selected_names: list[str] = []
    selected_scopes: list[str] = []
    selected_reasons: list[str] = []
    p_values: list[float] = []
    global_disagreement: list[float] = []

    ready_mask = latest[FEATURE_COLUMNS].notna().all(axis=1)
    latest["prediction_status"] = np.where(
        ready_mask,
        "READY",
        "DEFERRED_INCOMPLETE_FEATURES",
    )

    for _, row in latest.iterrows():
        if not bool(ready_mask.loc[row.name]):
            p_values.append(float("nan"))
            selected_names.append("")
            selected_scopes.append("")
            selected_reasons.append("deferred:incomplete_features")
            global_disagreement.append(float("nan"))
            continue

        asset = str(row["asset_class"])
        regime = regime_for_row(
            float(row["volatility_20"]),
            float(row["price_vs_sma60"]),
            threshold,
            float(row["gap_pct"]) if pd.notna(row["gap_pct"]) else None,
            float(row["volume_ratio_20"]) if pd.notna(row["volume_ratio_20"]) else None,
        ).value
        plan = route_plan(
            asset,
            regime,
            asset_regime_metrics=asset_regime_metrics,
            asset_metrics=asset_metrics,
            regime_metrics=regime_metrics,
            global_selected=global_selected,
            locked_asset_regime=frozen_routes.get("asset_regime_selected_models") if locked_mode else None,
            locked_asset=frozen_routes.get("asset_class_selected_models") if locked_mode else None,
            locked_regime=frozen_routes.get("regime_selected_models") if locked_mode else None,
            locked_global=frozen_routes.get("selected_model") if locked_mode else None,
        )
        model_name = plan.names[0]
        model, calibrator, actual_scope = fit_scoped_model(
            model_name, plan.scope, labeled, threshold, route_cache
        )
        raw_p = model.predict_proba(
            pd.DataFrame([row])[FEATURE_COLUMNS]
        )[:, 1]
        p_values.append(float(np.clip(calibrator.predict(raw_p)[0], 1e-5, 1 - 1e-5)))
        selected_names.append(model_name)
        selected_scopes.append(actual_scope)
        selected_reasons.append(plan.reason)

        g_probs = []
        one = pd.DataFrame([row])[FEATURE_COLUMNS]
        for candidate in models():
            gm, gc, _ = global_cache[(candidate, "global")]
            g_probs.append(float(gc.predict(gm.predict_proba(one)[:, 1])[0]))
        global_disagreement.append(float(np.std(g_probs)))

    latest["p_up_1d"] = p_values
    latest["model_id"] = selected_names
    latest["training_scope"] = selected_scopes
    latest["route_reason"] = selected_reasons
    latest["model_disagreement"] = global_disagreement

    # Return models are product-family specific when sufficient data exists;
    # otherwise fall back to the global return model.
    global_ret = make_return_model()
    global_ret.fit(labeled[FEATURE_COLUMNS], labeled["target_ret_1d"])
    latest_returns = []
    return_scope = []
    for asset, group in latest[ready_mask].groupby("asset_class", sort=False):
        subset = labeled[labeled["asset_class"].eq(asset)]
        model = global_ret
        scope = "global"
        if len(subset) >= 750:
            scoped = make_return_model()
            scoped.fit(subset[FEATURE_COLUMNS], subset["target_ret_1d"])
            model = scoped
            scope = f"asset:{asset}"
        pred = model.predict(group[FEATURE_COLUMNS])
        latest_returns.extend(zip(group.index, pred))
        return_scope.extend(zip(group.index, [scope] * len(group)))

    ret_by_index = {idx: value for idx, value in latest_returns}
    scope_by_index = {idx: value for idx, value in return_scope}
    latest["expected_return_1d"] = [
        float(ret_by_index.get(idx, np.nan)) for idx in latest.index
    ]
    latest["return_training_scope"] = [
        scope_by_index.get(idx, "") for idx in latest.index
    ]

    vol = latest["volatility_20"].clip(lower=0.0)
    latest["expected_close_1d"] = np.where(
        ready_mask,
        latest["close"] * (1 + latest["expected_return_1d"]),
        np.nan,
    )
    latest["range_low_1d"] = np.where(
        ready_mask,
        latest["close"] * np.exp(-1.96 * vol),
        np.nan,
    )
    latest["range_high_1d"] = np.where(
        ready_mask,
        latest["close"] * np.exp(1.96 * vol),
        np.nan,
    )
    latest["prediction_time"] = prediction_time
    latest["prediction_date"] = prediction_time.tz_convert("Asia/Tokyo").date()

    cols = [
        "symbol",
        "asset_class",
        "session_date",
        "close",
        "prediction_time",
        "prediction_date",
        "p_up_1d",
        "expected_return_1d",
        "expected_close_1d",
        "range_low_1d",
        "range_high_1d",
        "model_id",
        "training_scope",
        "return_training_scope",
        "route_reason",
        "model_disagreement",
        "prediction_status",
    ]

    out = cross_sectional_rank(latest[cols])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    stamp = prediction_time.strftime("%Y%m%dT%H%M%SZ")
    out.to_parquet(OUT.parent / f"prediction_{stamp}.parquet", index=False)
    print(
        f"prediction_rows={len(out)} "
        f"asset_classes={sorted(out['asset_class'].unique().tolist())}"
    )


if __name__ == "__main__":
    main()
