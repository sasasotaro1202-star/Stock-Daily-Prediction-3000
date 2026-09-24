from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.context import add_cross_sectional_context, add_market_context
from src.features.technical import FEATURE_COLUMNS, add_technical_features
from src.prediction.model_factories import models
from src.prediction.fit import fit_classifier
from src.prediction.regression import make_quantile_model, make_return_model
from src.prediction.targets import add_targets
from src.research.metrics import classification_metrics, cross_sectional_rank_ic
from src.ranking.cross_sectional import cross_sectional_rank
from src.research.router import route_plan, regime_for_row
from src.validation.calibration import make_calibrator
from src.validation.code_fingerprint import fingerprint_sha256
from src.validation.training_sample import cap_training_rows
from src.validation.training_window import restrict_to_lookback


def factories():
    return models()


def main():
    result = Path("data/research/frozen_holdout_result.json")
    if result.exists():
        print("frozen-holdout: already evaluated; no-op")
        return

    lock = Path("config/frozen_holdout.json")
    metrics = Path("data/research/latest_metrics.json")
    if not lock.exists() or not metrics.exists():
        raise SystemExit(
            "DEFERRED: frozen configuration and OOS metrics required"
        )

    frozen = json.loads(lock.read_text(encoding="utf-8"))
    selected = frozen["selected_model"]
    if selected not in factories():
        raise SystemExit("FAIL: frozen model is unavailable")

    df = pd.read_parquet("data/prices/canonical.parquet")
    context_path = Path("data/market_context.parquet")
    if not context_path.exists():
        raise SystemExit("DEFERRED: market context is absent")
    market_context = pd.read_parquet(context_path)
    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce").dt.date
    df["available_at"] = pd.to_datetime(
        df["available_at"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["available_at"])
    df = add_technical_features(df)
    df = add_market_context(df, market_context)
    df = add_targets(add_cross_sectional_context(df))
    cutoff = pd.Timestamp(frozen["cutoff_date"]).date()
    holdout_end = pd.Timestamp(frozen["holdout_end"]).date()
    df["date"] = pd.to_datetime(df["session_date"]).dt.date

    train = df[
        df["date"].le(cutoff)
    ].dropna(subset=FEATURE_COLUMNS + ["target_up_1d"])
    # The immutable holdout is exactly (cutoff, holdout_end], never the
    # entire post-cutoff history. Any rows after holdout_end are excluded so
    # future observations cannot silently enter one-time holdout evaluation.
    test = df[
        df["date"].gt(cutoff) & df["date"].le(holdout_end)
    ].dropna(subset=FEATURE_COLUMNS + ["target_up_1d", "target_ret_1d"])
    if not test.empty and max(test["date"]) > holdout_end:
        raise SystemExit("FAIL: frozen holdout contains rows after holdout_end")

    if len(train) < 1000 or len(test) < 500:
        raise SystemExit("DEFERRED: frozen holdout is too small")

    dates = sorted(train["date"].unique())
    cal_n = max(20, int(len(dates) * 0.2))
    core = train[train["date"].isin(set(dates[:-cal_n]))]
    cal = train[train["date"].isin(set(dates[-cal_n:]))]
    if core.target_up_1d.nunique() < 2 or cal.target_up_1d.nunique() < 2:
        raise SystemExit("DEFERRED: calibration split lacks both target classes")

    calibration_method = str(frozen.get("calibration_method", "platt"))
    if calibration_method not in {"platt", "beta", "isotonic"}:
        raise SystemExit("FAIL: frozen calibration method is invalid")

    training_window = int(frozen.get("classifier_training_window_sessions", 0))
    if training_window < 0:
        raise SystemExit("FAIL: frozen classifier training window is invalid")
    model_core = restrict_to_lookback(
        core,
        None if training_window == 0 else training_window,
    )
    model_core_dates = sorted(model_core["date"].unique())
    if len(model_core_dates) < 40:
        raise SystemExit("DEFERRED: frozen classifier training window is too small")
    model_cal_n = max(20, int(len(model_core_dates) * 0.2))
    model_fit = model_core[
        model_core["date"].isin(set(model_core_dates[:-model_cal_n]))
    ]
    model_cal = model_core[
        model_core["date"].isin(set(model_core_dates[-model_cal_n:]))
    ]
    if (
        model_fit.target_up_1d.nunique() < 2
        or model_cal.target_up_1d.nunique() < 2
    ):
        raise SystemExit("DEFERRED: frozen classifier window lacks both calibration classes")
    core_fit=cap_training_rows(model_fit,max_rows=300_000,recent_sessions=min(252, training_window or 252))

    # Fit every classifier actually referenced by the frozen routing policy,
    # while keeping the holdout completely untouched until final evaluation.
    required_classifiers = {"hgb", selected}
    for key in (
        "regime_selected_models",
        "asset_class_selected_models",
        "asset_regime_selected_models",
        "symbol_selected_models",
        "symbol_regime_selected_models",
    ):
        required_classifiers.update(
            str(value) for value in (frozen.get(key) or {}).values()
        )
    available_factories = factories()
    missing = sorted(set(required_classifiers) - set(available_factories))
    if missing:
        raise SystemExit(
            f"DEFERRED: frozen route classifiers unavailable: {missing}"
        )

    classifiers = {}
    for model_name in sorted(required_classifiers):
        model = available_factories[model_name]()
        fit_classifier(
            model,
            model_name,
            core_fit[FEATURE_COLUMNS],
            core_fit.target_up_1d.astype(int),
            core_fit["date"],
            half_life_sessions=int(
                __import__("yaml").safe_load(
                    Path("config/pipeline.yml").read_text(encoding="utf-8")
                ).get("models", {}).get("recency_weight_half_life_sessions", 252)
            ),
        )
        cal_p = model.predict_proba(model_cal[FEATURE_COLUMNS])[:, 1]
        calibrator = make_calibrator(calibration_method).fit(
            cal_p, model_cal.target_up_1d.astype(int)
        )
        classifiers[model_name] = {
            "model": model,
            "calibrator": calibrator,
        }

    selected_entry = classifiers[selected]
    p = selected_entry["calibrator"].predict(
        selected_entry["model"].predict_proba(test[FEATURE_COLUMNS])[:, 1]
    )
    global_model_metrics = classification_metrics(
        test.target_up_1d.astype(int), p
    )

    # Evaluate the exact frozen production routing policy on the immutable
    # holdout, rather than evaluating only the globally-selected classifier.
    vol_threshold = frozen.get("regime_vol_threshold")
    if not isinstance(vol_threshold, (int, float)) or not np.isfinite(float(vol_threshold)):
        raise SystemExit("FAIL: frozen regime volatility threshold is missing")
    vol_threshold = float(vol_threshold)
    frozen_routes = frozen
    routed_probabilities = []
    routed_models = []
    routed_regimes = []
    for _, row in test.iterrows():
        regime = regime_for_row(
            float(row["volatility_20"]) if pd.notna(row["volatility_20"]) else None,
            float(row["price_vs_sma60"]) if pd.notna(row["price_vs_sma60"]) else None,
            vol_threshold,
            float(row["gap_pct"]) if pd.notna(row["gap_pct"]) else None,
            float(row["volume_ratio_20"]) if pd.notna(row["volume_ratio_20"]) else None,
            float(row["vix_level_lag1"]) if pd.notna(row["vix_level_lag1"]) else None,
            float(row["breadth_up"]) if pd.notna(row["breadth_up"]) else None,
        ).value
        asset_class = str(row["asset_class"]) if "asset_class" in row and pd.notna(row["asset_class"]) else ""
        symbol_value = str(row["symbol"]) if "symbol" in row and pd.notna(row["symbol"]) else None
        plan = route_plan(
            asset_class,
            regime,
            symbol=symbol_value,
            locked_symbol_regime=frozen_routes.get("symbol_regime_selected_models"),
            locked_symbol=frozen_routes.get("symbol_selected_models"),
            locked_asset_regime=frozen_routes.get("asset_regime_selected_models"),
            locked_asset=frozen_routes.get("asset_class_selected_models"),
            locked_regime=frozen_routes.get("regime_selected_models"),
            locked_global=frozen_routes.get("selected_model"),
        )
        model_name = plan.names[0]
        entry = classifiers.get(model_name)
        if entry is None:
            raise SystemExit(
                f"FAIL: frozen route requires classifier absent from evaluator: {model_name}"
            )
        one = pd.DataFrame([row])[FEATURE_COLUMNS]
        raw = entry["model"].predict_proba(one)[:, 1]
        calibrated = entry["calibrator"].predict(raw)[0]
        routed_probabilities.append(float(np.clip(calibrated, 1e-5, 1 - 1e-5)))
        routed_models.append(model_name)
        routed_regimes.append(regime)

    routed_probabilities = np.asarray(routed_probabilities, dtype=float)
    routed_metrics = classification_metrics(
        test.target_up_1d.astype(int), routed_probabilities
    )
    routed_group_keys=(
        test["date"].astype(str)
        + "::"
        + test["asset_class"].astype(str)
        if "asset_class" in test.columns
        else test["date"].astype(str)
    )
    routed_metrics["rank_ic"]=cross_sectional_rank_ic(
        test["target_ret_1d"].astype(float),
        routed_probabilities,
        routed_group_keys,
    )
    route_usage = {
        name: int(sum(model == name for model in routed_models))
        for name in sorted(set(routed_models))
    }
    regime_usage = {
        name: int(sum(regime == name for regime in routed_regimes))
        for name in sorted(set(routed_regimes))
    }

    return_selected = frozen.get("return_selected_estimator", "q50")
    if return_selected not in {"mean", "q50", "blend_mean_q50"}:
        raise SystemExit("FAIL: frozen return estimator is invalid")

    mean_return_model = make_return_model()
    q_global = {
        "q10": make_quantile_model(0.10),
        "q50": make_quantile_model(0.50),
        "q90": make_quantile_model(0.90),
    }
    core_q=cap_training_rows(core,max_rows=250_000,recent_sessions=252)
    mean_return_model.fit(core_q[FEATURE_COLUMNS], core_q["target_ret_1d"])
    for qm in q_global.values():
        qm.fit(core_q[FEATURE_COLUMNS], core_q["target_ret_1d"])

    q_assets = {}
    for asset, subset in core.groupby("asset_class", sort=False):
        if len(subset) < 750:
            continue
        q_models = {
            "q10": make_quantile_model(0.10),
            "q50": make_quantile_model(0.50),
            "q90": make_quantile_model(0.90),
        }
        subset_fit=cap_training_rows(subset,max_rows=200_000,recent_sessions=252)
        for qm in q_models.values():
            qm.fit(subset_fit[FEATURE_COLUMNS], subset_fit["target_ret_1d"])
        q_assets[str(asset)] = q_models

    y_ret=test["target_ret_1d"].to_numpy(dtype=float)
    selected_pred_global_mean=mean_return_model.predict(test[FEATURE_COLUMNS])
    selected_pred_global_q50=q_global["q50"].predict(test[FEATURE_COLUMNS])
    if return_selected == "mean":
        selected_return_pred=selected_pred_global_mean
    elif return_selected == "q50":
        selected_return_pred=selected_pred_global_q50
    else:
        selected_return_pred=0.5*(selected_pred_global_mean+selected_pred_global_q50)

    lo=np.empty(len(test),dtype=float)
    hi=np.empty(len(test),dtype=float)
    for asset, subset in test.groupby("asset_class", sort=False):
        qmodels=q_assets.get(str(asset), q_global)
        idx=subset.index
        pos=test.index.get_indexer(idx)
        lo_vals=qmodels["q10"].predict(subset[FEATURE_COLUMNS])
        hi_vals=qmodels["q90"].predict(subset[FEATURE_COLUMNS])
        lo[pos]=lo_vals
        hi[pos]=hi_vals
    lo=np.minimum(lo,selected_return_pred)
    hi=np.maximum(hi,selected_return_pred)
    return_holdout_metrics={
        "selected_estimator": return_selected,
        "mae":float(__import__("sklearn.metrics",fromlist=["mean_absolute_error"]).mean_absolute_error(y_ret,selected_return_pred)),
        "rmse":float(__import__("sklearn.metrics",fromlist=["mean_squared_error"]).mean_squared_error(y_ret,selected_return_pred)**0.5),
        "sign_accuracy":float(np.mean((selected_return_pred>=0)==(y_ret>=0))),
        "rank_ic":float(cross_sectional_rank_ic(
            y_ret,
            selected_return_pred,
            (
                test["date"].astype(str)+"::"+test["asset_class"].astype(str)
                if "asset_class" in test.columns else test["date"].astype(str)
            ),
        )),
        "range_80_coverage":float(np.mean((y_ret>=lo)&(y_ret<=hi))),
    }

    rank_weight = frozen.get("rank_probability_weight", 0.50)
    if not isinstance(rank_weight, (int, float)) or not 0.0 <= float(rank_weight) <= 1.0:
        raise SystemExit("FAIL: frozen ranking probability weight is invalid")
    rank_uncertainty_penalty = frozen.get("rank_uncertainty_penalty", 0.0)
    if (
        not isinstance(rank_uncertainty_penalty, (int, float))
        or not 0.0 <= float(rank_uncertainty_penalty) <= 1.0
    ):
        raise SystemExit("FAIL: frozen ranking uncertainty penalty is invalid")

    ranking_frame = pd.DataFrame(
        {
            "prediction_date": test["date"].astype(str).to_numpy(),
            "asset_class": test["asset_class"].astype(str).to_numpy(),
            "p_up_1d": routed_probabilities,
            "expected_return_1d": selected_return_pred,
        },
        index=test.index,
    )
    ranking_frame["ranking_uncertainty"] = np.nan
    for asset, subset in test.groupby("asset_class", sort=False):
        qmodels = q_assets.get(str(asset), q_global)
        lo_vals = qmodels["q10"].predict(subset[FEATURE_COLUMNS])
        hi_vals = qmodels["q90"].predict(subset[FEATURE_COLUMNS])
        ranking_frame.loc[subset.index, "ranking_uncertainty"] = np.maximum(
            hi_vals - lo_vals,
            0.0,
        )

    ranked = cross_sectional_rank(
        ranking_frame,
        probability_col="p_up_1d",
        return_col="expected_return_1d",
        probability_weight=float(rank_weight),
        uncertainty_col="ranking_uncertainty",
        uncertainty_penalty=float(rank_uncertainty_penalty),
    )
    holdout_group_keys = (
        test["date"].astype(str)
        + "::"
        + test["asset_class"].astype(str)
        if "asset_class" in test.columns
        else test["date"].astype(str)
    )
    ranking_holdout = {
        "probability_weight": float(rank_weight),
        "uncertainty_penalty": float(rank_uncertainty_penalty),
        "rank_ic": float(
            cross_sectional_rank_ic(
                test["target_ret_1d"].astype(float),
                ranked["rank_score"].reindex(test.index).to_numpy(dtype=float),
                holdout_group_keys,
            )
        ),
    }

    base = float(core.target_up_1d.mean())
    baseline_metrics = classification_metrics(
        test.target_up_1d.astype(int),
        np.full(len(test), base),
    )

    max_ece = 0.20
    config = Path("config/pipeline.yml").read_text(encoding="utf-8")
    import re

    match = re.search(r"max_holdout_ece:\s*([0-9.]+)", config)
    if match:
        max_ece = float(match.group(1))

    payload = {
        "status": "EVALUATED_ONCE",
        "frozen_model": selected,
        "holdout_generation": int(frozen.get("holdout_generation", 1)),
        "code_fingerprint_sha256": fingerprint_sha256(),
        "research_code_fingerprint_sha256": frozen.get(
            "research_code_fingerprint_sha256"
        ),
        "holdout_start": frozen["holdout_start"],
        "holdout_end": frozen["holdout_end"],
        "holdout_rows": int(len(test)),
        "global_model_metrics": global_model_metrics,
        "production_route_metrics": routed_metrics,
        "production_route_model_usage": route_usage,
        "production_route_regime_usage": regime_usage,
        "baseline_metrics": baseline_metrics,
        "return_holdout_metrics": return_holdout_metrics,
        "ranking_holdout": ranking_holdout,
        "beats_baseline": bool(
            routed_metrics["logloss"] < baseline_metrics["logloss"]
        ),
        "calibration_within_limit": bool(
            routed_metrics["ece"] <= max_ece
        ),
        "max_ece": max_ece,
        "holdout_evaluation_mode": "frozen_production_routes",
        "calibration_method": calibration_method,
        "calibration_selection_source": "chronological_oos_only",
        "return_holdout_mode": "production_asset_quantile_routing",
        "feature_pipeline": "technical + market_context + cross_sectional_context",
    }
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
