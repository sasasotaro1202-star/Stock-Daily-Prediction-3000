from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
)

from src.prediction.regression import make_quantile_model, make_return_model
from src.prediction.model_factories import models
from src.prediction.fit import fit_classifier

from src.features.context import add_cross_sectional_context, add_market_context
from src.features.technical import FEATURE_COLUMNS, add_technical_features
from src.prediction.targets import add_targets
from src.research.metrics import aggregate_metric_rows, classification_metrics, cross_sectional_rank_ic
from src.research.return_selection import choose_return_estimator
from src.research.router import (
    ASSET_CANDIDATES,
    CANDIDATES,
    Regime,
    asset_plan,
    choose_from_oos,
    materially_better_than_parent,
    rebalance_global_oos_candidates,
)
from src.validation.calibration import CALIBRATION_METHODS, make_calibrator
from src.validation.training_sample import cap_training_rows
from src.validation.training_window import restrict_to_lookback
from src.research.regime_threshold import (
    aggregate_oos_training_thresholds,
    volatility_threshold_from_training,
)
from src.validation.leakage import audit_feature_columns, audit_target_separation
from src.validation.walk_forward import make_date_folds

PRICE_DIR = Path("data/prices")
OUT = Path("data/research/latest_metrics.json")
AUDIT = Path("data/research/leakage_audit.json")


def make_models():
    return models()


def aggregate_group(rows: list[dict[str, float]]) -> dict[str, float]:
    payload = aggregate_metric_rows(rows)
    payload["folds"] = float(len(rows))
    n_tests = [
        float(row["n_test"])
        for row in rows
        if "n_test" in row and np.isfinite(float(row["n_test"]))
    ]
    if n_tests:
        payload["n_test_min"] = float(min(n_tests))
        payload["n_test_max"] = float(max(n_tests))
    for key in ("logloss", "brier", "ece", "accuracy", "roc_auc", "rank_ic"):
        values = [float(r[key]) for r in rows if key in r and np.isfinite(r[key])]
        payload[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) >= 2 else 0.0
    return payload


def aggregate_model_rows(
    rows: list[tuple[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for model_name, row in rows:
        grouped.setdefault(model_name, []).append(row)
    return {model: aggregate_group(rs) for model, rs in grouped.items()}


def main():
    if not PRICE_DIR.exists():
        raise SystemExit("DEFERRED: price dataset is absent")

    df = pd.read_parquet(PRICE_DIR)
    context_path = Path("data/market_context.parquet")
    if not context_path.exists():
        raise SystemExit("DEFERRED: market context is absent")
    market_context = pd.read_parquet(context_path)
    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce").dt.date
    if len(df) < 2000:
        raise SystemExit(f"DEFERRED: insufficient price rows ({len(df)})")

    df = add_technical_features(df)
    df = add_market_context(df, market_context)
    df = add_targets(add_cross_sectional_context(df))

    fa = audit_feature_columns(FEATURE_COLUMNS)
    targets = [c for c in df.columns if c.startswith("target_")]
    ts = audit_target_separation(FEATURE_COLUMNS, targets)
    audit = {
        "feature_columns": list(FEATURE_COLUMNS),
        "ok": fa.ok and ts.ok,
        "violations": list(fa.violations + ts.violations),
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not audit["ok"]:
        raise SystemExit(f"FAIL: leakage audit {audit['violations']}")

    df = df.dropna(
        subset=FEATURE_COLUMNS + ["target_up_1d", "target_ret_1d"]
    ).copy()

    frozen_path = Path("config/frozen_holdout.json")
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        cutoff = pd.Timestamp(frozen["cutoff_date"]).date()
        df = df[pd.to_datetime(df["session_date"]).dt.date <= cutoff].copy()

    dates = sorted(pd.to_datetime(df["session_date"]).dt.date.unique())
    pipeline_cfg = yaml.safe_load(
        Path("config/pipeline.yml").read_text(encoding="utf-8")
    )
    model_cfg = pipeline_cfg.get("models", {})
    folds = make_date_folds(
        dates,
        min_train=252,
        test_size=21,
        step=21,
        embargo=1,
        purge=int(model_cfg.get("purge_sessions", 1)),
    )
    if len(folds) < 3:
        raise SystemExit(f"DEFERRED: only {len(folds)} OOS folds available")

    # Regime threshold is derived only from fold-local training distributions.
    # OOS/test observations never contribute to the frozen threshold.
    oos_regime_thresholds = []
    for fold in folds:
        train_dates = dates[: fold.train_end]
        cal_n = max(20, int(len(train_dates) * 0.2))
        core_dates = set(train_dates[:-cal_n])
        core = df[df.session_date.isin(core_dates)]
        if not core.empty:
            oos_regime_thresholds.append(
                volatility_threshold_from_training(core["volatility_20"])
            )
    if len(oos_regime_thresholds) < 3:
        raise SystemExit("DEFERRED: insufficient fold-local regime thresholds")

    return_estimators = {
        "mean": [],
        "q50": [],
        "blend_mean_q50": [],
    }
    interval_estimators = {
        "mean": [],
        "q50": [],
        "blend_mean_q50": [],
    }

    for fold in folds:
        train_dates = dates[: fold.train_end]
        cal_n = max(20, int(len(train_dates) * 0.2))
        core_dates = set(train_dates[:-cal_n])
        test_dates = set(dates[fold.test_start : fold.test_end])
        core = df[df.session_date.isin(core_dates)]
        test = df[df.session_date.isin(test_dates)]
        if min(len(core), len(test)) < 100:
            continue

        mean_model = make_return_model()
        q10 = make_quantile_model(0.10)
        q50 = make_quantile_model(0.50)
        q90 = make_quantile_model(0.90)
        core_q = cap_training_rows(core, max_rows=250_000, recent_sessions=252)
        mean_model.fit(core_q[FEATURE_COLUMNS], core_q["target_ret_1d"])
        q10.fit(core_q[FEATURE_COLUMNS], core_q["target_ret_1d"])
        q50.fit(core_q[FEATURE_COLUMNS], core_q["target_ret_1d"])
        q90.fit(core_q[FEATURE_COLUMNS], core_q["target_ret_1d"])

        mean_pred = mean_model.predict(test[FEATURE_COLUMNS])
        q50_pred = q50.predict(test[FEATURE_COLUMNS])
        blend_pred = 0.5 * mean_pred + 0.5 * q50_pred
        y_ret = test["target_ret_1d"].to_numpy(dtype=float)
        group_keys = (
            test["session_date"].astype(str)
            + "::"
            + test["asset_class"].astype(str)
            if "asset_class" in test.columns
            else test["session_date"].astype(str)
        )

        for name, pred in (
            ("mean", mean_pred),
            ("q50", q50_pred),
            ("blend_mean_q50", blend_pred),
        ):
            return_estimators[name].append({
                "mae": float(mean_absolute_error(y_ret, pred)),
                "rmse": float(mean_squared_error(y_ret, pred) ** 0.5),
                "sign_accuracy": float(np.mean((pred >= 0) == (y_ret >= 0))),
                "rank_ic": cross_sectional_rank_ic(
                    y_ret,
                    pred,
                    group_keys,
                ),
                "n_test": float(len(test)),
            })

        lo_base = q10.predict(test[FEATURE_COLUMNS])
        hi_base = q90.predict(test[FEATURE_COLUMNS])

        # Mirror production quantile routing: use asset-specific intervals
        # when the training slice has enough observations.
        for asset, subset in core.groupby("asset_class", sort=False):
            if len(subset) < 750:
                continue
            q_asset = {
                "q10": make_quantile_model(0.10),
                "q50": make_quantile_model(0.50),
                "q90": make_quantile_model(0.90),
            }
            subset_fit = cap_training_rows(
                subset, max_rows=200_000, recent_sessions=252
            )
            for qm in q_asset.values():
                qm.fit(subset_fit[FEATURE_COLUMNS], subset_fit["target_ret_1d"])
            mask = test["asset_class"].eq(asset).to_numpy()
            if mask.any():
                lo_base[mask] = q_asset["q10"].predict(
                    test.loc[mask, FEATURE_COLUMNS]
                )
                hi_base[mask] = q_asset["q90"].predict(
                    test.loc[mask, FEATURE_COLUMNS]
                )

        for name, pred in (
            ("mean", mean_pred),
            ("q50", q50_pred),
            ("blend_mean_q50", blend_pred),
        ):
            lo = np.minimum(lo_base, pred)
            hi = np.maximum(hi_base, pred)
            interval_estimators[name].append({
                "mae": float(mean_absolute_error(y_ret, pred)),
                "rmse": float(mean_squared_error(y_ret, pred) ** 0.5),
                "sign_accuracy": float(
                    np.mean((pred >= 0) == (y_ret >= 0))
                ),
                "q10_pinball": float(mean_pinball_loss(y_ret, lo, alpha=0.10)),
                "q90_pinball": float(mean_pinball_loss(y_ret, hi, alpha=0.90)),
                "range_80_coverage": float(
                    np.mean((y_ret >= lo) & (y_ret <= hi))
                ),
                "n_test": float(len(test)),
            })

    return_estimator_metrics = {
        name: aggregate_group(rows)
        for name, rows in return_estimators.items()
        if rows
    }
    return_mae_guard = float(
        model_cfg.get("return_estimator_mae_guard", 1.10)
    )
    return_rank_ic_tolerance = float(
        model_cfg.get("return_estimator_rank_ic_tolerance", 0.005)
    )
    selected_return_estimator = choose_return_estimator(
        return_estimator_metrics,
        min_folds=3,
        mae_guard=return_mae_guard,
        stability_penalty=0.25,
        rank_ic_tolerance=return_rank_ic_tolerance,
    )
    selected_interval_rows = interval_estimators.get(selected_return_estimator, [])
    interval_metrics = (
        aggregate_group(selected_interval_rows)
        if selected_interval_rows
        else {}
    )
    return_oos = {
        "folds": len(selected_interval_rows),
        "metrics": interval_metrics,
        "estimator_metrics": return_estimator_metrics,
        "selected_estimator": selected_return_estimator,
        "selection_guard": {
            "mae_guard": return_mae_guard,
            "rank_ic_tolerance": return_rank_ic_tolerance,
        },
        "status": "OOS_COMPLETE" if len(selected_interval_rows) >= 3 else "DEFERRED",
    }

    model_results = {}
    regime_rows = {reg.value: [] for reg in Regime if reg is not Regime.DATA_STRESSED}
    asset_rows: dict[str, list[tuple[str, dict[str, float]]]] = {}
    asset_regime_rows: dict[str, list[tuple[str, dict[str, float]]]] = {}
    symbol_rows: dict[str, list[tuple[str, dict[str, float]]]] = {}
    symbol_regime_rows: dict[str, list[tuple[str, dict[str, float]]]] = {}

    for name, factory in make_models().items():
        fold_rows = []
        for fold in folds:
            train_dates = dates[: fold.train_end]
            cal_n = max(20, int(len(train_dates) * 0.2))
            core_dates = set(train_dates[:-cal_n])
            cal_dates = set(train_dates[-cal_n:])
            test_dates = set(dates[fold.test_start : fold.test_end])

            core = df[df.session_date.isin(core_dates)]
            cal = df[df.session_date.isin(cal_dates)]
            test = df[df.session_date.isin(test_dates)].reset_index(drop=True)

            if min(len(core), len(cal), len(test)) < 50:
                continue
            if (
                core.target_up_1d.nunique() < 2
                or cal.target_up_1d.nunique() < 2
                or test.target_up_1d.nunique() < 2
            ):
                continue

            threshold = volatility_threshold_from_training(core["volatility_20"])

            core_fit = cap_training_rows(
                core,
                max_rows=300_000,
                recent_sessions=252,
            )
            model = factory()
            fit_classifier(
                model,
                name,
                core_fit[FEATURE_COLUMNS],
                core_fit.target_up_1d.astype(int),
                core_fit["session_date"],
                half_life_sessions=int(model_cfg.get("recency_weight_half_life_sessions", 252)),
            )
            cal_p = model.predict_proba(cal[FEATURE_COLUMNS])[:, 1]
            calibrator = make_calibrator("platt").fit(
                cal_p, cal.target_up_1d.astype(int)
            )
            p = calibrator.predict(
                model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            )

            row = classification_metrics(test.target_up_1d.astype(int), p)
            group_keys=(
                test["session_date"].astype(str)
                + "::"
                + test["asset_class"].astype(str)
                if "asset_class" in test.columns
                else test["session_date"].astype(str)
            )
            row["rank_ic"] = cross_sectional_rank_ic(
                test["target_ret_1d"].astype(float),
                p,
                group_keys,
            )
            row["n_test"] = float(len(test))
            fold_rows.append(row)

            regime = test.apply(
                lambda x: (
                    "event"
                    if (
                        abs(x["gap_pct"]) >= 0.03
                        or x["volume_ratio_20"] >= 3.0
                    )
                    else "data_stressed"
                    if (
                        pd.isna(x["volatility_20"])
                        or pd.isna(x["price_vs_sma60"])
                    )
                    else "high_vol"
                    if (
                        x["volatility_20"] >= threshold
                        or (
                            pd.notna(x["vix_level_lag1"])
                            and x["vix_level_lag1"] >= 30.0
                        )
                    )
                    else "trend"
                    if (
                        abs(x["price_vs_sma60"]) >= 0.02
                        or (
                            pd.notna(x["breadth_up"])
                            and (
                                x["breadth_up"] <= 0.25
                                or x["breadth_up"] >= 0.75
                            )
                        )
                    )
                    else "normal"
                ),
                axis=1,
            )

            for reg_name in regime.unique():
                mask = regime.eq(reg_name)
                if mask.sum() == 0:
                    continue
                subset = test.loc[mask]
                if subset.target_up_1d.nunique() < 2:
                    continue
                rr = classification_metrics(
                    subset.target_up_1d.astype(int), p[mask]
                )
                rr["rank_ic"] = cross_sectional_rank_ic(
                    subset["target_ret_1d"].astype(float),
                    p[mask],
                    subset["session_date"].astype(str),
                )
                rr["n_test"] = float(mask.sum())
                regime_rows.setdefault(reg_name, []).append((name, rr))

            if "asset_class" in test.columns:
                for asset_class in sorted(test["asset_class"].dropna().unique()):
                    mask = test["asset_class"].eq(asset_class)
                    subset = test.loc[mask]
                    if len(subset) < 30 or subset.target_up_1d.nunique() < 2:
                        continue
                    ar = classification_metrics(
                        subset.target_up_1d.astype(int), p[mask]
                    )
                    ar["rank_ic"] = cross_sectional_rank_ic(
                        subset["target_ret_1d"].astype(float),
                        p[mask],
                        subset["session_date"].astype(str),
                    )
                    ar["n_test"] = float(mask.sum())
                    asset_rows.setdefault(asset_class, []).append((name, ar))

                    for reg_name in sorted(regime[mask].dropna().unique()):
                        route_mask = mask & regime.eq(reg_name)
                        route_subset = test.loc[route_mask]
                        if (
                            len(route_subset) < 30
                            or route_subset.target_up_1d.nunique() < 2
                        ):
                            continue
                        kr = classification_metrics(
                            route_subset.target_up_1d.astype(int), p[route_mask]
                        )
                        kr["rank_ic"] = cross_sectional_rank_ic(
                            route_subset["target_ret_1d"].astype(float),
                            p[route_mask],
                            route_subset["session_date"].astype(str),
                        )
                        kr["n_test"] = float(route_mask.sum())
                        key = f"{asset_class}::{reg_name}"
                        asset_regime_rows.setdefault(key, []).append((name, kr))

            if "symbol" in test.columns and "asset_class" in test.columns:
                for (asset_class, symbol_value), subset in test.groupby(
                    ["asset_class", "symbol"], sort=False
                ):
                    if len(subset) < 10 or subset.target_up_1d.nunique() < 2:
                        continue
                    subset_positions = subset.index.to_numpy(dtype=int)
                    sr = classification_metrics(
                        subset.target_up_1d.astype(int),
                        p[subset_positions],
                    )
                    sr["rank_ic"] = cross_sectional_rank_ic(
                        subset["target_ret_1d"].astype(float),
                        p[subset_positions],
                        subset["session_date"].astype(str),
                    )
                    sr["n_test"] = float(len(subset))
                    symbol_key = f"{asset_class}::{symbol_value}"
                    symbol_rows.setdefault(symbol_key, []).append((name, sr))
                    subset_regime = regime.loc[subset.index]
                    for reg_name in sorted(subset_regime.dropna().unique()):
                        route_subset = subset.loc[subset_regime.eq(reg_name)]
                        if (
                            len(route_subset) < 10
                            or route_subset.target_up_1d.nunique() < 2
                        ):
                            continue
                        route_positions = route_subset.index.to_numpy(dtype=int)
                        rr_symbol = classification_metrics(
                            route_subset.target_up_1d.astype(int),
                            p[route_positions],
                        )
                        rr_symbol["rank_ic"] = cross_sectional_rank_ic(
                            route_subset["target_ret_1d"].astype(float),
                            p[route_positions],
                            route_subset["session_date"].astype(str),
                        )
                        rr_symbol["n_test"] = float(len(route_subset))
                        symbol_regime_key = f"{symbol_key}::{reg_name}"
                        symbol_regime_rows.setdefault(symbol_regime_key, []).append(
                            (name, rr_symbol)
                        )

        model_results[name] = {
            "folds": len(fold_rows),
            "metrics": aggregate_group(fold_rows) if fold_rows else {},
        }

    usable = {
        k: v
        for k, v in model_results.items()
        if v["folds"] >= 3 and "logloss" in v["metrics"]
    }
    if not usable:
        raise SystemExit("DEFERRED: no model has >=3 valid OOS folds")

    global_vol_threshold = aggregate_oos_training_thresholds(oos_regime_thresholds)

    regime_metrics = {
        reg: aggregate_model_rows(rows)
        for reg, rows in regime_rows.items()
    }
    asset_class_metrics = {
        asset: aggregate_model_rows(rows)
        for asset, rows in asset_rows.items()
    }
    asset_regime_metrics = {
        key: aggregate_model_rows(rows)
        for key, rows in asset_regime_rows.items()
    }

    pipeline_cfg = yaml.safe_load(
        Path("config/pipeline.yml").read_text(encoding="utf-8")
    )
    model_cfg = pipeline_cfg.get("models", {})
    rank_ic_tolerance = float(
        model_cfg.get("rank_ic_tiebreak_tolerance", 0.002)
    )

    routing_cfg = pipeline_cfg.get("routing", {})
    scope_improvement = float(
        routing_cfg.get("minimum_scoped_oos_improvement_logloss", 0.002)
    )
    if not 0.0 <= scope_improvement <= 1.0:
        raise SystemExit(
            "FAIL: minimum scoped OOS improvement LogLoss must be in [0, 1]"
        )

    global_candidates = {
        name: dict(value["metrics"], folds=float(value["folds"]))
        for name, value in usable.items()
    }
    balance_weight = float(model_cfg.get("asset_class_balance_weight", 0.50))
    min_asset_folds = int(model_cfg.get("minimum_asset_class_oos_folds", 3))
    balanced_candidates = rebalance_global_oos_candidates(
        global_candidates,
        asset_class_metrics,
        blend_weight=balance_weight,
        min_folds=min_asset_folds,
    )
    global_plan = choose_from_oos(
        "normal",
        balanced_candidates,
        rank_ic_tiebreak_tolerance=rank_ic_tolerance,
    )
    global_selected = global_plan.names[0]

    regime_selected = {}
    for reg_name, candidates in regime_metrics.items():
        if candidates:
            plan = choose_from_oos(
                reg_name,
                candidates,
                rank_ic_tiebreak_tolerance=rank_ic_tolerance,
            )
            if (
                not plan.reason.endswith("fallback")
                and materially_better_than_parent(
                    candidates,
                    plan.names[0],
                    global_selected,
                    min_improvement_logloss=scope_improvement,
                )
            ):
                regime_selected[reg_name] = plan.names[0]

    asset_selected = {}
    for asset_class, candidates in asset_class_metrics.items():
        if candidates:
            plan = asset_plan(
                asset_class,
                candidates,
                min_folds=3,
                rank_ic_tiebreak_tolerance=rank_ic_tolerance,
            )
            if (
                not plan.reason.endswith("fallback")
                and materially_better_than_parent(
                    candidates,
                    plan.names[0],
                    global_selected,
                    min_improvement_logloss=scope_improvement,
                )
            ):
                asset_selected[asset_class] = plan.names[0]

    asset_regime_selected = {}
    for key, candidates in asset_regime_metrics.items():
        asset_class, reg_name = key.split("::", 1)
        if candidates:
            plan = choose_from_oos(
                reg_name,
                candidates,
                candidates=ASSET_CANDIDATES.get(
                    asset_class, CANDIDATES[Regime.NORMAL]
                ),
                scope=key,
                min_folds=2,
                rank_ic_tiebreak_tolerance=rank_ic_tolerance,
            )
            parent_model = (
                asset_selected.get(asset_class)
                or regime_selected.get(reg_name)
                or global_selected
            )
            if (
                not plan.reason.endswith("fallback")
                and materially_better_than_parent(
                    candidates,
                    plan.names[0],
                    parent_model,
                    min_improvement_logloss=scope_improvement,
                )
            ):
                asset_regime_selected[key] = plan.names[0]



    # Security-level routes are selected only when each candidate has enough
    # chronological OOS evidence and clears the same minimum stable edge as
    # higher-level routes. This adds granularity without forcing noisy
    # per-security choices.
    symbol_metrics = {
        key: aggregate_model_rows(rows)
        for key, rows in symbol_rows.items()
    }
    symbol_regime_metrics = {
        key: aggregate_model_rows(rows)
        for key, rows in symbol_regime_rows.items()
    }

    def _eligible_security_candidates(
        candidates: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        return {
            name: metric
            for name, metric in candidates.items()
            if int(metric.get("folds", 0)) >= 4
            and float(metric.get("n_test_min", 0.0)) >= 10.0
            and np.isfinite(float(metric.get("logloss", float("nan"))))
        }

    symbol_selected = {}
    for key, candidates in symbol_metrics.items():
        asset_class, _symbol_value = key.split("::", 1)
        eligible = _eligible_security_candidates(candidates)
        if not eligible:
            continue
        plan = choose_from_oos(
            "normal",
            eligible,
            candidates=ASSET_CANDIDATES.get(asset_class),
            scope=f"symbol:{key}",
            min_folds=4,
            rank_ic_tiebreak_tolerance=rank_ic_tolerance,
        )
        parent_model = asset_selected.get(asset_class) or global_selected
        if (
            not plan.reason.endswith("fallback")
            and parent_model in eligible
            and materially_better_than_parent(
                eligible,
                plan.names[0],
                parent_model,
                min_improvement_logloss=scope_improvement,
            )
        ):
            symbol_selected[key] = plan.names[0]

    symbol_regime_selected = {}
    for key, candidates in symbol_regime_metrics.items():
        parts = key.split("::", 2)
        if len(parts) != 3:
            continue
        asset_class, _symbol_value, reg_name = parts
        eligible = _eligible_security_candidates(candidates)
        if not eligible:
            continue
        plan = choose_from_oos(
            reg_name,
            eligible,
            candidates=ASSET_CANDIDATES.get(asset_class),
            scope=f"symbol_regime:{key}",
            min_folds=4,
            rank_ic_tiebreak_tolerance=rank_ic_tolerance,
        )
        asset_regime_key = f"{asset_class}::{reg_name}"
        parent_model = (
            asset_regime_selected.get(asset_regime_key)
            or asset_selected.get(asset_class)
            or regime_selected.get(reg_name)
            or global_selected
        )
        if (
            not plan.reason.endswith("fallback")
            and parent_model in eligible
            and materially_better_than_parent(
                eligible,
                plan.names[0],
                parent_model,
                min_improvement_logloss=scope_improvement,
            )
        ):
            symbol_regime_selected[key] = plan.names[0]

    security_route_summary = {
        "symbol_routes": int(len(symbol_selected)),
        "symbol_regime_routes": int(len(symbol_regime_selected)),
        "total_security_routes": int(len(symbol_selected) + len(symbol_regime_selected)),
        "symbol_route_examples": sorted(symbol_selected)[:20],
        "symbol_regime_route_examples": sorted(symbol_regime_selected)[:20],
    }

    # Select classifier training-window length on chronological OOS after
    # model-family selection. 0 means all eligible history.
    window_candidates = (252, 504, 756, 0)
    window_metrics = {}
    for lookback in window_candidates:
        fold_rows = []
        for fold in folds:
            train_dates = dates[: fold.train_end]
            usable_train_dates = (
                train_dates if lookback == 0 else train_dates[-lookback:]
            )
            cal_n = max(20, int(len(usable_train_dates) * 0.2))
            if len(usable_train_dates) - cal_n < 40:
                continue
            core_dates = set(usable_train_dates[:-cal_n])
            cal_dates = set(usable_train_dates[-cal_n:])
            test_dates = set(dates[fold.test_start : fold.test_end])
            core = df[df.session_date.isin(core_dates)]
            cal = df[df.session_date.isin(cal_dates)]
            test = df[df.session_date.isin(test_dates)]
            if min(len(core), len(cal), len(test)) < 100:
                continue
            if (
                core.target_up_1d.nunique() < 2
                or cal.target_up_1d.nunique() < 2
                or test.target_up_1d.nunique() < 2
            ):
                continue
            factory = make_models().get(global_selected)
            if factory is None:
                continue
            fit_rows = restrict_to_lookback(
                core,
                None if lookback == 0 else lookback,
            )
            fit_rows = cap_training_rows(
                fit_rows,
                max_rows=300_000,
                recent_sessions=min(252, lookback or 252),
            )
            model = factory()
            fit_classifier(
                model,
                global_selected,
                fit_rows[FEATURE_COLUMNS],
                fit_rows.target_up_1d.astype(int),
                fit_rows["session_date"],
                half_life_sessions=int(model_cfg.get("recency_weight_half_life_sessions", 252)),
            )
            cal_p = model.predict_proba(cal[FEATURE_COLUMNS])[:, 1]
            calibrator = make_calibrator("platt").fit(
                cal_p,
                cal.target_up_1d.astype(int),
            )
            p = calibrator.predict(
                model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            )
            fold_rows.append(classification_metrics(
                test.target_up_1d.astype(int), p
            ))
        if fold_rows:
            logloss = np.asarray(
                [row["logloss"] for row in fold_rows],
                dtype=float,
            )
            window_metrics[str(lookback)] = {
                "logloss": float(np.mean(logloss)),
                "logloss_std": (
                    float(np.std(logloss, ddof=1))
                    if len(logloss) >= 2 else 0.0
                ),
                "folds": float(len(logloss)),
            }

    valid_windows = {
        key: value for key, value in window_metrics.items()
        if value["folds"] >= 3 and np.isfinite(value["logloss"])
    }
    if valid_windows:
        selected_window_key = min(
            valid_windows,
            key=lambda key: (
                valid_windows[key]["logloss"]
                + 0.25 * valid_windows[key]["logloss_std"],
                int(key) if int(key) > 0 else 10**9,
            ),
        )
        selected_training_window = int(selected_window_key)
    else:
        selected_training_window = 0

    # Select the probability calibration method on chronological OOS after
    # model-family and training-window selection. Every candidate is fit only
    # on the fold's calibration slice, and the test slice is used only for
    # scoring. This keeps the calibration choice out of the frozen holdout.
    calibration_rows = {method: [] for method in CALIBRATION_METHODS}
    asset_calibration_rows = {method: [] for method in CALIBRATION_METHODS}
    for fold in folds:
        train_dates = dates[: fold.train_end]
        usable_train_dates = (
            train_dates
            if selected_training_window == 0
            else train_dates[-selected_training_window:]
        )
        cal_n = max(20, int(len(usable_train_dates) * 0.2))
        if len(usable_train_dates) - cal_n < 40:
            continue
        core_dates = set(usable_train_dates[:-cal_n])
        cal_dates = set(usable_train_dates[-cal_n:])
        test_dates = set(dates[fold.test_start : fold.test_end])
        core = df[df.session_date.isin(core_dates)]
        cal = df[df.session_date.isin(cal_dates)]
        test = df[df.session_date.isin(test_dates)]
        if min(len(core), len(cal), len(test)) < 100:
            continue
        if (
            core.target_up_1d.nunique() < 2
            or cal.target_up_1d.nunique() < 2
            or test.target_up_1d.nunique() < 2
        ):
            continue
        factory = make_models().get(global_selected)
        if factory is None:
            continue
        fit_rows = cap_training_rows(
            core,
            max_rows=300_000,
            recent_sessions=min(252, selected_training_window or 252),
        )
        model = factory()
        fit_classifier(
            model,
            global_selected,
            fit_rows[FEATURE_COLUMNS],
            fit_rows.target_up_1d.astype(int),
            fit_rows["session_date"],
            half_life_sessions=int(
                model_cfg.get("recency_weight_half_life_sessions", 252)
            ),
        )
        cal_p = model.predict_proba(cal[FEATURE_COLUMNS])[:, 1]
        raw_test_p = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        for method in CALIBRATION_METHODS:
            calibrator = make_calibrator(method).fit(
                cal_p,
                cal.target_up_1d.astype(int),
            )
            p = np.clip(
                calibrator.predict(raw_test_p),
                1e-5,
                1 - 1e-5,
            )
            metrics = classification_metrics(
                test.target_up_1d.astype(int),
                p,
            )
            calibration_rows[method].append(metrics)

            # Measure calibration robustness across asset classes without
            # changing the model fit. Each fold contributes one macro score,
            # preventing a large universe segment from dominating method choice.
            asset_metrics = []
            if "asset_class" in test.columns:
                for asset_class in sorted(test["asset_class"].dropna().unique()):
                    mask = test["asset_class"].eq(asset_class).to_numpy()
                    subset = test.loc[mask]
                    if len(subset) < 30 or subset.target_up_1d.nunique() < 2:
                        continue
                    am = classification_metrics(
                        subset.target_up_1d.astype(int),
                        p[mask],
                    )
                    asset_metrics.append(am)
            if len(asset_metrics) >= 2:
                asset_calibration_rows[method].append({
                    "logloss": float(np.mean([m["logloss"] for m in asset_metrics])),
                    "brier": float(np.mean([m["brier"] for m in asset_metrics])),
                    "ece": float(np.mean([m["ece"] for m in asset_metrics])),
                    "assets": float(len(asset_metrics)),
                })

    calibration_candidates = {}
    calibration_balance_weight = float(
        np.clip(model_cfg.get("asset_class_balance_weight", 0.50), 0.0, 1.0)
    )
    for method, rows in calibration_rows.items():
        if len(rows) < 3:
            continue
        logloss = np.asarray([float(row["logloss"]) for row in rows], dtype=float)
        ece = np.asarray([float(row["ece"]) for row in rows], dtype=float)
        brier = np.asarray([float(row["brier"]) for row in rows], dtype=float)
        asset_rows = asset_calibration_rows.get(method, [])
        asset_logloss = np.asarray(
            [float(row["logloss"]) for row in asset_rows],
            dtype=float,
        )
        asset_brier = np.asarray(
            [float(row["brier"]) for row in asset_rows],
            dtype=float,
        )
        asset_ece = np.asarray(
            [float(row["ece"]) for row in asset_rows],
            dtype=float,
        )
        use_asset_macro = len(asset_rows) >= 3
        global_std = (
            float(np.std(logloss, ddof=1))
            if len(logloss) >= 2
            else 0.0
        )
        asset_std = (
            float(np.std(asset_logloss, ddof=1))
            if len(asset_logloss) >= 2
            else 0.0
        )
        global_mean = float(np.mean(logloss))
        asset_mean = float(np.mean(asset_logloss)) if use_asset_macro else global_mean
        selection_mean = (
            (1.0 - calibration_balance_weight) * global_mean
            + calibration_balance_weight * asset_mean
        )
        selection_std = (
            (1.0 - calibration_balance_weight) * global_std
            + calibration_balance_weight * asset_std
        )
        calibration_candidates[method] = {
            "logloss": global_mean,
            "logloss_std": global_std,
            "ece": float(np.mean(ece)),
            "brier": float(np.mean(brier)),
            "asset_macro_logloss": asset_mean,
            "asset_macro_brier": (
                float(np.mean(asset_brier)) if use_asset_macro else float(np.mean(brier))
            ),
            "asset_macro_ece": (
                float(np.mean(asset_ece)) if use_asset_macro else float(np.mean(ece))
            ),
            "asset_macro_folds": float(len(asset_rows)),
            "asset_balance_weight": calibration_balance_weight,
            "folds": float(len(rows)),
            "selection_score": selection_mean + 0.25 * selection_std,
        }

    if calibration_candidates:
        selected_calibration_method = min(
            calibration_candidates,
            key=lambda method: (
                calibration_candidates[method]["selection_score"],
                calibration_candidates[method]["ece"],
                {"platt": 0, "beta": 1, "isotonic": 2}[method],
            ),
        )
    else:
        selected_calibration_method = "platt"

    # Select the final production ranking blend on chronological OOS.
    # Each fold trains once; multiple score configurations are then evaluated,
    # so adding ranking candidates does not multiply model fitting cost.
    ranking_candidates = {}
    rank_weight_grid = (0.25, 0.50, 0.75)
    uncertainty_penalty_grid = (0.0, 0.05, 0.10)
    for fold in folds:
        train_dates = dates[: fold.train_end]
        selected_train_dates = (
            train_dates
            if selected_training_window == 0
            else train_dates[-selected_training_window:]
        )
        cal_n = max(20, int(len(selected_train_dates) * 0.2))
        core_dates = set(selected_train_dates[:-cal_n])
        cal_dates = set(selected_train_dates[-cal_n:])
        test_dates = set(dates[fold.test_start : fold.test_end])
        core = df[df.session_date.isin(core_dates)]
        cal = df[df.session_date.isin(cal_dates)]
        test = df[df.session_date.isin(test_dates)]
        if min(len(core), len(cal), len(test)) < 100:
            continue
        if (
            core.target_up_1d.nunique() < 2
            or cal.target_up_1d.nunique() < 2
            or test.target_up_1d.nunique() < 2
        ):
            continue

        classifier = make_models().get(global_selected)
        if classifier is None:
            continue
        model = classifier()
        core_fit = cap_training_rows(
            core, max_rows=300_000, recent_sessions=252
        )
        fit_classifier(
            model,
            global_selected,
            core_fit[FEATURE_COLUMNS],
            core_fit.target_up_1d.astype(int),
            core_fit["session_date"],
            half_life_sessions=int(model_cfg.get("recency_weight_half_life_sessions", 252)),
        )
        cal_p = model.predict_proba(cal[FEATURE_COLUMNS])[:, 1]
        calibrator = make_calibrator(selected_calibration_method).fit(
            cal_p, cal.target_up_1d.astype(int)
        )
        p = calibrator.predict(
            model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        )

        return_fit = cap_training_rows(
            core, max_rows=250_000, recent_sessions=252
        )
        if selected_return_estimator == "mean":
            return_model = make_return_model()
            return_model.fit(
                return_fit[FEATURE_COLUMNS],
                return_fit["target_ret_1d"],
            )
            expected = return_model.predict(test[FEATURE_COLUMNS])
        else:
            q50 = make_quantile_model(0.50)
            q50.fit(return_fit[FEATURE_COLUMNS], return_fit["target_ret_1d"])
            q50_pred = q50.predict(test[FEATURE_COLUMNS])
            if selected_return_estimator == "q50":
                expected = q50_pred
            else:
                mean_model = make_return_model()
                mean_model.fit(
                    return_fit[FEATURE_COLUMNS],
                    return_fit["target_ret_1d"],
                )
                expected = 0.5 * mean_model.predict(
                    test[FEATURE_COLUMNS]
                ) + 0.5 * q50_pred

        q10 = make_quantile_model(0.10)
        q90 = make_quantile_model(0.90)
        q10.fit(return_fit[FEATURE_COLUMNS], return_fit["target_ret_1d"])
        q90.fit(return_fit[FEATURE_COLUMNS], return_fit["target_ret_1d"])
        uncertainty = np.maximum(
            q90.predict(test[FEATURE_COLUMNS])
            - q10.predict(test[FEATURE_COLUMNS]),
            0.0,
        )

        temp = test[
            ["session_date"]
            + (["asset_class"] if "asset_class" in test.columns else [])
        ].copy()
        temp["p"] = p
        temp["expected"] = expected
        temp["uncertainty"] = uncertainty
        group_cols = (
            ["session_date"]
            + (["asset_class"] if "asset_class" in test.columns else [])
        )
        temp["rank_probability"] = temp.groupby(group_cols)["p"].rank(
            method="average", ascending=False, pct=True
        )
        temp["rank_expected"] = temp.groupby(group_cols)["expected"].rank(
            method="average", ascending=False, pct=True
        )
        temp["rank_uncertainty"] = temp.groupby(group_cols)["uncertainty"].rank(
            method="average", ascending=True, pct=True
        )
        group_keys = (
            temp["session_date"].astype(str)
            + "::"
            + temp["asset_class"].astype(str)
            if "asset_class" in temp.columns
            else temp["session_date"].astype(str)
        )

        for weight in rank_weight_grid:
            for penalty in uncertainty_penalty_grid:
                score = (
                    weight * temp["rank_probability"]
                    + (1.0 - weight) * temp["rank_expected"]
                    - penalty * temp["rank_uncertainty"]
                )
                key = f"{weight:.2f}::{penalty:.2f}"
                bucket = ranking_candidates.setdefault(
                    key, {"rank_ic_values": [], "n_tests": []}
                )
                bucket["rank_ic_values"].append(
                    cross_sectional_rank_ic(
                        test["target_ret_1d"].astype(float),
                        score,
                        group_keys,
                    )
                )
                bucket["n_tests"].append(float(len(test)))

    for key, bucket in ranking_candidates.items():
        vals = [
            value for value in bucket["rank_ic_values"]
            if np.isfinite(value)
        ]
        ranking_candidates[key] = {
            "probability_weight": float(key.split("::")[0]),
            "uncertainty_penalty": float(key.split("::")[1]),
            "rank_ic": float(np.mean(vals)) if vals else float("nan"),
            "rank_ic_std": (
                float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
            ),
            "folds": float(len(vals)),
            "selection_score": (
                float(np.mean(vals) - 0.25 * np.std(vals, ddof=1))
                if len(vals) >= 2
                else float(np.mean(vals)) if vals else float("nan")
            ),
        }

    valid_ranking = {
        key: value
        for key, value in ranking_candidates.items()
        if value["folds"] >= 3 and np.isfinite(value["selection_score"])
    }
    if valid_ranking:
        selected_rank_key = max(
            valid_ranking,
            key=lambda key: (
                valid_ranking[key]["selection_score"],
                -valid_ranking[key]["uncertainty_penalty"],
                valid_ranking[key]["probability_weight"],
            ),
        )
        selected_rank_weight = valid_ranking[selected_rank_key]["probability_weight"]
        selected_uncertainty_penalty = valid_ranking[selected_rank_key]["uncertainty_penalty"]
    else:
        selected_rank_weight = 0.50
        selected_uncertainty_penalty = 0.0

    payload = {
        "results": model_results,
        "return_oos": return_oos,
        "regime_metrics": regime_metrics,
        "regime_selected_models": regime_selected,
        "asset_class_metrics": asset_class_metrics,
        "asset_class_selected_models": asset_selected,
        "asset_regime_metrics": asset_regime_metrics,
        "asset_regime_selected_models": asset_regime_selected,
        "symbol_metrics": symbol_metrics,
        "symbol_regime_metrics": symbol_regime_metrics,
        "symbol_selected_models": symbol_selected,
        "symbol_regime_selected_models": symbol_regime_selected,
        "security_route_summary": security_route_summary,
        "selected_model": global_selected,
        "classifier_training_window_sessions": selected_training_window,
        "classifier_training_window_candidates": window_metrics,
        "calibration_method": selected_calibration_method,
        "calibration_method_candidates": calibration_candidates,
        "rank_probability_weight": selected_rank_weight,
        "rank_uncertainty_penalty": selected_uncertainty_penalty,
        "minimum_scoped_oos_improvement_logloss": scope_improvement,
        "ranking_weight_candidates": ranking_candidates,
        "global_selection_candidates": balanced_candidates,
        "regime_vol_threshold": global_vol_threshold,
        "regime_vol_threshold_source": "oos_fold_train_median",
        "regime_vol_threshold_folds": len(oos_regime_thresholds),
        "selection_basis": (
            "chronological walk-forward OOS only; global selection blends "
            "row-weighted LogLoss with a configurable macro asset-class blend "
            "to reduce universe-size dominance, then applies the 0.25 stability "
            "penalty; route hierarchy is asset_class+regime -> asset_class -> "
            "regime -> symbol_regime -> symbol, with security routes requiring " 
            ">=4 folds and >=10 observations/fold plus a stable parent edge; " 
            "calibration is fit inside each OOS training fold; "
            "frozen holdout remains unused during selection"
        ),
        "status": "OOS_COMPLETE",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
