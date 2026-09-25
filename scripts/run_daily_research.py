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
from src.research.online_ensemble import online_expert_average
from src.research.selection_evidence import paired_logloss_selection_evidence
from src.research.statistics import moving_block_bootstrap_mean
from src.research.sequential_selection import chronological_policy_oos
from src.research.nested_policy import nested_sequential_policy_oos
from src.research.conformal_classification import (
    conformal_prediction_sets,
    conformal_prediction_set_metrics,
    group_conformal_prediction_sets,
    group_conformal_prediction_set_metrics,
    adaptive_conformal_prediction_sets,
    adaptive_conformal_prediction_set_metrics,
)
from src.research.confidence_risk import (
    apply_confidence_risk_shrinkage,
    confidence_risk_features,
    fit_temporal_confidence_risk,
    predicted_error_risk,
    risk_bins,
)
from src.research.selective import (
    apply_confidence_shrinkage,
    select_confidence_shrinkage_parameters,
)
from src.research.router import (
    ASSET_CANDIDATES,
    CANDIDATES,
    Regime,
    asset_plan,
    choose_from_oos,
    materially_better_than_parent,
    rebalance_global_oos_candidates,
    regime_for_situation,
    situation_for_row,
)
from src.validation.calibration import CALIBRATION_METHODS, make_calibrator
from src.validation.training_sample import cap_training_rows
from src.validation.training_window import restrict_to_lookback
from src.research.calibration_routing import select_temporal_calibration_method
from src.research.drift_window import (
    robust_distribution_shift_score,
    select_drift_aware_window,
)
from src.research.regime_threshold import (
    aggregate_oos_training_thresholds,
    volatility_threshold_from_training,
)
from src.validation.leakage import audit_feature_columns, audit_target_separation
from src.validation.walk_forward import make_date_folds

PRICE_DIR = Path("data/prices/canonical.parquet")
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

    # Non-stationary markets can make a long-run fold average lag the
    # current regime. Keep every chronological fold, but give the newest
    # folds more influence in the model-family selection objective.
    logloss_rows = [
        float(r["logloss"])
        for r in rows
        if "logloss" in r and np.isfinite(float(r["logloss"]))
    ]
    if len(logloss_rows) >= 3:
        recent_weights = np.linspace(1.0, 2.0, num=len(logloss_rows), dtype=float)
        recent_logloss = float(
            np.average(np.asarray(logloss_rows, dtype=float), weights=recent_weights)
        )
        payload["recent_logloss"] = recent_logloss
        payload["recent_oos_selection_weight"] = 0.50
        payload["selection_logloss"] = (
            0.50 * float(payload["logloss"]) + 0.50 * recent_logloss
        )
    elif logloss_rows:
        payload["recent_logloss"] = float(payload["logloss"])
        payload["recent_oos_selection_weight"] = 0.0
        payload["selection_logloss"] = float(payload["logloss"])
    return payload


def aggregate_model_rows(
    rows: list[tuple[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for model_name, row in rows:
        grouped.setdefault(model_name, []).append(row)
    return {model: aggregate_group(rs) for model, rs in grouped.items()}


def _self_validate_conformal_research():
    """Fail fast on conformal research primitive corruption before expensive OOS work."""
    y_cal = np.asarray([0, 1] * 20, dtype=int)
    p_cal = np.asarray([0.2, 0.8] * 20, dtype=float)
    groups_cal = np.asarray(["stable"] * 40)
    p_test = np.asarray([0.95, 0.05], dtype=float)
    groups_test = np.asarray(["stable", "stable"])
    result = group_conformal_prediction_sets(
        y_cal,
        p_cal,
        groups_cal,
        p_test,
        groups_test,
        alpha=0.10,
        min_group_size=20,
    )
    if result["set_size"].shape != (2,):
        raise SystemExit("FAIL: group conformal self-check shape mismatch")
    if not np.isfinite(result["predicted_class_pvalue"]).all():
        raise SystemExit("FAIL: group conformal self-check produced non-finite p-values")


def main():
    _self_validate_conformal_research()

    if not PRICE_DIR.exists():
        raise SystemExit("DEFERRED: canonical price dataset is absent")

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
    situation_rows: dict[str, list[tuple[str, dict[str, float]]]] = {}
    asset_situation_rows: dict[str, list[tuple[str, dict[str, float]]]] = {}
    asset_rows: dict[str, list[tuple[str, dict[str, float]]]] = {}
    asset_regime_rows: dict[str, list[tuple[str, dict[str, float]]]] = {}
    symbol_rows: dict[str, list[tuple[str, dict[str, float]]]] = {}
    symbol_regime_rows: dict[str, list[tuple[str, dict[str, float]]]] = {}
    selective_rows_by_model: dict[str, list[dict[str, float]]] = {}
    conformal_rows_by_model_alpha: dict[str, dict[float, list[dict[str, float]]]] = {}
    group_conformal_rows_by_model: dict[str, list[dict[str, object]]] = {}
    adaptive_conformal_rows_by_model: dict[str, list[dict[str, float]]] = {}
    adaptive_conformal_situation_rows_by_model: dict[str, dict[str, list[dict[str, float]]]] = {}
    online_prediction_by_fold: dict[int, dict[str, object]] = {}

    for name, factory in make_models().items():
        fold_rows = []
        for fold_idx, fold in enumerate(folds):
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
            raw_test_p = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            p = calibrator.predict(raw_test_p)

            conformal_rows_by_model_alpha.setdefault(name, {})
            for alpha in (0.05, 0.10, 0.20):
                conformal_diag = conformal_prediction_set_metrics(
                    cal.target_up_1d.astype(int),
                    cal_p,
                    raw_test_p,
                    test.target_up_1d.astype(int),
                    alpha=alpha,
                )
                conformal_rows_by_model_alpha[name].setdefault(
                    float(alpha), []
                ).append(conformal_diag)
            conformal_case = conformal_prediction_sets(
                cal.target_up_1d.astype(int),
                cal_p,
                raw_test_p,
                alpha=0.10,
            )
            group_conformal_diag = group_conformal_prediction_set_metrics(
                cal.target_up_1d.astype(int),
                cal_p,
                cal["asset_class"].astype(str).to_numpy(),
                raw_test_p,
                test["asset_class"].astype(str).to_numpy(),
                test.target_up_1d.astype(int),
                alpha=0.10,
                min_group_size=50,
            )
            group_conformal_rows_by_model.setdefault(name, []).append(group_conformal_diag)
            group_conformal_case = group_conformal_prediction_sets(
                cal.target_up_1d.astype(int),
                cal_p,
                cal["asset_class"].astype(str).to_numpy(),
                raw_test_p,
                test["asset_class"].astype(str).to_numpy(),
                alpha=0.10,
                min_group_size=50,
            )
            adaptive_conformal_diag = adaptive_conformal_prediction_set_metrics(
                cal.target_up_1d.astype(int),
                cal_p,
                raw_test_p,
                test["session_date"].astype(str).to_numpy(),
                test.target_up_1d.astype(int),
                alpha=0.10,
                gamma=0.02,
                alpha_min=0.01,
                alpha_max=0.50,
            )
            adaptive_conformal_rows_by_model.setdefault(name, []).append(
                adaptive_conformal_diag
            )
            adaptive_conformal_case = adaptive_conformal_prediction_sets(
                cal.target_up_1d.astype(int),
                cal_p,
                raw_test_p,
                test["session_date"].astype(str).to_numpy(),
                test.target_up_1d.astype(int),
                alpha=0.10,
                gamma=0.02,
                alpha_min=0.01,
                alpha_max=0.50,
                observed_y_test=test.target_up_1d.astype(int).to_numpy(),
            )

            online_bank = online_prediction_by_fold.setdefault(
                fold_idx,
                {
                    "session_dates": test["session_date"].to_numpy(copy=True),
                    "y": test.target_up_1d.astype(int).to_numpy(copy=True),
                    "predictions": {},
                    "situations": None,
                    "risk_context": None,
                    "asset_classes": None,
                    "conformal_pred_pvalues": {},
                    "group_conformal_pred_pvalues": {},
                    "group_conformal_set_size": {},
                    "adaptive_conformal_pred_pvalues": {},
                    "adaptive_conformal_set_size": {},
                    "adaptive_conformal_alpha_used": {},
                },
            )
            online_bank["predictions"][name] = np.asarray(p, dtype=float)
            online_bank["conformal_pred_pvalues"][name] = np.asarray(
                conformal_case["predicted_class_pvalue"], dtype=float
            )
            online_bank["group_conformal_pred_pvalues"][name] = np.asarray(
                group_conformal_case["predicted_class_pvalue"], dtype=float
            )
            online_bank["group_conformal_set_size"][name] = np.asarray(
                group_conformal_case["set_size"], dtype=int
            )
            online_bank["adaptive_conformal_pred_pvalues"][name] = np.asarray(
                adaptive_conformal_case["predicted_class_pvalue"], dtype=float
            )
            online_bank["adaptive_conformal_set_size"][name] = np.asarray(
                adaptive_conformal_case["set_size"], dtype=int
            )
            online_bank["adaptive_conformal_alpha_used"][name] = np.asarray(
                adaptive_conformal_case["alpha_used"], dtype=float
            )
            if online_bank["asset_classes"] is None:
                online_bank["asset_classes"] = test["asset_class"].astype(str).to_numpy()
            if online_bank["risk_context"] is None:
                risk_columns = [
                    "volatility_20", "volume_ratio_20", "gap_pct", "breadth_up",
                    "market_dispersion_1d", "market_dispersion_vs_20d",
                    "vix_level_lag1", "return_z20", "drawdown_from_high_20",
                    "cs_ret_1d_rank", "cs_vol_rank",
                ]
                online_bank["risk_context"] = np.column_stack([
                    pd.to_numeric(test[col], errors="coerce").to_numpy(dtype=float)
                    if col in test.columns else np.full(len(test), np.nan)
                    for col in risk_columns
                ])

            # Research-only confidence shrinkage. Select parameters from
            # the fold-local calibration slice only; the OOS test slice remains
            # completely untouched until final evaluation.
            selective_rows_by_model.setdefault(name, [])
            base_rate = float(core_fit.target_up_1d.astype(int).mean())
            calibrated_cal_p = calibrator.predict(cal_p)
            selected = select_confidence_shrinkage_parameters(
                cal.target_up_1d.astype(int),
                calibrated_cal_p,
                base_rate=base_rate,
            )
            adjusted = apply_confidence_shrinkage(
                p,
                base_rate=base_rate,
                confidence_threshold=selected["confidence_threshold"],
                retained_weight=selected["retained_weight"],
            )
            baseline = classification_metrics(test.target_up_1d.astype(int), p)
            metrics = classification_metrics(
                test.target_up_1d.astype(int), adjusted
            )
            metrics["confidence_threshold"] = float(selected["confidence_threshold"])
            metrics["retained_weight"] = float(selected["retained_weight"])
            metrics["validation_logloss"] = float(selected["validation_logloss"])
            metrics["validation_brier"] = float(selected["validation_brier"])
            metrics["delta_logloss"] = float(
                baseline["logloss"] - metrics["logloss"]
            )
            metrics["n_test"] = float(len(test))
            selective_rows_by_model[name].append(metrics)

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
            row["fold"] = float(fold_idx)
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

            situations = test.apply(
                lambda x: situation_for_row(
                    str(regime.loc[x.name]),
                    gap_pct=float(x["gap_pct"]) if pd.notna(x["gap_pct"]) else None,
                    volume_ratio_20=float(x["volume_ratio_20"]) if pd.notna(x["volume_ratio_20"]) else None,
                    vix_level=float(x["vix_level_lag1"]) if pd.notna(x["vix_level_lag1"]) else None,
                    breadth_up=float(x["breadth_up"]) if pd.notna(x["breadth_up"]) else None,
                    price_vs_sma60=float(x["price_vs_sma60"]) if pd.notna(x["price_vs_sma60"]) else None,
                ),
                axis=1,
            )
            online_bank["situations"] = situations.astype(str).to_numpy(copy=True)
            adaptive_conformal_situation_rows_by_model.setdefault(name, {})
            adaptive_alpha = online_bank["adaptive_conformal_alpha_used"][name]
            adaptive_size = online_bank["adaptive_conformal_set_size"][name]
            adaptive_predicted = (p >= 0.5).astype(int)
            test_y = test.target_up_1d.to_numpy(dtype=int)
            for situation_name in sorted(situations.dropna().astype(str).unique()):
                mask = situations.astype(str).eq(situation_name).to_numpy()
                if mask.sum() < 30 or np.unique(test_y[mask]).size < 2:
                    continue
                y_s = test_y[mask]
                include_true = np.where(
                    y_s == 1,
                    adaptive_conformal_case["include_1"][mask],
                    adaptive_conformal_case["include_0"][mask],
                )
                size_s = adaptive_size[mask]
                singleton = size_s == 1
                adaptive_conformal_situation_rows_by_model[name].setdefault(
                    situation_name, []
                ).append({
                    "fold": float(fold_idx),
                    "n_test": float(mask.sum()),
                    "coverage": float(np.mean(include_true)),
                    "mean_set_size": float(np.mean(size_s)),
                    "singleton_rate": float(np.mean(singleton)),
                    "singleton_accuracy": (
                        float(np.mean(adaptive_predicted[mask][singleton] == y_s[singleton]))
                        if singleton.any() else float("nan")
                    ),
                    "empty_rate": float(np.mean(size_s == 0)),
                    "mean_alpha_used": float(np.mean(adaptive_alpha[mask])),
                })
            for situation_name in sorted(situations.dropna().unique()):
                mask = situations.eq(situation_name)
                subset = test.loc[mask]
                if len(subset) < 30 or subset.target_up_1d.nunique() < 2:
                    continue
                sm = classification_metrics(
                    subset.target_up_1d.astype(int), p[mask]
                )
                sm["rank_ic"] = cross_sectional_rank_ic(
                    subset["target_ret_1d"].astype(float),
                    p[mask],
                    subset["session_date"].astype(str),
                )
                sm["n_test"] = float(mask.sum())
                situation_rows.setdefault(str(situation_name), []).append((name, sm))

                if "asset_class" in subset.columns:
                    for asset_class in sorted(subset["asset_class"].dropna().unique()):
                        asset_mask = mask & test["asset_class"].eq(asset_class)
                        asset_subset = test.loc[asset_mask]
                        if (
                            len(asset_subset) < 30
                            or asset_subset.target_up_1d.nunique() < 2
                        ):
                            continue
                        asm = classification_metrics(
                            asset_subset.target_up_1d.astype(int),
                            p[asset_mask],
                        )
                        asm["rank_ic"] = cross_sectional_rank_ic(
                            asset_subset["target_ret_1d"].astype(float),
                            p[asset_mask],
                            asset_subset["session_date"].astype(str),
                        )
                        asm["n_test"] = float(asset_mask.sum())
                        key = f"{asset_class}::{situation_name}"
                        asset_situation_rows.setdefault(key, []).append((name, asm))

            if online_bank["situations"] is None:
                online_bank["situations"] = situations.to_numpy(dtype=str)

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
            "fold_metrics": fold_rows,
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
    situation_metrics = {
        situation: aggregate_model_rows(rows)
        for situation, rows in situation_rows.items()
    }
    asset_situation_metrics = {
        key: aggregate_model_rows(rows)
        for key, rows in asset_situation_rows.items()
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
    selection_evidence_min_folds = int(
        model_cfg.get("selection_evidence_min_common_oos_folds", 5)
    )
    selection_evidence_min_relative_improvement = float(
        model_cfg.get("selection_evidence_min_relative_improvement", 0.03)
    )
    selection_evidence_alpha = float(
        model_cfg.get("selection_evidence_alpha", 0.05)
    )
    if selection_evidence_min_folds < 5:
        raise SystemExit(
            "FAIL: selection evidence requires at least 5 common OOS folds"
        )
    if selection_evidence_min_relative_improvement < 0.03:
        raise SystemExit(
            "FAIL: selection evidence relative improvement floor cannot be below 3%"
        )
    if not 0.0 < selection_evidence_alpha <= 0.05:
        raise SystemExit(
            "FAIL: selection evidence alpha must be in (0, 0.05]"
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

    # Conservative selection evidence: compare the selected model with the
    # strongest OOS comparator on the same chronological folds. This evidence
    # never sees the frozen holdout and is consumed by lock_frozen_model.py
    # before a new production configuration can be frozen.
    global_selection_fold_rows = {
        name: value.get("fold_metrics", [])
        for name, value in model_results.items()
        if name in balanced_candidates
    }
    global_selection_evidence = paired_logloss_selection_evidence(
        global_selected,
        balanced_candidates,
        global_selection_fold_rows,
        trial_count=max(1, len(balanced_candidates)),
        min_folds=selection_evidence_min_folds,
        min_relative_improvement=selection_evidence_min_relative_improvement,
        alpha=selection_evidence_alpha,
    )

    sequential_cfg = pipeline_cfg.get("sequential_selection_research", {})
    if sequential_cfg.get("research_only", True) is not True:
        raise SystemExit("FAIL: sequential model selection audit must remain research-only")
    sequential_min_history = int(sequential_cfg.get("min_history_folds", 3))
    sequential_half_life = float(sequential_cfg.get("half_life_folds", 4.0))
    sequential_stability = float(sequential_cfg.get("stability_penalty", 0.25))
    sequential_baseline = sequential_cfg.get("baseline_model", "logistic")
    sequential_model_names = {
        name: value.get("fold_metrics", [])
        for name, value in model_results.items()
        if name in balanced_candidates
    }
    sequential_selection_research = chronological_policy_oos(
        sequential_model_names,
        min_history_folds=sequential_min_history,
        half_life_folds=sequential_half_life,
        stability_penalty=sequential_stability,
        baseline_model=str(sequential_baseline),
    )
    sequential_selection_research["selection_note"] = (
        "Each fold is evaluated only after the candidate model is selected "
        "from prior chronological OOS outcomes. Current-fold outcomes are "
        "never used by the selector. This research audit is not a production "
        "change and does not use the frozen holdout."
    )

    nested_cfg = pipeline_cfg.get("nested_policy_oos_research", {})
    if nested_cfg.get("research_only", True) is not True:
        raise SystemExit("FAIL: nested policy OOS audit must remain research-only")
    nested_min_outer = int(nested_cfg.get("min_outer_folds", 5))
    nested_outer_start = nested_cfg.get("outer_start_fold")
    if nested_outer_start is not None:
        nested_outer_start = int(nested_outer_start)
    nested_baseline = nested_cfg.get("baseline_model", "auto_inner_static")
    nested_selection_research = nested_sequential_policy_oos(
        sequential_model_names,
        outer_start_fold=nested_outer_start,
        min_history_folds=sequential_min_history,
        half_life_folds=sequential_half_life,
        stability_penalty=sequential_stability,
        baseline_model=str(nested_baseline),
        min_outer_folds=nested_min_outer,
    )
    nested_selection_research["selection_note"] = (
        "Outer-period performance is reserved for evaluating a fixed policy. "
        "Policy hyperparameters are fixed before the outer block; each outer "
        "fold sees only prior inner/outer outcomes, and the current outer "
        "outcome is added only after scoring. Research-only; no production change."
    )

    # Research-only temporal confidence-risk layer. It predicts the
    # probability that the selected model's directional decision will be
    # wrong, using only prior OOS prediction/context rows. It may shrink
    # overconfident probabilities, but never flips direction and never affects
    # the production artifact.
    risk_cfg = pipeline_cfg.get("confidence_risk", {})
    if risk_cfg.get("research_only", True) is not True:
        raise SystemExit("FAIL: confidence risk layer must remain research-only")
    risk_min_rows = int(risk_cfg.get("min_training_rows", 240))
    risk_threshold = float(risk_cfg.get("risk_threshold", 0.60))
    risk_max_shrink = float(risk_cfg.get("max_shrink", 0.35))
    risk_min_relative = float(
        risk_cfg.get("min_relative_oos_logloss_improvement", 0.03)
    )
    risk_min_positive_share = float(
        risk_cfg.get("min_positive_fold_share", 0.70)
    )
    risk_min_bootstrap = float(
        risk_cfg.get("min_bootstrap_probability", 0.90)
    )
    if not (0.0 <= risk_threshold < 1.0 and 0.0 <= risk_max_shrink <= 1.0):
        raise SystemExit("FAIL: invalid confidence risk bounds")
    confidence_risk_research = {
        "status": "INSUFFICIENT_OOS",
        "method": "temporal_correctness_meta_model",
        "research_only": True,
        "training_rows": 0,
        "folds": 0,
    }
    risk_history_x = []
    risk_history_y = []
    risk_history_target_y = []
    risk_conformal_history_x = []
    risk_conformal_history_y = []
    risk_adjusted_rows = []
    risk_raw_rows = []
    risk_fold_deltas = []
    risk_high_rows = []
    for fold_idx in sorted(online_prediction_by_fold):
        bank = online_prediction_by_fold[fold_idx]
        predictions = bank["predictions"]
        if global_selected not in predictions or bank.get("risk_context") is None:
            continue
        p_selected = np.asarray(predictions[global_selected], dtype=float)
        expert_names = sorted(predictions)
        expert_matrix = np.column_stack([
            np.asarray(predictions[n], dtype=float) for n in expert_names
        ])
        context = np.asarray(bank["risk_context"], dtype=float)
        meta_features = confidence_risk_features(
            p_selected, expert_matrix, context
        )
        conformal_pvalue = np.asarray(
            bank.get("conformal_pred_pvalues", {}).get(global_selected),
            dtype=float,
        )
        if conformal_pvalue.shape != (len(p_selected),):
            raise SystemExit(
                f"FAIL: conformal p-value shape mismatch in fold {fold_idx}"
            )
        conformal_context = np.column_stack([context, conformal_pvalue])
        conformal_meta_features = confidence_risk_features(
            p_selected,
            expert_matrix,
            conformal_context,
            include_conformal_pvalue=True,
        )
        selector = fit_temporal_confidence_risk(
            np.asarray(risk_history_x, dtype=float)
            if risk_history_x else np.empty((0, meta_features.shape[1])),
            np.asarray(risk_history_y, dtype=int)
            if risk_history_y else np.empty((0,), dtype=int),
            min_rows=risk_min_rows,
        )
        risk = predicted_error_risk(selector, meta_features)
        conformal_selector = fit_temporal_confidence_risk(
            np.asarray(risk_conformal_history_x, dtype=float)
            if risk_conformal_history_x
            else np.empty((0, conformal_meta_features.shape[1])),
            np.asarray(risk_conformal_history_y, dtype=int)
            if risk_conformal_history_y
            else np.empty((0,), dtype=int),
            min_rows=risk_min_rows,
        )
        conformal_risk = predicted_error_risk(
            conformal_selector, conformal_meta_features
        )
        y_fold = np.asarray(bank["y"], dtype=int)
        base_rate = (
            float(np.mean(np.asarray(risk_history_target_y, dtype=int)))
            if risk_history_target_y else 0.5
        )
        adjusted = apply_confidence_risk_shrinkage(
            p_selected, risk, base_rate=base_rate,
            risk_threshold=risk_threshold, max_shrink=risk_max_shrink,
        )
        conformal_adjusted = apply_confidence_risk_shrinkage(
            p_selected, conformal_risk, base_rate=base_rate,
            risk_threshold=risk_threshold, max_shrink=risk_max_shrink,
        )
        raw_m = classification_metrics(y_fold, p_selected)
        adjusted_m = classification_metrics(y_fold, adjusted)
        risk_raw_rows.append(raw_m)
        risk_adjusted_rows.append(adjusted_m)
        risk_fold_deltas.append(float(raw_m["logloss"] - adjusted_m["logloss"]))
        conformal_adjusted_m = classification_metrics(y_fold, conformal_adjusted)
        confidence_risk_research.setdefault(
            "conformal_feature_ablation_rows", []
        ).append({
            "fold": float(fold_idx),
            "raw_logloss": float(raw_m["logloss"]),
            "conformal_adjusted_logloss": float(conformal_adjusted_m["logloss"]),
            "raw_brier": float(raw_m["brier"]),
            "conformal_adjusted_brier": float(conformal_adjusted_m["brier"]),
            "raw_ece": float(raw_m["ece"]),
            "conformal_adjusted_ece": float(conformal_adjusted_m["ece"]),
        })

        high = risk >= risk_threshold
        if high.any() and np.unique(y_fold[high]).size >= 2:
            raw_high = classification_metrics(y_fold[high], p_selected[high])
            adj_high = classification_metrics(y_fold[high], adjusted[high])
            risk_high_rows.append({
                "fold": float(fold_idx),
                "n": float(high.sum()),
                "raw_logloss": float(raw_high["logloss"]),
                "adjusted_logloss": float(adj_high["logloss"]),
                "raw_accuracy": float(raw_high["accuracy"]),
                "adjusted_accuracy": float(adj_high["accuracy"]),
            })
        confidence_risk_research["last_fold_bins"] = risk_bins(
            risk, y_fold, p_selected
        )

        # Current-fold labels enter the meta-training history only after the
        # current fold has been scored, preserving chronological OOS order.
        current_correct = ((p_selected >= 0.5).astype(int) == y_fold).astype(int)
        risk_history_x.extend(meta_features.tolist())
        risk_history_y.extend(current_correct.tolist())
        risk_history_target_y.extend(y_fold.tolist())
        risk_conformal_history_x.extend(conformal_meta_features.tolist())
        risk_conformal_history_y.extend(current_correct.tolist())
        if selector is not None:
            confidence_risk_research["status"] = "EVALUATED"
            confidence_risk_research["training_rows"] = len(risk_history_y)
            confidence_risk_research["folds"] = int(
                confidence_risk_research.get("folds", 0) + 1
            )

    if risk_raw_rows:
        raw_risk = aggregate_group(risk_raw_rows)
        adjusted_risk = aggregate_group(risk_adjusted_rows)
        high_summary = {}
        if risk_high_rows:
            weights = np.asarray([r["n"] for r in risk_high_rows], dtype=float)
            high_summary = {
                "folds": len(risk_high_rows),
                "n_total": float(weights.sum()),
                "raw_logloss": float(np.average(
                    [r["raw_logloss"] for r in risk_high_rows], weights=weights
                )),
                "adjusted_logloss": float(np.average(
                    [r["adjusted_logloss"] for r in risk_high_rows], weights=weights
                )),
                "raw_accuracy": float(np.average(
                    [r["raw_accuracy"] for r in risk_high_rows], weights=weights
                )),
                "adjusted_accuracy": float(np.average(
                    [r["adjusted_accuracy"] for r in risk_high_rows], weights=weights
                )),
            }
        fold_array = np.asarray(risk_fold_deltas, dtype=float)
        bootstrap_probability = 0.0
        bootstrap_p05 = float("-inf")
        if len(fold_array) >= 5 and np.isfinite(fold_array).all():
            bootstrap_probability, bootstrap_p05 = moving_block_bootstrap_mean(
                fold_array,
                n_bootstrap=4000,
                seed=20260925,
            )
        positive_fold_share = (
            float(np.mean(fold_array > 0.0)) if len(fold_array) else 0.0
        )
        relative_logloss_improvement = (
            float(
                (raw_risk["logloss"] - adjusted_risk["logloss"])
                / max(abs(raw_risk["logloss"]), 1e-9)
            )
        )
        high_risk_improvement = (
            bool(
                high_summary
                and high_summary["adjusted_logloss"] <= high_summary["raw_logloss"]
            )
        )
        confidence_risk_research.update({
            "raw_oos": raw_risk,
            "risk_adjusted_oos": adjusted_risk,
            "logloss_improvement": float(
                raw_risk["logloss"] - adjusted_risk["logloss"]
            ),
            "relative_logloss_improvement": relative_logloss_improvement,
            "brier_improvement": float(
                raw_risk["brier"] - adjusted_risk["brier"]
            ),
            "ece_change": float(
                adjusted_risk["ece"] - raw_risk["ece"]
            ),
            "high_risk_cases": high_summary,
            "fold_logloss_improvements": [float(x) for x in risk_fold_deltas],
            "positive_fold_share": positive_fold_share,
            "bootstrap_probability_improvement": bootstrap_probability,
            "bootstrap_p05_improvement": bootstrap_p05,
            "research_positive": bool(
                len(fold_array) >= 5
                and relative_logloss_improvement >= risk_min_relative
                and positive_fold_share >= risk_min_positive_share
                and bootstrap_probability >= risk_min_bootstrap
                and bootstrap_p05 > 0.0
                and float(adjusted_risk["brier"] - raw_risk["brier"]) <= 0.001
                and float(adjusted_risk["ece"] - raw_risk["ece"]) <= 0.0
                and high_risk_improvement
            ),
            "policy": (
                "research_only; expanding chronological meta-learning; "
                "current-fold outcomes enter history only after scoring; "
                "high-risk predictions shrink toward prior base rate; no contrarian flip; "
                "positive status requires >=3% relative OOS LogLoss improvement, "
                ">=70% positive folds, stable bootstrap evidence, and no material calibration harm"
            ),
        })

    # Research-only online expert aggregation. Predictions for each session
    # use weights learned strictly before that session. We update weights only
    # after the whole session's outcomes are available, avoiding within-session
    # leakage. Learning-rate/share-rate panels are ablations only; none is
    # selected from these same OOS test results.
    online_expert_research = {}
    for learning_rate in (0.5, 1.0, 2.0, 4.0):
        for share_rate in (0.0, 0.02, 0.05, 0.10):
            fold_rows = []
            situation_rows_online: dict[str, list[dict[str, float]]] = {}
            for fold_idx in sorted(online_prediction_by_fold):
                bank = online_prediction_by_fold[fold_idx]
                predictions = bank["predictions"]
                if len(predictions) < 2:
                    continue
                try:
                    ensemble_p, final_weights, history = online_expert_average(
                        predictions,
                        bank["y"],
                        bank["session_dates"],
                        learning_rate=learning_rate,
                        share_rate=share_rate,
                    )
                except ValueError as exc:
                    raise SystemExit(
                        f"FAIL: online expert input validation failed in fold {fold_idx}: {exc}"
                    ) from exc
                y = np.asarray(bank["y"], dtype=int)
                metrics = classification_metrics(y, ensemble_p)
                metrics["rank_ic"] = cross_sectional_rank_ic(
                    y.astype(float),
                    ensemble_p,
                    pd.Series(bank["session_dates"]).astype(str).to_numpy(),
                )
                baseline = predictions.get(global_selected)
                if baseline is not None:
                    baseline_ll = classification_metrics(y, baseline)["logloss"]
                    metrics["delta_logloss_vs_global_selected"] = float(
                        baseline_ll - metrics["logloss"]
                    )
                else:
                    metrics["delta_logloss_vs_global_selected"] = float("nan")
                metrics["n_test"] = float(len(y))
                metrics["fold"] = float(fold_idx)
                metrics["learning_rate"] = float(learning_rate)
                metrics["share_rate"] = float(share_rate)
                metrics["final_weight_max"] = float(np.max(final_weights))
                metrics["final_weight_entropy"] = float(
                    -np.sum(final_weights * np.log(np.clip(final_weights, 1e-12, 1.0)))
                )
                metrics["online_sessions"] = float(len(history))
                fold_rows.append(metrics)

                situations = bank.get("situations")
                if situations is not None:
                    situations = np.asarray(situations, dtype=str)
                    for situation_name in sorted(set(situations.tolist())):
                        mask = situations == situation_name
                        if mask.sum() < 30 or len(np.unique(y[mask])) < 2:
                            continue
                        sm = classification_metrics(y[mask], ensemble_p[mask])
                        if baseline is not None:
                            sm["delta_logloss_vs_global_selected"] = float(
                                classification_metrics(y[mask], baseline[mask])["logloss"]
                                - sm["logloss"]
                            )
                        sm["n_test"] = float(mask.sum())
                        situation_rows_online.setdefault(situation_name, []).append(sm)

            if fold_rows:
                gains = np.asarray(
                    [r["delta_logloss_vs_global_selected"] for r in fold_rows],
                    dtype=float,
                )
                gains = gains[np.isfinite(gains)]
                result_key = (
                    str(learning_rate)
                    if share_rate == 0.0
                    else f"{learning_rate}::share={share_rate:g}"
                )
                online_expert_research[result_key] = {
                    "research_only": True,
                    "production_changed": False,
                    "promotion_allowed": False,
                    "learning_rate": float(learning_rate),
                    "share_rate": float(share_rate),
                    "selection_protocol": "fixed_learning_rate_and_share_rate_chronological_online_update_after_each_session",
                    "test_tuning_allowed": False,
                    "folds": len(fold_rows),
                    "mean_logloss": float(np.mean([r["logloss"] for r in fold_rows])),
                    "logloss_std": float(np.std([r["logloss"] for r in fold_rows], ddof=1)) if len(fold_rows) >= 2 else 0.0,
                    "mean_logloss_improvement_vs_global_selected": float(np.mean(gains)) if len(gains) else 0.0,
                    "positive_fold_ratio_vs_global_selected": float(np.mean(gains > 0.0)) if len(gains) else 0.0,
                    "metrics_by_fold": fold_rows,
                    "situation_metrics": {
                        key: {
                            "folds": len(rows),
                            "logloss": float(np.mean([r["logloss"] for r in rows])),
                            "delta_logloss_vs_global_selected": float(
                                np.nanmean([r.get("delta_logloss_vs_global_selected", np.nan) for r in rows])
                            ),
                            "n_test_min": float(min(r["n_test"] for r in rows)),
                        }
                        for key, rows in situation_rows_online.items()
                    },
                }
    # Research-only asset-balanced online updates. This ablation equalizes
    # asset-class influence within each session so large groups cannot dominate
    # expert adaptation. Current-session outcomes are used only after prediction.
    online_expert_balanced_research = {}
    for learning_rate in (1.0, 2.0, 4.0):
        fold_rows = []
        for fold_idx in sorted(online_prediction_by_fold):
            bank = online_prediction_by_fold[fold_idx]
            predictions = bank["predictions"]
            groups = bank.get("asset_classes")
            if len(predictions) < 2 or groups is None:
                continue
            try:
                ensemble_p, _, _ = online_expert_average(
                    predictions,
                    bank["y"],
                    bank["session_dates"],
                    learning_rate=learning_rate,
                    share_rate=0.05,
                    update_group_keys=groups,
                )
            except ValueError as exc:
                raise SystemExit(
                    f"FAIL: balanced online expert input validation failed in fold {fold_idx}: {exc}"
                ) from exc
            y = np.asarray(bank["y"], dtype=int)
            baseline = predictions.get(global_selected)
            gain = (
                classification_metrics(y, baseline)["logloss"]
                - classification_metrics(y, ensemble_p)["logloss"]
                if baseline is not None else float("nan")
            )
            fold_rows.append({
                "fold": float(fold_idx),
                "n_test": float(len(y)),
                "logloss": float(classification_metrics(y, ensemble_p)["logloss"]),
                "delta_logloss_vs_global_selected": float(gain),
                "learning_rate": float(learning_rate),
                "share_rate": 0.05,
            })
        if fold_rows:
            gains = np.asarray([r["delta_logloss_vs_global_selected"] for r in fold_rows], dtype=float)
            gains = gains[np.isfinite(gains)]
            online_expert_balanced_research[str(learning_rate)] = {
                "research_only": True,
                "production_changed": False,
                "promotion_allowed": False,
                "learning_rate": float(learning_rate),
                "share_rate": 0.05,
                "update_group": "asset_class",
                "selection_protocol": "fixed_parameters_prequential_asset_balanced_update_after_each_session",
                "test_tuning_allowed": False,
                "folds": len(fold_rows),
                "mean_logloss": float(np.mean([r["logloss"] for r in fold_rows])),
                "logloss_std": float(np.std([r["logloss"] for r in fold_rows], ddof=1)) if len(fold_rows) >= 2 else 0.0,
                "mean_logloss_improvement_vs_global_selected": float(np.mean(gains)) if len(gains) else 0.0,
                "positive_fold_ratio_vs_global_selected": float(np.mean(gains > 0.0)) if len(gains) else 0.0,
                "metrics_by_fold": fold_rows,
            }

    # Do not tune or promote an online learning rate from these same OOS folds.
    # A future promotion requires nested selection or a separate untouched period.

    # Each fold selects shrinkage hyperparameters from its calibration slice,
    # then evaluates once on an untouched chronological OOS test slice.
    selective_candidates = {}
    selective_selected = {
        "status": "NO_PROMOTION",
        "model": global_selected,
        "parameter_mode": "fold_local_calibration_selection",
        "mean_logloss_improvement": 0.0,
        "positive_fold_ratio": 0.0,
        "mean_logloss": 0.0,
        "logloss_std": 0.0,
        "folds": 0,
    }
    research_cfg = pipeline_cfg.get("research", {})
    min_selective_gain = float(
        research_cfg.get("minimum_selective_logloss_improvement", 0.001)
    )
    min_selective_positive_ratio = float(
        research_cfg.get("minimum_selective_positive_fold_ratio", 2.0 / 3.0)
    )

    for model_name, rows in selective_rows_by_model.items():
        if len(rows) < 3:
            continue
        finite_rows = [
            row for row in rows
            if np.isfinite(float(row.get("delta_logloss", np.nan)))
            and np.isfinite(float(row.get("logloss", np.nan)))
        ]
        if len(finite_rows) < 3:
            continue
        deltas = np.asarray(
            [float(row["delta_logloss"]) for row in finite_rows], dtype=float
        )
        lls = np.asarray(
            [float(row["logloss"]) for row in finite_rows], dtype=float
        )
        mean_gain = float(np.mean(deltas))
        positive_ratio = float(np.mean(deltas > 0.0))
        ll_std = float(np.std(lls, ddof=1))
        counts = {}
        for row in finite_rows:
            key = f'{float(row["confidence_threshold"]):.3f}::{float(row["retained_weight"]):.2f}'
            counts[key] = counts.get(key, 0) + 1
        modal_key = max(counts, key=lambda key: (counts[key], key))
        modal_threshold, modal_weight = modal_key.split("::")
        candidate = {
            "model": model_name,
            "parameter_mode": "fold_local_calibration_selection",
            "confidence_threshold_mode": float(modal_threshold),
            "retained_weight_mode": float(modal_weight),
            "parameter_selection_frequency": int(counts[modal_key]),
            "parameter_selection_counts": counts,
            "mean_logloss_improvement": mean_gain,
            "positive_fold_ratio": positive_ratio,
            "mean_logloss": float(np.mean(lls)),
            "logloss_std": ll_std,
            "selection_score": float(np.mean(lls) + 0.25 * ll_std),
            "folds": int(len(finite_rows)),
            "coverage": 1.0,
        }
        selective_candidates[model_name] = candidate
        if (
            model_name == global_selected
            and mean_gain >= min_selective_gain
            and positive_ratio >= min_selective_positive_ratio
            and (
                selective_selected["status"] == "NO_PROMOTION"
                or candidate["selection_score"]
                < float(selective_selected["selection_score"])
            )
        ):
            selective_selected = {**candidate, "status": "RESEARCH_CANDIDATE"}

    selective_probability_research = {
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "selected": selective_selected,
        "minimum_logloss_improvement": min_selective_gain,
        "minimum_positive_fold_ratio": min_selective_positive_ratio,
        "candidates": selective_candidates,
        "selection_protocol": "calibration_slice_selection_then_untouched_chronological_oos_test",
        "test_tuning_allowed": False,
        "method": "confidence_shrinkage_toward_fold_local_base_rate",
    }

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



    # Situation specialists are selected strictly from chronological OOS slices.
    # Global situation routes must beat the global model on the same situation
    # slice; asset+situation routes must beat the best non-situation parent on
    # the same asset+situation slice. Sparse situations never get promoted.
    def _eligible_situation_candidates(
        candidates: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        return {
            name: metric
            for name, metric in candidates.items()
            if int(metric.get("folds", 0)) >= 3
            and float(metric.get("n_test_min", 0.0)) >= 30.0
            and np.isfinite(float(metric.get("logloss", float("nan"))))
        }

    situation_selected = {}
    for situation_name, candidates in situation_metrics.items():
        parent_regime = regime_for_situation(situation_name)
        if parent_regime is None:
            continue
        eligible = _eligible_situation_candidates(candidates)
        if not eligible or global_selected not in eligible:
            continue
        plan = choose_from_oos(
            parent_regime,
            eligible,
            candidates=CANDIDATES.get(
                Regime(parent_regime),
                CANDIDATES[Regime.NORMAL],
            ),
            scope=f"situation:{situation_name}",
            min_folds=3,
            rank_ic_tiebreak_tolerance=rank_ic_tolerance,
        )
        if (
            not plan.reason.endswith("fallback")
            and materially_better_than_parent(
                eligible,
                plan.names[0],
                global_selected,
                min_improvement_logloss=scope_improvement,
            )
        ):
            situation_selected[str(situation_name)] = plan.names[0]

    asset_situation_selected = {}
    for key, candidates in asset_situation_metrics.items():
        asset_class, situation_name = key.split("::", 1)
        parent_regime = regime_for_situation(situation_name)
        if parent_regime is None:
            continue
        eligible = _eligible_situation_candidates(candidates)
        if not eligible:
            continue
        asset_regime_key = f"{asset_class}::{parent_regime}"
        parent_model = (
            asset_regime_selected.get(asset_regime_key)
            or asset_selected.get(asset_class)
            or regime_selected.get(parent_regime)
            or global_selected
        )
        if parent_model not in eligible:
            continue
        plan = choose_from_oos(
            parent_regime,
            eligible,
            candidates=ASSET_CANDIDATES.get(
                asset_class,
                CANDIDATES[Regime(parent_regime)],
            ),
            scope=f"asset_situation:{key}",
            min_folds=3,
            rank_ic_tiebreak_tolerance=rank_ic_tolerance,
        )
        if (
            not plan.reason.endswith("fallback")
            and materially_better_than_parent(
                eligible,
                plan.names[0],
                parent_model,
                min_improvement_logloss=scope_improvement,
            )
        ):
            asset_situation_selected[key] = plan.names[0]


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
        "situation_routes": int(len(situation_selected)),
        "asset_situation_routes": int(len(asset_situation_selected)),
        "symbol_routes": int(len(symbol_selected)),
        "symbol_regime_routes": int(len(symbol_regime_selected)),
        "total_security_routes": int(len(symbol_selected) + len(symbol_regime_selected)),
        "situation_route_examples": sorted(situation_selected)[:20],
        "asset_situation_route_examples": sorted(asset_situation_selected)[:20],
        "symbol_route_examples": sorted(symbol_selected)[:20],
        "symbol_regime_route_examples": sorted(symbol_regime_selected)[:20],
    }

    # Select classifier training-window length on chronological OOS after
    # model-family selection. 0 means all eligible history.
    window_candidates = (252, 504, 756, 0)
    window_metrics = {}
    window_fold_rows = {lookback: [] for lookback in window_candidates}
    for lookback in window_candidates:
        fold_rows = []
        for fold_idx, fold in enumerate(folds):
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
            metrics = classification_metrics(
                test.target_up_1d.astype(int), p
            )
            metrics["fold"] = float(fold_idx)
            metrics["drift"] = robust_distribution_shift_score(
                fit_rows[FEATURE_COLUMNS].to_numpy(dtype=float),
                test[FEATURE_COLUMNS].to_numpy(dtype=float),
            )
            fold_rows.append(metrics)
        window_fold_rows[lookback] = list(fold_rows)
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

    # Research-only drift-aware training-window routing. At fold t, the
    # router sees current-fold features (unlabeled) and chooses a window using
    # only prior OOS performance from similar drift conditions. Current-fold
    # outcomes are never used for the choice.
    window_cfg = pipeline_cfg.get("drift_aware_window", {})
    if window_cfg.get("research_only", True) is not True:
        raise SystemExit("FAIL: drift-aware window router must remain research-only")
    window_min_history = int(window_cfg.get("min_history_folds", 2))
    window_half_life = float(window_cfg.get("half_life_folds", 4.0))
    window_drift_scale = float(window_cfg.get("drift_scale", 0.50))
    window_stability_penalty = float(
        window_cfg.get("stability_penalty", 0.25)
    )
    dynamic_window_rows = []
    static_window_rows = []
    selected_window_by_fold = []
    window_history = {
        int(lookback): [
            row for row in window_fold_rows.get(lookback, [])
            if np.isfinite(float(row.get("fold", np.nan)))
        ]
        for lookback in window_candidates
    }
    common_folds = sorted({
        int(row["fold"])
        for rows in window_history.values()
        for row in rows
    })
    for fold_idx in common_folds:
        current_rows = {}
        current_drifts = {}
        for lookback in window_candidates:
            row = next(
                (
                    candidate for candidate in window_history[int(lookback)]
                    if int(candidate["fold"]) == fold_idx
                ),
                None,
            )
            if row is not None:
                current_rows[int(lookback)] = row
                current_drifts[int(lookback)] = float(row["drift"])
        prior_window_history = {
            lookback: [
                row for row in rows
                if int(row["fold"]) < fold_idx
            ]
            for lookback, rows in window_history.items()
        }
        selected_window, diagnostics = select_drift_aware_window(
            prior_window_history,
            fold_idx,
            current_drifts,
            min_history_folds=window_min_history,
            half_life_folds=window_half_life,
            drift_scale=window_drift_scale,
            stability_penalty=window_stability_penalty,
        )
        if selected_window is None or int(selected_training_window) not in current_rows:
            continue
        chosen_row = current_rows.get(int(selected_window))
        baseline_row = current_rows.get(int(selected_training_window))
        if chosen_row is None or baseline_row is None:
            continue
        dynamic_window_rows.append(chosen_row)
        static_window_rows.append(baseline_row)
        selected_window_by_fold.append({
            "fold": float(fold_idx),
            "selected_window": int(selected_window),
            "baseline_window": int(selected_training_window),
            "selection_score": float(diagnostics[int(selected_window)]["score"]),
            "current_drift": float(
                diagnostics[int(selected_window)]["current_drift"]
            ),
        })

    drift_aware_window_research = {
        "status": "INSUFFICIENT_OOS",
        "research_only": True,
        "method": "recent_oos_performance_conditioned_on_feature_drift",
        "selection_protocol": (
            "current-fold unlabeled feature distributions estimate drift; "
            "window selected only from prior chronological OOS losses with "
            "recency and drift similarity weighting; current outcomes enter "
            "history only after scoring"
        ),
        "min_history_folds": window_min_history,
        "half_life_folds": window_half_life,
        "drift_scale": window_drift_scale,
        "stability_penalty": window_stability_penalty,
    }
    if dynamic_window_rows:
        dynamic_agg = aggregate_group(dynamic_window_rows)
        static_agg = aggregate_group(static_window_rows)
        deltas = np.asarray(
            [
                float(base["logloss"] - dynamic["logloss"])
                for dynamic, base in zip(dynamic_window_rows, static_window_rows)
            ],
            dtype=float,
        )
        relative_improvement = float(
            (static_agg["logloss"] - dynamic_agg["logloss"])
            / max(abs(static_agg["logloss"]), 1e-9)
        )
        positive_share = float(np.mean(deltas > 0.0))
        bootstrap_probability = 0.0
        bootstrap_p05 = float("-inf")
        if len(deltas) >= 5 and np.isfinite(deltas).all():
            bootstrap_probability, bootstrap_p05 = moving_block_bootstrap_mean(
                deltas,
                n_bootstrap=4000,
                seed=20260925,
            )
        drift_aware_window_research.update({
            "status": "EVALUATED",
            "folds": len(dynamic_window_rows),
            "dynamic_oos": dynamic_agg,
            "static_baseline_oos": static_agg,
            "logloss_improvement": float(
                static_agg["logloss"] - dynamic_agg["logloss"]
            ),
            "relative_logloss_improvement": relative_improvement,
            "brier_improvement": float(
                static_agg["brier"] - dynamic_agg["brier"]
            ),
            "ece_change": float(
                dynamic_agg["ece"] - static_agg["ece"]
            ),
            "positive_fold_share": positive_share,
            "bootstrap_probability_improvement": bootstrap_probability,
            "bootstrap_p05_improvement": bootstrap_p05,
            "selected_window_by_fold": selected_window_by_fold,
            "research_positive": bool(
                len(deltas) >= 5
                and relative_improvement >= 0.03
                and positive_share >= 0.70
                and bootstrap_probability >= 0.90
                and bootstrap_p05 > 0.0
                and float(dynamic_agg["brier"] - static_agg["brier"]) <= 0.001
                and float(dynamic_agg["ece"] - static_agg["ece"]) <= 0.0
            ),
        })

    # Select the probability calibration method on chronological OOS after
    # model-family and training-window selection. Every candidate is fit only
    # on the fold's calibration slice, and the test slice is used only for
    # scoring. This keeps the calibration choice out of the frozen holdout.
    calibration_rows = {method: [] for method in CALIBRATION_METHODS}
    asset_calibration_rows = {method: [] for method in CALIBRATION_METHODS}

    # Research-only online calibration routing. A method is selected for fold
    # t using only calibration performance observed on folds < t; the current
    # test slice is scored after selection, then its outcomes enter history.
    calibration_cfg = pipeline_cfg.get("calibration", {})
    temporal_calibration_research_only = (
        calibration_cfg.get("temporal_router_research_only", True) is True
    )
    if not temporal_calibration_research_only:
        raise SystemExit("FAIL: temporal calibration router must remain research-only")
    temporal_min_history = int(
        calibration_cfg.get("temporal_min_history_folds", 2)
    )
    temporal_half_life = float(
        calibration_cfg.get("temporal_half_life_folds", 4.0)
    )
    temporal_stability_penalty = float(
        calibration_cfg.get("temporal_stability_penalty", 0.25)
    )
    temporal_calibration_history = {
        method: [] for method in CALIBRATION_METHODS
    }
    temporal_calibration_rows = []
    temporal_platt_rows = []

    for fold_idx, fold in enumerate(folds):
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

        # Candidate calibrators are all fit on the current pre-test
        # calibration slice, but the temporal router chooses among them from
        # prior OOS folds only.
        temporal_method = select_temporal_calibration_method(
            temporal_calibration_history,
            fold_idx,
            min_history_folds=temporal_min_history,
            half_life_folds=temporal_half_life,
            stability_penalty=temporal_stability_penalty,
            default_method="platt",
        )
        calibrated_test_probabilities = {}
        fold_candidate_metrics = {}
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
            calibrated_test_probabilities[method] = p
            metrics = classification_metrics(
                test.target_up_1d.astype(int),
                p,
            )
            fold_candidate_metrics[method] = metrics
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

        temporal_p = calibrated_test_probabilities[temporal_method]
        temporal_metrics = classification_metrics(
            test.target_up_1d.astype(int),
            temporal_p,
        )
        platt_metrics = fold_candidate_metrics["platt"]
        temporal_metrics["fold"] = float(fold_idx)
        temporal_metrics["method"] = temporal_method
        temporal_metrics["n_test"] = float(len(test))
        temporal_calibration_rows.append(temporal_metrics)
        temporal_platt_rows.append({
            **platt_metrics,
            "fold": float(fold_idx),
            "method": "platt",
            "n_test": float(len(test)),
        })

        # Current-fold candidate outcomes enter the routing history only after
        # this fold has been scored, preserving strict chronological OOS order.
        for method in CALIBRATION_METHODS:
            current = fold_candidate_metrics[method]
            temporal_calibration_history[method].append({
                "fold": float(fold_idx),
                "logloss": float(current["logloss"]),
                "ece": float(current["ece"]),
                "brier": float(current["brier"]),
            })

    temporal_calibration_research = {
        "status": "INSUFFICIENT_OOS",
        "research_only": True,
        "method": "expanding_recent_temporal_calibration_router",
        "selection_protocol": (
            "select method from prior chronological OOS calibration outcomes; "
            "fit selected method on current fold calibration slice; "
            "score untouched current fold test; only then update history"
        ),
        "min_history_folds": temporal_min_history,
        "half_life_folds": temporal_half_life,
        "stability_penalty": temporal_stability_penalty,
    }
    if temporal_calibration_rows:
        temporal_agg = aggregate_group(temporal_calibration_rows)
        platt_agg = aggregate_group(temporal_platt_rows)
        fold_delta = np.asarray(
            [
                float(base["logloss"] - dynamic["logloss"])
                for dynamic, base in zip(temporal_calibration_rows, temporal_platt_rows)
            ],
            dtype=float,
        )
        relative_improvement = float(
            (platt_agg["logloss"] - temporal_agg["logloss"])
            / max(abs(platt_agg["logloss"]), 1e-9)
        )
        positive_share = float(np.mean(fold_delta > 0.0))
        bootstrap_probability = 0.0
        bootstrap_p05 = float("-inf")
        if len(fold_delta) >= 5 and np.isfinite(fold_delta).all():
            bootstrap_probability, bootstrap_p05 = moving_block_bootstrap_mean(
                fold_delta,
                n_bootstrap=4000,
                seed=20260925,
            )
        temporal_calibration_research.update({
            "status": "EVALUATED",
            "folds": len(temporal_calibration_rows),
            "raw_oos": platt_agg,
            "temporal_oos": temporal_agg,
            "logloss_improvement": float(platt_agg["logloss"] - temporal_agg["logloss"]),
            "relative_logloss_improvement": relative_improvement,
            "brier_improvement": float(platt_agg["brier"] - temporal_agg["brier"]),
            "ece_change": float(temporal_agg["ece"] - platt_agg["ece"]),
            "positive_fold_share": positive_share,
            "bootstrap_probability_improvement": bootstrap_probability,
            "bootstrap_p05_improvement": bootstrap_p05,
            "selected_method_by_fold": [
                {"fold": float(row["fold"]), "method": str(row["method"])}
                for row in temporal_calibration_rows
            ],
            "research_positive": bool(
                len(fold_delta) >= 5
                and relative_improvement >= 0.03
                and positive_share >= 0.70
                and bootstrap_probability >= 0.90
                and bootstrap_p05 > 0.0
                and float(temporal_agg["brier"] - platt_agg["brier"]) <= 0.001
                and float(temporal_agg["ece"] - platt_agg["ece"]) <= 0.0
            ),
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
                {"platt": 0, "beta": 1, "isotonic": 2, "temperature": 3}[method],
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

    conformal_prediction_research = {
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "method": "split_conformal_classification_prediction_sets",
        "calibration_source": "raw_model_probability_on_fold_local_calibration_slice",
        "test_tuning_allowed": False,
        "alpha_candidates": {},
    }
    for model_name, by_alpha in conformal_rows_by_model_alpha.items():
        conformal_prediction_research["alpha_candidates"][model_name] = {}
        for alpha, rows in sorted(by_alpha.items()):
            conformal_prediction_research["alpha_candidates"][model_name][str(alpha)] = {
                "folds": len(rows),
                "mean_set_coverage": float(np.mean([r["set_coverage"] for r in rows])),
                "mean_set_size": float(np.mean([r["mean_set_size"] for r in rows])),
                "mean_singleton_rate": float(np.mean([r["singleton_rate"] for r in rows])),
                "mean_singleton_accuracy": float(np.nanmean([r["singleton_accuracy"] for r in rows])),
                "mean_empty_rate": float(np.mean([r["empty_rate"] for r in rows])),
                "mean_predicted_class_pvalue": float(np.mean([r["mean_predicted_class_pvalue"] for r in rows])),
                "fold_metrics": rows,
            }
    ablation_rows = confidence_risk_research.get("conformal_feature_ablation_rows", [])
    if ablation_rows:
        deltas = np.asarray(
            [r["raw_logloss"] - r["conformal_adjusted_logloss"] for r in ablation_rows],
            dtype=float,
        )
        conformal_prediction_research["risk_feature_ablation"] = {
            "folds": len(ablation_rows),
            "mean_logloss_improvement": float(np.mean(deltas)),
            "positive_fold_share": float(np.mean(deltas > 0.0)),
            "raw_logloss": float(np.mean([r["raw_logloss"] for r in ablation_rows])),
            "conformal_adjusted_logloss": float(np.mean([r["conformal_adjusted_logloss"] for r in ablation_rows])),
            "raw_brier": float(np.mean([r["raw_brier"] for r in ablation_rows])),
            "conformal_adjusted_brier": float(np.mean([r["conformal_adjusted_brier"] for r in ablation_rows])),
            "raw_ece": float(np.mean([r["raw_ece"] for r in ablation_rows])),
            "conformal_adjusted_ece": float(np.mean([r["conformal_adjusted_ece"] for r in ablation_rows])),
            "promotion_allowed": False,
        }

    adaptive_conformal_prediction_research = {
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "method": "session_batched_aci_style_adaptive_conformal_classification",
        "alpha": 0.10,
        "gamma": 0.02,
        "alpha_min": 0.01,
        "alpha_max": 0.50,
        "same_session_outcome_update": False,
        "models": {},
    }
    for model_name, rows in adaptive_conformal_rows_by_model.items():
        if not rows:
            continue
        adaptive_conformal_prediction_research["models"][model_name] = {
            "folds": len(rows),
            "mean_set_coverage": float(np.mean([r["set_coverage"] for r in rows])),
            "mean_set_size": float(np.mean([r["mean_set_size"] for r in rows])),
            "mean_singleton_rate": float(np.mean([r["singleton_rate"] for r in rows])),
            "mean_singleton_accuracy": float(np.nanmean([r["singleton_accuracy"] for r in rows])),
            "mean_empty_rate": float(np.mean([r["empty_rate"] for r in rows])),
            "mean_alpha_used": float(np.mean([r["mean_alpha_used"] for r in rows])),
            "mean_final_alpha": float(np.mean([r["final_alpha"] for r in rows])),
            "situation_metrics": {
                situation_name: {
                    "folds": len(srows),
                    "n_test_min": float(min(r["n_test"] for r in srows)),
                    "coverage": float(np.mean([r["coverage"] for r in srows])),
                    "mean_set_size": float(np.mean([r["mean_set_size"] for r in srows])),
                    "singleton_rate": float(np.mean([r["singleton_rate"] for r in srows])),
                    "singleton_accuracy": float(np.nanmean([r["singleton_accuracy"] for r in srows])),
                    "empty_rate": float(np.mean([r["empty_rate"] for r in srows])),
                    "mean_alpha_used": float(np.mean([r["mean_alpha_used"] for r in srows])),
                }
                for situation_name, srows in adaptive_conformal_situation_rows_by_model.get(model_name, {}).items()
            },
            "fold_metrics": rows,
        }

    group_conformal_prediction_research = {
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "method": "asset_class_conditional_split_conformal_classification",
        "min_group_size": 50,
        "alpha": 0.10,
        "models": {},
    }
    for model_name, rows in group_conformal_rows_by_model.items():
        if not rows:
            continue
        group_conformal_prediction_research["models"][model_name] = {
            "folds": len(rows),
            "mean_set_coverage": float(np.mean([r["set_coverage"] for r in rows])),
            "mean_set_size": float(np.mean([r["mean_set_size"] for r in rows])),
            "mean_singleton_rate": float(np.mean([r["singleton_rate"] for r in rows])),
            "mean_singleton_accuracy": float(np.nanmean([r["singleton_accuracy"] for r in rows])),
            "mean_empty_rate": float(np.mean([r["empty_rate"] for r in rows])),
            "mean_fallback_rate": float(np.mean([r["fallback_rate"] for r in rows])),
            "fold_metrics": rows,
        }

    payload = {
        "results": model_results,
        "return_oos": return_oos,
        "adaptive_conformal_prediction_research": adaptive_conformal_prediction_research,
        "group_conformal_prediction_research": group_conformal_prediction_research,
        "conformal_prediction_research": conformal_prediction_research,
        "regime_metrics": regime_metrics,
        "situation_metrics": situation_metrics,
        "asset_situation_metrics": asset_situation_metrics,
        "situation_selected_models": situation_selected,
        "asset_situation_selected_models": asset_situation_selected,
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
        "global_selection_evidence": global_selection_evidence,
        "sequential_selection_research": sequential_selection_research,
        "nested_sequential_selection_research": nested_selection_research,
        "classifier_training_window_sessions": selected_training_window,
        "classifier_training_window_candidates": window_metrics,
        "calibration_method": selected_calibration_method,
        "calibration_method_candidates": calibration_candidates,
        "rank_probability_weight": selected_rank_weight,
        "rank_uncertainty_penalty": selected_uncertainty_penalty,
        "minimum_scoped_oos_improvement_logloss": scope_improvement,
        "ranking_weight_candidates": ranking_candidates,
        "selective_probability_research": selective_probability_research,
        "online_expert_research": online_expert_research,
        "online_expert_balanced_research": online_expert_balanced_research,
        "confidence_risk_research": confidence_risk_research,
        "temporal_calibration_research": temporal_calibration_research,
        "drift_aware_window_research": drift_aware_window_research,
        "global_selection_candidates": balanced_candidates,
        "regime_vol_threshold": global_vol_threshold,
        "regime_vol_threshold_source": "oos_fold_train_median",
        "regime_vol_threshold_folds": len(oos_regime_thresholds),
        "selection_basis": (
            "chronological walk-forward OOS only; global selection blends "
            "row-weighted LogLoss with a configurable macro asset-class blend "
            "to reduce universe-size dominance, then applies the 0.25 stability "
            "penalty; route hierarchy is symbol_regime/symbol -> asset_situation/"
            "situation -> asset_class+regime -> asset_class -> regime -> global; "
            "situation routes require >=3 chronological OOS folds and >=30 rows/fold, "
            "and must clear the same stable parent edge as other scoped routes; "
            "security routes requiring "
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