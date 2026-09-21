from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.features.context import add_cross_sectional_context, add_market_context
from src.features.technical import FEATURE_COLUMNS, add_technical_features
from src.prediction.targets import add_targets
from src.research.metrics import aggregate_metric_rows, classification_metrics
from src.research.router import (
    ASSET_CANDIDATES,
    CANDIDATES,
    Regime,
    asset_plan,
    choose_from_oos,
)
from src.validation.calibration import PlattCalibrator
from src.validation.leakage import audit_feature_columns, audit_target_separation
from src.validation.walk_forward import make_date_folds

PRICE_DIR = Path("data/prices")
OUT = Path("data/research/latest_metrics.json")
AUDIT = Path("data/research/leakage_audit.json")


def make_models():
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
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    }


def aggregate_group(rows: list[dict[str, float]]) -> dict[str, float]:
    payload = aggregate_metric_rows(rows)
    payload["folds"] = float(len(rows))
    for key in ("logloss", "brier", "ece", "accuracy", "roc_auc"):
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

    df = df.dropna(subset=FEATURE_COLUMNS + ["target_up_1d"]).copy()

    frozen_path = Path("config/frozen_holdout.json")
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        cutoff = pd.Timestamp(frozen["cutoff_date"]).date()
        df = df[pd.to_datetime(df["session_date"]).dt.date <= cutoff].copy()

    dates = sorted(pd.to_datetime(df["session_date"]).dt.date.unique())
    folds = make_date_folds(
        dates,
        min_train=252,
        test_size=21,
        step=21,
        embargo=1,
    )
    if len(folds) < 3:
        raise SystemExit(f"DEFERRED: only {len(folds)} OOS folds available")

    model_results = {}
    regime_rows = {reg.value: [] for reg in Regime if reg is not Regime.DATA_STRESSED}
    asset_rows: dict[str, list[tuple[str, dict[str, float]]]] = {}
    asset_regime_rows: dict[str, list[tuple[str, dict[str, float]]]] = {}

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
            test = df[df.session_date.isin(test_dates)]

            if min(len(core), len(cal), len(test)) < 50:
                continue
            if (
                core.target_up_1d.nunique() < 2
                or cal.target_up_1d.nunique() < 2
                or test.target_up_1d.nunique() < 2
            ):
                continue

            threshold = (
                float(core["volatility_20"].dropna().quantile(0.75))
                if core["volatility_20"].notna().any()
                else 0.02
            )

            model = factory()
            model.fit(core[FEATURE_COLUMNS], core.target_up_1d.astype(int))
            cal_p = model.predict_proba(cal[FEATURE_COLUMNS])[:, 1]
            calibrator = PlattCalibrator().fit(
                cal_p, cal.target_up_1d.astype(int)
            )
            p = calibrator.predict(
                model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            )

            row = classification_metrics(test.target_up_1d.astype(int), p)
            row["n_test"] = float(len(test))
            fold_rows.append(row)

            regime = test.apply(
                lambda x: (
                    "normal"
                    if x["volatility_20"] < threshold
                    and abs(x["price_vs_sma60"]) < 0.02
                    else "high_vol"
                    if x["volatility_20"] >= threshold
                    else "trend"
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
                        kr["n_test"] = float(route_mask.sum())
                        key = f"{asset_class}::{reg_name}"
                        asset_regime_rows.setdefault(key, []).append((name, kr))

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

    regime_selected = {}
    for reg_name, candidates in regime_metrics.items():
        if candidates:
            plan = choose_from_oos(reg_name, candidates)
            if not plan.reason.endswith("fallback"):
                regime_selected[reg_name] = plan.names[0]

    asset_selected = {}
    for asset_class, candidates in asset_class_metrics.items():
        if candidates:
            plan = asset_plan(asset_class, candidates)
            if not plan.reason.endswith("fallback"):
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
            )
            if not plan.reason.endswith("fallback"):
                asset_regime_selected[key] = plan.names[0]

    global_candidates = {
        name: dict(value["metrics"], folds=float(value["folds"]))
        for name, value in usable.items()
    }
    global_plan = choose_from_oos("normal", global_candidates)
    global_selected = global_plan.names[0]

    payload = {
        "results": model_results,
        "regime_metrics": regime_metrics,
        "regime_selected_models": regime_selected,
        "asset_class_metrics": asset_class_metrics,
        "asset_class_selected_models": asset_selected,
        "asset_regime_metrics": asset_regime_metrics,
        "asset_regime_selected_models": asset_regime_selected,
        "selected_model": global_selected,
        "selection_basis": (
            "chronological walk-forward OOS only; selection penalizes unstable "
            "fold-to-fold LogLoss; route hierarchy is asset_class+regime -> "
            "asset_class -> regime -> global; calibration is fit inside each "
            "OOS training fold; frozen holdout remains unused during selection"
        ),
        "status": "OOS_COMPLETE",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
