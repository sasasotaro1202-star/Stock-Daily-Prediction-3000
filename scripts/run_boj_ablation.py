from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.context import add_cross_sectional_context, add_market_context
from src.features.technical import FEATURE_COLUMNS, add_technical_features
from src.prediction.fit import fit_classifier
from src.prediction.model_factories import models
from src.prediction.targets import add_targets
from src.research.boj_features import BOJ_FEATURE_COLUMNS, add_boj_context
from src.research.metrics import classification_metrics
from src.validation.walk_forward import make_date_folds
from src.validation.leakage import audit_feature_columns, audit_target_separation


PRICE = Path("data/prices/canonical.parquet")
CONTEXT = Path("data/market_context.parquet")
BOJ = Path("data/research/boj_timeseries.parquet")
OUT = Path("data/research/boj_ablation.json")


def _prepare() -> tuple[pd.DataFrame, list[str]]:
    if not PRICE.exists():
        raise SystemExit("DEFERRED: canonical price dataset missing")
    if not CONTEXT.exists():
        raise SystemExit("DEFERRED: market context missing")
    if not BOJ.exists():
        raise SystemExit("DEFERRED: BOJ artifact missing")

    prices = pd.read_parquet(PRICE)
    context = pd.read_parquet(CONTEXT)
    boj = pd.read_parquet(BOJ)
    frame = add_technical_features(prices)
    frame = add_market_context(frame, context)
    frame = add_cross_sectional_context(frame)
    frame = add_boj_context(frame, boj, min_history=20)
    frame = add_targets(frame)

    base_features = list(FEATURE_COLUMNS)
    augmented_features = base_features + BOJ_FEATURE_COLUMNS
    fa = audit_feature_columns(base_features)
    ts = audit_target_separation(base_features, [c for c in frame if c.startswith("target_")])
    if not fa.ok or not ts.ok:
        raise SystemExit("FAIL: baseline leakage audit")
    if not set(BOJ_FEATURE_COLUMNS).issubset(frame.columns):
        raise SystemExit("FAIL: BOJ feature columns missing")
    return frame, augmented_features


def _run_variant(
    frame: pd.DataFrame,
    feature_columns: list[str],
    folds,
    model_name: str = "logistic",
) -> dict[str, object]:
    factory = models().get(model_name)
    if factory is None:
        raise SystemExit("FAIL: logistic model factory unavailable")
    rows = []
    for fold in folds:
        dates = sorted(pd.to_datetime(frame["session_date"]).dt.date.unique())
        train_dates = dates[: fold.train_end]
        cal_n = max(20, int(len(train_dates) * 0.2))
        core_dates = set(train_dates[:-cal_n])
        cal_dates = set(train_dates[-cal_n:])
        test_dates = set(dates[fold.test_start : fold.test_end])
        needed = feature_columns + ["target_up_1d", "session_date"]
        core = frame[frame.session_date.isin(core_dates)].dropna(subset=needed)
        cal = frame[frame.session_date.isin(cal_dates)].dropna(subset=needed)
        test = frame[frame.session_date.isin(test_dates)].dropna(subset=needed)
        if min(len(core), len(cal), len(test)) < 100:
            continue
        if min(
            core["target_up_1d"].nunique(),
            cal["target_up_1d"].nunique(),
            test["target_up_1d"].nunique(),
        ) < 2:
            continue
        model = factory()
        fit_classifier(
            model,
            model_name,
            core[feature_columns],
            core["target_up_1d"].astype(int),
            core["session_date"],
            half_life_sessions=252,
        )
        cal_p = model.predict_proba(cal[feature_columns])[:, 1]
        test_p = model.predict_proba(test[feature_columns])[:, 1]
        from src.validation.calibration import make_calibrator
        calibrator = make_calibrator("platt").fit(
            cal_p, cal["target_up_1d"].astype(int)
        )
        p = calibrator.predict(test_p)
        m = classification_metrics(test["target_up_1d"].astype(int), p)
        m["n_test"] = float(len(test))
        rows.append(m)

    if len(rows) < 3:
        return {"status": "DEFERRED", "folds": len(rows)}
    metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("accuracy", "logloss", "brier", "ece", "roc_auc")
    }
    return {
        "status": "OOS_COMPLETE",
        "folds": len(rows),
        "metrics": metrics,
        "fold_rows": rows,
    }


def main() -> None:
    frame, augmented = _prepare()
    dates = sorted(pd.to_datetime(frame["session_date"]).dt.date.unique())
    folds = make_date_folds(
        dates,
        min_train=252,
        test_size=21,
        step=21,
        embargo=1,
        purge=1,
    )
    if len(folds) < 5:
        raise SystemExit(f"DEFERRED: only {len(folds)} folds")
    base = _run_variant(frame, list(FEATURE_COLUMNS), folds)
    boj = _run_variant(frame, augmented, folds)
    if base.get("status") != "OOS_COMPLETE" or boj.get("status") != "OOS_COMPLETE":
        raise SystemExit(
            "DEFERRED: both baseline and BOJ variants need >=3 valid OOS folds"
        )
    bm = base["metrics"]
    am = boj["metrics"]
    delta = {
        key: float(am[key] - bm[key])
        for key in ("accuracy", "logloss", "brier", "ece", "roc_auc")
    }
    result = {
        "status": "OOS_COMPLETE",
        "research_only": True,
        "production_changed": False,
        "model": "logistic",
        "feature_added": BOJ_FEATURE_COLUMNS,
        "folds_total": len(folds),
        "baseline": base,
        "boj_augmented": boj,
        "delta_augmented_minus_baseline": delta,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
