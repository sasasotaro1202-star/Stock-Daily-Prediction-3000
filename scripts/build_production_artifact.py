from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import numpy
import sklearn
import importlib.metadata as metadata

try:
    import lightgbm
except ImportError:
    lightgbm = None

from src.features.context import add_cross_sectional_context, add_market_context
from src.features.technical import FEATURE_COLUMNS, add_technical_features
from src.prediction.model_factories import models
from src.prediction.fit import fit_classifier
from src.prediction.production_artifact import ARTIFACT_PATH, release_signature
from src.prediction.regression import make_quantile_models, make_return_model
from src.prediction.targets import add_targets
from src.validation.calibration import make_calibrator
from src.validation.code_fingerprint import fingerprint_sha256, research_fingerprint_sha256
from src.validation.training_sample import cap_training_rows
from src.validation.training_window import restrict_to_lookback

PRICE = Path("data/prices")
METRICS = Path("data/research/latest_metrics.json")
GATE = Path("data/research/release_gate.json")
FROZEN = Path("config/frozen_holdout.json")


def production_eligible(df: pd.DataFrame) -> pd.DataFrame:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    dates = pd.to_datetime(df["session_date"], errors="coerce").dt.date
    cutoff = pd.Timestamp(frozen["cutoff_date"]).date()
    holdout_end = pd.Timestamp(frozen["holdout_end"]).date()
    return df.loc[(dates <= cutoff) | (dates > holdout_end)].copy()


def split_train_cal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(df["session_date"]).dt.date.unique())
    if len(dates) < 40:
        raise ValueError("insufficient chronological dates for calibration split")
    cal_n = max(20, int(len(dates) * 0.2))
    core = df[df.session_date.isin(set(dates[:-cal_n]))]
    cal = df[df.session_date.isin(set(dates[-cal_n:]))]
    if len(core) < 200 or len(cal) < 100:
        raise ValueError("insufficient rows for stable production fit")
    if core.target_up_1d.nunique() < 2 or cal.target_up_1d.nunique() < 2:
        raise ValueError("production calibration split lacks both target classes")
    return core, cal


def main():
    if not PRICE.exists() or not METRICS.exists() or not GATE.exists() or not FROZEN.exists():
        raise SystemExit("DEFERRED: production research/release state is missing")

    gate = json.loads(GATE.read_text(encoding="utf-8"))
    if not gate.get("approved", False):
        raise SystemExit(f"DEFERRED: release gate not approved: {gate.get('reasons', [])}")

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise SystemExit("DEFERRED: frozen model routing is not locked")

    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    if metrics.get("selected_model") != frozen.get("selected_model"):
        raise SystemExit("DEFERRED: frozen and research selected models differ")
    for key in (
        "regime_selected_models",
        "asset_class_selected_models",
        "asset_regime_selected_models",
        "situation_selected_models",
        "asset_situation_selected_models",
        "symbol_selected_models",
        "symbol_regime_selected_models",
        "rank_probability_weight",
        "rank_uncertainty_penalty",
        "classifier_training_window_sessions",
    ):
        if key in frozen and key in metrics and frozen[key] != metrics[key]:
            raise SystemExit(f"DEFERRED: frozen/research mismatch for {key}")
    df = pd.read_parquet(PRICE)
    context_path = Path("data/market_context.parquet")
    if not context_path.exists():
        raise SystemExit("DEFERRED: market context is absent")
    context = pd.read_parquet(context_path)
    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce").dt.date
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["available_at"].le(now)].copy()

    df = add_technical_features(df)
    df = add_market_context(df, context)
    df = add_cross_sectional_context(df)
    labeled = add_targets(df).dropna(
        subset=FEATURE_COLUMNS + ["target_up_1d", "target_ret_1d"]
    ).copy()
    labeled = production_eligible(labeled)
    if len(labeled) < 5000:
        raise SystemExit("DEFERRED: insufficient PIT-safe production training data")

    threshold = frozen.get("regime_vol_threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(float(threshold)):
        raise SystemExit("DEFERRED: frozen regime volatility threshold is missing")
    threshold = float(threshold)

    calibration_method = str(frozen.get("calibration_method", "platt"))
    if calibration_method not in {"platt", "beta", "isotonic", "temperature"}:
        raise SystemExit("DEFERRED: frozen calibration method is invalid")

    training_window = int(
        frozen.get("classifier_training_window_sessions", 0)
    )
    if training_window < 0:
        raise SystemExit("DEFERRED: frozen classifier training window is invalid")
    training_labeled = restrict_to_lookback(
        labeled,
        None if training_window == 0 else training_window,
    )
    core, cal = split_train_cal(training_labeled)
    core_fit = cap_training_rows(
        core,
        max_rows=300_000,
        recent_sessions=min(252, training_window or 252),
    )

    required_classifiers = {"hgb", str(frozen.get("selected_model", "hgb"))}
    for key in (
        "regime_selected_models",
        "asset_class_selected_models",
        "asset_regime_selected_models",
        "situation_selected_models",
        "asset_situation_selected_models",
        "symbol_selected_models",
        "symbol_regime_selected_models",
    ):
        required_classifiers.update(
            str(value) for value in (frozen.get(key) or {}).values()
        )
    classifiers = {}
    available_factories = models()
    missing_required = sorted(set(required_classifiers) - set(available_factories))
    if missing_required:
        raise SystemExit(
            f"DEFERRED: required production classifiers unavailable: {missing_required}"
        )
    pipeline_cfg = __import__("yaml").safe_load(
        Path("config/pipeline.yml").read_text(encoding="utf-8")
    )
    model_cfg = pipeline_cfg.get("models", {})
    recency_half_life = int(model_cfg.get("recency_weight_half_life_sessions", 252))

    for name in sorted(required_classifiers):
        factory = available_factories[name]
        model = factory()
        fit_classifier(
            model,
            name,
            core_fit[FEATURE_COLUMNS],
            core_fit["target_up_1d"].astype(int),
            core_fit["session_date"],
            half_life_sessions=recency_half_life,
        )
        cal_p = model.predict_proba(cal[FEATURE_COLUMNS])[:, 1]
        calibrator = make_calibrator(calibration_method).fit(
            cal_p,
            cal["target_up_1d"].astype(int),
        )
        classifiers[name] = {"model": model, "calibrator": calibrator}

    return_selected = frozen.get("return_selected_estimator", "q50")
    if return_selected not in {"mean", "q50", "blend_mean_q50"}:
        raise SystemExit("FAIL: frozen return estimator is invalid")
    mean_return_model = make_return_model()
    q_global = make_quantile_models()
    q_fit = cap_training_rows(labeled, max_rows=250_000, recent_sessions=252)
    mean_return_model.fit(q_fit[FEATURE_COLUMNS], q_fit["target_ret_1d"])
    for model in q_global.values():
        model.fit(q_fit[FEATURE_COLUMNS], q_fit["target_ret_1d"])

    q_assets = {}
    for asset, subset in labeled.groupby("asset_class", sort=False):
        if len(subset) < 750:
            continue
        q_models = make_quantile_models()
        subset_fit = cap_training_rows(
            subset, max_rows=200_000, recent_sessions=252
        )
        for model in q_models.values():
            model.fit(subset_fit[FEATURE_COLUMNS], subset_fit["target_ret_1d"])
        q_assets[str(asset)] = q_models

    payload = {
        "metadata": {
            "artifact_version": 1,
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "git_sha": os.getenv("GITHUB_SHA"),
            "code_fingerprint_sha256": fingerprint_sha256(),
            "research_code_fingerprint_sha256": research_fingerprint_sha256(),
            "release_signature": release_signature(),
            "feature_columns": list(FEATURE_COLUMNS),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "numpy_version": numpy.__version__,
            "sklearn_version": sklearn.__version__,
            "lightgbm_version": lightgbm.__version__ if lightgbm is not None else None,
            "runtime_dependency_versions": {
                name: metadata.version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "scipy",
                    "pyarrow",
                    "yfinance",
                    "PyYAML",
                    "curl_cffi",
                    "lightgbm",
                )
                if name != "lightgbm" or lightgbm is not None
            },
            "available_classifiers": sorted(classifiers),
            "required_classifiers": sorted(required_classifiers),
            "regime_vol_threshold": threshold,
            "rank_probability_weight": float(frozen.get("rank_probability_weight", 0.50)),
            "rank_uncertainty_penalty": float(frozen.get("rank_uncertainty_penalty", 0.0)),
            "situation_selected_models": dict(frozen.get("situation_selected_models") or {}),
            "asset_situation_selected_models": dict(frozen.get("asset_situation_selected_models") or {}),
            "symbol_selected_models": dict(frozen.get("symbol_selected_models") or {}),
            "symbol_regime_selected_models": dict(frozen.get("symbol_regime_selected_models") or {}),
            "selected_model": metrics.get("selected_model"),
            "return_selected_estimator": return_selected,
            "calibration_method": calibration_method,
            "classifier_training_window_sessions": training_window,
            "training_rows": int(len(training_labeled)),
            "training_latest_session": str(max(training_labeled["session_date"])),
        },
        "classifiers": classifiers,
        "return": {
            "selected": return_selected,
            "global": {
                "mean": mean_return_model,
                "q50": q_global["q50"],
            },
        },
        "quantile": {
            "global": q_global,
            "assets": q_assets,
        },
    }

    if not payload["metadata"]["selected_model"]:
        raise SystemExit("FAIL: selected production model is missing")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    Path("data/research/production_model_artifact.meta.json").write_text(
        json.dumps(payload["metadata"], indent=2),
        encoding="utf-8",
    )

    print(
        f"production-model-artifact: rows={len(labeled)} "
        f"latest={max(labeled['session_date'])} "
        f"selected={payload['metadata']['selected_model']} "
        f"assets={sorted(q_assets)}"
    )


if __name__ == "__main__":
    main()
