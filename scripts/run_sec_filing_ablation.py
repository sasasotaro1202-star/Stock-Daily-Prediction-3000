from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.context import add_cross_sectional_context, add_market_context
from src.features.sec_features import SEC_FEATURE_COLUMNS, add_sec_filing_features
from src.features.technical import FEATURE_COLUMNS, add_technical_features
from src.prediction.fit import fit_classifier
from src.prediction.model_factories import models
from src.prediction.targets import add_targets
from src.research.metrics import classification_metrics
from src.validation.training_sample import cap_training_rows
from src.validation.training_window import restrict_to_lookback
from src.validation.walk_forward import make_date_folds

PRICE = Path("data/prices/canonical.parquet")
SEC = Path("data/research/sec_filings_research.parquet")
METRICS = Path("data/research/latest_metrics.json")
FROZEN = Path("config/frozen_holdout.json")
OUT = Path("data/research/sec_filing_ablation.json")


def _deferred(reason: str) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "status": "DEFERRED",
                "reason": reason,
                "research_only": True,
                "production_changed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT.read_text(encoding="utf-8"))


def main() -> None:
    if not PRICE.exists():
        _deferred("canonical_price_dataset_missing")
        return
    if not SEC.exists():
        _deferred("sec_filings_dataset_missing")
        return
    if not METRICS.exists() or not FROZEN.exists():
        _deferred("current_oos_or_frozen_state_missing")
        return

    try:
        sec = pd.read_parquet(SEC)
        metrics = json.loads(METRICS.read_text(encoding="utf-8"))
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _deferred(f"state_read_failed:{type(exc).__name__}")
        return

    if sec.empty:
        _deferred("sec_dataset_empty")
        return
    if str(metrics.get("status")) != "OOS_COMPLETE":
        _deferred("direction_oos_not_complete")
        return

    selected = str(metrics.get("selected_model", "")).strip()
    if not selected:
        _deferred("selected_model_missing")
        return
    factory = models().get(selected)
    if factory is None:
        _deferred(f"selected_model_unavailable:{selected}")
        return

    df = pd.read_parquet(PRICE)
    df["session_date"] = pd.to_datetime(
        df["session_date"], errors="coerce"
    ).dt.date
    df["available_at"] = pd.to_datetime(
        df["available_at"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["session_date", "available_at"]).copy()
    market_context_path = Path("data/market_context.parquet")
    if not market_context_path.exists():
        _deferred("market_context_missing")
        return
    context = pd.read_parquet(market_context_path)
    df = add_technical_features(df)
    df = add_market_context(df, context)
    df = add_cross_sectional_context(df)
    df = add_sec_filing_features(df, sec)
    df = add_targets(df)

    cutoff_value = frozen.get("cutoff_date")
    if not cutoff_value:
        _deferred("frozen_cutoff_missing")
        return
    cutoff = pd.Timestamp(cutoff_value).date()
    df = df[df["session_date"] <= cutoff].copy()

    enhanced_columns = list(FEATURE_COLUMNS) + list(SEC_FEATURE_COLUMNS)
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_up_1d"]).copy()
    if len(df) < 5000:
        _deferred("insufficient_pit_safe_rows")
        return

    dates = sorted(pd.to_datetime(df["session_date"]).dt.date.unique())
    cfg = yaml.safe_load(
        Path("config/pipeline.yml").read_text(encoding="utf-8")
    )
    model_cfg = cfg.get("models", {})
    folds = make_date_folds(
        dates,
        min_train=252,
        test_size=21,
        step=21,
        embargo=1,
        purge=int(model_cfg.get("purge_sessions", 1)),
    )
    if len(folds) < 3:
        _deferred(f"insufficient_oos_folds:{len(folds)}")
        return

    lookback = int(metrics.get("classifier_training_window_sessions", 0))
    rows = []
    for fold_index, fold in enumerate(folds, start=1):
        train_dates = dates[: fold.train_end]
        chosen_train_dates = (
            train_dates if lookback == 0 else train_dates[-lookback:]
        )
        cal_n = max(20, int(len(chosen_train_dates) * 0.2))
        if len(chosen_train_dates) - cal_n < 40:
            continue
        core_dates = set(chosen_train_dates[:-cal_n])
        cal_dates = set(chosen_train_dates[-cal_n:])
        test_dates = set(dates[fold.test_start : fold.test_end])
        core = df[df["session_date"].isin(core_dates)].copy()
        cal = df[df["session_date"].isin(cal_dates)].copy()
        test = df[df["session_date"].isin(test_dates)].copy()
        if min(len(core), len(cal), len(test)) < 100:
            continue
        if (
            core["target_up_1d"].nunique() < 2
            or cal["target_up_1d"].nunique() < 2
            or test["target_up_1d"].nunique() < 2
        ):
            continue

        fit_core = restrict_to_lookback(
            core,
            None if lookback == 0 else lookback,
        )
        fit_core = cap_training_rows(
            fit_core,
            max_rows=300_000,
            recent_sessions=min(252, lookback or 252),
        )

        baseline_model = factory()
        sec_model = factory()
        fit_classifier(
            baseline_model,
            selected,
            fit_core[FEATURE_COLUMNS],
            fit_core["target_up_1d"].astype(int),
            fit_core["session_date"],
            half_life_sessions=int(
                model_cfg.get("recency_weight_half_life_sessions", 252)
            ),
        )
        fit_classifier(
            sec_model,
            selected,
            fit_core[enhanced_columns],
            fit_core["target_up_1d"].astype(int),
            fit_core["session_date"],
            half_life_sessions=int(
                model_cfg.get("recency_weight_half_life_sessions", 252)
            ),
        )

        baseline_cal = baseline_model.predict_proba(cal[FEATURE_COLUMNS])[:, 1]
        sec_cal = sec_model.predict_proba(cal[enhanced_columns])[:, 1]
        from src.validation.calibration import make_calibrator

        baseline_calibrator = make_calibrator("platt").fit(
            baseline_cal,
            cal["target_up_1d"].astype(int),
        )
        sec_calibrator = make_calibrator("platt").fit(
            sec_cal,
            cal["target_up_1d"].astype(int),
        )

        base_p = np.clip(
            baseline_calibrator.predict(
                baseline_model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            ),
            1e-5,
            1 - 1e-5,
        )
        sec_p = np.clip(
            sec_calibrator.predict(
                sec_model.predict_proba(test[enhanced_columns])[:, 1]
            ),
            1e-5,
            1 - 1e-5,
        )
        y = test["target_up_1d"].astype(int)
        base = classification_metrics(y, base_p)
        enhanced = classification_metrics(y, sec_p)
        rows.append(
            {
                "fold": fold_index,
                "n_test": int(len(test)),
                "baseline_logloss": base["logloss"],
                "sec_logloss": enhanced["logloss"],
                "logloss_delta_improvement": base["logloss"] - enhanced["logloss"],
                "baseline_brier": base["brier"],
                "sec_brier": enhanced["brier"],
                "brier_delta_improvement": base["brier"] - enhanced["brier"],
                "baseline_ece": base["ece"],
                "sec_ece": enhanced["ece"],
            }
        )

    if len(rows) < 3:
        _deferred(f"valid_oos_folds_too_few:{len(rows)}")
        return

    deltas = np.asarray(
        [float(row["logloss_delta_improvement"]) for row in rows],
        dtype=float,
    )
    brier_deltas = np.asarray(
        [float(row["brier_delta_improvement"]) for row in rows],
        dtype=float,
    )
    mean_delta = float(np.mean(deltas))
    positive_folds = int(np.sum(deltas > 0.0))
    # Keep promotion deliberately conservative. This ablation is only a
    # candidate; production code is unchanged unless a future release gate
    # explicitly validates the feature on a fresh frozen holdout.
    recommended = bool(
        mean_delta >= 0.002
        and positive_folds >= max(3, int(np.ceil(len(rows) * 0.67)))
    )

    payload = {
        "status": "OOS_COMPLETE",
        "selected_model": selected,
        "folds": rows,
        "summary": {
            "mean_logloss_delta_improvement": mean_delta,
            "median_logloss_delta_improvement": float(np.median(deltas)),
            "positive_folds": positive_folds,
            "mean_brier_delta_improvement": float(np.mean(brier_deltas)),
            "logloss_delta_std": (
                float(np.std(deltas, ddof=1)) if len(deltas) >= 2 else 0.0
            ),
        },
        "feature_columns_added": list(SEC_FEATURE_COLUMNS),
        "recommended_for_further_holdout": recommended,
        "research_only": True,
        "production_changed": False,
        "selection_note": (
            "same selected model and chronological folds as current OOS; "
            "SEC features admitted only at acceptance/available_at <= price available_at"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
