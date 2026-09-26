from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.context import add_cross_sectional_context, add_market_context
from src.research.event_intelligence import (
    EVENT_INTELLIGENCE_FEATURE_COLUMNS,
    add_event_intelligence_features,
    build_event_ledger,
)
from src.research.sec_features import SEC_FEATURE_COLUMNS, add_sec_filing_features
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


def _fit_and_predict(
    model_name: str,
    factory,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    model_cfg: dict,
) -> tuple[np.ndarray, dict]:
    model = factory()
    fit_classifier(
        model,
        model_name,
        train[columns],
        train["target_up_1d"].astype(int),
        train["session_date"],
        half_life_sessions=int(
            model_cfg.get("recency_weight_half_life_sessions", 252)
        ),
    )
    raw_cal = model.predict_proba(calibration[columns])[:, 1]
    from src.validation.calibration import make_calibrator

    calibrator = make_calibrator("platt").fit(
        raw_cal,
        calibration["target_up_1d"].astype(int),
    )
    p = np.clip(
        calibrator.predict(model.predict_proba(test[columns])[:, 1]),
        1e-5,
        1 - 1e-5,
    )
    return p, classification_metrics(test["target_up_1d"].astype(int), p)


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

    event_ledger, event_diag = build_event_ledger(sec)
    if event_diag.get("status") != "PASS":
        _deferred(
            f"event_ledger_not_pit_safe:{event_diag.get('reason') or 'unknown'}"
        )
        return
    if event_ledger.empty:
        _deferred("event_ledger_empty")
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
    df = add_event_intelligence_features(df, event_ledger)
    df = add_targets(df)

    cutoff_value = frozen.get("cutoff_date")
    if not cutoff_value:
        _deferred("frozen_cutoff_missing")
        return
    cutoff = pd.Timestamp(cutoff_value).date()
    df = df[df["session_date"] <= cutoff].copy()

    baseline_columns = list(FEATURE_COLUMNS)
    sec_columns = baseline_columns + list(SEC_FEATURE_COLUMNS)
    event_columns = sec_columns + list(EVENT_INTELLIGENCE_FEATURE_COLUMNS)
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
            core, None if lookback == 0 else lookback
        )
        fit_core = cap_training_rows(
            fit_core,
            max_rows=300_000,
            recent_sessions=min(252, lookback or 252),
        )

        _, base = _fit_and_predict(
            selected, factory, fit_core, cal, test, baseline_columns, model_cfg
        )
        _, sec_result = _fit_and_predict(
            selected, factory, fit_core, cal, test, sec_columns, model_cfg
        )
        _, event_result = _fit_and_predict(
            selected, factory, fit_core, cal, test, event_columns, model_cfg
        )

        rows.append(
            {
                "fold": fold_index,
                "n_test": int(len(test)),
                "baseline_logloss": base["logloss"],
                "sec_logloss": sec_result["logloss"],
                "event_logloss": event_result["logloss"],
                "sec_vs_baseline_logloss_delta_improvement": base["logloss"] - sec_result["logloss"],
                "event_vs_sec_logloss_delta_improvement": sec_result["logloss"] - event_result["logloss"],
                "baseline_brier": base["brier"],
                "sec_brier": sec_result["brier"],
                "event_brier": event_result["brier"],
                "sec_vs_baseline_brier_delta_improvement": base["brier"] - sec_result["brier"],
                "event_vs_sec_brier_delta_improvement": sec_result["brier"] - event_result["brier"],
                "baseline_ece": base["ece"],
                "sec_ece": sec_result["ece"],
                "event_ece": event_result["ece"],
                "sec_vs_baseline_ece_delta_improvement": base["ece"] - sec_result["ece"],
                "event_vs_sec_ece_delta_improvement": sec_result["ece"] - event_result["ece"],
                "baseline_accuracy": base["accuracy"],
                "sec_accuracy": sec_result["accuracy"],
                "event_accuracy": event_result["accuracy"],
            }
        )

    if len(rows) < 3:
        _deferred(f"valid_oos_folds_too_few:{len(rows)}")
        return

    sec_logloss_delta = np.asarray(
        [float(row["sec_vs_baseline_logloss_delta_improvement"]) for row in rows],
        dtype=float,
    )
    event_logloss_delta = np.asarray(
        [float(row["event_vs_sec_logloss_delta_improvement"]) for row in rows],
        dtype=float,
    )
    sec_brier_delta = np.asarray(
        [float(row["sec_vs_baseline_brier_delta_improvement"]) for row in rows],
        dtype=float,
    )
    event_brier_delta = np.asarray(
        [float(row["event_vs_sec_brier_delta_improvement"]) for row in rows],
        dtype=float,
    )
    event_ece_delta = np.asarray(
        [float(row["event_vs_sec_ece_delta_improvement"]) for row in rows],
        dtype=float,
    )

    mean_event_delta = float(np.mean(event_logloss_delta))
    positive_event_folds = int(np.sum(event_logloss_delta > 0.0))
    recommended = bool(
        mean_event_delta >= 0.002
        and positive_event_folds >= max(3, int(np.ceil(len(rows) * 0.67)))
    )

    payload = {
        "status": "OOS_COMPLETE",
        "selected_model": selected,
        "folds": rows,
        "summary": {
            "sec_mean_logloss_delta_improvement": float(np.mean(sec_logloss_delta)),
            "event_mean_logloss_delta_improvement_vs_sec": mean_event_delta,
            "event_median_logloss_delta_improvement_vs_sec": float(np.median(event_logloss_delta)),
            "event_positive_logloss_folds": positive_event_folds,
            "sec_mean_brier_delta_improvement": float(np.mean(sec_brier_delta)),
            "event_mean_brier_delta_improvement_vs_sec": float(np.mean(event_brier_delta)),
            "event_mean_ece_delta_improvement_vs_sec": float(np.mean(event_ece_delta)),
            "event_logloss_delta_std": (
                float(np.std(event_logloss_delta, ddof=1))
                if len(event_logloss_delta) >= 2
                else 0.0
            ),
        },
        "feature_columns_added": list(SEC_FEATURE_COLUMNS),
        "event_intelligence_feature_columns_added": list(
            EVENT_INTELLIGENCE_FEATURE_COLUMNS
        ),
        "event_intelligence": {
            "status": event_diag["status"],
            "ledger_rows": event_diag["usable_rows"],
            "ledger_symbols": event_diag["symbols"],
            "pit_invalid_rows": event_diag["pit_invalid_rows"],
            "deduplicated_rows": event_diag["deduplicated_rows"],
            "families": event_diag["families"],
            "availability_rule": "event.available_at <= price.available_at",
            "unknown_available_at": "FAIL_CLOSED",
        },
        "recommended_for_further_holdout": recommended,
        "research_only": True,
        "production_changed": False,
        "selection_note": (
            "nested chronological ablation: baseline -> SEC event counts -> "
            "Opta-like structured event signature; frozen holdout remains excluded from tuning"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
