from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.context import add_cross_sectional_context, add_market_context
from src.features.technical import FEATURE_COLUMNS, add_technical_features
from src.prediction.fit import fit_classifier
from src.prediction.model_factories import models
from src.prediction.production_artifact import ARTIFACT_PATH, load_production_artifact
from src.prediction.regression import make_quantile_models, make_return_model
from src.prediction.targets import add_targets
from src.ranking.cross_sectional import cross_sectional_rank
from src.research.router import regime_for_row
from src.validation.calibration import make_calibrator
from src.validation.training_sample import cap_training_rows
from src.validation.training_window import restrict_to_lookback

PRICE_DIR = Path("data/prices")
OUT = Path("data/predictions/latest.parquet")
FROZEN = Path("config/frozen_holdout.json")
METRICS = Path("data/research/latest_metrics.json")


def _production_eligible(df: pd.DataFrame) -> pd.DataFrame:
    if not FROZEN.exists():
        return df.copy()
    payload = json.loads(FROZEN.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN":
        raise SystemExit("DEFERRED: frozen holdout is not locked")
    cutoff = pd.Timestamp(payload["cutoff_date"]).date()
    end = pd.Timestamp(payload["holdout_end"]).date()
    dates = pd.to_datetime(df["session_date"], errors="coerce").dt.date
    return df.loc[(dates <= cutoff) | (dates > end)].copy()


def _load_model_choice() -> tuple[str, str]:
    if FROZEN.exists():
        payload = json.loads(FROZEN.read_text(encoding="utf-8"))
        selected = payload.get("selected_model")
        if selected:
            return str(selected), "frozen_selected_model"
    if METRICS.exists():
        payload = json.loads(METRICS.read_text(encoding="utf-8"))
        selected = payload.get("selected_model")
        if selected:
            return str(selected), "latest_oos_metrics"
    return "hgb", "deterministic_fallback"


def _calibration_method() -> str:
    if FROZEN.exists():
        payload = json.loads(FROZEN.read_text(encoding="utf-8"))
        method = str(payload.get("calibration_method", "platt"))
    else:
        method = "platt"
    if method not in {"platt", "beta", "isotonic"}:
        return "platt"
    return method


def _training_window() -> int:
    if FROZEN.exists():
        return int(json.loads(FROZEN.read_text(encoding="utf-8")).get(
            "classifier_training_window_sessions", 0
        ))
    return 0


def _latest_rows(df: pd.DataFrame, asset_filter: set[str]) -> pd.DataFrame:
    available = pd.to_datetime(df["available_at"], utc=True, errors="coerce")
    now = pd.Timestamp(datetime.now(timezone.utc))
    df = df.loc[available.le(now)].copy()
    if asset_filter:
        df = df[df["asset_class"].isin(asset_filter)].copy()
    latest = (
        df.sort_values(["asset_class", "symbol", "session_date"])
        .groupby(["asset_class", "symbol"], group_keys=False)
        .tail(1)
        .copy()
    )
    if latest.empty:
        raise SystemExit("DEFERRED: no PIT-safe latest security rows")
    return latest


def _predict_near_production(
    df: pd.DataFrame,
    latest: pd.DataFrame,
    *,
    model_name: str,
    selection_source: str,
) -> pd.DataFrame:
    factory_map = models()
    if model_name not in factory_map:
        model_name = "hgb"
        selection_source = "deterministic_hgb_fallback"

    asof = pd.Timestamp(datetime.now(timezone.utc))
    available = pd.to_datetime(df["available_at"], utc=True, errors="coerce")
    df = df.loc[available.le(asof)].copy()

    labeled = add_targets(df).dropna(
        subset=FEATURE_COLUMNS + ["target_up_1d", "target_ret_1d"]
    ).copy()
    labeled = _production_eligible(labeled)
    if len(labeled) < 5000:
        raise SystemExit("DEFERRED: insufficient PIT-safe near-production training rows")

    training_window = _training_window()
    labeled = restrict_to_lookback(
        labeled,
        None if training_window == 0 else training_window,
    )
    dates = sorted(pd.to_datetime(labeled["session_date"]).dt.date.unique())
    if len(dates) < 40:
        raise SystemExit("DEFERRED: insufficient chronological training dates")
    cal_n = max(20, int(len(dates) * 0.2))
    core = labeled[labeled["session_date"].isin(set(dates[:-cal_n]))].copy()
    cal = labeled[labeled["session_date"].isin(set(dates[-cal_n:]))].copy()
    core = cap_training_rows(
        core,
        max_rows=300_000,
        recent_sessions=min(252, training_window or 252),
    )
    model = factory_map[model_name]()
    half_life = 252
    if METRICS.exists():
        try:
            cfg = json.loads(METRICS.read_text(encoding="utf-8"))
            half_life = int(cfg.get("recency_weight_half_life_sessions", 252))
        except Exception:
            half_life = 252
    fit_classifier(
        model,
        model_name,
        core[FEATURE_COLUMNS],
        core["target_up_1d"].astype(int),
        core["session_date"],
        half_life_sessions=half_life,
    )
    raw_cal = model.predict_proba(cal[FEATURE_COLUMNS])[:, 1]
    calibrator = make_calibrator(_calibration_method()).fit(
        raw_cal,
        cal["target_up_1d"].astype(int),
    )

    ready = latest[FEATURE_COLUMNS].notna().all(axis=1)
    out = latest.copy()
    out["prediction_status"] = np.where(
        ready, "READY_NEAR_PRODUCTION", "DEFERRED_INCOMPLETE_FEATURES"
    )
    probs = np.full(len(out), np.nan, dtype=float)
    if ready.any():
        probs[ready.to_numpy()] = np.clip(
            calibrator.predict(model.predict_proba(out.loc[ready, FEATURE_COLUMNS])[:, 1]),
            1e-5,
            1 - 1e-5,
        )
    out["p_up_1d"] = probs

    ret = make_return_model()
    qmodels = make_quantile_models()
    qfit = cap_training_rows(labeled, max_rows=250_000, recent_sessions=252)
    ret.fit(qfit[FEATURE_COLUMNS], qfit["target_ret_1d"])
    for q in qmodels.values():
        q.fit(qfit[FEATURE_COLUMNS], qfit["target_ret_1d"])

    mid = np.full(len(out), np.nan, dtype=float)
    lo = np.full(len(out), np.nan, dtype=float)
    hi = np.full(len(out), np.nan, dtype=float)
    if ready.any():
        frame = out.loc[ready, FEATURE_COLUMNS]
        mid_v = ret.predict(frame)
        lo_v = qmodels["q10"].predict(frame)
        hi_v = qmodels["q90"].predict(frame)
        lo_v = np.minimum(lo_v, mid_v)
        hi_v = np.maximum(hi_v, mid_v)
        idx = np.flatnonzero(ready.to_numpy())
        mid[idx], lo[idx], hi[idx] = mid_v, lo_v, hi_v

    out["expected_return_1d"] = mid
    out["return_q10_1d"] = np.clip(lo, -0.99, None)
    out["return_q90_1d"] = hi
    out["expected_close_1d"] = out["close"] * (1 + out["expected_return_1d"])
    out["range_low_1d"] = out["close"] * (1 + out["return_q10_1d"])
    out["range_high_1d"] = out["close"] * (1 + out["return_q90_1d"])
    out["model_id"] = model_name
    out["training_scope"] = "near_production_global"
    out["return_training_scope"] = "near_production_global"
    out["route_reason"] = f"near_production:{selection_source}"
    out["regime"] = "near_production"
    out["model_disagreement"] = np.nan
    out["prediction_mode"] = "NEAR_PRODUCTION"
    out["prediction_time"] = pd.Timestamp(datetime.now(timezone.utc))
    out["prediction_date"] = out["prediction_time"].dt.tz_convert("Asia/Tokyo").dt.date
    out["model_version"] = "near-production-runtime"

    cols = [
        "symbol", "asset_class", "session_date", "close",
        "prediction_time", "prediction_date", "model_version",
        "p_up_1d", "expected_return_1d", "expected_close_1d",
        "range_low_1d", "range_high_1d", "model_id",
        "training_scope", "return_training_scope", "route_reason",
        "regime", "model_disagreement", "prediction_status",
        "prediction_mode",
    ]
    out["ranking_uncertainty"] = np.maximum(
        out["range_high_1d"] - out["range_low_1d"], 0.0
    )
    return cross_sectional_rank(
        out[cols + ["ranking_uncertainty"]],
        probability_weight=0.50,
        uncertainty_col="ranking_uncertainty",
        uncertainty_penalty=0.0,
    )


def main() -> None:
    if not PRICE_DIR.exists():
        raise SystemExit("DEFERRED: price store missing")
    asset_filter = {
        x.strip() for x in os.environ.get(
            "PREDICT_ASSET_CLASSES",
            "jp_stock,jp_etf,jp_reit,us_stock,us_etf",
        ).split(",") if x.strip()
    }

    # Preferred path: exact approved immutable production artifact. If an
    # artifact is present but fails validation, fail closed instead of silently
    # switching to a research/near-production model.
    if ARTIFACT_PATH.exists():
        try:
            artifact = load_production_artifact()
        except RuntimeError as exc:
            raise SystemExit(
                f"DEFERRED: approved production artifact failed validation: {exc}"
            ) from exc
    else:
        artifact = None

    df = pd.read_parquet(PRICE_DIR)
    context_path = Path("data/market_context.parquet")
    if not context_path.exists():
        raise SystemExit("DEFERRED: market context is absent")
    context = pd.read_parquet(context_path)
    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce").dt.date
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True, errors="coerce")
    df = add_technical_features(df)
    df = add_market_context(df, context)
    df = add_cross_sectional_context(df)

    if artifact is not None:
        # Reuse the exact production inference implementation and mark the mode.
        import scripts.run_daily_prediction as production
        old_filter = os.environ.get("PREDICT_ASSET_CLASSES")
        os.environ["PREDICT_ASSET_CLASSES"] = ",".join(sorted(asset_filter))
        try:
            production.main()
        finally:
            if old_filter is None:
                os.environ.pop("PREDICT_ASSET_CLASSES", None)
            else:
                os.environ["PREDICT_ASSET_CLASSES"] = old_filter
        produced = pd.read_parquet(production.OUT)
        produced["prediction_mode"] = "PRODUCTION"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        produced.to_parquet(OUT, index=False)
        print(
            f"prediction-mode=PRODUCTION rows={len(produced)} "
            f"selected={artifact['metadata']['selected_model']}"
        )
        return

    latest = _latest_rows(df, asset_filter)
    model_name, source = _load_model_choice()
    result = _predict_near_production(
        df,
        latest,
        model_name=model_name,
        selection_source=source,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT, index=False)
    print(
        f"prediction-mode=NEAR_PRODUCTION rows={len(result)} "
        f"model={model_name} source={source}"
    )


if __name__ == "__main__":
    main()
