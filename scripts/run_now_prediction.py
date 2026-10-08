from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from exchange_calendars import get_calendar

import numpy as np
import pandas as pd

from src.features.context import add_cross_sectional_context, add_market_context
from src.features.technical import FEATURE_COLUMNS, add_technical_features
from src.prediction.fit import fit_classifier
from src.prediction.model_factories import models
from src.prediction.hierarchical_fallback import (
    SYMBOL_FALLBACK_FEATURES,
    blend_probabilities,
    make_symbol_logistic_model,
)
from src.prediction.production_artifact import ARTIFACT_PATH, load_production_artifact
from src.prediction.regression import make_quantile_models, make_return_model
from src.prediction.targets import add_targets
from src.ranking.cross_sectional import cross_sectional_rank
from src.research.router import regime_for_row, situation_for_row
from src.validation.calibration import make_calibrator
from src.validation.training_sample import cap_training_rows
from src.validation.training_window import restrict_to_lookback

PRICE_DIR = Path("data/prices/canonical.parquet")
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

    # Hierarchical near-production fallback:
    #   global model -> asset-class model -> symbol-specific regularized logistic.
    # All component fits use only pre-test data. This path is intentionally
    # research/monitoring only; it does not bypass the frozen production gate.
    asset_experts: dict[str, tuple[object, object]] = {}
    for asset_class, asset_core in core.groupby("asset_class", sort=False):
        asset_name = str(asset_class)
        if len(asset_core) < 800 or asset_core["target_up_1d"].nunique() < 2:
            continue
        asset_cal = cal[cal["asset_class"].astype(str).eq(asset_name)]
        if len(asset_cal) < 40 or asset_cal["target_up_1d"].nunique() < 2:
            continue
        asset_model = factory_map[model_name]()
        fit_classifier(
            asset_model,
            model_name,
            asset_core[FEATURE_COLUMNS],
            asset_core["target_up_1d"].astype(int),
            asset_core["session_date"],
            half_life_sessions=half_life,
        )
        asset_raw_cal = asset_model.predict_proba(asset_cal[FEATURE_COLUMNS])[:, 1]
        asset_calibrator = make_calibrator(_calibration_method()).fit(
            asset_raw_cal,
            asset_cal["target_up_1d"].astype(int),
        )
        asset_experts[asset_name] = (asset_model, asset_calibrator)

    symbol_experts: dict[tuple[str, str], tuple[object, object]] = {}
    for (asset_class, symbol), symbol_core in core.groupby(
        ["asset_class", "symbol"], sort=False
    ):
        key = (str(asset_class), str(symbol))
        if len(symbol_core) < 160 or symbol_core["target_up_1d"].nunique() < 2:
            continue
        symbol_cal = cal[
            cal["asset_class"].astype(str).eq(key[0])
            & cal["symbol"].astype(str).eq(key[1])
        ]
        if len(symbol_cal) < 30 or symbol_cal["target_up_1d"].nunique() < 2:
            continue
        symbol_model = make_symbol_logistic_model()
        symbol_model.fit(
            symbol_core[SYMBOL_FALLBACK_FEATURES],
            symbol_core["target_up_1d"].astype(int),
        )
        symbol_raw_cal = symbol_model.predict_proba(
            symbol_cal[SYMBOL_FALLBACK_FEATURES]
        )[:, 1]
        symbol_calibrator = make_calibrator("platt").fit(
            symbol_raw_cal,
            symbol_cal["target_up_1d"].astype(int),
        )
        symbol_experts[key] = (symbol_model, symbol_calibrator)

    ready = latest[FEATURE_COLUMNS].notna().all(axis=1)
    out = latest.copy()
    out["prediction_status"] = np.where(
        ready, "READY_NEAR_PRODUCTION", "DEFERRED_INCOMPLETE_FEATURES"
    )
    probs = np.full(len(out), np.nan, dtype=float)
    disagreements = np.full(len(out), np.nan, dtype=float)
    selected_model_ids: list[str] = [""] * len(out)
    route_reasons: list[str] = ["deferred:incomplete_features"] * len(out)
    if ready.any():
        for idx, row in out.loc[ready].iterrows():
            one = pd.DataFrame([row])
            component_probs = [
                float(
                    np.clip(
                        calibrator.predict(
                            model.predict_proba(one[FEATURE_COLUMNS])[:, 1]
                        )[0],
                        1e-5,
                        1 - 1e-5,
                    )
                )
            ]
            component_weights = [0.55]
            component_names = [model_name]

            asset_key = str(row["asset_class"])
            asset_entry = asset_experts.get(asset_key)
            if asset_entry is not None:
                asset_model, asset_calibrator = asset_entry
                component_probs.append(
                    float(
                        np.clip(
                            asset_calibrator.predict(
                                asset_model.predict_proba(one[FEATURE_COLUMNS])[:, 1]
                            )[0],
                            1e-5,
                            1 - 1e-5,
                        )
                    )
                )
                component_weights.append(0.25)
                component_names.append(f"{model_name}:asset")

            symbol_key = (asset_key, str(row["symbol"]))
            symbol_entry = symbol_experts.get(symbol_key)
            if symbol_entry is not None:
                symbol_model, symbol_calibrator = symbol_entry
                component_probs.append(
                    float(
                        np.clip(
                            symbol_calibrator.predict(
                                symbol_model.predict_proba(
                                    one[SYMBOL_FALLBACK_FEATURES]
                                )[:, 1]
                            )[0],
                            1e-5,
                            1 - 1e-5,
                        )
                    )
                )
                component_weights.append(0.20)
                component_names.append("symbol:logistic")

            probs[idx] = blend_probabilities(component_probs, component_weights)
            disagreements[idx] = float(np.std(component_probs))
            selected_model_ids[idx] = "+".join(component_names)
            route_reasons[idx] = "near_production:hierarchical_global_asset_symbol"
    else:
        pass

    out["p_up_1d"] = probs

    ret = make_return_model()
    qmodels = make_quantile_models()
    qfit = cap_training_rows(labeled, max_rows=250_000, recent_sessions=252)
    ret.fit(qfit[FEATURE_COLUMNS], qfit["target_ret_1d"])
    for q in qmodels.values():
        q.fit(qfit[FEATURE_COLUMNS], qfit["target_ret_1d"])

    # Match the classifier hierarchy on the return side where there is enough
    # asset-class history. The global model remains a stabilizing reference.
    asset_return_experts: dict[str, tuple[object, dict[str, object]]] = {}
    for asset_class, asset_qfit in qfit.groupby("asset_class", sort=False):
        if len(asset_qfit) < 1200:
            continue
        asset_name = str(asset_class)
        asset_ret = make_return_model()
        asset_qs = make_quantile_models()
        asset_ret.fit(asset_qfit[FEATURE_COLUMNS], asset_qfit["target_ret_1d"])
        for q in asset_qs.values():
            q.fit(asset_qfit[FEATURE_COLUMNS], asset_qfit["target_ret_1d"])
        asset_return_experts[asset_name] = (asset_ret, asset_qs)

    mid = np.full(len(out), np.nan, dtype=float)
    lo = np.full(len(out), np.nan, dtype=float)
    hi = np.full(len(out), np.nan, dtype=float)
    q50_return = np.full(len(out), np.nan, dtype=float)
    if ready.any():
        frame = out.loc[ready, FEATURE_COLUMNS]
        global_mid_v = ret.predict(frame)
        global_lo_v = qmodels["q10"].predict(frame)
        global_q50_v = qmodels["q50"].predict(frame)
        global_hi_v = qmodels["q90"].predict(frame)

        asset_expert = asset_return_experts.get(str(asset))
        if asset_expert is not None:
            asset_ret, asset_qs = asset_expert
            asset_mid_v = asset_ret.predict(frame)
            asset_lo_v = asset_qs["q10"].predict(frame)
            asset_q50_v = asset_qs["q50"].predict(frame)
            asset_hi_v = asset_qs["q90"].predict(frame)
            mid_v = 0.30 * global_mid_v + 0.70 * asset_mid_v
            lo_v = 0.30 * global_lo_v + 0.70 * asset_lo_v
            q50_v = 0.30 * global_q50_v + 0.70 * asset_q50_v
            hi_v = 0.30 * global_hi_v + 0.70 * asset_hi_v
        else:
            mid_v = global_mid_v
            lo_v = global_lo_v
            q50_v = global_q50_v
            hi_v = global_hi_v
        # Keep the expected-return point estimate distinct from the
        # model-derived median quantile while enforcing monotone quantiles.
        lo_v = np.minimum(np.minimum(lo_v, mid_v), q50_v)
        hi_v = np.maximum(np.maximum(hi_v, mid_v), q50_v)
        q50_v = np.minimum(np.maximum(q50_v, lo_v), hi_v)
        idx = np.flatnonzero(ready.to_numpy())
        mid[idx], lo[idx], hi[idx] = mid_v, lo_v, hi_v
        q50_return[idx] = q50_v

    out["expected_return_1d"] = mid
    out["return_q10_1d"] = np.clip(lo, -0.99, None)
    out["return_q90_1d"] = hi
    q50_return = np.clip(q50_return, -0.99, None)
    out["expected_close_1d"] = out["close"] * (1 + out["expected_return_1d"])
    out["range_low_1d"] = out["close"] * (1 + out["return_q10_1d"])
    out["range_high_1d"] = out["close"] * (1 + out["return_q90_1d"])
    out["q10_1d"] = out["range_low_1d"]
    out["q50_1d"] = out["close"] * (1 + q50_return)
    out["q90_1d"] = out["range_high_1d"]
    out["model_id"] = selected_model_ids
    out["training_scope"] = "near_production_hierarchical"
    out["return_training_scope"] = np.where(
        out["asset_class"].astype(str).isin(set(asset_return_experts)),
        "near_production_asset_blend",
        "near_production_global",
    )
    out["route_reason"] = route_reasons
    frozen_payload = json.loads(FROZEN.read_text(encoding="utf-8")) if FROZEN.exists() else {}
    threshold = frozen_payload.get("regime_vol_threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(float(threshold)):
        threshold = float(labeled["volatility_20"].dropna().median())
    regimes = []
    situations = []
    for _, row in out.iterrows():
        if not bool(ready.loc[row.name]):
            regimes.append("data_stressed")
            situations.append("data_stressed")
            continue
        regime = regime_for_row(
            float(row["volatility_20"]),
            float(row["price_vs_sma60"]),
            float(threshold),
            gap_pct=float(row["gap_pct"]) if pd.notna(row["gap_pct"]) else None,
            volume_ratio_20=float(row["volume_ratio_20"]) if pd.notna(row["volume_ratio_20"]) else None,
            vix_level=float(row["vix_level_lag1"]) if pd.notna(row["vix_level_lag1"]) else None,
            breadth_up=float(row["breadth_up"]) if pd.notna(row["breadth_up"]) else None,
        )
        situation = situation_for_row(
            regime.value,
            gap_pct=float(row["gap_pct"]) if pd.notna(row["gap_pct"]) else None,
            volume_ratio_20=float(row["volume_ratio_20"]) if pd.notna(row["volume_ratio_20"]) else None,
            vix_level=float(row["vix_level_lag1"]) if pd.notna(row["vix_level_lag1"]) else None,
            breadth_up=float(row["breadth_up"]) if pd.notna(row["breadth_up"]) else None,
            price_vs_sma60=float(row["price_vs_sma60"]) if pd.notna(row["price_vs_sma60"]) else None,
        )
        regimes.append(regime.value)
        situations.append(situation)
    out["regime"] = regimes
    out["market_situation"] = situations
    out["model_disagreement"] = disagreements
    out["prediction_mode"] = "NEAR_PRODUCTION_HIERARCHICAL"
    out["prediction_time"] = pd.Timestamp(datetime.now(timezone.utc))
    out["prediction_date"] = out["prediction_time"].dt.tz_convert("Asia/Tokyo").dt.date

    # Explicitly materialize the next exchange session for each prediction row.
    # JP and US calendars are separated so the same runtime can be executed
    # after the JP close while predicting the still-future US session.
    def _target_date(row: pd.Series) -> str:
        asset_class = str(row.get("asset_class", ""))
        calendar_code = "XTKS" if asset_class in {"jp_stock", "jp_etf", "jp_reit"} else "XNYS"
        session = pd.Timestamp(row["session_date"]).normalize()
        calendar = get_calendar(calendar_code)
        sessions = calendar.sessions_in_range(session, session + pd.Timedelta(days=14))
        for candidate in sessions:
            candidate = pd.Timestamp(candidate)
            if candidate.date() > session.date():
                return str(candidate.date())
        raise ValueError(f"next exchange session unavailable: {calendar_code} {session.date()}")

    out["target_date"] = out.apply(_target_date, axis=1)
    out["model_version"] = "near-production-hierarchical-v1"

    cols = [
        "symbol", "asset_class", "session_date", "close", "available_at",
        "retrieved_at", "prediction_time", "prediction_date", "model_version",
        "p_up_1d", "expected_return_1d", "return_q10_1d", "q50_1d", "return_q90_1d",
        "expected_close_1d", "range_low_1d", "range_high_1d", "model_id",
        "training_scope", "return_training_scope", "route_reason",
        "regime", "market_situation", "model_disagreement", "prediction_status",
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
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    result.to_parquet(OUT.parent / f"prediction_{stamp}.parquet", index=False)
    print(
        f"prediction-mode=NEAR_PRODUCTION rows={len(result)} "
        f"model={model_name} source={source} snapshot={stamp}"
    )


if __name__ == "__main__":
    main()
