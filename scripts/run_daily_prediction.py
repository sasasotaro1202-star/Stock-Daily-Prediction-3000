from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import os

import numpy as np
import pandas as pd

from src.features.context import add_cross_sectional_context, add_market_context
from src.features.technical import FEATURE_COLUMNS, add_technical_features
from src.prediction.production_artifact import load_production_artifact
from src.prediction.targets import add_targets
from src.ranking.cross_sectional import cross_sectional_rank
from src.research.router import regime_for_row, route_plan
from src.validation.code_fingerprint import fingerprint_sha256

PRICE = Path("data/prices")
METRICS = Path("data/research/latest_metrics.json")
GATE = Path("data/research/release_gate.json")
FROZEN = Path("config/frozen_holdout.json")
OUT = Path("data/predictions/latest.parquet")



def production_eligible(df: pd.DataFrame) -> pd.DataFrame:
    if not FROZEN.exists():
        return df
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        return df

    dates = pd.to_datetime(df["session_date"], errors="coerce").dt.date
    cutoff = pd.Timestamp(frozen["cutoff_date"]).date()
    holdout_end = pd.Timestamp(frozen["holdout_end"]).date()
    # The acceptance holdout is immutable. Fresh observations after that
    # one-time holdout may be used for production updates.
    keep = (dates <= cutoff) | (dates > holdout_end)
    return df.loc[keep].copy()


def main():
    if not PRICE.exists() or not METRICS.exists() or not GATE.exists():
        raise SystemExit("DEFERRED: research/release artifacts are missing")

    gate = json.loads(GATE.read_text(encoding="utf-8"))
    if not gate.get("approved", False):
        raise SystemExit(
            f"DEFERRED: release gate not approved: {gate.get('reasons', [])}"
        )

    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    try:
        artifact = load_production_artifact()
    except RuntimeError as exc:
        raise SystemExit(f"DEFERRED: {exc}") from exc
    if artifact["metadata"]["selected_model"] != payload.get("selected_model"):
        raise SystemExit("DEFERRED: production artifact does not match selected research model")
    df = pd.read_parquet(PRICE)
    context_path = Path("data/market_context.parquet")
    if not context_path.exists():
        raise SystemExit("DEFERRED: market context is absent")
    market_context = pd.read_parquet(context_path)
    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce").dt.date
    asset_filter = [
        x.strip()
        for x in os.environ.get("PREDICT_ASSET_CLASSES", "").split(",")
        if x.strip()
    ]
    df["available_at"] = pd.to_datetime(
        df["available_at"], utc=True, errors="coerce"
    )
    prediction_time = pd.Timestamp(datetime.now(timezone.utc))

    def file_hash(path: Path) -> str | None:
        if not path.exists():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    model_version = fingerprint_sha256()
    universe_version = file_hash(Path("data/universe/latest.json"))
    market_context_version = file_hash(Path("data/market_context.parquet"))

    # End-to-end PIT gate for the production snapshot.
    df = df[df["available_at"].le(prediction_time)].copy()
    if df.empty:
        raise SystemExit("DEFERRED: no price rows satisfy available_at <= prediction_time")

    df = add_technical_features(df)
    df = add_market_context(df, market_context)
    df = add_cross_sectional_context(df)
    labeled = add_targets(df).dropna(
        subset=FEATURE_COLUMNS + ["target_up_1d", "target_ret_1d"]
    ).copy()
    labeled = production_eligible(labeled)

    if len(labeled) < 5000:
        raise SystemExit("DEFERRED: insufficient PIT-safe production training data")

    # Pick the latest actually-known session independently for each market family.
    latest = (
        df.sort_values(["asset_class", "session_date"])
        .groupby("asset_class", group_keys=False)
        .tail(1)
        .copy()
    )
    if asset_filter:
        latest = latest[latest["asset_class"].isin(asset_filter)].copy()
    if latest.empty:
        raise SystemExit("DEFERRED: no latest PIT-safe session rows")

    threshold = float(artifact["metadata"]["regime_vol_threshold"])

    metric_payload = payload
    frozen_routes = {}
    if FROZEN.exists():
        frozen_routes = json.loads(FROZEN.read_text(encoding="utf-8"))
    locked_mode = frozen_routes.get("status") == "FROZEN"
    regime_metrics = metric_payload.get("regime_metrics", {})
    asset_metrics = metric_payload.get("asset_class_metrics", {})
    asset_regime_metrics = metric_payload.get("asset_regime_metrics", {})
    global_selected = metric_payload.get("selected_model", "hgb")

    # Load the already-approved immutable model artifact. No classifier or
    # quantile model is retrained inside the production prediction job.
    classifiers = artifact["classifiers"]
    selected_names: list[str] = []
    selected_scopes: list[str] = []
    selected_reasons: list[str] = []
    p_values: list[float] = []
    global_disagreement: list[float] = []
    regimes: list[str] = []

    ready_mask = latest[FEATURE_COLUMNS].notna().all(axis=1)
    latest["prediction_status"] = np.where(
        ready_mask,
        "READY",
        "DEFERRED_INCOMPLETE_FEATURES",
    )

    for _, row in latest.iterrows():
        if not bool(ready_mask.loc[row.name]):
            p_values.append(float("nan"))
            selected_names.append("")
            selected_scopes.append("")
            selected_reasons.append("deferred:incomplete_features")
            regimes.append("data_stressed")
            global_disagreement.append(float("nan"))
            continue

        asset = str(row["asset_class"])
        regime = regime_for_row(
            float(row["volatility_20"]),
            float(row["price_vs_sma60"]),
            threshold,
            float(row["gap_pct"]) if pd.notna(row["gap_pct"]) else None,
            float(row["volume_ratio_20"]) if pd.notna(row["volume_ratio_20"]) else None,
            float(row["vix_level_lag1"]) if pd.notna(row["vix_level_lag1"]) else None,
            float(row["breadth_up"]) if pd.notna(row["breadth_up"]) else None,
        ).value
        plan = route_plan(
            asset,
            regime,
            asset_regime_metrics=asset_regime_metrics,
            asset_metrics=asset_metrics,
            regime_metrics=regime_metrics,
            global_selected=global_selected,
            locked_asset_regime=frozen_routes.get("asset_regime_selected_models") if locked_mode else None,
            locked_asset=frozen_routes.get("asset_class_selected_models") if locked_mode else None,
            locked_regime=frozen_routes.get("regime_selected_models") if locked_mode else None,
            locked_global=frozen_routes.get("selected_model") if locked_mode else None,
        )
        model_name = plan.names[0]
        entry = classifiers.get(model_name)
        if entry is None:
            raise SystemExit(f"DEFERRED: production classifier artifact missing {model_name}")
        model = entry["model"]
        calibrator = entry["calibrator"]
        actual_scope = plan.scope
        raw_p = model.predict_proba(
            pd.DataFrame([row])[FEATURE_COLUMNS]
        )[:, 1]
        p_values.append(float(np.clip(calibrator.predict(raw_p)[0], 1e-5, 1 - 1e-5)))
        selected_names.append(model_name)
        selected_scopes.append(actual_scope)
        selected_reasons.append(plan.reason)
        regimes.append(regime)

        g_probs = []
        one = pd.DataFrame([row])[FEATURE_COLUMNS]
        for candidate, entry in sorted(classifiers.items()):
            gm = entry["model"]
            gc = entry["calibrator"]
            g_probs.append(float(gc.predict(gm.predict_proba(one)[:, 1])[0]))
        global_disagreement.append(float(np.std(g_probs)))

    latest["p_up_1d"] = p_values
    latest["model_id"] = selected_names
    latest["training_scope"] = selected_scopes
    latest["route_reason"] = selected_reasons
    latest["regime"] = regimes
    latest["model_disagreement"] = global_disagreement

    # Load immutable quantile models from the same approved artifact.
    return_artifact = artifact["return"]
    return_kind = return_artifact["selected"]
    return_models = return_artifact["global"]

    def expected_return_predict(frame: pd.DataFrame) -> np.ndarray:
        mean_pred = return_models["mean"].predict(frame)
        q50_pred = return_models["q50"].predict(frame)
        if return_kind == "mean":
            return mean_pred
        if return_kind == "q50":
            return q50_pred
        if return_kind == "blend_mean_q50":
            return 0.5 * mean_pred + 0.5 * q50_pred
        raise RuntimeError(f"unknown return estimator: {return_kind}")

    q_artifact = artifact["quantile"]
    global_qmodels = q_artifact["global"]
    asset_qmodels = q_artifact["assets"]

    latest_returns = []
    return_scope = []
    latest_lows = []
    latest_highs = []

    for asset, group in latest[ready_mask].groupby("asset_class", sort=False):
        qmodels = asset_qmodels.get(str(asset), global_qmodels)
        scope = f"asset:{asset}" if str(asset) in asset_qmodels else "global"

        lo = qmodels["q10"].predict(group[FEATURE_COLUMNS])
        mid = expected_return_predict(group[FEATURE_COLUMNS])
        hi = qmodels["q90"].predict(group[FEATURE_COLUMNS])
        lo = np.minimum(lo, mid)
        hi = np.maximum(hi, mid)

        latest_returns.extend(zip(group.index, mid))
        latest_lows.extend(zip(group.index, lo))
        latest_highs.extend(zip(group.index, hi))
        return_scope.extend(zip(group.index, [scope] * len(group)))

    ret_by_index = {idx: value for idx, value in latest_returns}
    low_by_index = {idx: value for idx, value in latest_lows}
    high_by_index = {idx: value for idx, value in latest_highs}
    scope_by_index = {idx: value for idx, value in return_scope}

    latest["expected_return_1d"] = [
        float(ret_by_index.get(idx, np.nan)) for idx in latest.index
    ]
    latest["return_q10_1d"] = [
        float(low_by_index.get(idx, np.nan)) for idx in latest.index
    ]
    latest["return_q90_1d"] = [
        float(high_by_index.get(idx, np.nan)) for idx in latest.index
    ]
    latest["return_training_scope"] = [
        scope_by_index.get(idx, "") for idx in latest.index
    ]

    latest["return_q10_1d"] = latest["return_q10_1d"].clip(
        lower=-0.99
    )
    latest["return_q10_1d"] = np.minimum(
        latest["return_q10_1d"],
        latest["expected_return_1d"],
    )
    latest["return_q90_1d"] = np.maximum(
        latest["return_q90_1d"],
        latest["expected_return_1d"],
    )

    latest["expected_close_1d"] = np.where(
        ready_mask,
        latest["close"] * (1 + latest["expected_return_1d"]),
        np.nan,
    )
    latest["range_low_1d"] = np.where(
        ready_mask,
        latest["close"] * (1 + latest["return_q10_1d"]),
        np.nan,
    )
    latest["range_high_1d"] = np.where(
        ready_mask,
        latest["close"] * (1 + latest["return_q90_1d"]),
        np.nan,
    )
    latest["prediction_time"] = prediction_time
    latest["prediction_date"] = prediction_time.tz_convert("Asia/Tokyo").date()
    latest["model_version"] = model_version
    latest["universe_version"] = universe_version
    latest["market_context_version"] = market_context_version

    cols = [
        "symbol",
        "asset_class",
        "session_date",
        "close",
        "prediction_time",
        "prediction_date",
        "model_version",
        "universe_version",
        "market_context_version",
        "p_up_1d",
        "expected_return_1d",
        "expected_close_1d",
        "range_low_1d",
        "range_high_1d",
        "model_id",
        "training_scope",
        "return_training_scope",
        "route_reason",
        "regime",
        "model_disagreement",
        "prediction_status",
    ]

    rank_weight = float(artifact["metadata"].get("rank_probability_weight", 0.50))
    rank_uncertainty_penalty = float(
        artifact["metadata"].get("rank_uncertainty_penalty", 0.0)
    )
    # Ranking uncertainty uses the global q10-q90 interval so its definition
    # is identical in OOS, frozen holdout, and production.
    q10_global = global_qmodels["q10"].predict(
        latest[FEATURE_COLUMNS]
    )
    q90_global = global_qmodels["q90"].predict(
        latest[FEATURE_COLUMNS]
    )
    latest["ranking_uncertainty"] = np.maximum(q90_global - q10_global, 0.0)
    out = cross_sectional_rank(
        latest[cols + ["ranking_uncertainty"]],
        probability_weight=rank_weight,
        uncertainty_col="ranking_uncertainty",
        uncertainty_penalty=rank_uncertainty_penalty,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    stamp = prediction_time.strftime("%Y%m%dT%H%M%SZ")
    out.to_parquet(OUT.parent / f"prediction_{stamp}.parquet", index=False)
    print(
        f"prediction_rows={len(out)} "
        f"asset_classes={sorted(out['asset_class'].unique().tolist())}"
    )


if __name__ == "__main__":
    main()
