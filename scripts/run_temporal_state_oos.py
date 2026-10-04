from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.features.context import add_cross_sectional_context, add_market_context
from src.features.technical import FEATURE_COLUMNS, add_technical_features
from src.prediction.targets import add_targets
from src.research.metrics import classification_metrics
from src.research.pit_lineage import scheduled_prediction_time
from src.research.temporal_state import TEMPORAL_STATE_COLUMNS, build_temporal_state_features
from src.validation.code_fingerprint import evidence_fingerprint_sha256
from src.validation.leakage import audit_feature_columns, audit_target_separation
from src.validation.walk_forward import make_date_folds

PRICE_DIR = Path("data/prices/canonical.parquet")
MARKET_CONTEXT = Path("data/market_context.parquet")
PIPELINE = Path("config/pipeline.yml")
OUT = Path("data/research/temporal_state_oos.json")


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        LogisticRegression(max_iter=1000, C=0.5),
    )
    model.fit(train[features], train["target_up_1d"].astype(int))
    return model.predict_proba(test[features])[:, 1]


def _weighted_mean(rows: list[dict[str, object]], section: str, key: str) -> float:
    values = np.asarray([r[section][key] for r in rows], dtype=float)
    weights = np.asarray([r["n_test"] for r in rows], dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def main() -> int:
    if not PRICE_DIR.exists():
        raise SystemExit("DEFERRED: canonical price dataset is absent")
    if not MARKET_CONTEXT.exists():
        raise SystemExit("DEFERRED: market context is absent")
    if not PIPELINE.exists():
        raise SystemExit("FAIL: pipeline configuration is absent")

    cfg = yaml.safe_load(PIPELINE.read_text(encoding="utf-8")) or {}
    temporal_cfg = cfg.get("temporal_state_research", {}) or {}
    if temporal_cfg.get("enabled", True) is not True:
        raise SystemExit("DEFERRED: temporal state research is disabled by configuration")
    if temporal_cfg.get("production_changed", False) is not False:
        raise SystemExit("FAIL: temporal state configuration claims production mutation")
    if temporal_cfg.get("research_only", True) is not True:
        raise SystemExit("FAIL: temporal state configuration is not research_only")

    df = pd.read_parquet(PRICE_DIR).copy()
    required = {"symbol", "session_date", "open", "high", "low", "close", "volume", "available_at"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"DEFERRED: temporal state PIT input missing columns: {missing}")

    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce").dt.date
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True, errors="coerce")
    if df["session_date"].isna().any() or df["available_at"].isna().any():
        raise SystemExit("FAIL: temporal state PIT fields contain invalid values")
    if "asset_class" not in df.columns:
        raise SystemExit("DEFERRED: asset_class is required for the canonical PIT clock")

    # Reuse the canonical market-specific prediction clock. Resolve it only for
    # unique session/asset pairs, then apply the cutoff vectorially to every row.
    cutoff_rows = []
    for (session_date, asset_class) in (
        df[["session_date", "asset_class"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        prediction, source = scheduled_prediction_time(session_date, asset_class, cfg)
        if prediction is None:
            raise SystemExit(f"FAIL: temporal state PIT clock unavailable for {asset_class}")
        cutoff_rows.append({
            "session_date": session_date,
            "asset_class": asset_class,
            "_prediction_cutoff": prediction,
            "_prediction_clock_source": source,
        })
    cutoffs = pd.DataFrame(cutoff_rows)
    df = df.merge(cutoffs, on=["session_date", "asset_class"], how="left", validate="many_to_one")
    if df["_prediction_cutoff"].isna().any():
        raise SystemExit("FAIL: temporal state prediction cutoff could not be resolved")
    if (df["available_at"] > df["_prediction_cutoff"]).any():
        raise SystemExit("FAIL: temporal state input contains available_at after prediction cutoff")
    if "published_at" in df.columns:
        published = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        malformed_published = df["published_at"].notna() & published.isna()
        if malformed_published.any():
            raise SystemExit("FAIL: temporal state input contains malformed published_at")
        if ((published.notna()) & (published > df["available_at"])).any():
            raise SystemExit("FAIL: temporal state input contains published_at after available_at")
    if "retrieved_at" in df.columns:
        retrieved = pd.to_datetime(df["retrieved_at"], utc=True, errors="coerce")
        malformed_retrieved = df["retrieved_at"].notna() & retrieved.isna()
        if malformed_retrieved.any():
            raise SystemExit("FAIL: temporal state input contains malformed retrieved_at")
        if ((retrieved.notna()) & (retrieved < df["available_at"])).any():
            raise SystemExit("FAIL: temporal state input contains retrieved_at before available_at")
    for col in ("available_at_method", "source", "provider_symbol"):
        if col in df.columns and df[col].astype(str).str.strip().eq("").any():
            raise SystemExit(f"FAIL: temporal state PIT field {col} contains empty provenance")

    # Use the same preprocessing family as the main research pipeline.
    market_context = pd.read_parquet(MARKET_CONTEXT)
    df = add_technical_features(df)
    df = add_market_context(df, market_context)
    df = add_targets(add_cross_sectional_context(df))

    temporal = build_temporal_state_features(df)
    temporal = temporal.drop(columns=["_prediction_cutoff", "_prediction_clock_source"])
    base_features = list(FEATURE_COLUMNS)
    augmented_features = base_features + list(TEMPORAL_STATE_COLUMNS)

    base_audit = audit_feature_columns(base_features)
    temporal_names = audit_feature_columns(list(TEMPORAL_STATE_COLUMNS))
    target_audit = audit_target_separation(augmented_features, ["target_up_1d", "target_ret_1d"])
    if not base_audit.ok or not temporal_names.ok or not target_audit.ok:
        violations = list(base_audit.violations + temporal_names.violations + target_audit.violations)
        raise SystemExit(f"FAIL: temporal state leakage audit {violations}")

    # Prediction is after the session closes in the established daily pipeline.
    # Temporal features are causal within each instrument; explicit availability
    # is retained and audited rather than replacing available_at with retrieval time.
    if df["available_at"].isna().any():
        raise SystemExit("FAIL: temporal state PIT availability is unknown")

    frozen_path = Path("config/frozen_holdout.json")
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        cutoff = pd.Timestamp(frozen["cutoff_date"]).date()
        temporal = temporal[temporal["session_date"] <= cutoff].copy()

    required_model_columns = augmented_features + ["target_up_1d", "target_ret_1d"]
    temporal = temporal.dropna(subset=required_model_columns).copy()
    if temporal.empty:
        raise SystemExit("DEFERRED: no complete temporal-state observations")

    dates = sorted(temporal["session_date"].unique())
    folds = make_date_folds(
        dates,
        min_train=252,
        test_size=21,
        step=21,
        embargo=1,
        purge=int((cfg.get("models", {}) or {}).get("purge_sessions", 1)),
    )
    min_folds = max(3, int(temporal_cfg.get("min_oos_folds", 5)))
    if len(folds) < min_folds:
        raise SystemExit(f"DEFERRED: only {len(folds)} temporal OOS folds available; need {min_folds}")

    fold_rows: list[dict[str, object]] = []
    for fold_idx, fold in enumerate(folds):
        train_dates = dates[: fold.train_end]
        cal_n = max(20, int(len(train_dates) * 0.2))
        train_core = set(train_dates[:-cal_n])
        test_dates = set(dates[fold.test_start : fold.test_end])
        train = temporal[temporal["session_date"].isin(train_core)]
        test = temporal[temporal["session_date"].isin(test_dates)]
        if (
            len(train) < 500
            or len(test) < 50
            or train["target_up_1d"].nunique() < 2
            or test["target_up_1d"].nunique() < 2
        ):
            continue

        p_base = _fit_predict(train, test, base_features)
        p_augmented = _fit_predict(train, test, augmented_features)
        y = test["target_up_1d"].astype(int).to_numpy()
        base = classification_metrics(y, p_base)
        augmented = classification_metrics(y, p_augmented)
        fold_rows.append(
            {
                "fold": fold_idx,
                "test_start_date": str(min(test_dates)),
                "test_end_date": str(max(test_dates)),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "base": base,
                "temporal_augmented": augmented,
                "delta_augmented_minus_base": {
                    metric: float(augmented[metric] - base[metric])
                    for metric in ("logloss", "brier", "ece", "accuracy")
                    if metric in augmented and metric in base
                },
            }
        )

    if len(fold_rows) < min_folds:
        raise SystemExit(
            f"DEFERRED: only {len(fold_rows)} valid temporal OOS folds; need {min_folds}"
        )

    metric_names = ("logloss", "brier", "ece", "accuracy")
    base_metrics = {m: _weighted_mean(fold_rows, "base", m) for m in metric_names}
    augmented_metrics = {
        m: _weighted_mean(fold_rows, "temporal_augmented", m) for m in metric_names
    }
    delta = {m: augmented_metrics[m] - base_metrics[m] for m in metric_names}
    relative_improvement = (
        (base_metrics["logloss"] - augmented_metrics["logloss"]) / base_metrics["logloss"]
        if base_metrics["logloss"] > 0
        else float("nan")
    )

    payload = {
        "schema_version": 1,
        "status": "EVALUATED",
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "frozen_holdout_used": False,
        "method": "causal_multi_session_temporal_state_augmentation",
        "data_granularity": "daily_ohlcv",
        "intraday_data_used": False,
        "target": "next_business_day_up",
        "oos_protocol": "chronological_walk_forward_same_data_population",
        "min_oos_folds": min_folds,
        "folds": len(fold_rows),
        "temporal_state_features": list(TEMPORAL_STATE_COLUMNS),
        "base_feature_count": len(base_features),
        "augmented_feature_count": len(augmented_features),
        "metrics": {
            "base": base_metrics,
            "temporal_augmented": augmented_metrics,
            "delta_temporal_augmented_minus_base": delta,
            "relative_logloss_improvement": float(relative_improvement),
        },
        "fold_metrics": fold_rows,
        "pit": {
            "available_at_required": True,
            "available_at_valid": True,
            "market_specific_prediction_clock": True,
            "future_row_reference": False,
            "retrieval_time_not_used_as_availability": True,
        },
        "evidence_code_fingerprint_sha256": evidence_fingerprint_sha256(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
