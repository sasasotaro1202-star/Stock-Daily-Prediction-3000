from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.context import add_cross_sectional_context, add_market_context
from src.features.technical import FEATURE_COLUMNS, add_technical_features
from src.prediction.fit import fit_classifier
from src.prediction.model_factories import models
from src.prediction.targets import add_targets
from src.research.innovative_control_v6 import evaluate_v6
from src.research.regime_threshold import volatility_threshold_from_training
from src.validation.calibration import make_calibrator
from src.validation.leakage import audit_feature_columns, audit_target_separation
from src.validation.training_sample import cap_training_rows
from src.validation.walk_forward import make_date_folds


PRICE = Path("data/prices/canonical.parquet")
CONTEXT = Path("data/market_context.parquet")
OUT = Path("data/research/latest_metrics.json")
BANK_OUT = Path("data/research/innovative_v6_oos_bank.parquet")
RISK_COLUMNS = (
    "volatility_20", "volume_ratio_20", "gap_pct", "breadth_up",
    "market_dispersion_1d", "market_dispersion_vs_20d", "vix_level_lag1",
    "return_z20", "drawdown_from_high_20", "cs_ret_1d_rank", "cs_vol_rank",
)


def main() -> int:
    if not PRICE.exists() or not CONTEXT.exists():
        raise SystemExit("BLOCKED: reduced v6 inputs missing")

    df = pd.read_parquet(PRICE)
    market_context = pd.read_parquet(CONTEXT)
    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce").dt.date

    max_symbols = int(os.getenv("V6_REDUCED_MAX_SYMBOLS", "250"))
    symbols = sorted(df["symbol"].dropna().astype(str).unique())[:max_symbols]
    df = df[df["symbol"].astype(str).isin(symbols)].copy()

    df = add_technical_features(df)
    df = add_market_context(df, market_context)
    df = add_targets(add_cross_sectional_context(df))

    feature_audit = audit_feature_columns(FEATURE_COLUMNS)
    target_audit = audit_target_separation(
        FEATURE_COLUMNS,
        [c for c in df.columns if c.startswith("target_")],
    )
    if not (feature_audit.ok and target_audit.ok):
        raise SystemExit(f"FAIL: leakage audit {feature_audit.violations + target_audit.violations}")

    df = df.dropna(subset=FEATURE_COLUMNS + ["target_up_1d"]).copy()
    frozen_path = Path("config/frozen_holdout.json")
    if frozen_path.exists():
        cutoff = pd.Timestamp(
            json.loads(frozen_path.read_text(encoding="utf-8"))["cutoff_date"]
        ).date()
        df = df[df["session_date"] <= cutoff].copy()

    dates = sorted(df["session_date"].unique())
    all_folds = make_date_folds(
        dates, min_train=252, test_size=21, step=21, embargo=1, purge=1
    )
    folds = all_folds[:7]
    if len(folds) < 5:
        raise SystemExit(f"BLOCKED: only {len(folds)} reduced OOS folds")

    configured = models()
    selected = {
        name: configured[name]
        for name in ("logistic", "extra_trees", "hgb")
        if name in configured
    }
    if len(selected) < 2:
        raise SystemExit("BLOCKED: insufficient primary models")

    banks: dict[int, dict[str, object]] = {}
    base_rows: list[dict[str, float]] = []

    for fold_idx, fold in enumerate(folds):
        train_dates = dates[:fold.train_end]
        cal_n = max(20, int(len(train_dates) * 0.20))
        core_dates = set(train_dates[:-cal_n])
        cal_dates = set(train_dates[-cal_n:])
        test_dates = set(dates[fold.test_start:fold.test_end])

        core = df[df["session_date"].isin(core_dates)]
        cal = df[df["session_date"].isin(cal_dates)]
        test = df[df["session_date"].isin(test_dates)].reset_index(drop=True)
        if min(len(core), len(cal), len(test)) < 100:
            continue

        threshold = volatility_threshold_from_training(core["volatility_20"])
        regime = np.where(
            core.iloc[0:0].index.astype(int),
            "",
            "",
        )
        test_regime = np.where(
            pd.to_numeric(test["volatility_20"], errors="coerce").to_numpy(dtype=float) >= threshold,
            "high_vol",
            np.where(np.abs(pd.to_numeric(test["gap_pct"], errors="coerce").to_numpy(dtype=float)) >= 0.03,
                     "event",
                     np.where(np.abs(pd.to_numeric(test["price_vs_sma60"], errors="coerce").to_numpy(dtype=float)) >= 0.02,
                              "trend", "normal")),
        )

        bank = {
            "session_dates": test["session_date"].to_numpy(copy=True),
            "y": test["target_up_1d"].astype(int).to_numpy(copy=True),
            "predictions": {},
            "risk_context": np.column_stack([
                pd.to_numeric(test[col], errors="coerce").to_numpy(dtype=float)
                if col in test.columns else np.full(len(test), np.nan)
                for col in RISK_COLUMNS
            ]),
            "symbols": test["symbol"].astype(str).to_numpy(copy=True),
            "regimes": np.asarray(test_regime, dtype=str),
            "asset_classes": test["asset_class"].astype(str).to_numpy(copy=True)
                if "asset_class" in test.columns else np.array(["equity"] * len(test)),
            "situations": np.asarray(test_regime, dtype=str),
        }

        core_fit = cap_training_rows(core, max_rows=50_000, recent_sessions=126)
        for name, factory in selected.items():
            model = factory()
            fit_classifier(
                model,
                name,
                core_fit[FEATURE_COLUMNS],
                core_fit["target_up_1d"].astype(int),
                core_fit["session_date"],
                half_life_sessions=126,
            )
            cal_p = model.predict_proba(cal[FEATURE_COLUMNS])[:, 1]
            raw = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            calibrator = make_calibrator("platt").fit(
                cal_p, cal["target_up_1d"].astype(int)
            )
            p = np.clip(calibrator.predict(raw), 1e-6, 1 - 1e-6)
            bank["predictions"][name] = p
            base_rows.append({
                "fold": float(fold_idx),
                "model": name,
                "logloss": float(
                    -np.mean(
                        test["target_up_1d"].astype(int).to_numpy() * np.log(p)
                        + (1 - test["target_up_1d"].astype(int).to_numpy()) * np.log(1 - p)
                    )
                ),
                "accuracy": float(np.mean((p >= 0.5) == test["target_up_1d"].astype(int).to_numpy())),
            })
        banks[fold_idx] = bank

    if len(banks) < 5:
        raise SystemExit(f"BLOCKED: only {len(banks)} valid reduced banks")

    BANK_OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for fold_idx, bank in sorted(banks.items()):
        n = len(bank["y"])
        frame = pd.DataFrame({
            "fold": fold_idx,
            "session_date": np.asarray(bank["session_dates"]).astype(str),
            "y": np.asarray(bank["y"], dtype=int),
            "symbol": np.asarray(bank["symbols"], dtype=str),
            "asset_class": np.asarray(bank["asset_classes"], dtype=str),
            "regime": np.asarray(bank["regimes"], dtype=str),
            "situation": np.asarray(bank["situations"], dtype=str),
        })
        risk = np.asarray(bank["risk_context"], dtype=float)
        for i in range(risk.shape[1]):
            frame[f"risk_{i}"] = risk[:, i]
        for name, p in bank["predictions"].items():
            frame[f"p__{name}"] = np.asarray(p, dtype=float)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(BANK_OUT, index=False)

    v6 = evaluate_v6(banks, locked_folds=min(2, len(banks) - 1))
    v6["evaluation_mode"] = "REDUCED_EXPLORATORY_E2E"
    v6["reduced_parameters"] = {
        "max_symbols": max_symbols,
        "models": sorted(selected),
        "folds_requested": 7,
        "folds_valid": len(banks),
        "training_row_cap": 50000,
    }
    v6["run_metadata"] = {
        "git_sha": os.getenv("GITHUB_SHA", "unknown"),
        "run_id": os.getenv("GITHUB_RUN_ID", "unknown"),
        "python_version": os.getenv("RUNNER_TOOL_CACHE", "unknown"),
    }

    payload = {
        "schema_version": "v6-e2e-reduced-1",
        "reduced_oos": True,
        "base_model_metrics": base_rows,
        "innovative_prediction_control_v6": v6,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    Path("data/research/leakage_audit.json").write_text(
        json.dumps({
            "ok": True,
            "independent_feature_target_audit": "PASS",
            "meta_layer": v6.get("per_fold_meta_audit", []),
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
