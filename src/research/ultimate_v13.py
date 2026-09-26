"""Adapter that feeds the existing chronological OOS bank into v13 control research."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.research.ultimate_control_v13 import evaluate_v13
from src.research.ultimate_v13_extensions import augment_v13_result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _to_frame(fold: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    y = np.asarray(fold.get("y", []), dtype=int)
    dates = np.asarray(fold.get("session_dates", [""] * len(y)), dtype=str)
    situations = np.asarray(
        fold.get("regimes", fold.get("situations", ["unknown"] * len(y))),
        dtype=str,
    )
    symbols = np.asarray(fold.get("symbols", [""] * len(y)), dtype=str)
    assets = np.asarray(fold.get("asset_classes", ["unknown"] * len(y)), dtype=str)
    risk = np.asarray(
        fold.get("risk_context", np.zeros((len(y), 0), dtype=float)),
        dtype=float,
    )
    if risk.ndim != 2 or risk.shape[0] != len(y):
        raise ValueError("invalid risk_context shape")
    frame = pd.DataFrame({
        "session_date": dates,
        "symbol": symbols,
        "asset_class": assets,
        "regime": situations,
    })
    for i in range(risk.shape[1]):
        frame[f"risk_{i:02d}"] = risk[:, i]
    return frame, risk


def build_ultimate_intelligence(
    bank: Mapping[int, Mapping[str, Any]],
    out_dir: str | Path = "data/research/ultimate_v13",
) -> dict[str, Any]:
    ordered = []
    for key in sorted(bank, key=lambda x: int(x)):
        fold = dict(bank[key])
        frame, risk = _to_frame(fold)
        predictions = {
            str(name): np.asarray(pred, dtype=float)
            for name, pred in (fold.get("predictions") or {}).items()
        }
        y = np.asarray(fold.get("y", []), dtype=int)
        if len(y) != len(frame) or any(len(p) != len(y) for p in predictions.values()):
            raise ValueError(f"invalid OOS bank lengths for fold {key}")
        ordered.append({
            "y": y,
            "predictions": predictions,
            "frame": frame,
            "risk_matrix": risk,
        })

    result = evaluate_v13(
        ordered,
        locked_folds=2,
        min_folds=5,
    )
    result = augment_v13_result(bank, result)
    result["research_only"] = True
    result["production_changed"] = False
    result["promotion_allowed"] = False
    result["upstream_contract"] = {
        "PIT": "must be PASS from upstream independent audit",
        "Leakage": "must be PASS from upstream audits",
        "Meta-Leakage": "not independently revalidated by this layer",
        "Frozen_Holdout": "not used for v13 tuning",
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(result)
    (out / "ultimate_summary.json").write_text(
        json.dumps(safe, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for name, key in (
        ("model_disagreement.json", "core_three_layers"),
        ("predictability.json", "predictability"),
        ("future_failure.json", "future_failure"),
        ("time_to_failure.json", "time_to_failure"),
        ("error_correlation.json", "error_correlation"),
        ("regime_transition.json", "regime_transition"),
        ("retrieval.json", "retrieval"),
        ("uncertainty.json", "uncertainty"),
        ("prediction_strategy.json", "prediction_strategy"),
        ("prediction_output.json", "prediction_output"),
        ("adaptive_compute.json", "adaptive_compute"),
        ("metrics_locked.json", "metrics_locked"),
        ("statistical_validation.json", "statistical_validation"),
        ("revision_metrics.json", "revision_metrics"),
        ("worst_case.json", "worst_case_locked"),
        ("robustness.json", "robustness"),
        ("audits.json", "audits"),
        ("tta.json", "tta"),
        ("scenarios.json", "scenarios"),
        ("prediction_contracts.json", "prediction_contracts"),
        ("prediction_ledger.json", "prediction_ledger"),
        ("router_stability.json", "router_stability"),
        ("active_information.json", "active_information"),
        ("meta_label.json", "meta_label"),
    ):
        (out / name).write_text(
            json.dumps(_json_safe(result.get(key, {})), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    manifest = {
        "version": "ultimate_v13_control_plane",
        "schema_version": result.get("schema_version"),
        "fold_count": len(ordered),
        "artifacts": [
            "ultimate_summary.json",
            "model_disagreement.json",
            "predictability.json",
            "future_failure.json",
            "time_to_failure.json",
            "error_correlation.json",
            "regime_transition.json",
            "retrieval.json",
            "uncertainty.json",
            "prediction_strategy.json",
            "prediction_output.json",
            "adaptive_compute.json",
            "metrics_locked.json",
            "statistical_validation.json",
            "revision_metrics.json",
            "worst_case.json",
            "robustness.json",
            "audits.json",
            "tta.json",
            "scenarios.json",
            "prediction_contracts.json",
            "prediction_ledger.json",
            "router_stability.json",
            "active_information.json",
            "meta_label.json",
        ],
        "production_changed": False,
        "promotion_allowed": False,
    }
    (out / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return safe
