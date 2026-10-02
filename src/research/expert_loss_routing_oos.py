from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.ultimate_control_v13 import (
    _dynamic_routing_weights,
    _history_quality_weights,
)

SCHEMA_VERSION = 1
DEFAULT_MIN_TRAINING_ROWS = 240
DEFAULT_TEMPERATURE = 0.25
DEFAULT_UNIFORM_MIX = 0.10


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clip01(value: Any) -> float | None:
    if not _finite(value):
        return None
    return float(np.clip(float(value), 0.0, 1.0))


def _pit_ready(row: Mapping[str, Any]) -> bool:
    if row.get("pit_status") != "PASS":
        return False
    prediction_time = row.get("prediction_time")
    available_at = row.get("available_at")
    if not isinstance(prediction_time, str) or not isinstance(available_at, str):
        return False
    try:
        prediction = datetime.fromisoformat(
            prediction_time.replace("Z", "+00:00")
        )
        available = datetime.fromisoformat(
            available_at.replace("Z", "+00:00")
        )
    except ValueError:
        return False
    return (
        prediction.tzinfo is not None
        and prediction.utcoffset() is not None
        and available.tzinfo is not None
        and available.utcoffset() is not None
        and available <= prediction
    )


def _case_features(row: Mapping[str, Any], expert_prediction: float) -> np.ndarray | None:
    p = _clip01(expert_prediction)
    values = [
        _clip01(row.get("case_predictability")),
        _clip01(row.get("case_ood")),
        _clip01(row.get("case_failure_risk")),
        _clip01(row.get("case_disagreement")),
    ]
    if p is None or any(value is None for value in values):
        return None
    confidence = abs(p - 0.5) * 2.0
    entropy = -(
        p * math.log(max(p, 1e-12))
        + (1.0 - p) * math.log(max(1.0 - p, 1e-12))
    ) / math.log(2.0)
    return np.asarray([p, confidence, entropy, *values], dtype=float)


def _fit_failure_model(
    features: Sequence[np.ndarray],
    labels: Sequence[int],
    min_training_rows: int,
):
    if len(features) != len(labels) or len(labels) < int(min_training_rows):
        return None
    x = np.vstack(features).astype(float)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or not np.isfinite(x).all() or np.unique(y).size < 2:
        return None
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.25,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=20261002,
                ),
            ),
        ]
    ).fit(x, y)


def _smoothed_failure_rate(labels: Sequence[int]) -> float | None:
    if not labels:
        return None
    y = np.asarray(labels, dtype=float)
    if not np.isfinite(y).all():
        return None
    return float((float(y.sum()) + 0.5) / (len(y) + 1.0))


def _weights_from_predicted_loss(
    predicted_loss: np.ndarray,
    *,
    temperature: float,
    uniform_mix: float,
) -> np.ndarray:
    loss = np.asarray(predicted_loss, dtype=float)
    if loss.ndim != 2 or not np.isfinite(loss).all():
        raise ValueError("predicted expert loss must be finite 2D values")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must be in [0, 1]")
    shifted = -loss / float(temperature)
    shifted -= np.max(shifted, axis=1, keepdims=True)
    softmax = np.exp(shifted)
    softmax /= np.sum(softmax, axis=1, keepdims=True)
    experts = softmax.shape[1]
    return (
        (1.0 - uniform_mix) * softmax
        + uniform_mix * (1.0 / max(1, experts))
    )


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    if len(y) == 0 or len(y) != len(p):
        return {
            "n": float(len(y)),
            "accuracy": float("nan"),
            "logloss": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }
    pred = (p >= 0.5).astype(int)
    logloss = float(
        -np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p))
    )
    brier = float(np.mean((p - y) ** 2))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p <= hi if hi == 1.0 else p < hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {
        "n": float(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "logloss": logloss,
        "brier": brier,
        "ece": float(ece),
    }


def analyze_expert_loss_routing(
    ledger_rows: Sequence[Mapping[str, Any]],
    fold_results: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str] | None = None,
    locked_folds: int = 1,
    min_training_rows: int = DEFAULT_MIN_TRAINING_ROWS,
    temperature: float = DEFAULT_TEMPERATURE,
    uniform_mix: float = DEFAULT_UNIFORM_MIX,
) -> dict[str, Any]:
    """Research-only causal expert-loss routing audit.

    For each fold, every expert-loss model is fit only on strictly earlier
    folds. The current fold's outcomes are used only to score the resulting
    routing candidate. The locked suffix is never used for fitting or gating.
    """
    model_names = list(models or [])
    if len(model_names) < 2:
        raise ValueError("at least two expert models are required")
    if int(locked_folds) < 1:
        raise ValueError("locked_folds must be >= 1")

    base = {
        "schema_version": SCHEMA_VERSION,
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "frozen_holdout_used": False,
    }

    ordered_folds = [dict(row) for row in fold_results]
    if len(ordered_folds) <= int(locked_folds):
        return {
            **base,
            "status": "INSUFFICIENT_FOLDS",
            "folds": int(len(ordered_folds)),
        }

    locked_start = len(ordered_folds) - int(locked_folds)
    ledger: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in ledger_rows:
        try:
            key = (int(raw["fold"]), int(raw["row"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key in ledger:
            return {
                **base,
                "status": "BLOCKED_DUPLICATE_LEDGER_KEYS",
            }
        ledger[key] = dict(raw)

    expert_history: dict[str, list[tuple[np.ndarray, int]]] = {
        model: [] for model in model_names
    }
    expert_global_failures: dict[str, list[int]] = {
        model: [] for model in model_names
    }
    locked_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    fitted_models = 0

    for fold_index, fold in enumerate(ordered_folds):
        y = np.asarray(fold.get("y", []), dtype=int)
        predictions = fold.get("predictions")
        if not isinstance(predictions, Mapping):
            return {**base, "status": "BLOCKED_MISSING_EXPERT_PREDICTIONS"}

        p_columns = []
        for model in model_names:
            if model not in predictions:
                return {
                    **base,
                    "status": "BLOCKED_MISSING_EXPERT_PREDICTION",
                    "model": model,
                    "fold": int(fold_index),
                }
            values = np.asarray(predictions[model], dtype=float)
            if values.ndim != 1 or len(values) != len(y):
                return {
                    **base,
                    "status": "BLOCKED_MISALIGNED_EXPERT_PREDICTIONS",
                    "model": model,
                    "fold": int(fold_index),
                }
            p_columns.append(np.clip(values, 0.0, 1.0))
        p_matrix = np.column_stack(p_columns)

        current_rows = []
        for row_index in range(len(y)):
            meta = ledger.get((fold_index, row_index))
            if meta is None:
                return {
                    **base,
                    "status": "BLOCKED_MISSING_CASE_LINEAGE",
                    "fold": int(fold_index),
                    "row": int(row_index),
                }
            if not _pit_ready(meta):
                return {
                    **base,
                    "status": "BLOCKED_INVALID_CASE_LINEAGE",
                    "fold": int(fold_index),
                    "row": int(row_index),
                }
            current_rows.append(meta)

        predicted_loss = np.empty_like(p_matrix, dtype=float)
        model_status: dict[str, str] = {}
        for model_index, model in enumerate(model_names):
            history = expert_history[model]
            features = [item[0] for item in history]
            labels = [item[1] for item in history]
            fitted = _fit_failure_model(
                features,
                labels,
                min_training_rows=min_training_rows,
            )
            if fitted is not None:
                current_features = []
                for row_index, meta in enumerate(current_rows):
                    feature = _case_features(
                        meta,
                        float(p_matrix[row_index, model_index]),
                    )
                    if feature is None:
                        return {
                            **base,
                            "status": "BLOCKED_MISSING_CASE_FEATURES",
                            "fold": int(fold_index),
                            "row": int(row_index),
                            "model": model,
                        }
                    current_features.append(feature)
                predicted_loss[:, model_index] = fitted.predict_proba(
                    np.vstack(current_features)
                )[:, 1]
                model_status[model] = "EXECUTED_PRIOR_ONLY_FAILURE_MODEL"
                fitted_models += 1
            else:
                fallback = _smoothed_failure_rate(labels)
                if fallback is None:
                    predicted_loss[:, model_index] = 0.5
                    model_status[model] = "BLOCKED_NO_PRIOR_EXPERT_HISTORY"
                else:
                    predicted_loss[:, model_index] = fallback
                    model_status[model] = "EXECUTED_PRIOR_ONLY_GLOBAL_FAILURE_RATE"

        loss_weights = _weights_from_predicted_loss(
            predicted_loss,
            temperature=temperature,
            uniform_mix=uniform_mix,
        )

        quality = _history_quality_weights(ordered_folds[:fold_index], model_names)
        dynamic_weights = _dynamic_routing_weights(p_matrix, quality)
        dynamic_prediction = np.sum(p_matrix * dynamic_weights, axis=1)
        candidate_prediction = np.sum(p_matrix * loss_weights, axis=1)

        candidate_metrics = _binary_metrics(y, candidate_prediction)
        dynamic_metrics = _binary_metrics(y, dynamic_prediction)
        fold_result = {
            "fold": int(fold_index),
            "is_locked": bool(fold_index >= locked_start),
            "status": (
                "EXECUTED_PRIOR_ONLY"
                if all(value.startswith("EXECUTED") for value in model_status.values())
                else "FALLBACK_OR_BLOCKED_PRIOR_HISTORY"
            ),
            "model_status": model_status,
            "candidate": candidate_metrics,
            "dynamic": dynamic_metrics,
            "delta_candidate_minus_dynamic": {
                key: float(candidate_metrics[key] - dynamic_metrics[key])
                for key in ("accuracy", "logloss", "brier", "ece")
            },
            "predicted_loss_mean": {
                model: float(predicted_loss[:, i].mean())
                for i, model in enumerate(model_names)
            },
            "weight_means": {
                model: float(loss_weights[:, i].mean())
                for i, model in enumerate(model_names)
            },
            "weight_concentration_mean": float(np.mean(np.max(loss_weights, axis=1))),
        }
        fold_rows.append(fold_result)

        if fold_index >= locked_start:
            for row_index, meta in enumerate(current_rows):
                for model_index, model in enumerate(model_names):
                    actual_failed = int(
                        (p_matrix[row_index, model_index] >= 0.5) != bool(y[row_index])
                    )
                    locked_rows.append({
                        "fold": int(fold_index),
                        "row": int(row_index),
                        "model": model,
                        "predicted_failure": float(predicted_loss[row_index, model_index]),
                        "failed": actual_failed,
                    })

        # Only after the current fold has been fully evaluated may its outcomes
        # enter future expert-loss histories.
        for row_index, meta in enumerate(current_rows):
            for model_index, model in enumerate(model_names):
                feature = _case_features(
                    meta,
                    float(p_matrix[row_index, model_index]),
                )
                if feature is None:
                    return {
                        **base,
                        "status": "BLOCKED_MISSING_CASE_FEATURES_AFTER_EVALUATION",
                        "fold": int(fold_index),
                        "row": int(row_index),
                        "model": model,
                    }
                failed = int(
                    (p_matrix[row_index, model_index] >= 0.5) != bool(y[row_index])
                )
                expert_history[model].append((feature, failed))
                expert_global_failures[model].append(failed)

    locked_candidate = [r["candidate"] for r in fold_rows if r["is_locked"]]
    locked_dynamic = [r["dynamic"] for r in fold_rows if r["is_locked"]]
    if not locked_candidate:
        return {**base, "status": "NO_LOCKED_FOLDS"}

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
        total = sum(int(r["n"]) for r in rows)
        if total <= 0:
            return {
                "n": 0.0,
                "accuracy": float("nan"),
                "logloss": float("nan"),
                "brier": float("nan"),
                "ece": float("nan"),
            }
        # Aggregate by recomputing from per-case predictions is preferable, but
        # per-fold metric averaging would overweight small folds. The current
        # implementation keeps the locked suffix fold sizes explicit.
        weighted: dict[str, float] = {}
        for key in ("accuracy", "logloss", "brier", "ece"):
            weighted[key] = float(
                sum(float(r[key]) * int(r["n"]) for r in rows) / total
            )
        weighted["n"] = float(total)
        return weighted

    candidate_aggregate = aggregate(locked_candidate)
    dynamic_aggregate = aggregate(locked_dynamic)

    loss_y = np.asarray([row["failed"] for row in locked_rows], dtype=int)
    loss_p = np.asarray([row["predicted_failure"] for row in locked_rows], dtype=float)
    expert_loss_brier = (
        float(np.mean((loss_p - loss_y) ** 2))
        if len(loss_y)
        else float("nan")
    )

    locked_concentration = [
        float(r["weight_concentration_mean"])
        for r in fold_rows
        if r["is_locked"]
    ]

    delta = {
        "accuracy": float(candidate_aggregate["accuracy"] - dynamic_aggregate["accuracy"]),
        "logloss": float(candidate_aggregate["logloss"] - dynamic_aggregate["logloss"]),
        "brier": float(candidate_aggregate["brier"] - dynamic_aggregate["brier"]),
        "ece": float(candidate_aggregate["ece"] - dynamic_aggregate["ece"]),
    }

    return {
        **base,
        "status": "EVALUATED",
        "locked_folds": list(range(locked_start, len(ordered_folds))),
        "folds": int(len(ordered_folds)),
        "development_folds": list(range(locked_start)),
        "model_names": model_names,
        "development_rows": int(sum(len(np.asarray(f.get("y", []))) for f in ordered_folds[:locked_start])),
        "locked_rows": int(sum(len(np.asarray(f.get("y", []))) for f in ordered_folds[locked_start:])),
        "fitted_expert_loss_models": int(fitted_models),
        "method": {
            "name": "prequential_expert_loss_routing",
            "classifier": "standardized_logistic_regression_per_expert",
            "features": [
                "expert_prediction",
                "expert_prediction_confidence",
                "expert_prediction_entropy",
                "case_predictability",
                "case_ood",
                "case_failure_risk",
                "case_disagreement",
            ],
            "routing": "softmax_negative_predicted_failure_probability",
            "temperature": float(temperature),
            "uniform_mix": float(uniform_mix),
            "min_training_rows": int(min_training_rows),
            "fit_policy": "strictly_prior_folds_only",
            "locked_suffix_frozen": True,
        },
        "locked_metrics": {
            "expert_loss_routing": candidate_aggregate,
            "existing_dynamic_routing": dynamic_aggregate,
            "delta_candidate_minus_dynamic": delta,
        },
        "expert_loss_prediction": {
            "locked_brier": expert_loss_brier,
            "rows": int(len(loss_y)),
        },
        "routing_stability": {
            "locked_mean_weight_concentration": float(np.mean(locked_concentration)),
            "locked_max_weight_concentration": float(np.max(locked_concentration)),
        },
        "fold_results": fold_rows,
        "contracts": {
            "current_fold_outcomes_used_for_routing": False,
            "current_fold_outcomes_used_for_expert_loss_fit": False,
            "current_fold_outcomes_used_for_threshold_or_temperature_selection": False,
            "all_expert_loss_models_fit_only_on_prior_folds": True,
            "locked_suffix_routing_is_frozen_per_fold": True,
            "pit_requires_timezone_aware_available_at_le_prediction_time": True,
            "frozen_holdout_used": False,
            "production_changed": False,
            "promotion_allowed": False,
        },
    }


__all__ = ["analyze_expert_loss_routing"]
