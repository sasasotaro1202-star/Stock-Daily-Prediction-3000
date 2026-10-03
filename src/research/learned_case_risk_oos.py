from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCHEMA_VERSION = 1
DEFAULT_QUANTILE = 0.75
DEFAULT_MIN_TRAIN_ROWS = 240
_BASE_FEATURES = (
    "case_predictability",
    "case_ood",
    "case_failure_risk",
    "case_disagreement",
)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clip01(value: Any) -> float | None:
    if not _finite(value):
        return None
    return float(np.clip(float(value), 0.0, 1.0))


def _features(row: Mapping[str, Any]) -> np.ndarray | None:
    values = [_clip01(row.get(key)) for key in _BASE_FEATURES]
    p = _clip01(row.get("prediction"))
    if any(value is None for value in values) or p is None:
        return None
    confidence = abs(p - 0.5) * 2.0
    entropy = -(
        p * math.log(max(p, 1e-12))
        + (1.0 - p) * math.log(max(1.0 - p, 1e-12))
    ) / math.log(2.0)
    return np.asarray([*values, confidence, entropy], dtype=float)


def _fixed_risk_score(row: Mapping[str, Any]) -> float | None:
    values = [_clip01(row.get(key)) for key in _BASE_FEATURES]
    if any(value is None for value in values):
        return None
    predictability, ood, failure_risk, disagreement = values
    return float(
        np.mean(
            [
                1.0 - float(predictability),
                float(ood),
                float(failure_risk),
                float(disagreement),
            ]
        )
    )


def _parse_pit_timestamp(value: Any):
    if not isinstance(value, str):
        return None
    try:
        from datetime import datetime
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _prediction_cutoff(row: Mapping[str, Any]):
    """Resolve the explicit prediction cutoff without silently inventing time.

    Newer ledgers should provide prediction_cutoff. Legacy rows may carry only
    prediction_time, which is treated as the cutoff alias for backward
    compatibility. When both exist they must represent the same instant.
    """
    prediction_raw = row.get("prediction_cutoff")
    prediction = (
        _parse_pit_timestamp(prediction_raw)
        if prediction_raw is not None
        else _parse_pit_timestamp(row.get("prediction_time"))
    )
    if prediction_raw is not None and prediction is None:
        return None
    if row.get("prediction_time") is not None:
        generation = _parse_pit_timestamp(row.get("prediction_time"))
        if generation is None:
            return None
        if prediction is not None and generation != prediction:
            return None
    return prediction


def _pit_ready(row: Mapping[str, Any]) -> bool:
    if row.get("pit_status") != "PASS":
        return False

    cutoff = _prediction_cutoff(row)
    available = _parse_pit_timestamp(row.get("available_at"))
    if cutoff is None or available is None or available > cutoff:
        return False

    published_raw = row.get("published_at")
    retrieved_raw = row.get("retrieved_at")
    published = (
        _parse_pit_timestamp(published_raw)
        if published_raw is not None
        else None
    )
    retrieved = (
        _parse_pit_timestamp(retrieved_raw)
        if retrieved_raw is not None
        else None
    )

    # Optional provenance timestamps are fail-closed when present:
    # publication cannot occur after the prediction cutoff, and retrieval
    # cannot precede source availability.
    if published_raw is not None and published is None:
        return False
    if retrieved_raw is not None and retrieved is None:
        return False
    if published is not None and published > cutoff:
        return False
    if retrieved is not None and retrieved < available:
        return False

    # Defense in depth for derived case features: when the ledger carries
    # explicit feature-level provenance, never let a feature snapshot or
    # lineage status silently bypass the row-level PIT contract.
    feature_status_raw = row.get("feature_pit_status")
    if feature_status_raw is not None and str(feature_status_raw).strip() != "PASS":
        return False
    feature_cutoff_raw = row.get("feature_snapshot_cutoff")
    feature_cutoff = (
        _parse_pit_timestamp(feature_cutoff_raw)
        if feature_cutoff_raw is not None
        else None
    )
    if feature_cutoff_raw is not None and feature_cutoff is None:
        return False
    if feature_cutoff is not None and feature_cutoff > cutoff:
        return False

    feature_available_raw = row.get("feature_max_available_at")
    feature_available = (
        _parse_pit_timestamp(feature_available_raw)
        if feature_available_raw is not None
        else None
    )
    if feature_available_raw is not None and feature_available is None:
        return False
    if feature_available is not None and feature_available > cutoff:
        return False

    # Retrieval is itself an information-availability event for this
    # prediction path; a post-cutoff retrieval cannot make the row PIT-safe.
    if retrieved is not None and retrieved > cutoff:
        return False
    if (
        published is not None
        and retrieved is not None
        and retrieved < published
    ):
        return False
    return True


def _fit(
    features: np.ndarray,
    labels: np.ndarray,
    min_rows: int,
):
    if (
        features.ndim != 2
        or labels.ndim != 1
        or len(features) != len(labels)
        or len(labels) < int(min_rows)
        or np.unique(labels).size < 2
        or not np.isfinite(features).all()
    ):
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
    ).fit(features, labels)


def _quantile(values: Sequence[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    return float(np.quantile(array, q))


def _risk_metrics(scores: np.ndarray, failed: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    failed = np.asarray(failed, dtype=int)
    if len(scores) != len(failed):
        raise ValueError("score/failure arrays must be aligned")
    if len(scores) == 0:
        return {
            "accuracy": float("nan"),
            "logloss": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }

    p = np.clip(scores, 1e-6, 1.0 - 1e-6)
    y = failed.astype(float)
    logloss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
    brier = float(np.mean((p - y) ** 2))
    accuracy = float(np.mean((p >= 0.5) == failed.astype(bool)))

    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (p >= left) & (p < right if right < 1.0 else p <= right)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(
            float(p[mask].mean()) - float(y[mask].mean())
        )

    return {
        "accuracy": accuracy,
        "logloss": logloss,
        "brier": brier,
        "ece": float(ece),
    }


def _fold_stability(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray(
        [float(row[key]) for row in rows if _finite(row.get(key))],
        dtype=float,
    )
    if values.size == 0:
        return {
            "folds": 0,
            "mean": None,
            "std": None,
            "worst": None,
            "best": None,
        }
    return {
        "folds": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "worst": float(values.max()),
        "best": float(values.min()),
    }


def _evaluate(
    scores: np.ndarray,
    failed: np.ndarray,
    threshold: float | None,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=float)
    failed = np.asarray(failed, dtype=int)
    if len(scores) != len(failed):
        raise ValueError("score/failure arrays must be aligned")
    if len(scores) == 0:
        return {
            "n": 0,
            "high_risk_rows": 0,
            "high_risk_coverage": None,
            "high_risk_failure_rate": None,
            "low_risk_failure_rate": None,
            "failure_rate_lift_high_vs_low": None,
            "error_capture_rate": None,
        }
    high = (
        np.zeros(len(scores), dtype=bool)
        if threshold is None
        else scores >= float(threshold)
    )
    low = ~high
    high_n = int(high.sum())
    low_n = int(low.sum())
    high_failures = int(failed[high].sum())
    low_failures = int(failed[low].sum())
    total_failures = int(failed.sum())
    high_rate = high_failures / high_n if high_n else None
    low_rate = low_failures / low_n if low_n else None
    return {
        "n": int(len(scores)),
        "high_risk_rows": high_n,
        "high_risk_coverage": float(high.mean()),
        "high_risk_failure_rate": (
            float(high_rate) if high_rate is not None else None
        ),
        "low_risk_failure_rate": (
            float(low_rate) if low_rate is not None else None
        ),
        "failure_rate_lift_high_vs_low": (
            float(high_rate / low_rate)
            if high_rate is not None and low_rate is not None and low_rate > 0.0
            else None
        ),
        "error_capture_rate": (
            float(high_failures / total_failures) if total_failures else None
        ),
    }


def analyze_learned_case_risk(
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    risk_quantile: float = DEFAULT_QUANTILE,
    min_training_rows: int = DEFAULT_MIN_TRAIN_ROWS,
) -> dict[str, Any]:
    """Frozen-suffix audit for a learned prediction-time case-failure router.

    Development folds produce strictly prior out-of-fold learned-risk scores
    used to set a threshold. The model is then fit on all eligible development
    outcomes and frozen across the locked suffix.
    """
    if not 0.50 <= float(risk_quantile) < 1.0:
        raise ValueError("risk_quantile must be in [0.50, 1.0)")
    if int(min_training_rows) < 30:
        raise ValueError("min_training_rows must be >= 30")

    ordered = sorted(
        (dict(row) for row in ledger_rows),
        key=lambda row: (
            int(row.get("fold", 0) or 0),
            int(row.get("row", 0) or 0),
        ),
    )
    folds = sorted({int(row.get("fold", 0) or 0) for row in ordered})
    locked_folds = sorted(
        {
            fold
            for fold in folds
            if any(
                bool(row.get("is_locked"))
                and int(row.get("fold", 0) or 0) == fold
                for row in ordered
            )
        }
    )
    base = {
        "schema_version": SCHEMA_VERSION,
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
    }
    if not locked_folds:
        return {
            **base,
            "status": "NO_LOCKED_FOLDS",
            "frozen_holdout_used": False,
        }

    first_locked = min(locked_folds)
    development = [
        row for row in ordered
        if int(row.get("fold", 0) or 0) < first_locked
    ]
    locked = [
        row
        for row in ordered
        if int(row.get("fold", 0) or 0) >= first_locked
        and bool(row.get("is_locked"))
    ]

    def valid(row: Mapping[str, Any]):
        if not _pit_ready(row):
            return None
        features = _features(row)
        try:
            failed = int(row["failed"])
        except (KeyError, TypeError, ValueError):
            return None
        if features is None or failed not in (0, 1):
            return None
        return features, failed

    development_by_fold: dict[int, list[tuple[np.ndarray, int]]] = {}
    invalid_development_pit_rows = 0
    development_rows_without_usable_features_or_outcomes = 0
    for row in development:
        if not _pit_ready(row):
            invalid_development_pit_rows += 1
            continue
        item = valid(row)
        if item is not None:
            development_by_fold.setdefault(
                int(row.get("fold", 0) or 0), []
            ).append(item)
        else:
            development_rows_without_usable_features_or_outcomes += 1

    if invalid_development_pit_rows:
        return {
            **base,
            "status": "BLOCKED_INVALID_DEVELOPMENT_PIT",
            "block_reason": (
                "development rows with invalid or missing PIT timestamp "
                "provenance are fail-closed rather than silently excluded"
            ),
            "frozen_holdout_used": True,
            "development_rows": int(len(development)),
            "invalid_development_pit_rows": int(invalid_development_pit_rows),
            "development_rows_without_usable_features_or_outcomes": int(
                development_rows_without_usable_features_or_outcomes
            ),
            "scored_rows": 0,
        }

    history_features: list[np.ndarray] = []
    history_labels: list[int] = []
    development_oof_learned: list[float] = []
    development_oof_fixed: list[float] = []
    development_oof_by_fold: dict[int, dict[str, list[float] | list[int]]] = {}
    training_folds = 0

    for fold in sorted(development_by_fold):
        model = _fit(
            np.vstack(history_features)
            if history_features
            else np.empty((0, len(_BASE_FEATURES) + 2)),
            np.asarray(history_labels, dtype=int)
            if history_labels
            else np.empty((0,), dtype=int),
            min_training_rows,
        )
        for row in development:
            if int(row.get("fold", 0) or 0) != fold:
                continue
            item = valid(row)
            if item is None:
                continue
            features, failed = item
            fold_bucket = development_oof_by_fold.setdefault(
                int(fold),
                {"learned": [], "fixed": [], "failed": []},
            )
            if model is not None:
                learned_score = float(
                    model.predict_proba(features.reshape(1, -1))[0, 1]
                )
                development_oof_learned.append(learned_score)
                fold_bucket["learned"].append(learned_score)
            fixed = _fixed_risk_score(row)
            if fixed is not None:
                development_oof_fixed.append(fixed)
                fold_bucket["fixed"].append(fixed)
            fold_bucket["failed"].append(int(failed))
            history_features.append(features)
            history_labels.append(failed)
        if model is not None:
            training_folds += 1

    trained = _fit(
        np.vstack(history_features)
        if history_features
        else np.empty((0, len(_BASE_FEATURES) + 2)),
        np.asarray(history_labels, dtype=int)
        if history_labels
        else np.empty((0,), dtype=int),
        min_training_rows,
    )

    valid_locked = []
    for row in locked:
        item = valid(row)
        if item is not None:
            valid_locked.append((row, item[0], item[1]))

    if len(valid_locked) != len(locked):
        return {
            **base,
            "status": "BLOCKED_INVALID_LOCKED_CASES",
            "block_reason": (
                "all locked rows require valid prediction-time features, "
                "outcomes, and PIT lineage"
            ),
            "frozen_holdout_used": True,
            "locked_rows": int(len(locked)),
            "scored_rows": 0,
            "invalid_rows": int(len(locked) - len(valid_locked)),
        }

    if trained is None:
        return {
            **base,
            "status": "INSUFFICIENT_DEVELOPMENT_HISTORY",
            "frozen_holdout_used": True,
            "development_rows": int(len(history_labels)),
            "development_oof_risk_rows": int(len(development_oof_learned)),
        }

    learned_threshold = _quantile(
        development_oof_learned,
        risk_quantile,
    )
    fixed_threshold = _quantile(development_oof_fixed, risk_quantile)
    if learned_threshold is None or fixed_threshold is None:
        return {
            **base,
            "status": "INSUFFICIENT_DEVELOPMENT_RISK_HISTORY",
            "frozen_holdout_used": True,
            "development_rows": int(len(history_labels)),
            "development_oof_risk_rows": int(len(development_oof_learned)),
        }

    locked_features = np.vstack([features for _, features, _ in valid_locked])
    locked_failed = np.asarray(
        [failed for _, _, failed in valid_locked],
        dtype=int,
    )
    learned_scores = trained.predict_proba(locked_features)[:, 1]
    fixed_scores = np.asarray(
        [_fixed_risk_score(row) for row, _, _ in valid_locked],
        dtype=float,
    )
    learned_eval = _evaluate(
        learned_scores,
        locked_failed,
        learned_threshold,
    )
    fixed_eval = _evaluate(
        fixed_scores,
        locked_failed,
        fixed_threshold,
    )
    learned_risk_metrics = _risk_metrics(learned_scores, locked_failed)
    fixed_risk_metrics = _risk_metrics(fixed_scores, locked_failed)

    locked_rows_by_fold: dict[int, list[tuple[float, float, int]]] = {}
    for (row, _, failed), learned_score, fixed_score in zip(
        valid_locked,
        learned_scores,
        fixed_scores,
    ):
        locked_rows_by_fold.setdefault(
            int(row.get("fold", 0) or 0),
            [],
        ).append((float(learned_score), float(fixed_score), int(failed)))

    fold_metrics: list[dict[str, Any]] = []
    for fold in sorted(locked_rows_by_fold):
        fold_rows = locked_rows_by_fold[fold]
        fold_learned = np.asarray([r[0] for r in fold_rows], dtype=float)
        fold_fixed = np.asarray([r[1] for r in fold_rows], dtype=float)
        fold_failed = np.asarray([r[2] for r in fold_rows], dtype=int)
        fold_metrics.append(
            {
                "fold": int(fold),
                "rows": int(len(fold_rows)),
                "learned": {
                    "risk_metrics": _risk_metrics(fold_learned, fold_failed),
                    "high_risk": _evaluate(
                        fold_learned,
                        fold_failed,
                        learned_threshold,
                    ),
                },
                "fixed": {
                    "risk_metrics": _risk_metrics(fold_fixed, fold_failed),
                    "high_risk": _evaluate(
                        fold_fixed,
                        fold_failed,
                        fixed_threshold,
                    ),
                },
            }
        )

    return {
        **base,
        "status": "EVALUATED",
        "frozen_holdout_used": True,
        "locked_folds": locked_folds,
        "development_folds": sorted(development_by_fold),
        "development_rows": int(len(history_labels)),
        "development_oof_risk_rows": int(len(development_oof_learned)),
        "development_rows_without_usable_features_or_outcomes": int(
            development_rows_without_usable_features_or_outcomes
        ),
        "model_training_folds": int(training_folds),
        "locked_rows": int(len(locked_failed)),
        "method": {
            "name": "frozen_prequential_learned_case_risk",
            "classifier": "standardized_logistic_regression",
            "features": [
                *_BASE_FEATURES,
                "prediction_confidence",
                "normalized_entropy",
            ],
            "threshold_source": "development_only_out_of_fold_risk_quantile",
            "risk_quantile": float(risk_quantile),
            "min_training_rows": int(min_training_rows),
            "locked_suffix_frozen": True,
        },
        "fixed_case_risk": {
            "threshold": float(fixed_threshold),
            "evaluation": fixed_eval,
            "risk_metrics": fixed_risk_metrics,
        },
        "learned_case_risk": {
            "threshold": float(learned_threshold),
            "evaluation": learned_eval,
            "risk_metrics": learned_risk_metrics,
        },
        "development_fold_metrics": [
            {
                "fold": int(fold),
                "rows": int(len(bucket["failed"])),
                "learned": {
                    "risk_metrics": _risk_metrics(
                        np.asarray(bucket["learned"], dtype=float),
                        np.asarray(bucket["failed"], dtype=int),
                    ),
                },
                "fixed": {
                    "risk_metrics": _risk_metrics(
                        np.asarray(bucket["fixed"], dtype=float),
                        np.asarray(bucket["failed"], dtype=int),
                    ),
                },
            }
            for fold, bucket in sorted(development_oof_by_fold.items())
            if bucket["learned"] and bucket["fixed"]
        ],
        "fold_metrics": fold_metrics,
        "stability": {
            "development_fold_count": int(
                sum(
                    1
                    for bucket in development_oof_by_fold.values()
                    if bucket["learned"] and bucket["fixed"]
                )
            ),
            "development_learned_logloss": _fold_stability(
                [
                    {"logloss": _risk_metrics(
                        np.asarray(bucket["learned"], dtype=float),
                        np.asarray(bucket["failed"], dtype=int),
                    )["logloss"]}
                    for bucket in development_oof_by_fold.values()
                    if bucket["learned"] and bucket["fixed"]
                ],
                "logloss",
            ),
            "development_fixed_logloss": _fold_stability(
                [
                    {"logloss": _risk_metrics(
                        np.asarray(bucket["fixed"], dtype=float),
                        np.asarray(bucket["failed"], dtype=int),
                    )["logloss"]}
                    for bucket in development_oof_by_fold.values()
                    if bucket["learned"] and bucket["fixed"]
                ],
                "logloss",
            ),
            "locked_fold_count": int(len(fold_metrics)),
            "locked_learned_logloss": _fold_stability(
                [
                    {"logloss": row["learned"]["risk_metrics"]["logloss"]}
                    for row in fold_metrics
                ],
                "logloss",
            ),
            "locked_fixed_logloss": _fold_stability(
                [
                    {"logloss": row["fixed"]["risk_metrics"]["logloss"]}
                    for row in fold_metrics
                ],
                "logloss",
            ),
            "locked_learned_brier": _fold_stability(
                [
                    {"brier": row["learned"]["risk_metrics"]["brier"]}
                    for row in fold_metrics
                ],
                "brier",
            ),
            "locked_fixed_brier": _fold_stability(
                [
                    {"brier": row["fixed"]["risk_metrics"]["brier"]}
                    for row in fold_metrics
                ],
                "brier",
            ),
        },
        "delta_learned_minus_fixed": {
            "high_risk_coverage": float(
                learned_eval["high_risk_coverage"]
                - fixed_eval["high_risk_coverage"]
            ),
            "error_capture_rate": (
                float(
                    learned_eval["error_capture_rate"]
                    - fixed_eval["error_capture_rate"]
                )
                if learned_eval["error_capture_rate"] is not None
                and fixed_eval["error_capture_rate"] is not None
                else None
            ),
            "failure_rate_lift": (
                float(
                    learned_eval["failure_rate_lift_high_vs_low"]
                    - fixed_eval["failure_rate_lift_high_vs_low"]
                )
                if learned_eval["failure_rate_lift_high_vs_low"] is not None
                and fixed_eval["failure_rate_lift_high_vs_low"] is not None
                else None
            ),
        },
        "contracts": {
            "development_model_training_uses_only_prior_folds": True,
            "development_threshold_uses_oof_risk_only": True,
            "locked_suffix_model_is_frozen": True,
            "locked_outcomes_used_for_fit_or_threshold": False,
            "pit_requires_timezone_aware_available_at_le_prediction_time": True,
            "development_pit_invalid_rows_fail_closed": True,
            "production_changed": False,
            "promotion_allowed": False,
        },
    }


__all__ = ["analyze_learned_case_risk"]
