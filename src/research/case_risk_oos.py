from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1
RISK_QUANTILE = 0.75
REQUIRED_COMPONENTS = (
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


def case_risk_score(row: Mapping[str, Any]) -> float | None:
    """Build a fixed, research-only case-risk score from prediction-time signals.

    The score intentionally uses no outcome-derived quantity. All four
    per-case signals are required; missing/invalid state fails closed instead
    of silently imputing a risk level. Legacy fold-level summary fields are
    deliberately ignored so a fold aggregate cannot masquerade as case risk.
    """
    predictability = _clip01(row.get("case_predictability"))
    ood = _clip01(row.get("case_ood"))
    failure_risk = _clip01(row.get("case_failure_risk"))
    disagreement = _clip01(row.get("case_disagreement"))
    values = (predictability, ood, failure_risk, disagreement)
    if any(value is None for value in values):
        return None
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


def _binary_metrics(y: Sequence[int], probabilities: Sequence[float]) -> dict[str, float | int | None]:
    yy = np.asarray(y, dtype=int)
    pp = np.asarray(probabilities, dtype=float)
    if len(yy) != len(pp):
        raise ValueError("y/probability length mismatch")
    if len(yy) == 0:
        return {
            "n": 0,
            "accuracy": None,
            "logloss": None,
            "brier": None,
            "ece": None,
            "error_rate": None,
        }
    if np.unique(yy).size < 2:
        # Accuracy/logloss/brier remain meaningful for a one-class slice.
        pass
    if not np.isfinite(pp).all() or np.any((pp < 0.0) | (pp > 1.0)):
        raise ValueError("probability values must be finite and in [0, 1]")
    clipped = np.clip(pp, 1e-12, 1.0 - 1e-12)
    pred = clipped >= 0.5
    accuracy = float(np.mean(pred == yy))
    logloss = float(
        -np.mean(
            yy * np.log(clipped) + (1 - yy) * np.log(1.0 - clipped)
        )
    )
    brier = float(np.mean((clipped - yy) ** 2))

    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (clipped >= left) & (
            (clipped < right) if right < 1.0 else (clipped <= right)
        )
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(
            float(clipped[mask].mean()) - float(yy[mask].mean())
        )

    return {
        "n": int(len(yy)),
        "accuracy": accuracy,
        "logloss": logloss,
        "brier": brier,
        "ece": float(ece),
        "error_rate": float(1.0 - accuracy),
    }




def _pit_ready(row: Mapping[str, Any]) -> bool:
    """Use the canonical row-level fail-closed PIT contract."""
    from src.research.pit_contract import audit_pit_row

    return bool(audit_pit_row(row)["ok"])


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    x = np.asarray(list(values), dtype=float)
    if x.size == 0 or not np.isfinite(x).all():
        return None
    return float(np.quantile(x, float(quantile)))


def analyze_case_risk(
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    risk_quantile: float = RISK_QUANTILE,
) -> dict[str, Any]:
    """Evaluate whether prior-known risk signals concentrate future errors.

    Locked/OOS folds are evaluated chronologically. For every locked fold, the
    high-risk threshold is derived only from scores in strictly earlier folds.
    Outcomes are used only after the risk assignment is fixed, so the diagnostic
    is prequential with respect to the risk threshold.

    This is diagnostic-only. It never changes model, routing, calibration, or
    production state.
    """
    if not 0.50 <= float(risk_quantile) < 1.0:
        raise ValueError("risk_quantile must be in [0.50, 1.0)")

    prior_scores: list[float] = []
    locked_rows = 0
    scored_rows = 0
    invalid_rows = 0
    per_fold: list[dict[str, Any]] = []
    pooled: list[tuple[int, float, int, float, bool]] = []

    ordered = sorted(
        (dict(row) for row in ledger_rows),
        key=lambda row: (
            int(row.get("fold", 0) or 0),
            int(row.get("row", 0) or 0),
        ),
    )

    # Build the score history over every chronological OOS fold, but evaluate
    # risk concentration only on the frozen/locked suffix. PIT lineage is a
    # hard precondition: unknown prediction/availability timestamps fail closed.
    for fold in sorted({int(row.get("fold", 0) or 0) for row in ordered}):
        fold_rows = [
            row for row in ordered if int(row.get("fold", 0) or 0) == fold
        ]
        current_scores = [
            score
            for row in fold_rows
            if _pit_ready(row) and (score := case_risk_score(row)) is not None
        ]

        locked = [row for row in fold_rows if bool(row.get("is_locked"))]
        if locked:
            threshold = _percentile(prior_scores, risk_quantile)
            fold_scores: list[float] = []
            fold_y: list[int] = []
            fold_p: list[float] = []
            fold_high: list[bool] = []
            fold_invalid = 0

            for row in locked:
                score = case_risk_score(row)
                if score is None or not _pit_ready(row):
                    fold_invalid += 1
                    continue
                try:
                    outcome = int(row["result"])
                    probability = float(row["prediction"])
                except (KeyError, TypeError, ValueError):
                    fold_invalid += 1
                    continue
                if outcome not in (0, 1) or not _finite(probability):
                    fold_invalid += 1
                    continue

                high = bool(threshold is not None and score >= threshold)
                fold_scores.append(score)
                fold_y.append(outcome)
                fold_p.append(float(np.clip(probability, 0.0, 1.0)))
                fold_high.append(high)
                pooled.append((fold, score, outcome, probability, high))

            locked_rows += len(locked)
            scored_rows += len(fold_scores)
            invalid_rows += fold_invalid

            high_mask = np.asarray(fold_high, dtype=bool)
            low_mask = ~high_mask
            fold_metrics: dict[str, Any] = {
                "fold": fold,
                "locked_rows": int(len(locked)),
                "scored_rows": int(len(fold_scores)),
                "invalid_rows": int(fold_invalid),
                "prior_score_count": int(len(prior_scores)),
                "threshold_source": (
                    "strictly_prior_oos_scores"
                    if threshold is not None
                    else "NO_PRIOR_SCORE_THRESHOLD"
                ),
                "high_risk_threshold": threshold,
                "high_risk_coverage": (
                    float(high_mask.mean()) if len(high_mask) else None
                ),
            }
            if fold_scores:
                fold_metrics["risk_score_mean"] = float(np.mean(fold_scores))
                fold_metrics["risk_score_std"] = float(np.std(fold_scores))

            if len(fold_scores):
                fold_metrics["all"] = _binary_metrics(fold_y, fold_p)
            else:
                fold_metrics["all"] = _binary_metrics([], [])

            fold_metrics["high_risk"] = _binary_metrics(
                np.asarray(fold_y)[high_mask],
                np.asarray(fold_p)[high_mask],
            )
            fold_metrics["low_risk"] = _binary_metrics(
                np.asarray(fold_y)[low_mask],
                np.asarray(fold_p)[low_mask],
            )
            fold_metrics["status"] = (
                "EVALUATED" if threshold is not None else "WARMUP_NO_PRIOR_THRESHOLD"
            )
            per_fold.append(fold_metrics)

        # Scores from the current fold become available only after its outcome
        # evaluation has finished; they can define the next fold's threshold.
        prior_scores.extend(current_scores)

    if invalid_rows:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "BLOCKED_INVALID_LOCKED_CASES",
            "block_reason": (
                "locked case aggregation is fail-closed: every locked row must "
                "have valid per-case signals and explicit PIT lineage"
            ),
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
            "frozen_holdout_used": False,
            "locked_rows": int(locked_rows),
            "scored_rows": 0,
            "invalid_rows": int(invalid_rows),
            "per_fold": per_fold,
            "aggregate": {},
        }

    if not scored_rows:
        pit_blocked = any(
            bool(row.get("is_locked"))
            and not _pit_ready(row)
            for row in ordered
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "BLOCKED_PIT_LINEAGE" if pit_blocked else "NO_VALID_LOCKED_CASES",
            "block_reason": (
                "explicit timezone-aware available_at <= prediction_time and pit_status=PASS are required"
                if pit_blocked else None
            ),
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
            "frozen_holdout_used": False,
            "locked_rows": int(locked_rows),
            "scored_rows": 0,
            "invalid_rows": int(invalid_rows),
            "per_fold": per_fold,
            "aggregate": {},
        }

    scores = np.asarray([x[1] for x in pooled], dtype=float)
    y = np.asarray([x[2] for x in pooled], dtype=int)
    p = np.asarray([x[3] for x in pooled], dtype=float)
    high = np.asarray([x[4] for x in pooled], dtype=bool)
    failures = (p >= 0.5) != (y.astype(bool))

    all_metrics = _binary_metrics(y, p)
    high_metrics = _binary_metrics(y[high], p[high])
    low_metrics = _binary_metrics(y[~high], p[~high])

    high_failures = int(failures[high].sum())
    total_failures = int(failures.sum())
    error_capture = (
        float(high_failures / total_failures)
        if total_failures
        else None
    )
    high_failure_rate = (
        float(failures[high].mean()) if high.any() else None
    )
    low_failure_rate = (
        float(failures[~high].mean()) if (~high).any() else None
    )
    failure_rate_lift = (
        float(high_failure_rate / low_failure_rate)
        if high_failure_rate is not None
        and low_failure_rate is not None
        and low_failure_rate > 0.0
        else None
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "EVALUATED",
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "frozen_holdout_used": False,
        "method": {
            "name": "prequential_case_risk_audit",
            "risk_score": (
                "mean(1-predictability, ood, failure_risk, disagreement)"
            ),
            "threshold": "strictly_prior_oos_score_quantile",
            "risk_quantile": float(risk_quantile),
            "outcomes_used_for_threshold": False,
        },
        "locked_rows": int(locked_rows),
        "scored_rows": int(scored_rows),
        "invalid_rows": int(invalid_rows),
        "risk_score": {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "high_risk_rows": int(high.sum()),
            "high_risk_coverage": float(high.mean()),
        },
        "aggregate": {
            "all": all_metrics,
            "high_risk": high_metrics,
            "low_risk": low_metrics,
            "error_capture_rate": error_capture,
            "failure_rate_lift_high_vs_low": failure_rate_lift,
            "high_minus_low_logloss": (
                float(high_metrics["logloss"] - low_metrics["logloss"])
                if high_metrics["logloss"] is not None
                and low_metrics["logloss"] is not None
                else None
            ),
            "high_minus_low_brier": (
                float(high_metrics["brier"] - low_metrics["brier"])
                if high_metrics["brier"] is not None
                and low_metrics["brier"] is not None
                else None
            ),
            "high_minus_low_ece": (
                float(high_metrics["ece"] - low_metrics["ece"])
                if high_metrics["ece"] is not None
                and low_metrics["ece"] is not None
                else None
            ),
        },
        "per_fold": per_fold,
        "contracts": {
            "only_locked_oos_rows_evaluated": True,
            "threshold_uses_strictly_prior_oos_scores": True,
            "current_fold_scores_added_after_evaluation": True,
            "outcomes_used_for_threshold": False,
            "production_changed": False,
            "promotion_allowed": False,
        },
    }
