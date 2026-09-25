from __future__ import annotations

from math import isfinite, sqrt
from statistics import NormalDist
from typing import Iterable, Mapping

from .statistics import moving_block_bootstrap_mean


def _selection_score(metric: Mapping[str, object]) -> float:
    mean = float(
        metric.get("selection_logloss", metric.get("logloss", float("inf")))
    )
    std = float(metric.get("selection_logloss_std", metric.get("logloss_std", 0.0)))
    if not isfinite(mean) or not isfinite(std):
        return float("inf")
    return mean + 0.25 * max(0.0, std)


def _fold_values(rows: Iterable[Mapping[str, object]]) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in rows:
        try:
            fold = int(float(row["fold"]))
            value = float(row["logloss"])
        except (KeyError, TypeError, ValueError):
            continue
        if isfinite(value):
            values[fold] = value
    return values


def paired_logloss_selection_evidence(
    selected_model: str,
    candidate_metrics: Mapping[str, Mapping[str, object]],
    fold_rows_by_model: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    trial_count: int,
    min_folds: int = 5,
    min_relative_improvement: float = 0.03,
    alpha: float = 0.05,
) -> dict[str, object]:
    """
    Conservative paired-fold evidence for a model selected from chronological OOS.

    Positive improvement means the selected model has lower LogLoss than the
    comparator on the same OOS folds. The confidence bound is multiple-comparison
    adjusted using Bonferroni and is never computed from the frozen holdout.
    """
    candidates = {
        str(name): metric
        for name, metric in candidate_metrics.items()
        if str(name) != str(selected_model)
    }
    if selected_model not in candidate_metrics:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "eligible_for_freeze": False,
            "reason": "selected_model_metrics_missing",
            "selected_model": str(selected_model),
        }
    if not candidates:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "eligible_for_freeze": False,
            "reason": "no_comparator_candidate",
            "selected_model": str(selected_model),
            "trial_count": int(max(1, trial_count)),
        }

    comparator_model = min(
        candidates,
        key=lambda name: _selection_score(candidates[name]),
    )
    selected_rows = _fold_values(fold_rows_by_model.get(selected_model, ()))
    comparator_rows = _fold_values(fold_rows_by_model.get(comparator_model, ()))
    common_folds = sorted(set(selected_rows).intersection(comparator_rows))

    if len(common_folds) < int(min_folds):
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "eligible_for_freeze": False,
            "reason": "too_few_common_oos_folds",
            "selected_model": str(selected_model),
            "comparator_model": str(comparator_model),
            "common_folds": int(len(common_folds)),
            "minimum_common_folds": int(min_folds),
            "trial_count": int(max(1, trial_count)),
        }

    # Comparator - selected: positive values are improvements in LogLoss.
    differences = [
        comparator_rows[fold] - selected_rows[fold]
        for fold in common_folds
    ]
    n = len(differences)
    mean_gain = sum(differences) / n
    variance = (
        sum((value - mean_gain) ** 2 for value in differences) / (n - 1)
        if n >= 2
        else 0.0
    )
    std = sqrt(max(0.0, variance))
    se = std / sqrt(n) if n > 0 else float("inf")

    comparator_mean = sum(comparator_rows[fold] for fold in common_folds) / n
    selected_mean = sum(selected_rows[fold] for fold in common_folds) / n
    relative_improvement = (
        mean_gain / abs(comparator_mean)
        if isfinite(comparator_mean) and comparator_mean != 0.0
        else float("nan")
    )

    trials = max(1, int(trial_count))
    adjusted_alpha = min(0.5, max(1e-6, float(alpha) / trials))
    z = NormalDist().inv_cdf(1.0 - adjusted_alpha / 2.0)
    ci_lower = mean_gain - z * se
    ci_upper = mean_gain + z * se
    required_gain = max(
        0.0,
        abs(comparator_mean) * float(min_relative_improvement),
    )

    block_bootstrap_probability, block_bootstrap_p05 = moving_block_bootstrap_mean(
        differences,
        n_bootstrap=4000,
        seed=20260925,
    )

    eligible = bool(
        isfinite(mean_gain)
        and isfinite(relative_improvement)
        and relative_improvement >= float(min_relative_improvement)
        and ci_lower >= required_gain
        and block_bootstrap_probability >= 0.90
        and block_bootstrap_p05 >= required_gain
    )
    status = "SUPPORTED" if eligible else "NOT_SIGNIFICANT"

    return {
        "status": status,
        "eligible_for_freeze": eligible,
        "method": "paired_oos_fold_logloss_ci_bonferroni_plus_moving_block_bootstrap",
        "selected_model": str(selected_model),
        "comparator_model": str(comparator_model),
        "common_folds": int(n),
        "minimum_common_folds": int(min_folds),
        "trial_count": trials,
        "alpha": float(alpha),
        "adjusted_alpha": float(adjusted_alpha),
        "z": float(z),
        "selected_mean_logloss": float(selected_mean),
        "comparator_mean_logloss": float(comparator_mean),
        "mean_logloss_improvement": float(mean_gain),
        "relative_logloss_improvement": float(relative_improvement),
        "paired_logloss_std": float(std),
        "paired_logloss_se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "block_bootstrap_probability_improvement": float(block_bootstrap_probability),
        "block_bootstrap_p05_improvement": float(block_bootstrap_p05),
        "block_bootstrap_block_length": int(max(1, min(len(differences), int(__import__("math").ceil(len(differences) ** (1.0 / 3.0)))))),
        "required_relative_improvement": float(min_relative_improvement),
        "required_absolute_improvement": float(required_gain),
        "evidence_note": (
            "Paired differences use only common chronological OOS folds. "
            "The frozen/blind holdout is excluded. This is a conservative "
            "selection-evidence gate, not a guarantee of future performance. "
            "The moving-block bootstrap preserves local chronological dependence "
            "when assessing uncertainty across OOS folds."
        ),
    }
