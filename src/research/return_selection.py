from __future__ import annotations

import math


def choose_return_estimator(
    candidates: dict[str, dict[str, float]],
    *,
    min_folds: int = 3,
    mae_guard: float = 1.10,
    stability_penalty: float = 0.25,
    rank_ic_tolerance: float = 0.005,
) -> str:
    """Select an expected-return estimator from chronological OOS evidence.

    Rank IC is the primary objective because expected return feeds
    cross-sectional ranking. MAE is a guardrail and fold instability is
    penalized before the final tie-break.
    """
    usable = {}
    for name, metric in candidates.items():
        try:
            folds = int(metric.get("folds", 0))
            mae = float(metric["mae"])
            rank_ic = float(metric["rank_ic"])
            rank_ic_std = float(metric.get("rank_ic_std", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if (
            folds < min_folds
            or not math.isfinite(mae)
            or not math.isfinite(rank_ic)
            or not math.isfinite(rank_ic_std)
        ):
            continue
        usable[name] = {
            "mae": mae,
            "rank_ic": rank_ic,
            "rank_ic_score": rank_ic - stability_penalty * rank_ic_std,
            "folds": folds,
        }

    if not usable:
        return "q50"

    best_mae = min(v["mae"] for v in usable.values())
    eligible = {
        name: metric
        for name, metric in usable.items()
        if metric["mae"] <= best_mae * max(1.0, float(mae_guard))
    }
    if not eligible:
        eligible = usable

    best_score = max(v["rank_ic_score"] for v in eligible.values())
    tied = [
        (name, metric)
        for name, metric in eligible.items()
        if metric["rank_ic_score"] >= best_score - max(0.0, float(rank_ic_tolerance))
    ]
    tied.sort(
        key=lambda item: (
            item[1]["mae"],
            -item[1]["rank_ic_score"],
            item[0],
        )
    )
    return tied[0][0]
