from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


def _weighted_selection_score(
    rows: Sequence[Mapping[str, object]],
    current_fold: int,
    half_life_folds: float,
    stability_penalty: float,
) -> tuple[float, float, float]:
    folds = np.asarray([int(float(row["fold"])) for row in rows], dtype=float)
    losses = np.asarray([float(row["logloss"]) for row in rows], dtype=float)
    if (
        folds.size == 0
        or not np.isfinite(losses).all()
        or not np.isfinite(folds).all()
        or any(int(fold) >= current_fold for fold in folds)
    ):
        raise ValueError("sequential selector requires finite prior-fold rows only")
    weights = np.power(
        0.5, (float(current_fold) - folds) / float(half_life_folds)
    )
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise ValueError("invalid sequential selector weights")
    mean_loss = float(np.average(losses, weights=weights))
    variance = float(np.average((losses - mean_loss) ** 2, weights=weights))
    weighted_std = float(np.sqrt(max(variance, 0.0)))
    return (
        mean_loss + float(stability_penalty) * weighted_std,
        mean_loss,
        weighted_std,
    )


def select_prior_oos_model(
    fold_rows_by_model: Mapping[str, Iterable[Mapping[str, object]]],
    current_fold: int,
    *,
    min_history_folds: int = 3,
    half_life_folds: float = 4.0,
    stability_penalty: float = 0.25,
) -> tuple[str | None, dict[str, dict[str, float]]]:
    """Select a model for a fold using only prior chronological OOS outcomes."""
    if current_fold < 0:
        raise ValueError("current_fold must be >= 0")
    if min_history_folds < 1:
        raise ValueError("min_history_folds must be >= 1")
    if not np.isfinite(half_life_folds) or half_life_folds <= 0:
        raise ValueError("half_life_folds must be > 0")
    if stability_penalty < 0:
        raise ValueError("stability_penalty must be >= 0")

    diagnostics: dict[str, dict[str, float]] = {}
    for model_name, raw_rows in fold_rows_by_model.items():
        prior_rows = []
        seen_folds = set()
        for raw_row in raw_rows:
            try:
                fold = int(float(raw_row["fold"]))
                logloss = float(raw_row["logloss"])
            except (KeyError, TypeError, ValueError):
                continue
            if fold >= current_fold:
                raise ValueError(
                    "sequential model selector received non-prior fold history"
                )
            if fold in seen_folds or not np.isfinite(logloss):
                continue
            seen_folds.add(fold)
            prior_rows.append({"fold": float(fold), "logloss": logloss})
        if len(prior_rows) < min_history_folds:
            continue
        score, mean_loss, weighted_std = _weighted_selection_score(
            prior_rows,
            current_fold,
            half_life_folds,
            stability_penalty,
        )
        diagnostics[str(model_name)] = {
            "selection_score": float(score),
            "weighted_logloss": float(mean_loss),
            "weighted_std": float(weighted_std),
            "history_folds": float(len(prior_rows)),
        }

    if not diagnostics:
        return None, diagnostics
    selected = min(
        diagnostics,
        key=lambda name: (
            diagnostics[name]["selection_score"],
            diagnostics[name]["weighted_logloss"],
            str(name),
        ),
    )
    return selected, diagnostics


def chronological_policy_oos(
    fold_rows_by_model: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    min_history_folds: int = 3,
    half_life_folds: float = 4.0,
    stability_penalty: float = 0.25,
    baseline_model: str | None = None,
) -> dict[str, object]:
    """Evaluate prior-OOS model selection on untouched current folds.

    Each fold is scored only after the selector has made its decision from
    earlier folds. No current-fold outcome contributes to the selection.
    """
    row_maps: dict[str, dict[int, Mapping[str, object]]] = {}
    all_folds = set()
    for model_name, raw_rows in fold_rows_by_model.items():
        mapped: dict[int, Mapping[str, object]] = {}
        for raw_row in raw_rows:
            try:
                fold = int(float(raw_row["fold"]))
                logloss = float(raw_row["logloss"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(logloss):
                continue
            if fold in mapped:
                raise ValueError("duplicate fold for model in chronological policy")
            mapped[fold] = raw_row
            all_folds.add(fold)
        row_maps[str(model_name)] = mapped

    selected_rows: list[dict[str, float]] = []
    baseline_rows: list[dict[str, float]] = []
    decisions: list[dict[str, object]] = []
    for fold in sorted(all_folds):
        if fold < min_history_folds:
            continue
        selected, diagnostics = select_prior_oos_model(
            row_maps,
            fold,
            min_history_folds=min_history_folds,
            half_life_folds=half_life_folds,
            stability_penalty=stability_penalty,
        )
        if selected is None or fold not in row_maps[selected]:
            continue
        selected_row = row_maps[selected][fold]
        selected_rows.append({
            "fold": float(fold),
            "logloss": float(selected_row["logloss"]),
        })
        decision: dict[str, object] = {
            "fold": int(fold),
            "selected_model": str(selected),
            "history_folds": int(diagnostics[selected]["history_folds"]),
            "selection_score": float(diagnostics[selected]["selection_score"]),
        }
        if baseline_model is not None and baseline_model in row_maps:
            baseline_row = row_maps[baseline_model].get(fold)
            if baseline_row is not None:
                baseline_rows.append({
                    "fold": float(fold),
                    "logloss": float(baseline_row["logloss"]),
                })
                decision["baseline_model"] = str(baseline_model)
                decision["baseline_logloss"] = float(baseline_row["logloss"])
        decisions.append(decision)

    def _mean(rows: Sequence[Mapping[str, float]]) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([float(row["logloss"]) for row in rows]))

    selected_mean = _mean(selected_rows)
    baseline_mean = _mean(baseline_rows)
    delta = (
        baseline_mean - selected_mean
        if np.isfinite(selected_mean) and np.isfinite(baseline_mean)
        else float("nan")
    )
    relative = (
        delta / abs(baseline_mean)
        if np.isfinite(delta) and baseline_mean != 0.0
        else float("nan")
    )
    return {
        "status": "EVALUATED" if selected_rows else "INSUFFICIENT_OOS",
        "method": "prior_oos_sequential_model_selection",
        "min_history_folds": int(min_history_folds),
        "half_life_folds": float(half_life_folds),
        "stability_penalty": float(stability_penalty),
        "folds": int(len(selected_rows)),
        "selected_oos_logloss": selected_mean,
        "baseline_model": baseline_model,
        "baseline_oos_logloss": baseline_mean,
        "logloss_improvement": delta,
        "relative_logloss_improvement": relative,
        "selected_model_by_fold": decisions,
        "research_only": True,
    }
