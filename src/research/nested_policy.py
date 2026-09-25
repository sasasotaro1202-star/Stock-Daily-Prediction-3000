from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from src.research.sequential_selection import select_prior_oos_model


def nested_sequential_policy_oos(
    fold_rows_by_model: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    outer_start_fold: int | None = None,
    min_history_folds: int = 3,
    half_life_folds: float = 4.0,
    stability_penalty: float = 0.25,
    baseline_model: str | None = None,
    min_outer_folds: int = 5,
) -> dict[str, object]:
    """Evaluate a fixed sequential selection policy on a later outer block.

    The policy hyperparameters are fixed before the outer block. For each
    outer fold, model selection uses only outcomes strictly before that fold.
    Earlier outer outcomes may be used for later outer decisions, matching
    genuine prequential operation. The outer block is never used to tune the
    policy hyperparameters.
    """
    if min_history_folds < 1:
        raise ValueError("min_history_folds must be >= 1")
    if min_outer_folds < 1:
        raise ValueError("min_outer_folds must be >= 1")
    if outer_start_fold is not None and outer_start_fold < 0:
        raise ValueError("outer_start_fold must be >= 0 when provided")

    row_maps: dict[str, dict[int, Mapping[str, object]]] = {}
    all_folds: set[int] = set()
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
                raise ValueError("duplicate fold for model in nested policy")
            mapped[fold] = raw_row
            all_folds.add(fold)
        row_maps[str(model_name)] = mapped

    sorted_folds = sorted(all_folds)
    if len(sorted_folds) < min_history_folds + min_outer_folds:
        return {
            "status": "INSUFFICIENT_OOS",
            "method": "nested_outer_prior_oos_sequential_model_selection",
            "reason": "insufficient_total_folds",
            "total_folds": len(sorted_folds),
            "outer_folds": 0,
            "research_only": True,
        }

    if outer_start_fold is None:
        outer_start_fold = sorted_folds[len(sorted_folds) // 2]
    outer_start_fold = int(outer_start_fold)

    outer_folds = [fold for fold in sorted_folds if fold >= outer_start_fold]
    inner_folds = [fold for fold in sorted_folds if fold < outer_start_fold]
    if len(inner_folds) < min_history_folds or len(outer_folds) < min_outer_folds:
        return {
            "status": "INSUFFICIENT_OOS",
            "method": "nested_outer_prior_oos_sequential_model_selection",
            "reason": "invalid_outer_split",
            "total_folds": len(sorted_folds),
            "inner_folds": len(inner_folds),
            "outer_folds": len(outer_folds),
            "outer_start_fold": outer_start_fold,
            "min_history_folds": min_history_folds,
            "min_outer_folds": min_outer_folds,
            "research_only": True,
        }

    history_by_model: dict[str, list[Mapping[str, object]]] = {}
    for model_name, rows in row_maps.items():
        history_by_model[model_name] = [
            rows[fold] for fold in inner_folds if fold in rows
        ]

    fixed_inner_baseline = baseline_model
    if baseline_model in (None, "auto_inner_static"):
        fixed_inner_baseline, _ = select_prior_oos_model(
            history_by_model,
            outer_start_fold,
            min_history_folds=min_history_folds,
            half_life_folds=half_life_folds,
            stability_penalty=stability_penalty,
        )

    outer_deltas: list[float] = []
    selected_rows: list[Mapping[str, object]] = []
    baseline_rows: list[Mapping[str, object]] = []
    decisions: list[dict[str, object]] = []

    for fold in outer_folds:
        selected, diagnostics = select_prior_oos_model(
            history_by_model,
            fold,
            min_history_folds=min_history_folds,
            half_life_folds=half_life_folds,
            stability_penalty=stability_penalty,
        )
        if selected is None or fold not in row_maps.get(selected, {}):
            continue

        selected_row = row_maps[selected][fold]
        selected_rows.append(selected_row)

        decision: dict[str, object] = {
            "fold": fold,
            "selected_model": str(selected),
            "history_max_fold": max(
                int(float(row["fold"]))
                for rows in history_by_model.values()
                for row in rows
            ),
            "history_folds": int(
                diagnostics[selected]["history_folds"]
            ),
            "selection_score": float(
                diagnostics[selected]["selection_score"]
            ),
        }

        if fixed_inner_baseline is not None and fixed_inner_baseline in row_maps:
            baseline = row_maps[fixed_inner_baseline].get(fold)
            if baseline is not None:
                baseline_rows.append(baseline)
                gain = float(baseline["logloss"]) - float(selected_row["logloss"])
                if np.isfinite(gain):
                    outer_deltas.append(gain)
                decision["baseline_model"] = str(fixed_inner_baseline)
                decision["baseline_logloss"] = float(baseline["logloss"])
                decision["logloss_gain_vs_baseline"] = gain

        decisions.append(decision)

        # Only after the outer fold has been scored do its outcomes become
        # available to the online policy for subsequent outer folds.
        for model_name, rows in row_maps.items():
            if fold in rows:
                history_by_model.setdefault(model_name, []).append(rows[fold])

    def _mean(rows: list[Mapping[str, object]], metric: str) -> float:
        values = []
        for row in rows:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else float("nan")

    selected_mean = _mean(selected_rows, "logloss")
    baseline_mean = _mean(baseline_rows, "logloss")
    improvement = (
        baseline_mean - selected_mean
        if np.isfinite(selected_mean) and np.isfinite(baseline_mean)
        else float("nan")
    )
    relative = (
        improvement / abs(baseline_mean)
        if np.isfinite(improvement) and baseline_mean != 0.0
        else float("nan")
    )

    deltas = np.asarray(outer_deltas, dtype=float)
    bootstrap_probability = 0.0
    bootstrap_p05 = float("-inf")
    if len(deltas) >= 5 and np.isfinite(deltas).all():
        rng = np.random.default_rng(20260925)
        idx = rng.integers(0, len(deltas), size=(2000, len(deltas)))
        boot = deltas[idx].mean(axis=1)
        bootstrap_probability = float(np.mean(boot > 0.0))
        bootstrap_p05 = float(np.quantile(boot, 0.05))

    positive_share = (
        float(np.mean(deltas > 0.0)) if len(deltas) else 0.0
    )

    research_positive = bool(
        len(deltas) >= min_outer_folds
        and np.isfinite(relative)
        and relative >= 0.03
        and positive_share >= 0.70
        and bootstrap_probability >= 0.90
        and bootstrap_p05 > 0.0
    )

    return {
        "status": "EVALUATED" if selected_rows else "INSUFFICIENT_OOS",
        "method": "nested_outer_prior_oos_sequential_model_selection",
        "research_only": True,
        "outer_start_fold": outer_start_fold,
        "inner_folds": len(inner_folds),
        "outer_folds": len(selected_rows),
        "min_history_folds": min_history_folds,
        "half_life_folds": float(half_life_folds),
        "stability_penalty": float(stability_penalty),
        "baseline_model": fixed_inner_baseline,
        "baseline_model_policy": str(baseline_model) if baseline_model is not None else "auto_inner_static",
        "selected_oos_logloss": selected_mean,
        "baseline_oos_logloss": baseline_mean,
        "logloss_improvement": improvement,
        "relative_logloss_improvement": relative,
        "positive_fold_share": positive_share,
        "bootstrap_probability_improvement": bootstrap_probability,
        "bootstrap_p05_improvement": bootstrap_p05,
        "research_positive": research_positive,
        "selected_model_by_outer_fold": decisions,
        "selection_hyperparameters_frozen_before_outer": True,
        "outer_outcomes_used_for_later_outer_adaptation": True,
    }
