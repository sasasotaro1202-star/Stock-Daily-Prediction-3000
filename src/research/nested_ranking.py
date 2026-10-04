from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _weighted_loss_score(
    fold_rows: Sequence[Mapping[str, object]],
    current_fold: int,
    half_life_folds: float,
    stability_penalty: float,
) -> float:
    if not fold_rows:
        return float("inf")
    folds = np.asarray([int(row["fold"]) for row in fold_rows], dtype=float)
    losses = np.asarray([float(row["logloss"]) for row in fold_rows], dtype=float)
    if (
        not np.isfinite(folds).all()
        or not np.isfinite(losses).all()
        or np.any(folds >= current_fold)
    ):
        raise ValueError("ranking model history must contain strictly prior folds")
    weights = np.power(
        0.5,
        (float(current_fold) - folds) / float(half_life_folds),
    )
    mean = float(np.average(losses, weights=weights))
    variance = float(np.average((losses - mean) ** 2, weights=weights))
    return mean + float(stability_penalty) * float(np.sqrt(max(variance, 0.0)))


def _select_prior_model(
    history: Mapping[str, list[dict[str, object]]],
    current_fold: int,
    *,
    min_history_folds: int,
    half_life_folds: float,
    stability_penalty: float,
) -> str | None:
    candidates: dict[str, float] = {}
    for name, rows in history.items():
        prior = [row for row in rows if int(row["fold"]) < current_fold]
        if len(prior) < min_history_folds:
            continue
        candidates[str(name)] = _weighted_loss_score(
            prior,
            current_fold,
            half_life_folds,
            stability_penalty,
        )
    if not candidates:
        return None
    return min(candidates, key=lambda name: (candidates[name], name))


def _select_prior_return_estimator(
    history: Mapping[str, list[dict[str, object]]],
    current_fold: int,
    *,
    min_history_folds: int,
) -> str | None:
    scores: dict[str, tuple[float, float]] = {}
    for name, rows in history.items():
        prior = [row for row in rows if int(row["fold"]) < current_fold]
        if len(prior) < min_history_folds:
            continue
        values = np.asarray(
            [float(row["rank_ic"]) for row in prior],
            dtype=float,
        )
        if not np.isfinite(values).all():
            continue
        scores[str(name)] = (
            float(np.mean(values) - 0.25 * np.std(values, ddof=1))
            if len(values) >= 2
            else float(values.mean()),
            float(np.mean(values)),
        )
    if not scores:
        return None
    return max(scores, key=lambda name: (scores[name][0], scores[name][1], name))


def _rank_score(
    probability: np.ndarray,
    expected: np.ndarray,
    uncertainty: np.ndarray,
    session_dates: np.ndarray,
    asset_classes: np.ndarray | None,
    probability_weight: float,
    uncertainty_penalty: float,
) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    expected = np.asarray(expected, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    session_dates = np.asarray(session_dates, dtype=str)
    if asset_classes is None:
        group_keys = session_dates
        groups = session_dates
    else:
        groups = np.asarray(asset_classes, dtype=str)
        group_keys = session_dates + "::" + groups

    result = np.full(probability.shape, np.nan, dtype=float)
    for key in np.unique(group_keys):
        mask = group_keys == key
        if mask.sum() == 0:
            continue
        def pct_rank(values: np.ndarray, ascending: bool) -> np.ndarray:
            order = np.argsort(values if ascending else -values, kind="mergesort")
            ranks = np.empty(len(values), dtype=float)
            ranks[order] = np.arange(1, len(values) + 1, dtype=float)
            return ranks / float(len(values))

        rp = pct_rank(probability[mask], ascending=False)
        re = pct_rank(expected[mask], ascending=False)
        ru = pct_rank(uncertainty[mask], ascending=True)
        result[mask] = (
            float(probability_weight) * rp
            + (1.0 - float(probability_weight)) * re
            - float(uncertainty_penalty) * ru
        )
    return result


def nested_prequential_ranking_oos(
    predictions_by_fold: Mapping[int, Mapping[str, Sequence[float]]],
    return_predictions_by_fold: Mapping[
        int, Mapping[str, Mapping[str, Sequence[float]]]
    ],
    *,
    min_history_folds: int = 3,
    model_half_life_folds: float = 4.0,
    model_stability_penalty: float = 0.25,
    weights: Sequence[float] = (0.25, 0.50, 0.75),
    uncertainty_penalties: Sequence[float] = (0.0, 0.05, 0.10),
) -> dict[str, object]:
    """Evaluate ranking selection with a genuinely prequential outer loop.

    At outer fold t:
      1. select the classifier only from model log-losses on folds < t;
      2. select the return estimator only from rank-IC on folds < t;
      3. select ranking weight/penalty only from ranking scores on earlier
         outer folds using the already-chosen model/return path;
      4. score the untouched current fold;
      5. only then make the current outcome available to later folds.

    This function intentionally does not accept a globally selected model or
    training-window value. It therefore cannot silently inherit same-OOS
    global model/window selection.
    """
    if min_history_folds < 1:
        raise ValueError("min_history_folds must be >= 1")
    if not np.isfinite(model_half_life_folds) or model_half_life_folds <= 0:
        raise ValueError("model_half_life_folds must be > 0")
    if model_stability_penalty < 0:
        raise ValueError("model_stability_penalty must be >= 0")

    folds = sorted(set(int(fold) for fold in predictions_by_fold))
    if len(folds) <= min_history_folds:
        return {
            "status": "INSUFFICIENT_OOS",
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
            "method": "nested_prequential_ranking_selection",
            "reason": "insufficient_folds",
            "folds": 0,
        }

    model_history: dict[str, list[dict[str, object]]] = {}
    return_history: dict[str, list[dict[str, object]]] = {}
    selected_model_by_fold: dict[int, str] = {}
    selected_return_by_fold: dict[int, str] = {}
    selected_rank_by_fold: dict[int, tuple[float, float]] = {}
    rank_history_by_candidate: dict[str, list[dict[str, object]]] = {
        f"{float(w):.2f}::{float(p):.2f}": []
        for w in weights
        for p in uncertainty_penalties
    }
    outer_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []

    for fold in folds:
        bank = predictions_by_fold.get(fold) or {}
        y = np.asarray(bank.get("y", []), dtype=int)
        if y.size == 0:
            continue
        model_rows = {
            name: rows
            for name, rows in model_history.items()
            if len([r for r in rows if int(r["fold"]) < fold]) >= min_history_folds
        }
        model = _select_prior_model(
            model_rows,
            fold,
            min_history_folds=min_history_folds,
            half_life_folds=model_half_life_folds,
            stability_penalty=model_stability_penalty,
        )
        return_estimator = _select_prior_return_estimator(
            return_history,
            fold,
            min_history_folds=min_history_folds,
        )
        if model is None or return_estimator is None:
            # Warmup folds are still admitted into history, but are not scored
            # as selection evidence because no prior decision was possible.
            model_losses = bank.get("model_loglosses") or {}
            for name, loss in model_losses.items():
                try:
                    loss = float(loss)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(loss):
                    model_history.setdefault(str(name), []).append(
                        {"fold": fold, "logloss": loss}
                    )
            return_metrics = bank.get("return_metrics") or {}
            for name, metric in return_metrics.items():
                try:
                    rank_ic = float(metric["rank_ic"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(rank_ic):
                    return_history.setdefault(str(name), []).append(
                        {"fold": fold, "rank_ic": rank_ic}
                    )
            continue

        prior_candidates = {
            key: rows
            for key, rows in rank_history_by_candidate.items()
            if len([r for r in rows if int(r["fold"]) < fold]) >= min_history_folds
        }
        if prior_candidates:
            rank_scores = {}
            for key, rows in prior_candidates.items():
                vals = np.asarray(
                    [
                        float(row["rank_ic"])
                        for row in rows
                        if int(row["fold"]) < fold
                        and np.isfinite(float(row["rank_ic"]))
                    ],
                    dtype=float,
                )
                if vals.size:
                    rank_scores[key] = float(
                        vals.mean() - 0.25 * (vals.std(ddof=1) if vals.size >= 2 else 0.0)
                    )
            rank_key = (
                max(rank_scores, key=lambda key: (rank_scores[key], key))
                if rank_scores
                else "0.50::0.00"
            )
        else:
            rank_key = "0.50::0.00"
        rank_weight, rank_penalty = map(float, rank_key.split("::"))

        probabilities = np.asarray(bank["predictions"][model], dtype=float)
        return_bank = return_predictions_by_fold.get(fold, {}).get(return_estimator)
        if return_bank is None:
            continue
        expected = np.asarray(return_bank["pred"], dtype=float)
        interval = np.asarray(return_bank.get("interval"), dtype=float)
        if interval.ndim != 2 or interval.shape[0] != expected.size or interval.shape[1] != 2:
            continue
        uncertainty = np.maximum(interval[:, 1] - interval[:, 0], 0.0)
        session_dates = np.asarray(bank["session_dates"], dtype=str)
        asset_classes = (
            np.asarray(bank["asset_classes"], dtype=str)
            if bank.get("asset_classes") is not None
            else None
        )
        score = _rank_score(
            probabilities,
            expected,
            uncertainty,
            session_dates,
            asset_classes,
            rank_weight,
            rank_penalty,
        )
        group_keys = (
            session_dates + "::" + asset_classes
            if asset_classes is not None
            else session_dates
        )
        deltas = []
        baseline_score = _rank_score(
            probabilities,
            expected,
            uncertainty,
            session_dates,
            asset_classes,
            0.50,
            0.0,
        )

        def rank_ic(values: np.ndarray) -> float:
            vals = []
            for key in np.unique(group_keys):
                mask = group_keys == key
                if mask.sum() < 2:
                    continue
                a = values[mask]
                b = None
                target = np.asarray(bank["target_returns"], dtype=float)
                b = target[mask]
                if np.allclose(a, a[0]) or np.allclose(b, b[0]):
                    continue
                vals.append(float(np.corrcoef(a, b, rowvar=False)[0, 1]))
            return float(np.mean(vals)) if vals else float("nan")

        selected_rank_ic = rank_ic(score)
        baseline_rank_ic = rank_ic(baseline_score)
        selected_rank_by_fold = {
            "fold": fold,
            "model": model,
            "return_estimator": return_estimator,
            "probability_weight": rank_weight,
            "uncertainty_penalty": rank_penalty,
            "rank_ic": selected_rank_ic,
            "baseline_rank_ic": baseline_rank_ic,
        }
        outer_rows.append(selected_rank_by_fold)
        if np.isfinite(selected_rank_ic) and np.isfinite(baseline_rank_ic):
            deltas.append(selected_rank_ic - baseline_rank_ic)
        if deltas:
            rank_key = f"{rank_weight:.2f}::{rank_penalty:.2f}"

        # Current fold is added to model/return/ranking histories only after
        # its untouched OOS score has been recorded.
        model_losses = bank.get("model_loglosses") or {}
        for name, loss in model_losses.items():
            try:
                loss = float(loss)
            except (TypeError, ValueError):
                continue
            if np.isfinite(loss):
                model_history.setdefault(str(name), []).append(
                    {"fold": fold, "logloss": loss}
                )

        return_metrics = bank.get("return_metrics") or {}
        for name, metric in return_metrics.items():
            try:
                value = float(metric["rank_ic"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                return_history.setdefault(str(name), []).append(
                    {"fold": fold, "rank_ic": value}
                )

        for key in rank_history_by_candidate:
            candidate_weight, candidate_penalty = map(float, key.split("::"))
            candidate_score = _rank_score(
                probabilities,
                expected,
                uncertainty,
                session_dates,
                asset_classes,
                candidate_weight,
                candidate_penalty,
            )
            candidate_rank_ic = rank_ic(candidate_score)
            if np.isfinite(candidate_rank_ic):
                rank_history_by_candidate[key].append(
                    {"fold": fold, "rank_ic": candidate_rank_ic}
                )
        selected_model_by_fold[fold] = model
        selected_return_by_fold[fold] = return_estimator
        selected_rank_by_fold[fold] = (rank_weight, rank_penalty)

    if len(outer_rows) < min_history_folds:
        return {
            "status": "INSUFFICIENT_OOS",
            "research_only": True,
            "production_changed": False,
            "promotion_allowed": False,
            "method": "nested_prequential_ranking_selection",
            "reason": "insufficient_outer_decisions",
            "folds": len(outer_rows),
        }

    selected_deltas = np.asarray(
        [float(r["rank_ic"] - r["baseline_rank_ic"]) for r in outer_rows],
        dtype=float,
    )
    finite = selected_deltas[np.isfinite(selected_deltas)]
    positive_share = float(np.mean(finite > 0.0)) if finite.size else 0.0
    mean_delta = float(np.mean(finite)) if finite.size else float("nan")
    baseline_mean = float(
        np.mean(
            [
                float(r["baseline_rank_ic"])
                for r in outer_rows
                if np.isfinite(float(r["baseline_rank_ic"]))
            ]
        )
    )
    relative = (
        mean_delta / abs(baseline_mean)
        if np.isfinite(mean_delta) and np.isfinite(baseline_mean) and baseline_mean != 0.0
        else float("nan")
    )

    return {
        "status": "EVALUATED",
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "method": "nested_prequential_ranking_selection",
        "folds": len(outer_rows),
        "min_history_folds": min_history_folds,
        "model_half_life_folds": float(model_half_life_folds),
        "model_stability_penalty": float(model_stability_penalty),
        "outer_metrics": outer_rows,
        "selected_model_by_fold": selected_model_by_fold,
        "selected_return_estimator_by_fold": selected_return_by_fold,
        "selected_ranking_parameters_by_fold": {
            str(fold): {
                "probability_weight": float(values[0]),
                "uncertainty_penalty": float(values[1]),
            }
            for fold, values in sorted(selected_rank_by_fold.items())
        },
        "mean_rank_ic_improvement_vs_fixed_baseline": mean_delta,
        "relative_rank_ic_improvement_vs_fixed_baseline": relative,
        "positive_fold_share": positive_share,
        "training_window_policy": "not_selected_from_same OOS; ranking evaluation consumes fold-local model predictions",
        "same_oos_global_model_or_window_reuse": False,
        "ranking_weight_selection_prequential": True,
        "model_selection_prequential": True,
        "return_estimator_selection_prequential": True,
        "research_positive": bool(
            finite.size >= 5
            and positive_share >= 0.70
            and np.isfinite(mean_delta)
            and mean_delta > 0.0
        ),
    }
