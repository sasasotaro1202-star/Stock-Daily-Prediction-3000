from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from src.research.statistics import moving_block_bootstrap_mean


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
    model_fold_rows: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    min_history_folds: int = 3,
    production_identity: Mapping[str, object] | None = None,
    prediction_generation_training_window_sessions: int | None = 0,
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

    This function intentionally does not use a globally selected model or
    training-window value for the nested selection itself. An optional
    production_identity is used only after scoring to audit whether the nested
    evidence is aligned with the eventual frozen production configuration.
    """
    The prediction-bank window uses the same semantics as the research
    runner: 0 means the full eligible pre-test core history, subject only
    to the deterministic row-count cap. The recent_sessions=252 argument
    used by cap_training_rows is a sampling safeguard, not a 252-session
    training-window restriction.
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

    def _safe_float(value: object) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None

    def _rank_ic(
        values: np.ndarray,
        target: np.ndarray,
        group_keys: np.ndarray,
    ) -> float:
        vals: list[float] = []
        for key in np.unique(group_keys):
            mask = group_keys == key
            if int(mask.sum()) < 2:
                continue
            a = np.asarray(values[mask], dtype=float)
            b = np.asarray(target[mask], dtype=float)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                continue
            if np.allclose(a, a[0]) or np.allclose(b, b[0]):
                continue
            corr = float(np.corrcoef(a, b, rowvar=False)[0, 1])
            if np.isfinite(corr):
                vals.append(corr)
        return float(np.mean(vals)) if vals else float("nan")

    model_rows_normalized: dict[str, list[dict[str, object]]] = {}
    for name, rows in model_fold_rows.items():
        normalized: list[dict[str, object]] = []
        for row in rows:
            fold = _safe_float(row.get("fold"))
            loss = _safe_float(row.get("logloss"))
            if fold is None or loss is None:
                continue
            normalized.append({
                "fold": int(fold),
                "logloss": float(loss),
            })
        model_rows_normalized[str(name)] = normalized

    for fold in folds:
        bank = predictions_by_fold.get(fold) or {}
        y = np.asarray(bank.get("y", []), dtype=int)
        if y.size == 0:
            continue
        session_dates = np.asarray(bank.get("session_dates", []), dtype=str)
        asset_classes = (
            np.asarray(bank["asset_classes"], dtype=str)
            if bank.get("asset_classes") is not None
            else None
        )
        group_keys = (
            session_dates + "::" + asset_classes
            if asset_classes is not None
            else session_dates
        )

        model = (
            _select_prior_model(
                model_history,
                fold,
                min_history_folds=min_history_folds,
                half_life_folds=model_half_life_folds,
                stability_penalty=model_stability_penalty,
            )
            if fold >= min_history_folds
            else None
        )
        return_estimator = (
            _select_prior_return_estimator(
                return_history,
                fold,
                min_history_folds=min_history_folds,
            )
            if fold >= min_history_folds
            else None
        )

        # All ranking hyperparameters are selected from ranking outcomes of
        # earlier outer folds only. There is no access to the current fold
        # outcome when rank_weight/rank_penalty is chosen.
        rank_key = "0.50::0.00"
        if model is not None and return_estimator is not None:
            prior_scores: dict[str, float] = {}
            for key, rows in rank_history_by_candidate.items():
                prior = [
                    float(row["rank_ic"])
                    for row in rows
                    if int(row["fold"]) < fold
                    and np.isfinite(float(row["rank_ic"]))
                ]
                if len(prior) < min_history_folds:
                    continue
                values = np.asarray(prior, dtype=float)
                prior_scores[key] = float(
                    values.mean()
                    - 0.25 * (values.std(ddof=1) if values.size >= 2 else 0.0)
                )
            if prior_scores:
                rank_key = max(
                    prior_scores,
                    key=lambda key: (prior_scores[key], key),
                )

        rank_weight, rank_penalty = map(float, rank_key.split("::"))
        return_bank = (
            return_predictions_by_fold.get(fold, {}).get(return_estimator)
            if return_estimator is not None
            else None
        )
        probabilities = (
            np.asarray(bank.get("predictions", {}).get(model, []), dtype=float)
            if model is not None
            else np.empty((0,), dtype=float)
        )
        expected = (
            np.asarray(return_bank.get("pred", []), dtype=float)
            if return_bank is not None
            else np.empty((0,), dtype=float)
        )
        target_returns = (
            np.asarray(return_bank.get("y", []), dtype=float)
            if return_bank is not None
            else np.empty((0,), dtype=float)
        )
        interval = (
            np.asarray(return_bank.get("interval"), dtype=float)
            if return_bank is not None
            else np.empty((0, 2), dtype=float)
        )

        valid_current = (
            model is not None
            and return_estimator is not None
            and probabilities.size == y.size
            and expected.size == y.size
            and target_returns.size == y.size
            and interval.shape == (y.size, 2)
            and session_dates.size == y.size
        )
        if valid_current:
            uncertainty = np.maximum(interval[:, 1] - interval[:, 0], 0.0)
            score = _rank_score(
                probabilities,
                expected,
                uncertainty,
                session_dates,
                asset_classes,
                rank_weight,
                rank_penalty,
            )
            baseline_score = _rank_score(
                probabilities,
                expected,
                uncertainty,
                session_dates,
                asset_classes,
                0.50,
                0.0,
            )
            selected_rank_ic = _rank_ic(score, target_returns, group_keys)
            baseline_rank_ic = _rank_ic(
                baseline_score,
                target_returns,
                group_keys,
            )
            outer_rows.append({
                "fold": fold,
                "model": model,
                "return_estimator": return_estimator,
                "probability_weight": rank_weight,
                "uncertainty_penalty": rank_penalty,
                "rank_ic": selected_rank_ic,
                "baseline_rank_ic": baseline_rank_ic,
            })
            selected_model_by_fold[fold] = model
            selected_return_by_fold[fold] = return_estimator
            selected_rank_by_fold[fold] = (rank_weight, rank_penalty)

            # Current outcomes are appended only after the current fold was
            # scored, preserving strict prequential ordering.
            for name, rows in model_rows_normalized.items():
                row = next((r for r in rows if r["fold"] == fold), None)
                if row is not None:
                    model_history.setdefault(name, []).append(row)

            for estimator_name, estimator_bank in (
                return_predictions_by_fold.get(fold, {}) or {}
            ).items():
                pred = np.asarray(estimator_bank.get("pred", []), dtype=float)
                target = np.asarray(estimator_bank.get("y", []), dtype=float)
                if pred.size != y.size or target.size != y.size:
                    continue
                estimator_rank_ic = _rank_ic(pred, target, group_keys)
                if np.isfinite(estimator_rank_ic):
                    return_history.setdefault(str(estimator_name), []).append({
                        "fold": fold,
                        "rank_ic": estimator_rank_ic,
                    })

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
                candidate_rank_ic = _rank_ic(
                    candidate_score,
                    target_returns,
                    group_keys,
                )
                if np.isfinite(candidate_rank_ic):
                    rank_history_by_candidate[key].append({
                        "fold": fold,
                        "rank_ic": candidate_rank_ic,
                    })
        else:
            # Warmup still contributes prior model/return evidence but never
            # creates ranking-selection evidence for production.
            for name, rows in model_rows_normalized.items():
                row = next((r for r in rows if r["fold"] == fold), None)
                if row is not None:
                    model_history.setdefault(name, []).append(row)
            for estimator_name, estimator_bank in (
                return_predictions_by_fold.get(fold, {}) or {}
            ).items():
                pred = np.asarray(estimator_bank.get("pred", []), dtype=float)
                target = np.asarray(estimator_bank.get("y", []), dtype=float)
                if pred.size != y.size or target.size != y.size:
                    continue
                estimator_rank_ic = _rank_ic(pred, target, group_keys)
                if np.isfinite(estimator_rank_ic):
                    return_history.setdefault(str(estimator_name), []).append({
                        "fold": fold,
                        "rank_ic": estimator_rank_ic,
                    })

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
    bootstrap_probability = 0.0
    bootstrap_p05 = float("-inf")
    if finite.size >= 5 and np.isfinite(finite).all():
        bootstrap_probability, bootstrap_p05 = moving_block_bootstrap_mean(
            finite,
            n_bootstrap=4000,
            seed=20261004,
        )

    final_rank_parameters = None
    if outer_rows:
        last = outer_rows[-1]
        final_rank_parameters = {
            "probability_weight": float(last["probability_weight"]),
            "uncertainty_penalty": float(last["uncertainty_penalty"]),
            "selection_fold": int(last["fold"]),
            "model": str(last["model"]),
            "return_estimator": str(last["return_estimator"]),
        }

    production_identity_alignment = {
        "provided": production_identity is not None,
        "aligned": False,
        "checks": {
            "selected_model_matches_final_prequential_model": False,
            "selected_return_estimator_matches_final_prequential_estimator": False,
            "classifier_training_window_matches_prediction_generation": False,
            "rank_probability_weight_matches_final_prequential_parameter": False,
            "rank_uncertainty_penalty_matches_final_prequential_parameter": False,
        },
    }
    if production_identity is not None and final_rank_parameters is not None:
        production_model = str(production_identity.get("selected_model", "")).strip()
        production_return = str(production_identity.get("return_estimator", "")).strip()
        try:
            production_window = int(
                production_identity.get("classifier_training_window_sessions")
            )
        except (TypeError, ValueError):
            production_window = None
        try:
            production_weight = float(
                production_identity.get("rank_probability_weight")
            )
            production_penalty = float(
                production_identity.get("rank_uncertainty_penalty")
            )
        except (TypeError, ValueError):
            production_weight = None
            production_penalty = None

        production_identity_alignment["checks"] = {
            "selected_model_matches_final_prequential_model": (
                production_model == str(final_rank_parameters["model"])
            ),
            "selected_return_estimator_matches_final_prequential_estimator": (
                production_return
                == str(final_rank_parameters["return_estimator"])
            ),
            "classifier_training_window_matches_prediction_generation": (
                production_window is not None
                and prediction_generation_training_window_sessions is not None
                and production_window
                == int(prediction_generation_training_window_sessions)
            ),
            "rank_probability_weight_matches_final_prequential_parameter": (
                production_weight is not None
                and abs(
                    production_weight
                    - float(final_rank_parameters["probability_weight"])
                )
                <= 1e-12
            ),
            "rank_uncertainty_penalty_matches_final_prequential_parameter": (
                production_penalty is not None
                and abs(
                    production_penalty
                    - float(final_rank_parameters["uncertainty_penalty"])
                )
                <= 1e-12
            ),
        }
        production_identity_alignment["aligned"] = all(
            production_identity_alignment["checks"].values()
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
        "bootstrap_probability_improvement": bootstrap_probability,
        "bootstrap_p05_improvement": bootstrap_p05,
        "bootstrap_method": "moving_block",
        "final_prequential_ranking_parameters": final_rank_parameters,
        "production_identity_alignment": production_identity_alignment,
        "prediction_generation_training_window_sessions": (
            int(prediction_generation_training_window_sessions)
            if prediction_generation_training_window_sessions is not None
            else None
        ),
        "training_window_policy": "not_selected_from same OOS; ranking evaluation consumes fold-local model predictions",
        "same_oos_global_model_or_window_reuse": False,
        "ranking_weight_selection_prequential": True,
        "model_selection_prequential": True,
        "return_estimator_selection_prequential": True,
        "research_positive": bool(
            finite.size >= 5
            and positive_share >= 0.70
            and np.isfinite(mean_delta)
            and mean_delta > 0.0
            and bootstrap_probability >= 0.90
            and bootstrap_p05 > 0.0
        ),
    }
